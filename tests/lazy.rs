//! Stores fetched as they are read (`ome_zarr::lazy`): only the metadata
//! looked at and the chunks read are fetched, from either Zarr format.

use std::cell::RefCell;
use std::sync::Arc;

use futures::executor::block_on;
use ome_zarr::array::{DType, LazyArray, NdArray};
use ome_zarr::classes::{OmeZarrImage, OmeZarrMultiscale, Version};
use ome_zarr::lazy::{Fetched, LazyStore, Progress, https_url, is_metadata, root_of, settle};
use ome_zarr::scale::ScaleFactors;
use ome_zarr::store::{Store, ZipStore};
use ome_zarr::writer::GroupLike;
use ome_zarr::zarr::{Group, Mode};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

/// A 3-level `zyx` pyramid written in `version`'s format to a memory store.
fn remote(version: Version) -> (Store, NdArray) {
    let shape = [8usize, 96, 96];
    let n: usize = shape.iter().product();
    let data = NdArray::from_f64_vec(
        DType::UInt16,
        &shape,
        (0..n).map(|i| (i % 4093) as f64).collect(),
    )
    .unwrap();
    let img = OmeZarrImage::from_axes(LazyArray::from_ndarray(data.clone()), "zyx").unwrap();
    let ms = OmeZarrMultiscale::builder(img)
        .scale_factors(Some(ScaleFactors::Ints(vec![2, 4])))
        .build()
        .unwrap();
    let store = Store::memory();
    let fmt = if version == Version::V04 { 2 } else { 3 };
    let g = Group::open(&store, "", Mode::Write, Some(fmt)).unwrap();
    ms.to_ome_zarr(GroupLike::Group(g), None, version, true, true)
        .unwrap();
    (store, data)
}

#[test]
fn reads_one_level_fetching_only_its_chunks() {
    for version in [Version::V04, Version::V05, Version::V06] {
        let (remote, data) = remote(version);
        let asked = RefCell::new(Vec::<String>::new());
        let fetch = |k: String| {
            asked.borrow_mut().push(k.clone());
            let r: Fetched = Ok(remote.get(&k).unwrap().map(|b| b.to_vec()));
            async move { r }
        };
        let lazy = Arc::new(LazyStore::new());
        let store = lazy.store();
        let mut p = Progress::default();
        let mut heard = 0;
        let ms = block_on(settle(
            &lazy,
            || {
                OmeZarrMultiscale::from_ome_zarr(GroupLike::Group(Group::open(
                    &store,
                    "",
                    Mode::Read,
                    None,
                )?))
            },
            &fetch,
            4,
            &mut |_| heard += 1,
            &mut p,
        ))
        .unwrap();
        assert_eq!(ms.images().len(), 3);
        assert!(
            asked.borrow().iter().all(|k| is_metadata(k)),
            "{version:?}: {:?}",
            asked.borrow()
        );
        // once the format is known, the other format's documents are not asked for
        let other = |k: &String| {
            if version == Version::V04 {
                k.ends_with("zarr.json")
            } else {
                k.ends_with(".zattrs") || k.ends_with(".zarray")
            }
        };
        let probes = asked.borrow().iter().filter(|k| other(k)).count();
        assert!(
            probes <= 2,
            "{version:?}: {probes} probes of the other format: {:?}",
            asked.borrow()
        );

        let before = asked.borrow().len();
        let full = block_on(settle(
            &lazy,
            || ms.images()[0].data.compute(),
            &fetch,
            4,
            &mut |_| heard += 1,
            &mut p,
        ))
        .unwrap();
        assert_eq!(full, data);
        let chunks = asked.borrow()[before..].to_vec();
        assert!(
            !chunks.is_empty() && chunks.iter().all(|k| !is_metadata(k)),
            "{chunks:?}"
        );
        let level = |k: &String| k.split('/').next().unwrap().to_string();
        let first = level(&chunks[0]);
        assert!(
            chunks.iter().all(|k| level(k) == first),
            "{version:?}: one level's chunks only: {chunks:?}"
        );
        assert_eq!(
            (p.pending, p.files, heard),
            (0, asked.borrow().len(), asked.borrow().len())
        );
    }
}

#[test]
fn over_a_zip_nothing_is_missing() {
    let (remote, data) = remote(Version::V05);
    // the pyramid zipped, as a host would hand over a dropped .zip
    let mut buf = std::io::Cursor::new(Vec::new());
    {
        let mut w = zip::ZipWriter::new(&mut buf);
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        let mut keys = vec![String::new()];
        while let Some(prefix) = keys.pop() {
            let (files, dirs) = remote.list_dir(&prefix).unwrap();
            for f in files {
                let k = if prefix.is_empty() {
                    f
                } else {
                    format!("{prefix}/{f}")
                };
                w.start_file(&k, opts).unwrap();
                std::io::Write::write_all(&mut w, &remote.get(&k).unwrap().unwrap()).unwrap();
            }
            keys.extend(dirs.into_iter().map(|d| {
                if prefix.is_empty() {
                    d
                } else {
                    format!("{prefix}/{d}")
                }
            }));
        }
        w.finish().unwrap();
    }
    let lazy = Arc::new(LazyStore::over(Arc::new(
        ZipStore::from_bytes(buf.into_inner()).unwrap(),
    )));
    let g = Group::open(&lazy.store(), "", Mode::Read, None).unwrap();
    let ms = OmeZarrMultiscale::from_ome_zarr(GroupLike::Group(g)).unwrap();
    assert_eq!(ms.images()[0].data.compute().unwrap(), data);
    assert!(!lazy.has_missing());
}

#[test]
fn urls_and_roots() {
    assert_eq!(
        https_url("s3://bucket/a/b.zarr/").unwrap(),
        "https://bucket.s3.amazonaws.com/a/b.zarr"
    );
    assert_eq!(
        https_url("gs://bucket/b.zarr").unwrap(),
        "https://storage.googleapis.com/bucket/b.zarr"
    );
    assert_eq!(
        https_url(" https://x.org/s.zarr/ ").unwrap(),
        "https://x.org/s.zarr"
    );
    assert!(https_url("ftp://x").is_err());
    assert_eq!(
        root_of(["a.zarr/s0/zarr.json", "a.zarr/zarr.json", "a.zarr/s0/c/0"]).unwrap(),
        "a.zarr/"
    );
    assert_eq!(root_of(["zarr.json", "0/.zattrs"]).unwrap(), "");
    assert!(root_of(["readme.txt"]).is_err());
}

#[test]
fn prefetching_opens_in_two_round_trips() {
    let (remote, _) = remote(Version::V06);
    let fetch = |k: String| {
        let r: Fetched = Ok(remote.get(&k).unwrap().map(|b| b.to_vec()));
        async move { r }
    };
    let rounds = |prefetch: bool| {
        let lazy = Arc::new(LazyStore::new());
        let store = lazy.store();
        let mut p = Progress::default();
        block_on(settle(
            &lazy,
            || {
                let g = Group::open(&store, "", Mode::Read, None)?;
                if prefetch {
                    lazy.prefetch_multiscales("", &g.attrs()?, g.zarr_format());
                }
                OmeZarrMultiscale::from_ome_zarr(GroupLike::Group(g))
            },
            &fetch,
            8,
            &mut |_| {},
            &mut p,
        ))
        .unwrap();
        p.rounds
    };
    // the root, then each of 3 levels and the labels one round each; with
    // the prefetch, the root and then all of them at once
    assert!(rounds(false) >= 4);
    assert_eq!(rounds(true), 2);
}
