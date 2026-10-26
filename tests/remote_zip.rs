//! A zip read in byte ranges (`ome_zarr::remote_zip`): members come out as
//! written, from small ranges only, and an image streams out of one.

use std::cell::RefCell;
use std::io::{Cursor, Write};
use std::sync::Arc;

use futures::executor::block_on;
use ome_zarr::array::{DType, LazyArray, NdArray};
use ome_zarr::classes::{OmeZarrImage, OmeZarrMultiscale, Version};
use ome_zarr::lazy::{LazyStore, Progress, settle};
use ome_zarr::remote_zip::{Opened, RemoteZip, Tail};
use ome_zarr::store::Store;
use ome_zarr::writer::GroupLike;
use ome_zarr::zarr::{Group, Mode};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

/// `files` zipped, each stored or deflated as asked; zip64 records if `large`.
fn archive(files: &[(&str, Vec<u8>, bool)], large: bool, comment: &str) -> Vec<u8> {
    let mut buf = Cursor::new(Vec::new());
    let mut w = zip::ZipWriter::new(&mut buf);
    for (name, data, deflate) in files {
        let method = if *deflate {
            zip::CompressionMethod::Deflated
        } else {
            zip::CompressionMethod::Stored
        };
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(method)
            .large_file(large);
        w.start_file(*name, opts).unwrap();
        w.write_all(data).unwrap();
    }
    w.set_comment(comment);
    w.finish().unwrap();
    buf.into_inner()
}

/// A server over `bytes`: suffix and span reads, each logged (start, len).
struct Server {
    bytes: Vec<u8>,
    log: RefCell<Vec<(u64, u64)>>,
    ranges: bool,
}

impl Server {
    fn new(bytes: Vec<u8>) -> Server {
        Server {
            bytes,
            log: RefCell::new(vec![]),
            ranges: true,
        }
    }
    fn tail(&self, n: u64) -> Result<Tail, String> {
        if !self.ranges {
            return Ok(Tail::Whole(self.bytes.clone()));
        }
        let start = (self.bytes.len() as u64).saturating_sub(n);
        self.log.borrow_mut().push((start, n));
        Ok(Tail::Suffix(self.bytes[start as usize..].to_vec()))
    }
    fn range(&self, start: u64, len: u64) -> Result<Vec<u8>, String> {
        self.log.borrow_mut().push((start, len));
        let end = ((start + len) as usize).min(self.bytes.len());
        Ok(self
            .bytes
            .get(start as usize..end)
            .ok_or("past the end")?
            .to_vec())
    }
    fn open(&self) -> Opened {
        block_on(RemoteZip::open(
            &|n| std::future::ready(self.tail(n)),
            &|s, l| std::future::ready(self.range(s, l)),
        ))
        .unwrap()
    }
    fn get(&self, z: &RemoteZip, key: &str) -> Option<Vec<u8>> {
        block_on(z.get(key, &|s, l| std::future::ready(self.range(s, l)))).unwrap()
    }
}

fn pattern(n: usize, seed: u8) -> Vec<u8> {
    (0..n)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

#[test]
fn members_from_small_ranges() {
    for large in [false, true] {
        let files = [
            ("a/zarr.json", b"{\"zarr_format\":3}".to_vec(), false),
            ("a/s0/c/0", pattern(200_000, 1), true),
            ("a/s0/c/1", pattern(5000, 2), false),
        ];
        let server = Server::new(archive(&files, large, "a comment"));
        let Opened::Index(z) = server.open() else {
            panic!("whole")
        };
        assert_eq!(
            server.log.borrow().len(),
            1,
            "the index from the tail alone"
        );
        let mut names: Vec<_> = z.names().collect();
        names.sort();
        assert_eq!(names, ["a/s0/c/0", "a/s0/c/1", "a/zarr.json"]);
        for (name, data, _) in &files {
            server.log.borrow_mut().clear();
            assert_eq!(
                server.get(&z, name).as_ref(),
                Some(data),
                "{name} (zip64 {large})"
            );
            let log = server.log.borrow();
            assert_eq!(log.len(), 1, "one range for {name}");
            assert!(
                log[0].1 < z.member(name).unwrap().compressed + 400,
                "{name}: a range just over the member"
            );
        }
        assert_eq!(server.get(&z, "a/nothing"), None);
    }
}

#[test]
fn a_central_directory_past_the_tail() {
    // more than 64 KiB of central directory: read with a second range
    let names: Vec<String> = (0..1500)
        .map(|i| format!("store.zarr/s0/c/{i}/0/0/a-long-name-to-make-the-directory-large"))
        .collect();
    let files: Vec<(&str, Vec<u8>, bool)> = names
        .iter()
        .map(|n| (n.as_str(), pattern(10, n.len() as u8), false))
        .collect();
    let server = Server::new(archive(&files, false, ""));
    let Opened::Index(z) = server.open() else {
        panic!("whole")
    };
    assert_eq!(server.log.borrow().len(), 2);
    assert_eq!(z.names().count(), 1500);
    assert_eq!(server.get(&z, &names[777]), Some(files[777].1.clone()));
}

#[test]
fn a_server_without_ranges_sends_it_whole() {
    let mut server = Server::new(archive(&[("x", vec![1, 2, 3], false)], false, ""));
    server.ranges = false;
    assert!(matches!(server.open(), Opened::Whole(b) if b == server.bytes));
}

#[test]
fn an_image_streams_out_of_a_ranged_zip() {
    // a pyramid zipped (stored, as OME-Zarr zips are), read lazily
    let shape = [8usize, 64, 64];
    let n: usize = shape.iter().product();
    let data = NdArray::from_f64_vec(
        DType::UInt16,
        &shape,
        (0..n).map(|i| (i % 1000) as f64).collect(),
    )
    .unwrap();
    let img = OmeZarrImage::from_axes(LazyArray::from_ndarray(data.clone()), "zyx").unwrap();
    let ms = OmeZarrMultiscale::builder(img).build().unwrap();
    let src = Store::memory();
    let g = Group::open(&src, "img.zarr", Mode::Write, Some(3)).unwrap();
    ms.to_ome_zarr(GroupLike::Group(g), None, Version::V06, true, true)
        .unwrap();
    let mut keys = vec![];
    let mut todo = vec!["img.zarr".to_string()];
    while let Some(p) = todo.pop() {
        let (files, dirs) = src.list_dir(&p).unwrap();
        keys.extend(files.into_iter().map(|f| format!("{p}/{f}")));
        todo.extend(dirs.into_iter().map(|d| format!("{p}/{d}")));
    }
    let files: Vec<(&str, Vec<u8>, bool)> = keys
        .iter()
        .map(|k| (k.as_str(), src.get(k).unwrap().unwrap().to_vec(), false))
        .collect();
    let server = Server::new(archive(&files, false, ""));
    let Opened::Index(z) = server.open() else {
        panic!("whole")
    };

    let lazy = Arc::new(LazyStore::new());
    let store = lazy.store();
    let range = |s, l| std::future::ready(server.range(s, l));
    let (z, range) = (&z, &range);
    let fetch = |k: String| async move { z.get(&format!("img.zarr/{k}"), range).await };
    let mut p = Progress::default();
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
        8,
        &mut |_| {},
        &mut p,
    ))
    .unwrap();
    let level = ms.images()[0].data.clone();
    let got = block_on(settle(
        &lazy,
        || level.compute(),
        &fetch,
        8,
        &mut |_| {},
        &mut p,
    ))
    .unwrap();
    assert_eq!(got, data);
    // after the tail, every range is a member's, nothing like the archive
    let biggest = server
        .log
        .borrow()
        .iter()
        .skip(1)
        .map(|r| r.1)
        .max()
        .unwrap();
    assert!(
        biggest < server.bytes.len() as u64 / 2,
        "{biggest} of {}",
        server.bytes.len()
    );
}
