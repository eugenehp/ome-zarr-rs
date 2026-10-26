//! Byte-level parity with ome-zarr-py: the stores in `tests/fixtures/py` were
//! written by `scripts/gen_zarr_fixtures.py`; here the same pipelines run in
//! Rust and every metadata document and array must match.
#![cfg(not(target_arch = "wasm32"))]

use std::path::{Path, PathBuf};

use ome_zarr::array::LazyArray;
use ome_zarr::classes::{
    ChannelColor, LabelsArg, OmeZarrImage, OmeZarrLabels, OmeZarrMultiscale, ScaleMap, Version,
};
use ome_zarr::format::Format;
use ome_zarr::scale::{Methods, ScaleFactors};
use ome_zarr::store::Store;
use ome_zarr::writer::{self, ImageOptions, StorageOptions};
use ome_zarr::zarr::{Group, Mode, ZArray};
use serde_json::{Value, json};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/py")
        .join(name)
}

fn read_input(store: &Path, path: &str) -> LazyArray {
    let s = Store::local(store, true).unwrap();
    // the Python fixtures downsampled in-memory numpy arrays (auto-chunked)
    LazyArray::from_ndarray(ZArray::open(&s, path).unwrap().read_all().unwrap())
}

/// Remove random label colours before comparing.
fn mask(v: &mut Value) {
    match v {
        Value::Object(m) => {
            if m.contains_key("rgba") {
                m.insert("rgba".into(), json!("masked"));
            }
            for (_, x) in m.iter_mut() {
                mask(x);
            }
        }
        Value::Array(a) => a.iter_mut().for_each(mask),
        _ => {}
    }
}

fn walk(dir: &Path, base: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            walk(&p, base, out);
        } else {
            out.push(p.strip_prefix(base).unwrap().to_path_buf());
        }
    }
}

/// Compare two stores: the same files; every metadata document byte-identical
/// (modulo random label colours); every array's data equal.
fn compare_stores(py: &Path, rs: &Path) {
    let mut files = vec![];
    walk(py, py, &mut files);
    files.sort();
    let mut rs_files = vec![];
    walk(rs, rs, &mut rs_files);
    rs_files.sort();
    assert_eq!(
        files,
        rs_files,
        "different files in {} and {}",
        py.display(),
        rs.display()
    );
    let mut checked_arrays = 0;
    for f in &files {
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        if !matches!(
            name.as_str(),
            "zarr.json" | ".zattrs" | ".zgroup" | ".zarray"
        ) {
            continue;
        }
        let py_bytes = std::fs::read(py.join(f)).unwrap();
        let rs_bytes = std::fs::read(rs.join(f)).unwrap();
        let a: Value = serde_json::from_slice(&py_bytes).unwrap();
        let b: Value = serde_json::from_slice(&rs_bytes).unwrap();
        // our serializer reproduces Python's text for Python's own documents
        assert_eq!(
            ome_zarr::pyjson::dumps(&a, Some(2)).as_bytes(),
            py_bytes.as_slice(),
            "pyjson round trip of {}",
            f.display()
        );
        let (mut am, mut bm) = (a.clone(), b);
        mask(&mut am);
        mask(&mut bm);
        let (ta, tb) = (
            ome_zarr::pyjson::dumps(&am, Some(2)),
            ome_zarr::pyjson::dumps(&bm, Some(2)),
        );
        if ta != tb {
            let (la, lb): (Vec<&str>, Vec<&str>) = (ta.lines().collect(), tb.lines().collect());
            let i = la
                .iter()
                .zip(&lb)
                .position(|(x, y)| x != y)
                .unwrap_or(la.len().min(lb.len()));
            let ctx = |l: &[&str]| l[i.saturating_sub(3)..(i + 4).min(l.len())].join("\n");
            panic!(
                "metadata differs in {} at line {i}:\n--- python\n{}\n--- rust\n{}",
                f.display(),
                ctx(&la),
                ctx(&lb)
            );
        }
        if am == a {
            assert!(py_bytes == rs_bytes, "file bytes differ in {}", f.display());
        }
        let is_array =
            name == ".zarray" || a.get("node_type").and_then(|n| n.as_str()) == Some("array");
        if is_array {
            let arr_path = f.parent().unwrap().to_string_lossy().to_string();
            let pa = ZArray::open(&Store::local(py, true).unwrap(), &arr_path)
                .unwrap()
                .read_all()
                .unwrap();
            let ra = ZArray::open(&Store::local(rs, true).unwrap(), &arr_path)
                .unwrap()
                .read_all()
                .unwrap();
            assert_eq!(pa.dtype(), ra.dtype(), "dtype {arr_path}");
            assert!(
                pa == ra,
                "array data differs in {} ({})",
                arr_path,
                py.display()
            );
            checked_arrays += 1;
        }
    }
    assert!(checked_arrays > 0, "no arrays compared in {}", py.display());
}

fn tmp() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

fn czyx() -> Vec<String> {
    "czyx".chars().map(|c| c.to_string()).collect()
}

#[test]
fn class_api_all_versions_and_methods() {
    for v in ["0.4", "0.5", "0.6"] {
        for method in ["resize", "nearest", "local_mean", "zoom"] {
            let name = format!("class_{v}_{method}.zarr");
            let py = fixture(&name);
            let data = read_input(&py, "s0");
            let img = OmeZarrImage::new(
                data,
                &czyx(),
                Some(ScaleMap::from([("z", 2.0), ("y", 0.5), ("x", 0.5)])),
                Some(
                    [
                        ("y".to_string(), "micrometer".to_string()),
                        ("x".to_string(), "micrometer".to_string()),
                    ]
                    .into(),
                ),
                None,
                "image",
            )
            .unwrap();
            let ms = OmeZarrMultiscale::builder(img)
                .scale_factors(Some(ScaleFactors::Ints(vec![2, 4])))
                .method(Some(Methods::parse(method).unwrap()))
                .channel_names(&["A", "B"])
                .channel_colors(vec![
                    ChannelColor::from("FF0000"),
                    ChannelColor::Rgb(vec![0, 255, 0]),
                ])
                .contrast_limits(vec![(0.0, 20.0), (1.0, 30.0)])
                .build()
                .unwrap();
            let dir = tmp();
            let out = dir.path().join(&name);
            ms.to_ome_zarr(out.as_path(), None, Version::parse(v).unwrap(), true, true)
                .unwrap();
            compare_stores(&py, &out);
        }
    }
}

#[test]
fn chunked_downsampling_matches_dask() {
    for method in ["resize", "nearest", "local_mean", "zoom"] {
        let name = format!("chunked_{method}.zarr");
        let py = fixture(&name);
        let data = read_input(&py, "s0").rechunk(&[1, 32, 32]);
        let axes: Vec<String> = vec!["c".into(), "y".into(), "x".into()];
        let img = OmeZarrImage::new(data, &axes, None, None, None, "image").unwrap();
        let sf = ScaleFactors::per_dim(vec![
            vec![("y", 2.0), ("x", 2.0)],
            vec![("y", 4.0), ("x", 3.0)],
        ]);
        let ms = OmeZarrMultiscale::builder(img)
            .scale_factors(Some(sf))
            .method(Some(Methods::parse(method).unwrap()))
            .build()
            .unwrap();
        let dir = tmp();
        let out = dir.path().join(&name);
        ms.to_ome_zarr(out.as_path(), None, Version::V05, true, true)
            .unwrap();
        compare_stores(&py, &out);
    }
}

#[test]
fn float32_pyramid() {
    let py = fixture("float32.zarr");
    let data = read_input(&py, "s0");
    let img = OmeZarrImage::from_axes(data, "yx").unwrap();
    let ms = OmeZarrMultiscale::builder(img)
        .scale_factors(Some(ScaleFactors::Ints(vec![2, 4, 8])))
        .build()
        .unwrap();
    let dir = tmp();
    let out = dir.path().join("float32.zarr");
    ms.to_ome_zarr(out.as_path(), None, Version::V06, true, true)
        .unwrap();
    compare_stores(&py, &out);
}

#[test]
fn class_api_with_labels() {
    let py = fixture("with_labels.zarr");
    let data = read_input(&py, "s0");
    let lbl = read_input(&py, "labels/lab/s0");
    let li = OmeZarrImage::new(lbl, &czyx(), None, None, None, "lab").unwrap();
    let labels = OmeZarrLabels::new(
        li,
        Some(ScaleFactors::Ints(vec![2])),
        Some(Methods::Nearest),
        true,
    )
    .unwrap();
    let ms = OmeZarrMultiscale::builder(
        OmeZarrImage::new(data, &czyx(), None, None, None, "image").unwrap(),
    )
    .scale_factors(Some(ScaleFactors::Ints(vec![2])))
    .labels(LabelsArg::One(labels))
    .build()
    .unwrap();
    let dir = tmp();
    let out = dir.path().join("with_labels.zarr");
    ms.to_ome_zarr(out.as_path(), None, Version::V05, true, true)
        .unwrap();
    compare_stores(&py, &out);
}

#[test]
fn writer_api() {
    for fmt in [Format::V04, Format::V05] {
        let name = format!("writer_{}.zarr", fmt.version());
        let py = fixture(&name);
        let data = read_input(&py, "s0");
        let lbl = read_input(&py, "labels/cells/s0");
        let dir = tmp();
        let out = dir.path().join(&name);
        let g =
            Group::open_url(out.to_str().unwrap(), Mode::Write, Some(fmt.zarr_format())).unwrap();
        let opts = ImageOptions {
            axes: Some("czyx".into()),
            storage_options: Some(StorageOptions::chunks(&[16, 16]).into()),
            scale: Some(ScaleMap::from([("y", 0.25), ("x", 0.25)])),
            fmt: Some(fmt),
            ..Default::default()
        };
        writer::write_image(&data, &g, &opts).unwrap();
        let lopts = ImageOptions {
            axes: Some("czyx".into()),
            fmt: Some(fmt),
            ..Default::default()
        };
        writer::write_labels(&lbl, &g, "cells", &lopts).unwrap();
        compare_stores(&py, &out);
    }
}

#[test]
fn plate_api() {
    for fmt in [Format::V04, Format::V05] {
        let name = format!("plate_{}.zarr", fmt.version());
        let py = fixture(&name);
        let dir = tmp();
        let out = dir.path().join(&name);
        let p =
            Group::open_url(out.to_str().unwrap(), Mode::Write, Some(fmt.zarr_format())).unwrap();
        let rows = vec!["A".to_string(), "B".to_string()];
        let cols = vec!["1".to_string(), "2".to_string(), "3".to_string()];
        let wells = vec![json!("A/1"), json!("A/3"), json!("B/2")];
        writer::write_plate_metadata(
            &p,
            &rows,
            &cols,
            &wells,
            Some(fmt),
            None,
            None,
            Some("plate"),
        )
        .unwrap();
        for wp in ["A/1", "A/3", "B/2"] {
            let well = p.require_group(wp).unwrap();
            writer::write_well_metadata(&well, &[json!("0"), json!("1")], Some(fmt)).unwrap();
            for f in 0..2 {
                let data = read_input(&py, &format!("{wp}/{f}/s0"));
                let opts = ImageOptions {
                    axes: Some("cyx".into()),
                    fmt: Some(fmt),
                    scale_factors: Some(ScaleFactors::Ints(vec![2])),
                    ..Default::default()
                };
                writer::write_image(&data, well.require_group(&f.to_string()).unwrap(), &opts)
                    .unwrap();
            }
        }
        compare_stores(&py, &out);
    }
}

#[test]
fn sharded_writer() {
    let py = fixture("sharded.zarr");
    let data = read_input(&py, "s0");
    let dir = tmp();
    let out = dir.path().join("sharded.zarr");
    let g = Group::open_url(out.to_str().unwrap(), Mode::Write, Some(3)).unwrap();
    let opts = ImageOptions {
        axes: Some("cyx".into()),
        storage_options: Some(
            StorageOptions {
                chunks: Some(vec![16, 16]),
                shards: Some(vec![32, 32]),
                ..Default::default()
            }
            .into(),
        ),
        scale_factors: Some(ScaleFactors::Ints(vec![2])),
        ..Default::default()
    };
    writer::write_image(&data, &g, &opts).unwrap();
    compare_stores(&py, &out);
}
