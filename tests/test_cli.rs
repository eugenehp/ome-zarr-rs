//! Ports of ome-zarr-py's test_cli.py, test_ome_zarr.py and CSV tests.
#![cfg(not(target_arch = "wasm32"))]

use std::path::{Path, PathBuf};

use ome_zarr::cli::main_with_args;
use ome_zarr::format::{CURRENT_FORMAT, Format};
use ome_zarr::utils::{find_multiscales, finder, info_to, strip_common_prefix, view};
use ome_zarr::writer;
use ome_zarr::zarr::{Group, Mode};
use serde_json::{Value, json};

fn tmp() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

fn info(path: &str) -> String {
    let mut out = String::new();
    info_to(path, false, &mut out).unwrap();
    out
}

#[test]
fn coins_info() {
    for fmt in [Some(Format::V04), Some(Format::V05), None] {
        let d = tmp();
        let f = d.path().join("data-1");
        let mut args = vec![
            "create".to_string(),
            "--method=coins".into(),
            f.to_string_lossy().to_string(),
        ];
        if let Some(fm) = fmt {
            args.extend(["--format".into(), fm.version().into()]);
        }
        main_with_args(args).unwrap();
        main_with_args(["info", f.to_str().unwrap()]).unwrap();
        let out = info(f.to_str().unwrap());
        assert!(
            out.contains(
                &Path::new("labels")
                    .join("coins")
                    .to_string_lossy()
                    .to_string()
            )
        );
        let v = fmt.unwrap_or(CURRENT_FORMAT).version();
        assert!(out.contains(&format!("- version: {v}")), "{out}");
    }
}

#[test]
fn astronaut_info_and_download() {
    for fmt in [Some(Format::V04), Some(Format::V05), None] {
        let d = tmp();
        let f = d.path().join("data-3");
        let out_dir = d.path().join("out");
        let mut args = vec![
            "create".to_string(),
            "--method=astronaut".into(),
            f.to_string_lossy().to_string(),
        ];
        if let Some(fm) = fmt {
            args.extend(["--format".into(), fm.version().into()]);
        }
        main_with_args(args).unwrap();
        main_with_args([
            "download".to_string(),
            f.to_string_lossy().to_string(),
            format!("--output={}", out_dir.display()),
        ])
        .unwrap();
        let dl = out_dir.join("data-3");
        main_with_args(["info", dl.to_str().unwrap()]).unwrap();
        if fmt.is_some_and(|f| f.zarr_format() == 2) {
            for item in [".zattrs", ".zgroup", "labels", "s0", "s1", "s2", "s3", "s4"] {
                assert!(dl.join(item).exists(), "{item}");
            }
            for item in [".zarray", ".zattrs", "0", "1", "2"] {
                assert!(dl.join("s1").join(item).exists(), "s1/{item}");
            }
        } else {
            for item in ["labels", "s0", "s1", "s2", "s3", "s4", "zarr.json"] {
                assert!(dl.join(item).exists(), "{item}");
            }
            for item in ["c", "zarr.json"] {
                assert!(dl.join("s1").join(item).exists(), "s1/{item}");
            }
        }
        // downloaded data equals the original
        let a = Group::open_url(f.to_str().unwrap(), Mode::Read, None)
            .unwrap()
            .array("s2")
            .unwrap()
            .read_all()
            .unwrap();
        let b = Group::open_url(dl.to_str().unwrap(), Mode::Read, None)
            .unwrap()
            .array("s2")
            .unwrap()
            .read_all()
            .unwrap();
        assert_eq!(a, b);
    }
}

#[test]
fn astronaut_info_log_format() {
    // test_ome_zarr.py: shapes and dtypes of the levels
    let d = tmp();
    let f = d.path().join("data");
    ome_zarr::data::create_zarr(
        f.to_str().unwrap(),
        ome_zarr::data::SampleData::Astronaut,
        CURRENT_FORMAT,
    )
    .unwrap();
    let out = info(f.to_str().unwrap());
    assert!(out.contains("(3, 1024, 1024)"));
    assert!(out.contains("(3, 512, 512)"));
    let lbl = Group::open_url(
        f.join("labels/astronaut_labels").to_str().unwrap(),
        Mode::Read,
        None,
    )
    .unwrap();
    assert_eq!(
        lbl.array("s1").unwrap().dtype(),
        ome_zarr::array::DType::Int8
    );
    // download preserves the root zarr.json
    let target = d.path().join("out");
    ome_zarr::utils::download(f.to_str().unwrap(), target.to_str().unwrap()).unwrap();
    let root: Value =
        serde_json::from_str(&std::fs::read_to_string(target.join("data/zarr.json")).unwrap())
            .unwrap();
    assert_eq!(root["node_type"], "group");
    assert_eq!(root["zarr_format"], 3);
    assert!(root["attributes"].get("attributes").is_none());
    assert!(root["attributes"].get("node_type").is_none());
    assert_eq!(root["attributes"]["ome"]["version"], "0.5");
}

fn rotate_and_test(hierarchy: &[PathBuf], reverse: bool) {
    let results: Vec<Vec<String>> = vec![
        vec!["d".into()],
        vec!["d".into(), "e".into()],
        vec!["d".into(), "e".into(), "f".into()],
    ];
    let mut h: Vec<PathBuf> = hierarchy.to_vec();
    for _ in 0..3 {
        h.rotate_right(1);
        let mut copy: Vec<Vec<String>> = h.iter().map(|p| ome_zarr::utils::splitall(p)).collect();
        let common = strip_common_prefix(&mut copy).unwrap();
        assert_eq!(common, "d");
        let mut a = copy.clone();
        a.sort();
        let mut b = results.clone();
        b.sort();
        assert_eq!(a, b);
    }
    if reverse {
        let mut r = hierarchy.to_vec();
        r.reverse();
        rotate_and_test(&r, false);
    }
}

#[test]
fn strip_prefix() {
    rotate_and_test(
        &[
            PathBuf::from("d"),
            PathBuf::from("d/e"),
            PathBuf::from("d/e/f"),
        ],
        true,
    );
    rotate_and_test(
        &[
            PathBuf::from("/a/b/c/d"),
            PathBuf::from("/a/b/c/d/e"),
            PathBuf::from("/a/b/c/d/e/f"),
        ],
        true,
    );
    let mut none = vec![vec!["a".to_string()], vec!["b".to_string()]];
    assert!(strip_common_prefix(&mut none).is_err());
}

#[test]
fn view_dry_run() {
    let d = tmp();
    let p = d.path().join("data");
    std::fs::create_dir_all(&p).unwrap();
    assert!(
        view(p.to_str().unwrap(), 8000, true, false)
            .unwrap()
            .is_none()
    );
    assert!(
        view(p.to_str().unwrap(), 8000, true, true)
            .unwrap()
            .is_some()
    );
    let f = d.path().join("data-4");
    main_with_args(["create", "--method=astronaut", f.to_str().unwrap()]).unwrap();
    let url = view(f.to_str().unwrap(), 8000, true, false)
        .unwrap()
        .unwrap();
    assert_eq!(
        url,
        "https://ome.github.io/ome-ngff-validator/?source=http://localhost:8000/data-4"
    );
    let scene = p.join("scene.zarr");
    std::fs::create_dir_all(&scene).unwrap();
    std::fs::write(scene.join("zarr.json"), json!({"attributes": {"ome": {"scene": {"coordinateTransformations": [{"input": {"path": "image.zarr"}}]}}}}).to_string()).unwrap();
    assert!(
        view(scene.to_str().unwrap(), 8000, true, false)
            .unwrap()
            .is_some()
    );
    assert_eq!(find_multiscales(&scene)[0].0, scene.join("image.zarr"));
}

#[test]
fn finder_csv() {
    for fmt in [Format::V04, Format::V05] {
        let d = tmp();
        let img_dir = d.path().join("data/images");
        std::fs::create_dir_all(&img_dir).unwrap();
        assert!(
            finder(img_dir.to_str().unwrap(), 8000, true)
                .unwrap()
                .is_none()
        );
        assert!(!img_dir.join("biofile_finder.csv").exists());
        let dir2 = img_dir.join("dir2");
        let bf = img_dir.join("bf2raw.zarr");
        std::fs::create_dir_all(&dir2).unwrap();
        std::fs::create_dir_all(bf.join("OME")).unwrap();
        main_with_args([
            "create",
            "--method=astronaut",
            img_dir.join("astronaut").to_str().unwrap(),
            "--format",
            fmt.version(),
        ])
        .unwrap();
        main_with_args([
            "create",
            "--method=coins",
            dir2.join("coins").to_str().unwrap(),
            "--format",
            fmt.version(),
        ])
        .unwrap();
        std::fs::write(bf.join(".zattrs"), r#"{"bioformats2raw.layout" : 3}"#).unwrap();
        std::fs::write(
            bf.join("OME/METADATA.ome.xml"),
            r#"<?xml version="1.0" encoding="UTF-8"?>
                <OME><Image ID="Image:1" Name="test.fake"></Image></OME>
                "#,
        )
        .unwrap();
        let plate = Group::open_url(
            dir2.join("plate").to_str().unwrap(),
            Mode::Write,
            Some(fmt.zarr_format()),
        )
        .unwrap();
        writer::write_plate_metadata_simple(&plate, &["A"], &["1"], &["A/1"], None).unwrap();
        let url = finder(img_dir.to_str().unwrap(), 8000, true)
            .unwrap()
            .unwrap();
        assert!(url.starts_with("https://bff.allencell.org/app?source=%7B%22uri%22%3A%20%22http%3A//localhost%3A8000/images/biofile_finder.csv%22"));
        let csv = std::fs::read_to_string(img_dir.join("biofile_finder.csv")).unwrap();
        assert!(
            csv.contains("File Path,File Name,Folders,Uploaded"),
            "{csv}"
        );
        assert!(csv.contains("dir2/plate/A/1/0,plate,dir2"), "{csv}");
        assert!(csv.contains("coins,dir2"), "{csv}");
        assert!(csv.contains("test.fake"), "{csv}");
    }
}

#[test]
fn find_multiscales_empty() {
    let d = tmp();
    assert!(find_multiscales(d.path()).is_empty());
}

#[test]
fn csv_to_labels() {
    for fmt in [Format::V04, Format::V05] {
        let d = tmp();
        let f = d.path().join("img");
        ome_zarr::data::create_zarr(f.to_str().unwrap(), ome_zarr::data::SampleData::Coins, fmt)
            .unwrap();
        // labels must be at labels/0 for csv_to_labels
        std::fs::rename(f.join("labels/coins"), f.join("labels/0")).unwrap();
        let csv = d.path().join("props.csv");
        std::fs::write(&csv, "id,area,name,flag\n1,10.5,one,x\n2,20,two,\n").unwrap();
        let r = main_with_args([
            "csv_to_labels",
            csv.to_str().unwrap(),
            "id",
            "area#d,name,flag#b,missing#l",
            f.to_str().unwrap(),
            "label-value",
        ]);
        if fmt == Format::V05 {
            // as ome-zarr-py: v0.5 metadata (under "ome") is not found
            assert!(
                r.unwrap_err()
                    .to_string()
                    .contains("zarr_path must be to plate.zarr or image.zarr")
            );
            continue;
        }
        r.unwrap();
        let g = Group::open_url(f.join("labels/0").to_str().unwrap(), Mode::Read, None).unwrap();
        let md = writer::get_metadata(&g).unwrap();
        let props = md["image-label"]["properties"].as_array().unwrap();
        let p1 = props.iter().find(|p| p["label-value"] == 1).unwrap();
        assert_eq!(p1["area"], 10.5);
        assert_eq!(p1["name"], "one");
        assert_eq!(p1["flag"], true);
        let p2 = props.iter().find(|p| p["label-value"] == 2).unwrap();
        assert_eq!(p2["area"], 20.0);
        assert_eq!(p2["flag"], false);
    }
}

#[test]
fn scale_command() {
    let d = tmp();
    let f = d.path().join("img");
    ome_zarr::data::create_zarr(
        f.to_str().unwrap(),
        ome_zarr::data::SampleData::Coins,
        CURRENT_FORMAT,
    )
    .unwrap();
    let out = d.path().join("scaled");
    main_with_args([
        "scale",
        f.join("s0").to_str().unwrap(),
        out.to_str().unwrap(),
        "yx",
        "--method",
        "nearest",
        "--max_layer",
        "2",
        "--copy-metadata",
    ])
    .unwrap();
    let g = Group::open_url(out.to_str().unwrap(), Mode::Read, None).unwrap();
    assert_eq!(
        writer::get_metadata(&g).unwrap()["multiscales"][0]["datasets"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    assert!(
        main_with_args([
            "scale",
            f.join("s0").to_str().unwrap(),
            d.path().join("x").to_str().unwrap(),
            "yx",
            "--method",
            "laplacian"
        ])
        .is_err()
    );
}

#[test]
#[ignore = "requires network access to EMBL-EBI"]
fn s3_info() {
    for url in [
        "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001240.zarr",
        "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.2/6001240.zarr",
        "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.3/9836842.zarr",
        "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr",
        "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0062A/6001240_labels.zarr",
    ] {
        main_with_args(["info", url]).unwrap();
    }
}

#[test]
#[ignore = "requires network access to EMBL-EBI"]
fn class_reader_legacy() {
    for (url, labels) in [
        (
            "https://livingobjects.ebi.ac.uk/idr/zarr/v0.1/6001237.zarr",
            true,
        ),
        (
            "https://livingobjects.ebi.ac.uk/idr/zarr/v0.2/6001240.zarr",
            true,
        ),
        (
            "https://livingobjects.ebi.ac.uk/idr/zarr/v0.3/idr0052A/5514375.zarr",
            true,
        ),
        (
            "https://livingobjects.ebi.ac.uk/idr/zarr/v0.3/idr0109A/12922361.zarr",
            false,
        ),
    ] {
        let ms = ome_zarr::classes::OmeZarrMultiscale::from_ome_zarr(url).unwrap();
        assert!(ms.omero.is_some(), "{url}");
        if labels {
            assert!(ms.labels().is_some_and(|l| !l.is_empty()), "{url}");
        }
    }
}
