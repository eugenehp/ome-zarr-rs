//! Round 2 of the differential parity tests with ome-zarr-py (the second half
//! of scripts/gen_parity_fixtures.py, observations2.json).
#![cfg(not(target_arch = "wasm32"))]

mod common;

use std::path::Path;

use common::*;
use ome_zarr::array::{DType, LazyArray};
use ome_zarr::classes::{
    LabelsArg, OmeZarrImage, OmeZarrLabels, OmeZarrMultiscale, OmeZarrScene, SceneImages,
};
use ome_zarr::models::transforms::{CoordinateSystem, Transform};
use ome_zarr::scale::{Methods, ScaleFactors, Scaler};
use ome_zarr::writer::{self, ImageOptions};
use ome_zarr::zarr::{Group, Mode};
use serde_json::{Value, json};

fn observations2() -> Value {
    serde_json::from_str(&std::fs::read_to_string(fixtures().join("observations2.json")).unwrap())
        .unwrap()
}

fn group(dir: &Path, name: &str) -> (std::path::PathBuf, Group) {
    let p = dir.join(name);
    let g = Group::open_url(p.to_str().unwrap(), Mode::Write, None).unwrap();
    (p, g)
}

fn ints(v: &[u64]) -> Option<ScaleFactors> {
    Some(ScaleFactors::Ints(v.to_vec()))
}

#[test]
fn chunked_inputs_every_method() {
    let d = tempfile::tempdir().unwrap();
    for (name, dtype) in [
        ("uint8", DType::UInt8),
        ("int16", DType::Int16),
        ("float32", DType::Float32),
    ] {
        for method in ["resize", "nearest", "local_mean", "zoom"] {
            let off = if name == "int16" { 100 } else { 0 };
            let data = LazyArray::from_ndarray(gen_data(&[3, 70, 53], dtype, 251, off))
                .rechunk(&[1, 17, 23]);
            let store = format!("chunked_{name}_{method}.zarr");
            let (p, g) = group(d.path(), &store);
            let opts = ImageOptions {
                axes: Some("cyx".into()),
                method: Some(Methods::parse(method).unwrap()),
                scale_factors: ints(&[2, 4]),
                ..Default::default()
            };
            writer::write_image(&data, &g, &opts).unwrap();
            compare_stores(&py_store(&store), &p);
        }
    }
}

#[test]
fn delayed_and_bool() {
    let obs = observations2();
    let d = tempfile::tempdir().unwrap();
    let (p, g) = group(d.path(), "delayed.zarr");
    let opts = ImageOptions {
        axes: Some("cyx".into()),
        compute: false,
        scale_factors: ints(&[2]),
        ..Default::default()
    };
    let jobs =
        writer::write_image(&LazyArray::from_ndarray(gen_u8(&[2, 40, 40])), &g, &opts).unwrap();
    assert_eq!(json!(jobs.len()), obs["delayed:jobs"]);
    writer::compute_all(&jobs).unwrap();
    compare_stores(&py_store("delayed.zarr"), &p);

    let (p, g) = group(d.path(), "bool_nearest.zarr");
    let opts = ImageOptions {
        axes: Some("yx".into()),
        method: Some(Methods::Nearest),
        scale_factors: ints(&[2]),
        ..Default::default()
    };
    writer::write_image(
        &LazyArray::from_ndarray(gen_data(&[32, 32], DType::Bool, 251, 0)),
        &g,
        &opts,
    )
    .unwrap();
    compare_stores(&py_store("bool_nearest.zarr"), &p);
}

#[test]
fn scene_with_image_transforms_and_labels() {
    let obs = observations2();
    let rot = CoordinateSystem::from_json(&json!({"name": "rot", "axes": [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]})).unwrap();
    let img_tf = Transform::from_json(&json!({"type": "affine", "affine": [[0.0, 1.0, 5.0], [-1.0, 0.0, 2.0]], "input": {"name": "physical"}, "output": {"name": "rot"}})).unwrap();
    let lab_img = OmeZarrImage::from_axes(
        LazyArray::from_ndarray(gen_data(&[32, 32], DType::UInt8, 3, 0)),
        "yx",
    )
    .unwrap()
    .with_name("lab");
    let labels = OmeZarrLabels::new(lab_img, ints(&[2]), Some(Methods::Nearest), true).unwrap();
    let ia = OmeZarrMultiscale::builder(
        OmeZarrImage::from_axes(
            LazyArray::from_ndarray(gen_data(&[32, 32], DType::UInt16, 251, 0)),
            "yx",
        )
        .unwrap()
        .with_name("ia"),
    )
    .scale_factors(ints(&[2]))
    .coordinate_systems(vec![rot])
    .coordinate_transformations(vec![img_tf])
    .labels(LabelsArg::One(labels))
    .build()
    .unwrap();
    let ib = OmeZarrMultiscale::builder(
        OmeZarrImage::from_axes(LazyArray::from_ndarray(gen_u8(&[32, 32])), "yx")
            .unwrap()
            .with_name("ib"),
    )
    .scale_factors(ints(&[2]))
    .build()
    .unwrap();
    let tf = Transform::from_json(&json!({"type": "translation", "translation": [1.5, -2.0], "input": {"name": "physical", "path": "ia"}, "output": {"name": "physical", "path": "ib"}})).unwrap();
    let scene = OmeZarrScene::new(SceneImages::List(vec![ia, ib]), vec![tf], vec![], None).unwrap();
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("scene2.zarr");
    scene.to_ome_zarr(p.to_str().unwrap(), true, true).unwrap();
    compare_stores(&py_store("scene2.zarr"), &p);

    let gobs = &obs["scene2:graph"];
    let mut nodes: Vec<Value> = scene
        .graph()
        .nodes()
        .iter()
        .map(|(a, b)| json!([a, b]))
        .collect();
    nodes.sort_by_key(|v| v.to_string());
    let mut want = gobs["nodes"].as_array().unwrap().clone();
    want.sort_by_key(|v| v.to_string());
    assert_eq!(nodes, want);
    let pts = vec![
        vec![1.0, 2.0],
        vec![10.5, 3.25],
        vec![20.0, 30.0],
        vec![-3.0, 7.0],
    ];
    for (src, tgt) in [
        (("ia", "rot"), ("ib", "physical")),
        (("ib", "physical"), ("ia", "rot")),
    ] {
        let key = format!("('{}', '{}')->('{}', '{}')", src.0, src.1, tgt.0, tgt.1);
        let w = &gobs["paths"][&key];
        let (s, t) = (
            (src.0.to_string(), src.1.to_string()),
            (tgt.0.to_string(), tgt.1.to_string()),
        );
        let seq = scene.graph().get_sequence(&s, &t, false).unwrap();
        assert_eq!(json!(seq.len()), w["len"], "{key}");
        assert_eq!(
            json!(seq.to_affine().unwrap().matrix),
            w["affine"],
            "{key} affine"
        );
        assert_eq!(
            json!(scene.graph().transform(&s, &t, &pts).unwrap()),
            w["points"],
            "{key} points"
        );
    }
}

#[test]
fn downloads_of_plate_and_scene() {
    let d = tempfile::tempdir().unwrap();
    for (src, out) in [
        ("plate_0.4.zarr", "download_plate"),
        ("scene.zarr", "download_scene"),
    ] {
        let o = d.path().join(out);
        ome_zarr::utils::download(py_store(src).to_str().unwrap(), o.to_str().unwrap()).unwrap();
        compare_stores(&py_store(out), &o);
    }
}

#[test]
fn finder_view_and_find_multiscales() {
    let obs = observations2();
    let d = tempfile::tempdir().unwrap();
    // the same tree, created in the same order, as the Python script
    let f = d.path().join("finder");
    std::fs::create_dir_all(f.join("sub")).unwrap();
    copy_dir(&py_store("coins_0.4.zarr"), &f.join("coins.zarr"));
    copy_dir(&py_store("plate_0.5.zarr"), &f.join("sub/plate.zarr"));
    copy_dir(&py_store("scene.zarr"), &f.join("sub/scene.zarr"));
    ome_zarr::utils::finder(f.to_str().unwrap(), 8123, true).unwrap();
    let csv = std::fs::read_to_string(f.join("biofile_finder.csv")).unwrap();
    let rows: Vec<String> = csv
        .lines()
        .map(|r| r.rsplit_once(',').unwrap().0.to_string())
        .collect();
    let want: Vec<String> = obs["finder:csv"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    // rows follow the directory order of the file system, as os.scandir
    assert_eq!(rows, want);

    for (name, w) in obs["find_multiscales"].as_object().unwrap() {
        let got: Vec<Value> = ome_zarr::utils::find_multiscales(&py_store(name))
            .into_iter()
            .map(|(p, n, dir)| json!([p.to_string_lossy(), n, dir]))
            .collect();
        assert_eq!(&json!(got), w, "{name}");
    }
    let plate_well = py_store("plate_0.4.zarr").join("A").join("1");
    let r = ome_zarr::utils::view(plate_well.to_str().unwrap(), 8000, true, false).unwrap();
    assert!(r.is_none());
    let msg = format!(
        "No OME-Zarr images found in {0}. Try $ ome_zarr finder {0} or use -f to force open in browser.\n",
        plate_well.display()
    );
    assert_eq!(json!(msg), obs["view:stdout"]);
}

fn copy_dir(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for e in std::fs::read_dir(src).unwrap() {
        let p = e.unwrap().path();
        let t = dst.join(p.file_name().unwrap());
        if p.is_dir() {
            copy_dir(&p, &t);
        } else {
            std::fs::copy(&p, &t).unwrap();
        }
    }
}

#[test]
fn scale_cli() {
    let obs = observations2();
    assert_eq!(obs["cli_scale:error"], Value::Null);
    let d = tempfile::tempdir().unwrap();
    let out = d.path().join("cli_scale.zarr");
    let input = py_store("coins_0.4.zarr").join("s0");
    let args = [
        "scale",
        input.to_str().unwrap(),
        out.to_str().unwrap(),
        "yx",
        "--method",
        "nearest",
        "--max_layer",
        "2",
        "--copy-metadata",
    ];
    ome_zarr::cli::main_with_args(args.iter().map(|s| s.to_string())).unwrap();
    compare_stores(&py_store("cli_scale.zarr"), &out);
}

#[test]
fn more_errors() {
    let obs = observations2();
    let d = tempfile::tempdir().unwrap();
    let fmt03 = ome_zarr::format::Format::V03;
    let g03 = Group::open_url(
        d.path().join("e_v03.zarr").to_str().unwrap(),
        Mode::Write,
        Some(2),
    )
    .unwrap();
    let (_, gb) = group(d.path(), "e_bool.zarr");
    let cases: Vec<(&str, ome_zarr::Result<()>)> = vec![
        (
            "plate_from_ome_zarr",
            OmeZarrMultiscale::from_ome_zarr(py_store("plate_0.4.zarr").as_path()).map(|_| ()),
        ),
        (
            "well_from_ome_zarr",
            OmeZarrMultiscale::from_ome_zarr(py_store("plate_0.5.zarr/A/1").as_path()).map(|_| ()),
        ),
        (
            "resize_bool",
            writer::write_image(
                &LazyArray::from_ndarray(gen_data(&[16, 16], DType::Bool, 251, 0)),
                &gb,
                &ImageOptions {
                    axes: Some("yx".into()),
                    ..Default::default()
                },
            )
            .map(|_| ()),
        ),
        (
            "v03_axes",
            writer::write_multiscales_metadata(
                &g03,
                &[json!({"path": "0"})],
                Some(fmt03),
                Some(&"xy".into()),
                None,
                None,
                None,
            ),
        ),
        (
            "scaler_unknown",
            Scaler {
                method: "nope".into(),
                ..Default::default()
            }
            .func(&gen_u8(&[8, 8]))
            .map(|_| ()),
        ),
        (
            "info_missing",
            ome_zarr::utils::info(py_store("does_not_exist.zarr").to_str().unwrap(), false)
                .map(|_| ()),
        ),
        (
            "download_exists",
            ome_zarr::utils::download(
                py_store("coins_0.4.zarr").to_str().unwrap(),
                py_store("download_0.4").to_str().unwrap(),
            ),
        ),
    ];
    let mut failures = vec![];
    for (name, r) in cases {
        let want = &obs["errors"][name];
        let (wt, wm) = (
            want["type"].as_str().unwrap(),
            want["msg"].as_str().unwrap(),
        );
        match r {
            Ok(()) => failures.push(format!("{name}: no error, python raised {wt}: {wm}")),
            Err(e) if e.py_type() != wt || e.py_str() != wm => failures.push(format!(
                "{name}:\n  python {wt}: {wm}\n  rust   {}: {}",
                e.py_type(),
                e.py_str()
            )),
            Err(_) => {}
        }
    }
    // pydantic's ValidationError for float label values
    let r = OmeZarrLabels::from_image(
        OmeZarrImage::from_axes(
            LazyArray::from_ndarray(gen_data(&[8, 8], DType::Float64, 251, 0)),
            "yx",
        )
        .unwrap(),
    );
    let want = &obs["errors"]["labels_float"];
    match r {
        Ok(_) => failures.push("labels_float: no error".into()),
        Err(e) => {
            if e.py_type() != want["type"] || e.py_str() != want["msg"] {
                let m = e.py_str();
                failures.push(format!(
                    "labels_float: rust {}: {}",
                    e.py_type(),
                    &m[..m.len().min(300)]
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
