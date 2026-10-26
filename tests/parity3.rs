//! Round 3 of the differential parity tests with ome-zarr-py: exception
//! messages that interpolate Python objects (pydantic models, tuples, numpy
//! arrays, graph nodes). See observations3.json.
#![cfg(not(target_arch = "wasm32"))]

mod common;

use common::*;
use ome_zarr::array::LazyArray;
use ome_zarr::classes::{OmeZarrImage, OmeZarrMultiscale, OmeZarrScene, SceneImages};
use ome_zarr::models::transforms::{CoordinateSystem, Transform};
use ome_zarr::scale::ScaleFactors;
use ome_zarr::writer::{self, ImageOptions};
use ome_zarr::zarr::{Group, JsonMap, Mode};
use serde_json::{Value, json};

fn observations3() -> Value {
    serde_json::from_str(&std::fs::read_to_string(fixtures().join("observations3.json")).unwrap())
        .unwrap()
}

fn tf(v: Value) -> Transform {
    Transform::from_json(&v).unwrap()
}

fn cs(name: &str) -> CoordinateSystem {
    CoordinateSystem::from_json(&json!({"name": name, "axes": [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]})).unwrap()
}

fn img(name: &str) -> OmeZarrImage {
    OmeZarrImage::from_axes(LazyArray::from_ndarray(gen_u8(&[16, 16])), "yx")
        .unwrap()
        .with_name(name)
}

fn ms(css: Vec<CoordinateSystem>, tfs: Vec<Transform>) -> ome_zarr::Result<()> {
    OmeZarrMultiscale::builder(img("image"))
        .scale_factors(Some(ScaleFactors::Ints(vec![2])))
        .coordinate_systems(css)
        .coordinate_transformations(tfs)
        .build()
        .map(|_| ())
}

fn mk(name: &str) -> OmeZarrMultiscale {
    OmeZarrMultiscale::builder(
        OmeZarrImage::from_axes(LazyArray::from_ndarray(gen_u8(&[32, 32])), "yx")
            .unwrap()
            .with_name(name),
    )
    .scale_factors(Some(ScaleFactors::Ints(vec![2])))
    .build()
    .unwrap()
}

fn dicts(v: Value) -> Vec<JsonMap> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|a| a.as_object().unwrap().clone())
        .collect()
}

#[test]
fn messages_with_python_objects() {
    let obs = observations3();
    let noio = tf(json!({"type": "scale", "scale": [1.0, 2.0]}));
    let nodef = tf(
        json!({"type": "scale", "scale": [1.0, 2.0], "input": {"name": "a"}, "output": {"name": "b"}}),
    );
    let unknown = tf(
        json!({"type": "translation", "translation": [1.0, 2.0], "input": {"name": "physical"}, "output": {"name": "nope"}}),
    );
    let aff = tf(
        json!({"type": "affine", "affine": [[1.0, 0.5, 2.0], [0.0, 1.0, -1.0]], "input": {"name": "physical", "path": "x"}, "output": {"name": "zz"}}),
    );
    let seq = tf(
        json!({"type": "sequence", "input": {"name": "physical"}, "output": {"name": "other"},
        "transformations": [{"type": "scale", "scale": [2.0, 2.0]}, {"type": "mapAxis", "mapAxis": [1, 0]}]}),
    );
    let d = tempfile::tempdir().unwrap();
    let g = Group::open_url(
        d.path().join("e3.zarr").to_str().unwrap(),
        Mode::Write,
        None,
    )
    .unwrap();
    let world2 = cs("world2");
    let s2 = |p: &str| {
        tf(
            json!({"type": "scale", "scale": [2.0, 2.0], "input": {"name": "physical", "path": p}, "output": {"name": if p == "a" { "world" } else { "world2" }}}),
        )
    };
    let sc3 = OmeZarrScene::new(
        SceneImages::List(vec![mk("a"), mk("b")]),
        vec![s2("a"), s2("b")],
        vec![cs("world"), world2],
        None,
    )
    .unwrap();
    let sp = |a: &str, b: &str| (a.to_string(), b.to_string());
    let cases: Vec<(&str, ome_zarr::Result<()>)> = vec![
        ("image_tf_no_input", ms(vec![cs("other")], vec![noio.clone()])),
        ("image_tf_not_default", ms(vec![cs("other")], vec![nodef])),
        ("image_tf_unknown_cs", ms(vec![cs("other")], vec![unknown])),
        ("image_tf_unknown_cs_affine", ms(vec![], vec![aff])),
        ("image_tf_unknown_cs_seq", ms(vec![], vec![seq])),
        ("scene_tf_no_io", OmeZarrScene::new(SceneImages::List(vec![mk("a")]), vec![noio], vec![], None).map(|_| ())),
        ("nearest_too_large", ome_zarr::scale::downscale_nearest(&LazyArray::from_ndarray(gen_u8(&[4, 4])), &[1, 8]).map(|_| ())),
        ("nearest_dims", ome_zarr::scale::downscale_nearest(&LazyArray::from_ndarray(gen_u8(&[4, 4])), &[1]).map(|_| ())),
        (
            "axes_unknown_types",
            writer::write_image(
                &LazyArray::from_ndarray(gen_u8(&[2, 2, 8, 8])),
                &g,
                &ImageOptions {
                    axes: Some(dicts(json!([{"name": "a", "type": "foo"}, {"name": "b", "type": "bar"}, {"name": "y", "type": "space"}, {"name": "x", "type": "space"}])).into()),
                    ..Default::default()
                },
            )
            .map(|_| ()),
        ),
        ("tnd_not_affine", ome_zarr::transformnd::Affine::new(vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.5, 0.0, 1.0]]).map(|_| ())),
        ("tnd_not_affine2", ome_zarr::transformnd::Affine::new(vec![vec![1.0, 0.0], vec![-12.25, 1e-3]]).map(|_| ())),
        ("tnd_not_affine3", ome_zarr::transformnd::Affine::new(vec![vec![1.0, 0.0, 0.0], vec![3.0, 100.0, 2.0]]).map(|_| ())),
        ("graph_unknown_source", sc3.graph().get_sequence(&sp("q", "physical"), &sp("a", "physical"), false).map(|_| ())),
        ("graph_unknown_target", sc3.graph().get_sequence(&sp("a", "physical"), &sp("q", "physical"), false).map(|_| ())),
        ("graph_no_path", sc3.graph().get_sequence(&sp("a", "physical"), &sp("b", "physical"), false).map(|_| ())),
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
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn axes_without_name_are_written() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("axes_no_name.zarr");
    let g = Group::open_url(p.to_str().unwrap(), Mode::Write, None).unwrap();
    writer::write_multiscales_metadata(
        &g,
        &[json!({"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1, 1]}]})],
        None,
        Some(&dicts(json!([{"type": "space"}, {"name": "x", "type": "space"}])).into()),
        None,
        None,
        None,
    )
    .unwrap();
    compare_stores(&py_store("axes_no_name.zarr"), &p);
}
