//! Differential parity with ome-zarr-py: every scenario of
//! scripts/gen_parity_fixtures.py, run in Rust and compared with what Python
//! produced (stores byte for byte, observations value for value).
#![cfg(not(target_arch = "wasm32"))]

mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use common::*;
use ome_zarr::array::{DType, LazyArray, NdArray};
use ome_zarr::classes::{OmeZarrImage, OmeZarrMultiscale, ScaleMap, Version};
use ome_zarr::format::Format;
use ome_zarr::scale::{Methods, ScaleFactors, Scaler};
use ome_zarr::writer::{
    self, ImageOptions, LabelsOptions, MultiscaleOptions, StorageOptions, StorageOptionsArg,
};
use ome_zarr::zarr::{Compressor, Group, JsonMap, Mode};
use serde_json::{Value, json};

fn tmp() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

fn obj(v: Value) -> JsonMap {
    v.as_object().unwrap().clone()
}

fn units(pairs: &[(&str, &str)]) -> JsonMap {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), json!(v)))
        .collect()
}

fn group(dir: &Path, name: &str, fmt: Option<Format>) -> (PathBuf, Group) {
    let p = dir.join(name);
    let g = Group::open_url(
        p.to_str().unwrap(),
        Mode::Write,
        fmt.map(|f| f.zarr_format()),
    )
    .unwrap();
    (p, g)
}

fn per_dim(levels: &[&[(&str, f64)]]) -> ScaleFactors {
    ScaleFactors::per_dim(levels.iter().map(|l| l.to_vec()).collect::<Vec<_>>())
}

#[test]
fn dtypes_and_methods() {
    let d = tmp();
    for (name, dtype) in [
        ("uint8", DType::UInt8),
        ("int8", DType::Int8),
        ("int16", DType::Int16),
        ("uint16", DType::UInt16),
        ("uint32", DType::UInt32),
        ("int32", DType::Int32),
        ("float32", DType::Float32),
        ("float64", DType::Float64),
    ] {
        for method in ["resize", "nearest", "local_mean", "zoom"] {
            let off = if name.starts_with("int") { 100 } else { 0 };
            let data = gen_data(&[2, 9, 70, 53], dtype, 251, off);
            let store = format!("dtype_{name}_{method}.zarr");
            let (p, g) = group(d.path(), &store, None);
            let opts = ImageOptions {
                axes: Some("czyx".into()),
                method: Some(Methods::parse(method).unwrap()),
                scale_factors: Some(per_dim(&[
                    &[("z", 2.0), ("y", 2.0), ("x", 2.0)],
                    &[("z", 2.0), ("y", 4.0), ("x", 3.0)],
                ])),
                ..Default::default()
            };
            writer::write_image(&LazyArray::from_ndarray(data), &g, &opts).unwrap();
            compare_stores(&py_store(&store), &p);
        }
    }
}

#[test]
fn functional_writer_api() {
    for fmt in [Format::V04, Format::V05] {
        let v = fmt.version();
        let d = tmp();
        // write_multiscale
        let (p, g) = group(d.path(), &format!("write_multiscale_{v}.zarr"), Some(fmt));
        let pyr: Vec<LazyArray> = [[16, 64, 64], [16, 32, 32], [16, 16, 16]]
            .iter()
            .map(|s| LazyArray::from_ndarray(gen_data(s, DType::UInt16, 251, 0)))
            .collect();
        let mo = MultiscaleOptions {
            fmt: Some(fmt),
            axes: Some("zyx".into()),
            name: Some("ms".into()),
            scale: Some(ScaleMap::from([("z", 2.0), ("y", 0.5), ("x", 0.5)])),
            axes_units: Some(units(&[
                ("z", "micrometer"),
                ("y", "micrometer"),
                ("x", "micrometer"),
            ])),
            storage_options: Some(StorageOptionsArg::PerLevel(vec![
                StorageOptions::chunks(&[8, 8]),
                StorageOptions::chunks(&[16, 16]),
                StorageOptions::chunks(&[4, 4]),
            ])),
            metadata: Some(obj(json!({
                "omero": {"channels": [{"color": "FF0000", "window": {"start": 0, "end": 10, "min": 0, "max": 255}}]},
                "method": "manual", "version": "1"
            }))),
            ..MultiscaleOptions::new()
        };
        writer::write_multiscale(&pyr, &g, &mo).unwrap();
        compare_stores(&py_store(&format!("write_multiscale_{v}.zarr")), &p);

        // write_multiscales_metadata only
        let (p, g) = group(
            d.path(),
            &format!("multiscales_metadata_{v}.zarr"),
            Some(fmt),
        );
        let datasets = vec![
            json!({"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 1e-05]}]}),
            json!({"path": "1", "coordinateTransformations": [{"type": "scale", "scale": [1, 2, 2e-05]}, {"type": "translation", "translation": [0, 0.5, 1e16]}]}),
        ];
        let axes = ome_zarr::axes::AxesInput::from_json(
            &json!([{"name": "c", "type": "channel"}, {"name": "y", "type": "space", "unit": "nanometer"}, "x"]),
        );
        // Python accepts a mixed list of dicts and names
        let axes = axes.unwrap_or_else(|_| {
            ome_zarr::axes::AxesInput::Dicts(vec![
                obj(json!({"name": "c", "type": "channel"})),
                obj(json!({"name": "y", "type": "space", "unit": "nanometer"})),
                obj(json!({"name": "x", "type": "space"})),
            ])
        });
        writer::write_multiscales_metadata(
            &g,
            &datasets,
            Some(fmt),
            Some(&axes),
            None,
            Some("named"),
            Some(&obj(json!({"description": "café"}))),
        )
        .unwrap();
        compare_stores(&py_store(&format!("multiscales_metadata_{v}.zarr")), &p);

        // write_multiscale_labels + write_label_metadata
        let (p, g) = group(d.path(), &format!("multiscale_labels_{v}.zarr"), Some(fmt));
        let lp = vec![
            LazyArray::from_ndarray(gen_data(&[32, 32], DType::UInt32, 5, 0)),
            LazyArray::from_ndarray(gen_data(&[16, 16], DType::UInt32, 5, 0)),
        ];
        let lo = LabelsOptions {
            multiscale: MultiscaleOptions {
                fmt: Some(fmt),
                axes: Some("yx".into()),
                storage_options: Some(StorageOptions::chunks(&[8, 8]).into()),
                ..MultiscaleOptions::new()
            },
            label_metadata: Some(obj(json!({
                "colors": [{"label-value": 1, "rgba": [255, 0, 0, 255]}],
                "properties": [{"label-value": 1, "class": "cell"}]
            }))),
        };
        writer::write_multiscale_labels(&lp, &g, "seg", &lo).unwrap();
        writer::write_label_metadata(
            g.group("labels").unwrap(),
            "seg",
            Some(&[json!({"label-value": 2, "rgba": [0, 0, 255, 128]})]),
            None,
            Some(fmt),
            Some(&obj(json!({"extra": "kept"}))),
        )
        .unwrap();
        compare_stores(&py_store(&format!("multiscale_labels_{v}.zarr")), &p);

        // compressor
        let (p, g) = group(d.path(), &format!("compressed_{v}.zarr"), Some(fmt));
        let comp = Compressor::Blosc {
            cname: "zstd".into(),
            clevel: 3,
            shuffle: 2,
            blocksize: 0,
        };
        let opts = ImageOptions {
            fmt: Some(fmt),
            axes: Some("cyx".into()),
            storage_options: Some(
                StorageOptions {
                    compressor: Some(comp),
                    chunks: Some(vec![32, 32]),
                    ..Default::default()
                }
                .into(),
            ),
            ..Default::default()
        };
        writer::write_image(
            &LazyArray::from_ndarray(gen_data(&[3, 64, 64], DType::UInt16, 251, 0)),
            &g,
            &opts,
        )
        .unwrap();
        compare_stores(&py_store(&format!("compressed_{v}.zarr")), &p);

        // legacy scaler
        for method in [
            "nearest",
            "gaussian",
            "laplacian",
            "local_mean",
            "zoom",
            "resize_image",
        ] {
            let (p, g) = group(d.path(), &format!("scaler_{method}_{v}.zarr"), Some(fmt));
            let opts = ImageOptions {
                fmt: Some(fmt),
                axes: Some("cyx".into()),
                scaler: Some(Scaler {
                    max_layer: 2,
                    method: method.into(),
                    ..Default::default()
                }),
                ..Default::default()
            };
            writer::write_image(&LazyArray::from_ndarray(gen_u8(&[2, 50, 45])), &g, &opts).unwrap();
            compare_stores(&py_store(&format!("scaler_{method}_{v}.zarr")), &p);
        }

        // write_labels
        let (p, g) = group(d.path(), &format!("labels_{v}.zarr"), Some(fmt));
        let opts = ImageOptions {
            fmt: Some(fmt),
            axes: Some("cyx".into()),
            scale_factors: Some(ScaleFactors::Ints(vec![2, 4])),
            label_metadata: Some(obj(
                json!({"properties": [{"label-value": 3, "kind": "x"}]}),
            )),
            scale: Some(ScaleMap::from([("y", 0.2), ("x", 0.2)])),
            axes_units: Some(units(&[("y", "micrometer"), ("x", "micrometer")])),
            ..Default::default()
        };
        writer::write_labels(
            &LazyArray::from_ndarray(gen_data(&[1, 40, 40], DType::Int16, 7, 0)),
            &g,
            "lab",
            &opts,
        )
        .unwrap();
        compare_stores(&py_store(&format!("labels_{v}.zarr")), &p);
    }
}

#[test]
fn dask_5d_with_z_downsampling() {
    let d = tmp();
    let (p, g) = group(d.path(), "dask5d.zarr", None);
    let data = LazyArray::from_ndarray(gen_data(&[2, 3, 10, 66, 66], DType::UInt16, 251, 0))
        .rechunk(&[1, 1, 3, 20, 20]);
    let opts = ImageOptions {
        axes: Some("tczyx".into()),
        scale_factors: Some(per_dim(&[
            &[("z", 2.0), ("y", 2.0), ("x", 2.0)],
            &[("z", 2.0), ("y", 4.0), ("x", 4.0)],
        ])),
        coordinate_transformations: Some(vec![
            vec![
                json!({"type": "scale", "scale": [1, 1, 1, 1, 1]})
            ];
            3
        ]),
        ..Default::default()
    };
    writer::write_image(&data, &g, &opts).unwrap();
    compare_stores(&py_store("dask5d.zarr"), &p);
}

#[test]
fn sample_data_download_and_csv() {
    for fmt in [Format::V04, Format::V05] {
        let v = fmt.version();
        let d = tmp();
        for (name, sd) in [
            ("coins", ome_zarr::data::SampleData::Coins),
            ("astronaut", ome_zarr::data::SampleData::Astronaut),
        ] {
            let p = d.path().join(format!("{name}_{v}.zarr"));
            ome_zarr::data::create_zarr(p.to_str().unwrap(), sd, fmt).unwrap();
            compare_stores(&py_store(&format!("{name}_{v}.zarr")), &p);
        }
        // download of the Python-written store
        let out = d.path().join(format!("download_{v}"));
        ome_zarr::utils::download(
            py_store(&format!("coins_{v}.zarr")).to_str().unwrap(),
            out.to_str().unwrap(),
        )
        .unwrap();
        compare_stores(&py_store(&format!("download_{v}")), &out);
    }
    // csv_to_labels on a copy of the Python coins store
    let d = tmp();
    let target = d.path().join("csv_0.4.zarr");
    copy_dir(&py_store("coins_0.4.zarr"), &target);
    std::fs::rename(target.join("labels/coins"), target.join("labels/0")).unwrap();
    ome_zarr::csv::csv_to_zarr(
        fixtures().join("props.csv").to_str().unwrap(),
        "id",
        "area#d,name,flag#b,n#l,missing#s",
        target.to_str().unwrap(),
        "label-value",
    )
    .unwrap();
    compare_stores(&py_store("csv_0.4.zarr"), &target);
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
fn plates() {
    for fmt in [Format::V04, Format::V05] {
        let v = fmt.version();
        let d = tmp();
        let (p, g) = group(d.path(), &format!("plate_{v}.zarr"), Some(fmt));
        let s = |v: &[&str]| -> Vec<String> { v.iter().map(|x| x.to_string()).collect() };
        let wells = ["A/1", "A/2", "A/4", "B/2", "B/3", "C/1", "C/3", "C/4"];
        writer::write_plate_metadata(
            &g,
            &s(&["A", "B", "C"]),
            &s(&["1", "2", "3", "4"]),
            &wells.iter().map(|w| json!(w)).collect::<Vec<_>>(),
            Some(fmt),
            Some(&[json!({"id": 1, "name": "a", "starttime": 10})]),
            Some(3),
            Some("plate"),
        )
        .unwrap();
        for (i, wp) in wells.iter().enumerate() {
            let well = g.require_group(wp).unwrap();
            writer::write_well_metadata(
                &well,
                &[
                    json!("0"),
                    json!({"path": "1", "acquisition": 1}),
                    json!("2"),
                ],
                Some(fmt),
            )
            .unwrap();
            for f in 0..3 {
                let data = gen_data(&[1, 2, 1, 24, 30], DType::UInt8, 97 + i as u64 + f, 0);
                let opts = ImageOptions {
                    fmt: Some(fmt),
                    scale_factors: Some(ScaleFactors::Ints(vec![2])),
                    ..Default::default()
                };
                writer::write_image(
                    &LazyArray::from_ndarray(data),
                    well.require_group(&f.to_string()).unwrap(),
                    &opts,
                )
                .unwrap();
            }
        }
        compare_stores(&py_store(&format!("plate_{v}.zarr")), &p);
    }
}

fn stores_prefix() -> String {
    std::fs::canonicalize(fixtures().join("stores"))
        .unwrap()
        .to_string_lossy()
        .to_string()
}

#[test]
fn reader_nodes_and_info() {
    let obs = observations();
    for name in [
        "coins_0.4.zarr",
        "coins_0.5.zarr",
        "plate_0.4.zarr",
        "plate_0.5.zarr",
        "plate_0.4.zarr/B/3",
        "coins_0.5.zarr/labels/coins",
        "astronaut_0.4.zarr/labels",
    ] {
        let path = py_store(name);
        let loc = ome_zarr::io::parse_url_r(path.to_str().unwrap())
            .unwrap()
            .unwrap();
        let nodes = ome_zarr::reader::Reader::new(loc).unwrap().nodes().unwrap();
        let mut got = vec![];
        for n in &nodes {
            let data: Vec<Value> = n
                .data
                .iter()
                .map(|d| {
                    let sum: f64 = d.compute().unwrap().to_f64_vec().iter().sum();
                    json!({"shape": d.shape(), "dtype": d.dtype().name(), "chunks": d.chunks(), "sum": sum})
                })
                .collect();
            got.push(json!({
                "repr": n.to_string(),
                "visible": n.visible(),
                "specs": n.specs.iter().map(|s| s.kind().to_string()).collect::<Vec<_>>(),
                "metadata": Value::Object(n.metadata.clone()),
                "data": data,
            }));
        }
        let want = &obs[format!("nodes:{name}")];
        assert_eq!(&Value::Array(got), want, "nodes of {name}");
        for stats in [false, true] {
            let mut out = String::new();
            ome_zarr::utils::info_to(path.to_str().unwrap(), stats, &mut out).unwrap();
            let want = obs[format!("info:{name}:{}", if stats { "True" } else { "False" })]
                .as_str()
                .unwrap();
            assert_eq!(
                out.replace(&stores_prefix(), "STORES"),
                want.replace(&stores_prefix(), "STORES"),
                "info of {name} (stats={stats})"
            );
        }
    }
}

#[test]
fn from_ome_zarr_versions() {
    let obs = observations();
    for name in [
        "legacy_0.1.zarr",
        "legacy_0.2.zarr",
        "legacy_0.3.zarr",
        "conv_v04_swap.zarr",
        "conv_v05_top.zarr",
        "coins_0.4.zarr",
        "astronaut_0.5.zarr",
        "write_multiscale_0.5.zarr",
    ] {
        let want = &obs[format!("from_ome_zarr:{name}")];
        let got = match OmeZarrMultiscale::from_ome_zarr(py_store(name).as_path()) {
            Err(e) => {
                assert_eq!(
                    want["error"], "ValidationError",
                    "{name}: unexpected error {e}"
                );
                assert_eq!(e.py_type(), "ValidationError", "{name}: {e}");
                continue;
            }
            Ok(ms) => ms,
        };
        assert!(
            want.get("error").is_none(),
            "{name}: python failed but rust succeeded"
        );
        let opt_map = |m: &Option<BTreeMap<String, String>>| {
            m.as_ref().map(|m| json!(m)).unwrap_or(Value::Null)
        };
        let images: Vec<Value> = got
            .images()
            .iter()
            .map(|i| {
                let scale: JsonMap = i
                    .scale
                    .0
                    .iter()
                    .map(|(k, v)| (k.clone(), json!(v)))
                    .collect();
                json!({
                    "axes": i.axes, "scale": scale, "units": opt_map(&i.axes_units),
                    "types": i.axes_types, "name": i.name, "shape": i.data.shape(),
                })
            })
            .collect();
        let got_v = json!({
            "type": "OMEZarrMultiscale",
            "metadata": serde_json::to_value(got.metadata()).unwrap(),
            "images": images,
            "omero": got.omero.as_ref().map(|o| serde_json::to_value(o).unwrap()).unwrap_or(Value::Null),
            "labels": got.labels().map(|l| { let mut v: Vec<String> = l.iter().map(|(k, _)| k.clone()).collect(); v.sort(); json!(v) }).unwrap_or(Value::Null),
        });
        // scales compare by value (Python ints vs floats are equal)
        assert_eq!(normalize(&got_v), normalize(want), "from_ome_zarr {name}");
    }
}

/// Numbers compared as f64 (Python keeps `1` and `1.0` distinct in JSON).
fn normalize(v: &Value) -> Value {
    match v {
        Value::Number(n) => json!(n.as_f64().unwrap()),
        Value::Null => json!("NaN"),
        Value::Array(a) => Value::Array(a.iter().map(normalize).collect()),
        Value::Object(m) => {
            Value::Object(m.iter().map(|(k, v)| (k.clone(), normalize(v))).collect())
        }
        o => o.clone(),
    }
}

#[test]
fn legacy_scaler_arrays() {
    let obs = observations();
    for shape in [vec![64usize, 64], vec![3, 64, 64], vec![1, 2, 1, 64, 64]] {
        let key = shape
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join("x");
        let data = gen_u8(&shape);
        for method in [
            "nearest",
            "gaussian",
            "laplacian",
            "local_mean",
            "zoom",
            "resize_image",
        ] {
            let max_layer = if method == "zoom" { 2 } else { 3 };
            let sc = Scaler {
                max_layer,
                method: method.into(),
                ..Default::default()
            };
            let levels = if method == "resize_image" {
                vec![Scaler::default().resize_image(&data).unwrap()]
            } else {
                sc.func(&data).unwrap()
            };
            let want = obs[format!("scaler:{key}:{method}")].as_array().unwrap();
            assert_eq!(levels.len(), want.len(), "{key} {method}");
            for (l, w) in levels.iter().zip(want) {
                assert_eq!(json!(l.shape()), w["shape"], "{key} {method}");
                assert_eq!(l.dtype().name(), w["dtype"], "{key} {method}");
                let wd: Vec<f64> = w["data"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap())
                    .collect();
                assert_eq!(l.to_f64_vec(), wd, "{key} {method}");
            }
        }
    }
}

#[test]
fn scene_store_and_graph() {
    use ome_zarr::classes::{OmeZarrScene, SceneImages};
    use ome_zarr::models::transforms::{CoordinateSystem, Transform};
    let obs = observations();
    let mk = |name: &str| {
        OmeZarrMultiscale::builder(
            OmeZarrImage::from_axes(LazyArray::from_ndarray(gen_u8(&[32, 32])), "yx")
                .unwrap()
                .with_name(name),
        )
        .scale_factors(Some(ScaleFactors::Ints(vec![2])))
        .build()
        .unwrap()
    };
    let world = CoordinateSystem::from_json(&json!({"name": "world", "axes": [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]})).unwrap();
    let tfs: Vec<Transform> = [
        json!({"type": "affine", "affine": [[1.5, 0.2, 3.0], [-0.1, 0.9, -2.0]], "input": {"name": "physical", "path": "a"}, "output": {"name": "world"}}),
        json!({"type": "sequence", "input": {"name": "world"}, "output": {"name": "physical", "path": "b"},
            "transformations": [{"type": "rotation", "rotation": [[0.0, -1.0], [1.0, 0.0]]}, {"type": "scale", "scale": [2.0, 0.5]}, {"type": "mapAxis", "mapAxis": [1, 0]}]}),
        json!({"type": "byDimension", "input": {"name": "physical", "path": "b"}, "output": {"name": "physical", "path": "c"}, "name": "bd",
            "transformations": [{"inputAxes": [0], "outputAxes": [1], "transformation": {"type": "translation", "translation": [4.0]}},
                                {"inputAxes": [1], "outputAxes": [0], "transformation": {"type": "scale", "scale": [3.0]}}]}),
        json!({"type": "displacements", "input": {"name": "physical", "path": "c"}, "output": {"name": "physical", "path": "d"}, "path": "coordinateTransformations/field"}),
    ]
    .iter()
    .map(|t| Transform::from_json(t).unwrap())
    .collect();
    let field = gen_data(&[2, 32, 32], DType::Float32, 13, 0)
        .to_f64()
        .mapv(|v| v - 1.0);
    let field = NdArray::Float32(field.mapv(|v| v as f32));
    let ax: Vec<String> = "cyx".chars().map(|c| c.to_string()).collect();
    let types = [
        ("c".to_string(), Some("displacement".to_string())),
        ("y".into(), Some("space".into())),
        ("x".into(), Some("space".into())),
    ]
    .into();
    let fimg = OmeZarrImage::new(
        LazyArray::from_ndarray(field),
        &ax,
        Some(ScaleMap::from([("y", 1.0), ("x", 1.0)])),
        None,
        Some(types),
        "field",
    )
    .unwrap();
    let fms = OmeZarrMultiscale::builder(fimg)
        .scale_factors(Some(ScaleFactors::Ints(vec![])))
        .build()
        .unwrap();
    let scene = OmeZarrScene::new(
        SceneImages::List(vec![mk("a"), mk("b"), mk("c"), mk("d")]),
        tfs,
        vec![world],
        Some(vec![("field".into(), fms)]),
    )
    .unwrap();
    let d = tmp();
    let p = d.path().join("scene.zarr");
    scene.to_ome_zarr(p.to_str().unwrap(), true, true).unwrap();
    compare_stores(&py_store("scene.zarr"), &p);

    let gobs = &obs["scene:graph"];
    let mut nodes: Vec<Value> = scene
        .graph()
        .nodes()
        .iter()
        .map(|(a, b)| json!([a, b]))
        .collect();
    nodes.sort_by_key(|v| v.to_string());
    let mut want_nodes = gobs["nodes"].as_array().unwrap().clone();
    want_nodes.sort_by_key(|v| v.to_string());
    assert_eq!(nodes, want_nodes);
    let pts = vec![
        vec![1.0, 2.0],
        vec![10.5, 3.25],
        vec![20.0, 30.0],
        vec![-3.0, 7.0],
    ];
    for (src, tgt) in [
        (("a", "physical"), ("b", "physical")),
        (("b", "physical"), ("a", "physical")),
        (("a", "physical"), ("c", "physical")),
        (("c", "physical"), ("d", "physical")),
        (("", "world"), ("d", "physical")),
    ] {
        let key = format!("('{}', '{}')->('{}', '{}')", src.0, src.1, tgt.0, tgt.1);
        let want = &gobs["paths"][&key];
        let s = (src.0.to_string(), src.1.to_string());
        let t = (tgt.0.to_string(), tgt.1.to_string());
        let seq = scene.graph().get_sequence(&s, &t, false).unwrap();
        assert_eq!(json!(seq.len()), want["len"], "{key}");
        let aff = seq
            .to_affine()
            .map(|a| json!(a.matrix))
            .unwrap_or(Value::Null);
        assert_eq!(normalize(&aff), normalize(&want["affine"]), "{key} affine");
        let got = scene.graph().transform(&s, &t, &pts).unwrap();
        assert_eq!(
            normalize(&json!(got)),
            normalize(&want["points"]),
            "{key} points"
        );
    }
    let read = OmeZarrScene::from_ome_zarr(py_store("scene.zarr").to_str().unwrap()).unwrap();
    let mut imgs: Vec<&String> = read.images.iter().map(|(k, _)| k).collect();
    imgs.sort();
    assert_eq!(json!(imgs), obs["scene:read"]["images"]);
    assert_eq!(json!(read.graph().node_count()), obs["scene:read"]["nodes"]);
}

#[test]
fn error_types_and_messages() {
    let obs = observations();
    let d = tmp();
    let (_, g2) = group(d.path(), "errors_v2.zarr", Some(Format::V04));
    let (_, g3) = group(d.path(), "errors_v3.zarr", Some(Format::V05));
    let img = |s: &[usize]| LazyArray::from_ndarray(gen_u8(s));
    let wi = |data: LazyArray,
              g: &Group,
              fmt: Option<Format>,
              axes: Option<ome_zarr::axes::AxesInput>| {
        writer::write_image(
            &data,
            g,
            &ImageOptions {
                fmt,
                axes,
                ..Default::default()
            },
        )
        .map(|_| ())
    };
    let s = |v: &[&str]| -> Vec<String> { v.iter().map(|x| x.to_string()).collect() };
    let plate = |rows: &[&str], cols: &[&str], wells: Vec<Value>, acq: Option<Vec<Value>>| {
        writer::write_plate_metadata(
            &g2,
            &s(rows),
            &s(cols),
            &wells,
            None,
            acq.as_deref(),
            None,
            None,
        )
    };
    let mm = |ds: Vec<Value>, md: Option<JsonMap>| {
        writer::write_multiscales_metadata(
            &g2,
            &ds,
            None,
            Some(&"yx".into()),
            None,
            None,
            md.as_ref(),
        )
    };
    let ct = |t: Value| vec![json!({"path": "0", "coordinateTransformations": t})];
    let sc11 = || ct(json!([{"type": "scale", "scale": [1, 1]}]));
    let omero = |o: Value| Some(obj(json!({ "omero": o })));
    let cases: Vec<(&str, ome_zarr::Result<()>)> = vec![
        (
            "fmt_mismatch_v2",
            wi(img(&[8, 8]), &g2, Some(Format::V05), None),
        ),
        (
            "fmt_mismatch_v3",
            wi(img(&[8, 8]), &g3, Some(Format::V04), None),
        ),
        ("axes_3d_missing", wi(img(&[2, 8, 8]), &g3, None, None)),
        (
            "axes_len",
            wi(img(&[2, 8, 8]), &g3, None, Some("yx".into())),
        ),
        (
            "axes_time_order",
            wi(img(&[2, 8, 8]), &g3, None, Some("ytx".into())),
        ),
        (
            "axes_two_channels",
            wi(img(&[2, 2, 8, 8]), &g3, None, Some("ccyx".into())),
        ),
        (
            "axes_channel_after_space",
            wi(img(&[8, 8, 2]), &g3, None, Some("yxc".into())),
        ),
        (
            "axes_unknown",
            wi(
                img(&[2, 2, 8, 8]),
                &g3,
                None,
                Some(vec!["a", "b", "y", "x"].into()),
            ),
        ),
        (
            "deprecated_v03",
            wi(img(&[8, 8]), &g2, Some(Format::V03), None),
        ),
        (
            "rows_dup",
            plate(&["A", "A"], &["1"], vec![json!("A/1")], None),
        ),
        (
            "rows_alnum",
            plate(&["A", "&"], &["1"], vec![json!("A/1")], None),
        ),
        ("wells_empty", plate(&["A"], &["1"], vec![], None)),
        ("well_row", plate(&["A"], &["1"], vec![json!("B/1")], None)),
        ("well_col", plate(&["A"], &["1"], vec![json!("A/2")], None)),
        (
            "well_parts",
            plate(&["A"], &["1"], vec![json!("A/1/0")], None),
        ),
        (
            "well_dict_index",
            plate(
                &["A", "B"],
                &["1"],
                vec![json!({"path": "B/1", "rowIndex": 0, "columnIndex": 0})],
                None,
            ),
        ),
        (
            "well_dict_missing",
            plate(&["A"], &["1"], vec![json!({"path": "A/1"})], None),
        ),
        ("well_type", plate(&["A"], &["1"], vec![json!(1)], None)),
        (
            "acq_id",
            plate(
                &["A"],
                &["1"],
                vec![json!("A/1")],
                Some(vec![json!({"name": "x"})]),
            ),
        ),
        (
            "acq_id_type",
            plate(
                &["A"],
                &["1"],
                vec![json!("A/1")],
                Some(vec![json!({"id": "1"})]),
            ),
        ),
        (
            "image_path",
            writer::write_well_metadata(&g2, &[json!({"acquisition": 0})], None),
        ),
        (
            "image_acq",
            writer::write_well_metadata(&g2, &[json!({"path": "0", "acquisition": "0"})], None),
        ),
        (
            "image_type",
            writer::write_well_metadata(&g2, &[json!(0)], None),
        ),
        ("datasets_empty", mm(vec![], None)),
        ("datasets_type", mm(vec![json!("0")], None)),
        ("datasets_path", mm(vec![json!({"key": 1})], None)),
        ("ct_count", mm(vec![json!({"path": "0"})], None)),
        (
            "ct_first",
            mm(
                ct(
                    json!([{"type": "translation", "translation": [1, 1]}, {"type": "scale", "scale": [1, 1]}]),
                ),
                None,
            ),
        ),
        (
            "ct_two_scales",
            mm(
                ct(json!([{"type": "scale", "scale": [1, 1]}, {"type": "scale", "scale": [1, 1]}])),
                None,
            ),
        ),
        (
            "ct_scale_len",
            mm(ct(json!([{"type": "scale", "scale": [1, 1, 1]}])), None),
        ),
        (
            "ct_scale_type",
            mm(ct(json!([{"type": "scale", "scale": [1, "1"]}])), None),
        ),
        ("ct_missing_type", mm(ct(json!([{"scale": [1, 1]}])), None)),
        (
            "ct_tr_len",
            mm(
                ct(
                    json!([{"type": "scale", "scale": [1, 1]}, {"type": "translation", "translation": [1]}]),
                ),
                None,
            ),
        ),
        ("omero_none", mm(sc11(), omero(Value::Null))),
        (
            "omero_color",
            mm(sc11(), omero(json!({"channels": [{"color": "F00"}]}))),
        ),
        (
            "omero_window_type",
            mm(sc11(), omero(json!({"channels": [{"window": []}]}))),
        ),
        (
            "omero_window_key",
            mm(
                sc11(),
                omero(json!({"channels": [{"window": {"start": 0, "end": 1, "min": 0}}]})),
            ),
        ),
        (
            "omero_window_value",
            mm(
                sc11(),
                omero(
                    json!({"channels": [{"window": {"start": 0, "end": 1, "min": 0, "max": "1"}}]}),
                ),
            ),
        ),
        (
            "image_dims",
            OmeZarrImage::from_axes(img(&[2, 8, 8]), "czyx").map(|_| ()),
        ),
        (
            "image_scale_axes",
            OmeZarrImage::new(
                img(&[2, 8, 8]),
                &s(&["c", "y", "x"]),
                Some(ScaleMap::from([("q", 1.0)])),
                None,
                None,
                "image",
            )
            .map(|_| ()),
        ),
        (
            "channel_names_len",
            OmeZarrMultiscale::builder(OmeZarrImage::from_axes(img(&[2, 8, 8]), "cyx").unwrap())
                .channel_names(&["a"])
                .build()
                .map(|_| ()),
        ),
        (
            "version_unknown",
            ome_zarr::format::format_from_version("9.9").map(|_| ()),
        ),
        ("store_exists", {
            let ms =
                OmeZarrMultiscale::builder(OmeZarrImage::from_axes(img(&[32, 32]), "yx").unwrap())
                    .scale_factors(Some(ScaleFactors::Ints(vec![2])))
                    .build()
                    .unwrap();
            let p = d.path().join("exists.zarr");
            ms.to_ome_zarr(p.as_path(), None, Version::V06, true, false)
                .unwrap();
            ms.to_ome_zarr(p.as_path(), None, Version::V06, true, false)
                .map(|_| ())
        }),
        ("unsupported_version", Version::parse("0.7").map(|_| ())),
    ];
    let mut failures = vec![];
    for (name, r) in cases {
        let want = &obs["errors"][name];
        match r {
            Ok(()) => failures.push(format!("{name}: no error, python raised {want}")),
            Err(e) => {
                let (t, m) = (e.py_type(), e.py_str());
                let wt = want["type"].as_str().unwrap();
                let wm = want["msg"].as_str().unwrap();
                // Python's set ordering is randomised per process
                let same_msg =
                    m == wm || (name == "image_scale_axes" && sorted_chars(&m) == sorted_chars(wm));
                if t != wt || !same_msg {
                    failures.push(format!("{name}:\n  python {wt}: {wm}\n  rust   {t}: {m}"));
                }
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

fn sorted_chars(s: &str) -> Vec<char> {
    let mut v: Vec<char> = s.chars().collect();
    v.sort();
    v
}
