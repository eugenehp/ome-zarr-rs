//! Ports of ome-zarr-py's test_image_class.py, test_multiscale_coords.py and
//! test_scene.py.
#![cfg(not(target_arch = "wasm32"))]

use ome_zarr::array::{DType, LazyArray, NdArray};
use ome_zarr::classes::scene::{OmeZarrScene, SceneImages};
use ome_zarr::classes::tnd_utils::{ozmp_tf_to_tnd, setup_vectorfield_args};
use ome_zarr::classes::{
    ChannelColor, LabelsArg, OmeZarrImage, OmeZarrLabels, OmeZarrMultiscale, ScaleMap, Version,
};
use ome_zarr::models::transforms::{
    Axis, CoordinateSystem, CoordinateSystemIdentifier as Csid, Transform,
};
use ome_zarr::models::{ImageLabel, Multiscale};
use ome_zarr::scale::ScaleFactors;
use ome_zarr::writer::{StorageOptions, StorageOptionsArg, get_metadata};
use ome_zarr::zarr::{Group, Mode};
use serde_json::{Value, json};

fn data(shape: &[usize]) -> NdArray {
    let n: usize = shape.iter().product();
    NdArray::from_f64_vec(
        DType::UInt8,
        shape,
        (0..n).map(|v| ((v * 7919) % 23) as f64).collect(),
    )
    .unwrap()
}

fn img(shape: &[usize], axes: &str, scale: &[(&str, f64)]) -> OmeZarrImage {
    let ax: Vec<String> = axes.chars().map(|c| c.to_string()).collect();
    let sm = ScaleMap(scale.iter().map(|(k, v)| (k.to_string(), *v)).collect());
    OmeZarrImage::new(
        LazyArray::from_ndarray(data(shape)),
        &ax,
        Some(sm),
        None,
        None,
        "image",
    )
    .unwrap()
}

type DimCase = (Vec<usize>, &'static str, Vec<(&'static str, f64)>);

fn dims() -> Vec<DimCase> {
    vec![
        (vec![128, 128], "yx", vec![("y", 0.5), ("x", 0.5)]),
        (vec![2, 128, 128], "cyx", vec![("y", 0.5), ("x", 0.5)]),
        (
            vec![2, 10, 128, 128],
            "czyx",
            vec![("z", 1.0), ("y", 0.5), ("x", 0.5)],
        ),
        (
            vec![3, 2, 10, 128, 128],
            "tczyx",
            vec![("z", 1.0), ("y", 0.5), ("x", 0.5)],
        ),
    ]
}

#[test]
fn additional_transforms() {
    for (shape, axes, scale) in dims() {
        let image = img(&shape, axes, &scale);
        let meta = |c: char| match c {
            'c' => ("channel", "none"),
            't' => ("time", "second"),
            _ => ("space", "micrometer"),
        };
        let world_scale: Vec<f64> = axes
            .chars()
            .map(|c| if "ct".contains(c) { 1.0 } else { 0.5 })
            .collect();
        let world_tr: Vec<f64> = axes
            .chars()
            .map(|c| if "ct".contains(c) { 0.0 } else { 10.0 })
            .collect();
        let t = Transform::from_json(&json!({"type": "sequence", "input": {"name": "physical"}, "output": {"name": "world"},
            "transformations": [{"type": "scale", "scale": world_scale}, {"type": "translation", "translation": world_tr}]}))
        .unwrap();
        let cs = CoordinateSystem::from_json(&json!({"name": "world", "axes": axes.chars().map(|c| json!({"name": c.to_string(), "type": meta(c).0, "unit": meta(c).1})).collect::<Vec<_>>()})).unwrap();
        // missing the "world" coordinate system
        assert!(
            OmeZarrMultiscale::builder(image.clone())
                .coordinate_transformations(vec![t.clone()])
                .build()
                .unwrap_err()
                .is_value()
        );
        let ms = OmeZarrMultiscale::builder(image.clone())
            .coordinate_transformations(vec![t.clone()])
            .coordinate_systems(vec![cs.clone()])
            .build()
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let g = Group::open_url(
            dir.path().join("t.zarr").to_str().unwrap(),
            Mode::Write,
            None,
        )
        .unwrap();
        ms.to_ome_zarr(g.clone(), None, Version::V06, true, true)
            .unwrap();
        let md = get_metadata(&g).unwrap();
        assert_eq!(
            md["multiscales"][0]["coordinateTransformations"][0]["output"]["name"],
            "world"
        );
        let bad = t.with_io(t.input().cloned(), Some(Csid::name("nonexistent")));
        assert!(
            OmeZarrMultiscale::builder(image)
                .coordinate_transformations(vec![bad])
                .coordinate_systems(vec![cs])
                .build()
                .is_err()
        );
    }
}

#[test]
fn image_class_versions() {
    for (shape, axes, scale) in dims() {
        for v in [Version::V04, Version::V05, Version::V06] {
            let ms = OmeZarrMultiscale::new(img(&shape, axes, &scale)).unwrap();
            let dir = tempfile::tempdir().unwrap();
            let zf = if v == Version::V04 { 2 } else { 3 };
            let g = Group::open_url(
                dir.path().join("v.zarr").to_str().unwrap(),
                Mode::Write,
                Some(zf),
            )
            .unwrap();
            ms.to_ome_zarr(g.clone(), None, v, true, true).unwrap();
            let attrs = g.attrs().unwrap();
            let version = if v == Version::V04 {
                attrs["multiscales"][0]["version"].clone()
            } else {
                attrs["ome"]["version"].clone()
            };
            assert_eq!(version, v.as_str());
            let back = OmeZarrMultiscale::from_ome_zarr(g).unwrap();
            assert_eq!(back.images().len(), 5);
            assert_eq!(back.images()[0].data.compute().unwrap(), data(&shape));
            assert_eq!(back.images()[0].scale.get("y"), Some(0.5));
        }
    }
}

#[test]
fn image_class_bad_args() {
    let d = LazyArray::from_ndarray(data(&[2, 128, 128]));
    let ax = |s: &str| s.chars().map(|c| c.to_string()).collect::<Vec<_>>();
    assert!(
        OmeZarrImage::new(d.clone(), &ax("czyx"), None, None, None, "i")
            .unwrap_err()
            .is_value()
    );
    let sm = ScaleMap::from([("c", 1.0), ("z", 0.5), ("y", 0.5), ("x", 0.5)]);
    assert!(
        OmeZarrImage::new(d.clone(), &ax("zyx"), Some(sm), None, None, "i")
            .unwrap_err()
            .is_value()
    );
    let image = OmeZarrImage::new(
        d,
        &ax("cyx"),
        Some(ScaleMap::from([("y", 0.5), ("x", 0.5)])),
        None,
        None,
        "i",
    )
    .unwrap();
    assert_eq!(image.scale.get("c"), Some(1.0));
    assert!(
        OmeZarrMultiscale::builder(image.clone())
            .channel_names(&["Channel 0"])
            .build()
            .unwrap_err()
            .is_type()
    );
    assert!(
        OmeZarrMultiscale::builder(image.clone())
            .channel_names(&["a", "b"])
            .channel_colors(vec!["#ff0000".into()])
            .build()
            .unwrap_err()
            .is_type()
    );
    assert!(
        OmeZarrMultiscale::builder(image.clone())
            .channel_names(&["a", "b"])
            .contrast_limits(vec![(0.0, 255.0)])
            .build()
            .unwrap_err()
            .is_type()
    );
    let ms = OmeZarrMultiscale::builder(image)
        .channel_names(&["Channel 0", "Channel 1"])
        .channel_colors(vec![
            ChannelColor::Rgb(vec![255, 0, 0]),
            ChannelColor::Rgb(vec![0, 255, 0]),
        ])
        .contrast_limits(vec![(0.0, 255.0), (0.0, 255.0)])
        .build()
        .unwrap();
    assert_eq!(ms.images().len(), 5);
    assert_eq!(ms.omero.as_ref().unwrap().channels[1].color, "00ff00");
    let dir = tempfile::tempdir().unwrap();
    ms.to_ome_zarr(
        dir.path().join("bad_args.zarr").as_path(),
        None,
        Version::V05,
        true,
        false,
    )
    .unwrap();
}

#[test]
fn image_class_writer_default_scale_and_storage_options() {
    let image = img(&[32, 256, 256], "zyx", &[]);
    assert!(
        "zyx"
            .chars()
            .all(|d| image.scale.get(&d.to_string()) == Some(1.0))
    );
    for (shape, axes, scale) in dims() {
        for per_level in [true, false] {
            let ms = OmeZarrMultiscale::builder(img(&shape, axes, &scale))
                .scale_factors(Some(ScaleFactors::per_dim(vec![
                    axes.chars()
                        .map(|d| (d.to_string(), if "xy".contains(d) { 2.0 } else { 1.0 }))
                        .collect(),
                ])))
                .build()
                .unwrap();
            let chunks: Vec<Vec<u64>> = vec![
                shape.iter().map(|&s| (s as u64).min(32)).collect(),
                shape.iter().map(|&s| (s as u64).min(16)).collect(),
            ];
            let so: StorageOptionsArg = if per_level {
                StorageOptionsArg::PerLevel(
                    chunks.iter().map(|c| StorageOptions::chunks(c)).collect(),
                )
            } else {
                StorageOptions::chunks(&chunks[0]).into()
            };
            let dir = tempfile::tempdir().unwrap();
            let p = dir.path().join("test");
            ms.to_ome_zarr(p.as_path(), Some(&so), Version::V06, true, true)
                .unwrap();
            let g = Group::open_url(p.to_str().unwrap(), Mode::Read, None).unwrap();
            for (level, path) in ["s0", "s1"].iter().enumerate() {
                let a = g.array(path).unwrap();
                let expected = if per_level {
                    &chunks[level]
                } else {
                    &chunks[0]
                };
                assert_eq!(
                    a.chunks(),
                    ome_zarr::array::retuple(expected, a.shape()).as_slice()
                );
            }
        }
    }
}

#[test]
fn image_class_writer_labels() {
    for (shape, axes, scale) in dims() {
        for v in [Version::V04, Version::V05, Version::V06] {
            let d = data(&shape);
            let lbl = NdArray::UInt8(d.to_f64().mapv(|x| if x > 11.0 { 1 } else { 0 }));
            let ax: Vec<String> = axes.chars().map(|c| c.to_string()).collect();
            let sm = ScaleMap(scale.iter().map(|(k, v)| (k.to_string(), *v)).collect());
            let li = OmeZarrImage::new(
                LazyArray::from_ndarray(lbl),
                &ax,
                Some(sm.clone()),
                None,
                None,
                "test_labels",
            )
            .unwrap();
            let labels = OmeZarrLabels::new(li, None, None, true).unwrap();
            let ms = OmeZarrMultiscale::builder(
                OmeZarrImage::new(
                    LazyArray::from_ndarray(d),
                    &ax,
                    Some(sm),
                    None,
                    None,
                    "image",
                )
                .unwrap(),
            )
            .labels(LabelsArg::One(labels))
            .build()
            .unwrap();
            let dir = tempfile::tempdir().unwrap();
            let p = dir.path().join("test");
            ms.to_ome_zarr(p.as_path(), None, v, true, true).unwrap();
            let lg = Group::open_url(p.join("labels").to_str().unwrap(), Mode::Read, None).unwrap();
            assert_eq!(get_metadata(&lg).unwrap()["labels"], json!(["test_labels"]));
            let back = OmeZarrMultiscale::from_ome_zarr(p.as_path()).unwrap();
            assert!(back.label("test_labels").is_some());
            if axes.contains('c') {
                assert!(back.omero.is_some());
            }
            assert!(back.label("test_labels").unwrap().image_label.is_some());
        }
    }
}

#[test]
fn bad_image_label_metadata() {
    let bad = json!({"colors": [1, 2, 3, 4], "properties": {"prop1": "value1"}});
    assert!(ImageLabel::from_json(&bad).is_err());
    // `LabelBase.model_validate([1])`: pydantic's ValidationError
    let e = ImageLabel::from_json(&json!([1])).unwrap_err();
    assert_eq!(e.py_type(), "ValidationError");
    assert!(
        e.py_str()
            .contains("Input should be a valid dictionary or instance of LabelBase")
    );
}

#[test]
fn image_class_writer_transformations() {
    for shape in [
        vec![256usize, 256],
        vec![32, 256, 256],
        vec![2, 32, 256, 256],
        vec![1, 2, 1, 256, 256],
    ] {
        for v in [Version::V04, Version::V05, Version::V06] {
            let axes: String = "tczyx"[5 - shape.len()..].to_string();
            let scale0 = [1.0, 1.0, 0.5, 0.18, 0.18];
            let s0 = &scale0[5 - shape.len()..];
            let sm = ScaleMap(
                axes.chars()
                    .map(|c| c.to_string())
                    .zip(s0.iter().copied())
                    .collect(),
            );
            let ax: Vec<String> = axes.chars().map(|c| c.to_string()).collect();
            let image = OmeZarrImage::new(
                LazyArray::from_ndarray(data(&shape)),
                &ax,
                Some(sm),
                None,
                None,
                "image",
            )
            .unwrap();
            let sf = ScaleFactors::per_dim(
                (1..5)
                    .map(|i| {
                        ax.iter()
                            .map(|d| {
                                (
                                    d.clone(),
                                    if d == "x" || d == "y" {
                                        2f64.powi(i)
                                    } else {
                                        1.0
                                    },
                                )
                            })
                            .collect()
                    })
                    .collect::<Vec<_>>(),
            );
            let ms = OmeZarrMultiscale::builder(image)
                .scale_factors(Some(sf))
                .build()
                .unwrap();
            let dir = tempfile::tempdir().unwrap();
            let p = dir.path().join("test");
            ms.to_ome_zarr(p.as_path(), None, v, true, true).unwrap();
            let g = Group::open_url(p.to_str().unwrap(), Mode::Read, None).unwrap();
            let md = get_metadata(&g).unwrap();
            let datasets = md["multiscales"][0]["datasets"].as_array().unwrap().clone();
            let shape0 = g.array("s0").unwrap().shape().to_vec();
            for (level, ds) in datasets.iter().enumerate() {
                let cts: Vec<Value> = if v == Version::V06 {
                    ds["coordinateTransformations"][0]["transformations"]
                        .as_array()
                        .unwrap()
                        .clone()
                } else {
                    ds["coordinateTransformations"].as_array().unwrap().clone()
                };
                assert_eq!(cts.len(), 2);
                let sc: Vec<f64> = cts[0]["scale"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap())
                    .collect();
                let a = g.array(ds["path"].as_str().unwrap()).unwrap();
                for (i, d) in axes.chars().enumerate() {
                    let rel = shape0[i] as f64 / a.shape()[i] as f64;
                    let want = s0[i] * rel;
                    if level == 0 {
                        assert_eq!(sc[i], s0[i]);
                    } else {
                        assert_eq!(sc[i], want, "{d} level {level}");
                    }
                    if "tcz".contains(d) {
                        assert_eq!(rel, 1.0);
                    }
                }
            }
        }
    }
}

fn two_cs() -> (CoordinateSystem, CoordinateSystem) {
    (
        CoordinateSystem::new(
            "cs1",
            vec![Axis::new("y", Some("space")), Axis::new("x", Some("space"))],
        )
        .unwrap(),
        CoordinateSystem::new(
            "cs2",
            vec![Axis::new("x", Some("space")), Axis::new("y", Some("space"))],
        )
        .unwrap(),
    )
}

fn zeros_image() -> OmeZarrImage {
    let a = NdArray::zeros(DType::Float64, &[32, 64]);
    OmeZarrImage::from_axes(LazyArray::from_ndarray(a), "yx").unwrap()
}

#[test]
fn local_and_descendant_coordinate_systems() {
    let (cs1, cs2) = two_cs();
    let ms = OmeZarrMultiscale::builder(zeros_image())
        .coordinate_systems(vec![cs1, cs2.clone()])
        .build()
        .unwrap();
    assert_eq!(ms.base.get_local_coordinate_system("cs2"), Some(&cs2));
    assert!(
        ms.base
            .get_local_coordinate_system("notacoordinatesystem")
            .is_none()
    );
    assert_eq!(
        ms.get_descendant_coordinate_system(&Csid::name("cs2")),
        Some(&cs2)
    );
    assert!(
        ms.get_descendant_coordinate_system(&Csid::name("notacoordinatesystem"))
            .is_none()
    );

    let mut img = OmeZarrMultiscale::new(zeros_image()).unwrap();
    assert!(
        img.get_descendant_coordinate_system(&Csid::name_path("lbl", "labels/fake"))
            .is_none()
    );
    img.set_labels(Some(LabelsArg::Map(vec![])));
    assert!(
        img.get_descendant_coordinate_system(&Csid::name_path("lbl", "labels/fake"))
            .is_none()
    );
    let mut lbls = OmeZarrLabels::from_image(
        OmeZarrImage::from_axes(
            LazyArray::from_ndarray(NdArray::zeros(DType::UInt64, &[32, 64])),
            "yx",
        )
        .unwrap(),
    )
    .unwrap();
    let realcs = CoordinateSystem::new(
        "physical",
        vec![Axis::new("y", Some("space")), Axis::new("x", Some("space"))],
    )
    .unwrap();
    let md = Multiscale {
        coordinate_systems: vec![realcs.clone()],
        datasets: lbls.base.metadata.datasets.clone(),
        coordinate_transformations: None,
        metadata: None,
        name: None,
        type_: None,
        extra: Default::default(),
    };
    md.validate().unwrap();
    lbls.base.metadata = md;
    img.set_labels(Some(LabelsArg::Map(vec![("labels/real".into(), lbls)])));
    assert_eq!(
        img.get_descendant_coordinate_system(&Csid::name_path("physical", "labels/real")),
        Some(&realcs)
    );
}

fn tf(v: Value) -> Transform {
    Transform::from_json(&v).unwrap()
}

fn transforms() -> Vec<Transform> {
    vec![
        tf(json!({"type": "scale", "scale": [1.0, 1.0]})),
        tf(json!({"type": "translation", "translation": [0.0, 0.0]})),
        tf(json!({"type": "rotation", "rotation": [[0.0, -1.0], [1.0, 0.0]]})),
        tf(json!({"type": "affine", "affine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]})),
        tf(json!({"type": "mapAxis", "mapAxis": [1, 0]})),
        tf(
            json!({"type": "sequence", "transformations": [{"type": "scale", "scale": [1.0, 1.0]}, {"type": "translation", "translation": [0.0, 0.0]}]}),
        ),
        tf(json!({"type": "byDimension", "transformations": [
            {"inputAxes": [1], "outputAxes": [1], "transformation": {"type": "translation", "translation": [0.0]}},
            {"inputAxes": [0], "outputAxes": [0], "transformation": {"type": "scale", "scale": [1.0]}}]})),
    ]
}

fn yx_image(name: &str) -> OmeZarrMultiscale {
    let d = data(&[64, 64]);
    OmeZarrMultiscale::new(
        OmeZarrImage::from_axes(LazyArray::from_ndarray(d), "yx")
            .unwrap()
            .with_name(name),
    )
    .unwrap()
}

#[test]
fn scene_without_coordinate_systems() {
    for t in transforms() {
        let a = yx_image("imageA");
        let additional_cs = CoordinateSystem::from_json(&json!({"name": "additional", "axes": [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]})).unwrap();
        let additional_tf = tf(
            json!({"type": "rotation", "rotation": [[0.0, -1.0], [1.0, 0.0]], "input": {"name": "physical"}, "output": {"name": "additional"}}),
        );
        let b = OmeZarrMultiscale::builder(
            OmeZarrImage::from_axes(LazyArray::from_ndarray(data(&[64, 64])), "yx")
                .unwrap()
                .with_name("imageB"),
        )
        .coordinate_systems(vec![additional_cs])
        .coordinate_transformations(vec![additional_tf])
        .build()
        .unwrap();
        let t = t.with_io(
            Some(Csid::name_path("physical", "imageA")),
            Some(Csid::name_path("physical", "imageB")),
        );
        let scene = OmeZarrScene::new(SceneImages::List(vec![a, b]), vec![t.clone()], vec![], None)
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("test_scene.zarr");
        scene.to_ome_zarr(p.to_str().unwrap(), true, true).unwrap();
        ome_zarr::utils::download(
            p.to_str().unwrap(),
            dir.path().join("downloaded").to_str().unwrap(),
        )
        .unwrap();
        assert!(
            dir.path()
                .join("downloaded/test_scene.zarr/imageA/s0")
                .exists()
        );
        assert_eq!(scene.graph().node_count(), 3);
        let seq = scene
            .graph()
            .get_sequence(
                &("imageA".into(), "physical".into()),
                &("imageB".into(), "physical".into()),
                false,
            )
            .unwrap();
        if t.type_name() != "byDimension" {
            let m = seq.to_affine().unwrap().matrix;
            assert_eq!(m.len(), m[0].len());
        }
        let read = OmeZarrScene::from_ome_zarr(p.to_str().unwrap()).unwrap();
        assert_eq!(read.graph().node_count(), 3);
        let g = Group::open_url(p.to_str().unwrap(), Mode::Read, None).unwrap();
        let ome = g.attrs().unwrap()["ome"].clone();
        assert_eq!(ome["version"], "0.6");
        assert_eq!(
            ome["scene"]["coordinateTransformations"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(ome["scene"]["coordinateTransformations"][0], t.to_json());
    }
}

#[test]
fn scene_with_coordinate_systems() {
    for t in transforms() {
        let a = yx_image("imageA");
        let b = yx_image("imageB");
        let axes = a.metadata().coordinate_systems[0].axes.clone();
        let world = CoordinateSystem::new("world", axes.clone()).unwrap();
        let world2 = CoordinateSystem::new("world2", axes).unwrap();
        let t1 = t.with_io(
            Some(Csid::name_path("physical", "imageA")),
            Some(Csid::name("world")),
        );
        let t2 = t.with_io(Some(Csid::name("world")), Some(Csid::name("world2")));
        let t3 = t.with_io(
            Some(Csid::name("world2")),
            Some(Csid::name_path("physical", "imageB")),
        );
        let scene = OmeZarrScene::new(
            SceneImages::List(vec![a, b]),
            vec![t1, t2, t3],
            vec![world, world2],
            None,
        )
        .unwrap();
        assert_eq!(scene.graph().node_count(), 4);
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("cs.zarr");
        scene.to_ome_zarr(p.to_str().unwrap(), true, true).unwrap();
        ome_zarr::utils::download(p.to_str().unwrap(), dir.path().join("dl").to_str().unwrap())
            .unwrap();
        let read = OmeZarrScene::from_ome_zarr(p.to_str().unwrap()).unwrap();
        assert_eq!(read.graph().node_count(), 4);
        let ome = Group::open_url(p.to_str().unwrap(), Mode::Read, None)
            .unwrap()
            .attrs()
            .unwrap()["ome"]
            .clone();
        assert_eq!(
            ome["scene"]["coordinateSystems"].as_array().unwrap().len(),
            2
        );
    }
}

#[test]
fn coordinate_system_retrieval_and_appending() {
    let a = yx_image("imageA");
    let b = yx_image("imageB");
    let c = yx_image("imageC");
    let axes: Vec<Value> = a.metadata().coordinate_systems[0]
        .axes
        .iter()
        .map(|x| serde_json::to_value(x).unwrap())
        .collect();
    let world = json!({"name": "world", "axes": axes});
    let world2 = json!({"name": "world2", "axes": axes});
    let t =
        |i: Value, o: Value| json!({"type": "scale", "scale": [1.0, 1.0], "input": i, "output": o});
    let ts = vec![
        t(
            json!({"name": "physical", "path": "imageA"}),
            json!({"name": "world"}),
        ),
        t(json!({"name": "world"}), json!({"name": "world2"})),
        t(
            json!({"name": "world2"}),
            json!({"name": "physical", "path": "imageB"}),
        ),
    ];
    let scene = OmeZarrScene::from_json_parts(
        SceneImages::List(vec![a.clone(), b.clone()]),
        &ts,
        &[world.clone(), world2],
    )
    .unwrap();
    let keys = |v: Vec<((String, String), CoordinateSystem)>| {
        v.into_iter().map(|(k, _)| k).collect::<Vec<_>>()
    };
    let k = |p: &str, n: &str| (p.to_string(), n.to_string());
    let all = keys(scene.get_coordinate_system(None, None));
    for want in [
        k("", "world"),
        k("", "world2"),
        k("imageA", "physical"),
        k("imageB", "physical"),
    ] {
        assert!(all.contains(&want));
    }
    let top = keys(scene.get_coordinate_system(Some(""), None));
    assert!(top.contains(&k("", "world")) && !top.contains(&k("imageA", "physical")));
    let ia = keys(scene.get_coordinate_system(Some("imageA"), None));
    assert!(
        ia.contains(&k("imageA", "physical"))
            && !ia.contains(&k("", "world"))
            && !ia.contains(&k("imageB", "physical"))
    );
    let phys = keys(scene.get_coordinate_system(None, Some("physical")));
    assert!(!phys.contains(&k("", "world")) && phys.len() == 2);

    // appending
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("append.zarr");
    let tf1 = t(
        json!({"name": "physical", "path": "imageA"}),
        json!({"name": "world"}),
    );
    let tf2 = t(
        json!({"name": "world"}),
        json!({"name": "physical", "path": "imageB"}),
    );
    let tf3 = t(
        json!({"name": "world"}),
        json!({"name": "physical", "path": "imageC"}),
    );
    let scene = OmeZarrScene::from_json_parts(
        SceneImages::List(vec![a, b]),
        &[tf1, tf2],
        std::slice::from_ref(&world),
    )
    .unwrap();
    scene.to_ome_zarr(p.to_str().unwrap(), true, true).unwrap();
    let read = OmeZarrScene::from_ome_zarr(p.to_str().unwrap()).unwrap();
    let mut imgs: Vec<OmeZarrMultiscale> = read.images.iter().map(|(_, i)| i.clone()).collect();
    imgs.push(c);
    let mut tfs: Vec<Transform> = read.coordinate_transformations().to_vec();
    tfs.push(tf(tf3));
    let new_scene = OmeZarrScene::new(
        SceneImages::List(imgs),
        tfs,
        vec![CoordinateSystem::from_json(&world).unwrap()],
        None,
    )
    .unwrap();
    new_scene
        .to_ome_zarr(p.to_str().unwrap(), false, true)
        .unwrap();
    assert_eq!(new_scene.graph().node_count(), 4);
    let g = Group::open_url(p.to_str().unwrap(), Mode::Read, None).unwrap();
    for i in ["imageA", "imageB", "imageC"] {
        assert!(g.contains(i) && g.group(i).unwrap().contains("s0"));
    }
    let sm = g.attrs().unwrap()["ome"]["scene"].clone();
    assert_eq!(sm["coordinateTransformations"].as_array().unwrap().len(), 3);
    assert_eq!(sm["coordinateSystems"].as_array().unwrap().len(), 1);
}

#[test]
fn scene_with_displacements() {
    let a = yx_image("imageA");
    let b = yx_image("imageB");
    let vf = NdArray::zeros(DType::Float32, &[2, 64, 64]);
    let ax: Vec<String> = "cyx".chars().map(|c| c.to_string()).collect();
    let types = [
        ("c".to_string(), Some("displacement".to_string())),
        ("y".into(), Some("space".into())),
        ("x".into(), Some("space".into())),
    ]
    .into();
    let dimg = OmeZarrImage::new(
        LazyArray::from_ndarray(vf),
        &ax,
        Some(ScaleMap::from([("y", 1.0), ("x", 1.0)])),
        None,
        Some(types),
        "displacementField",
    )
    .unwrap();
    let dms = OmeZarrMultiscale::new(dimg).unwrap();
    let t = tf(
        json!({"type": "displacements", "input": {"name": "physical", "path": "imageA"}, "output": {"name": "physical", "path": "imageB"}, "path": "coordinateTransformations/displacementField"}),
    );
    let scene = OmeZarrScene::new(
        SceneImages::List(vec![a, b]),
        vec![t],
        vec![],
        Some(vec![("displacementField".into(), dms)]),
    )
    .unwrap();
    // zero displacements: points map to themselves
    let out = scene
        .graph()
        .transform(
            &("imageA".into(), "physical".into()),
            &("imageB".into(), "physical".into()),
            &vec![vec![10.0, 20.0]],
        )
        .unwrap();
    assert_eq!(out, vec![vec![10.0, 20.0]]);
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("disp.zarr");
    scene.to_ome_zarr(p.to_str().unwrap(), true, true).unwrap();
    let g = Group::open_url(p.to_str().unwrap(), Mode::Read, None).unwrap();
    assert!(
        g.contains("coordinateTransformations") && g.contains("imageA") && g.contains("imageB")
    );
    let dg = g
        .group("coordinateTransformations/displacementField")
        .unwrap();
    let ome = dg.attrs().unwrap()["ome"].clone();
    assert_eq!(ome["version"], "0.6");
    let ax0 = &ome["multiscales"][0]["coordinateSystems"][0]["axes"][0];
    assert_eq!(ax0["type"], "displacement");
    assert_eq!(ax0["discrete"], true);
    let read = OmeZarrScene::from_ome_zarr(p.to_str().unwrap()).unwrap();
    let d = &read
        .coordinate_displacements
        .iter()
        .find(|(k, _)| k == "displacementField")
        .unwrap()
        .1;
    for i in d.images() {
        assert_eq!(i.axes_types["c"].as_deref(), Some("displacement"));
    }
    assert_eq!(
        d.metadata().coordinate_systems[0].axes[0].discrete,
        Some(true)
    );
}

fn check_equivalent(ozm: &Transform, t: &ome_zarr::transformnd::Transform) {
    let ndim = t.ndims().source;
    let pts: Vec<Vec<f64>> = (0..10)
        .map(|i| {
            (0..ndim)
                .map(|j| ((i * 37 + j * 11) % 200) as f64 - 100.0)
                .collect()
        })
        .collect();
    let got = t.apply(&pts).unwrap();
    match pts
        .iter()
        .map(|p| ozm.transform_point(p))
        .collect::<ome_zarr::Result<Vec<_>>>()
    {
        Ok(want) => {
            for (a, b) in got.iter().zip(&want) {
                for (x, y) in a.iter().zip(b) {
                    assert!((x - y).abs() < 1e-9, "{ozm:?}: {a:?} vs {b:?}");
                }
            }
        }
        Err(ome_zarr::Error::NotImplemented(_)) => {}
        Err(e)
            if matches!(ozm, Transform::Affine(_)) && e.to_string().contains("imensionality") => {}
        Err(e) => panic!("{e}"),
    }
    if let Ok(true) = ozm.has_inverse() {
        assert!(
            t.invert().is_some(),
            "ome-zarr-models implements inversion but transformnd does not"
        );
    }
}

#[test]
fn convert_transformations() {
    let cases: Vec<(Value, Option<usize>)> = vec![
        (json!({"type": "identity"}), Some(3)),
        (json!({"type": "scale", "scale": [2, 3]}), None),
        (json!({"type": "translation", "translation": [2, 3]}), None),
        (
            json!({"type": "affine", "affine": [[1, 0, 0], [0, 1, 0]]}),
            None,
        ),
        (
            json!({"type": "affine", "affine": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}),
            None,
        ),
        (json!({"type": "mapAxis", "mapAxis": [2, 1, 0]}), None),
        (
            json!({"type": "projectAxis", "createdOutputs": [1], "droppedInputs": [0]}),
            Some(3),
        ),
        (
            json!({"type": "rotation", "rotation": [[1, 0], [0, 1]]}),
            None,
        ),
        (
            json!({"type": "byDimension", "transformations": [
            {"inputAxes": [0, 2], "outputAxes": [0, 2], "transformation": {"type": "identity"}},
            {"inputAxes": [1], "outputAxes": [1], "transformation": {"type": "translation", "translation": [10]}}]}),
            None,
        ),
        (
            json!({"type": "sequence", "transformations": [{"type": "translation", "translation": [2, 3]}, {"type": "scale", "scale": [10, 100]}]}),
            None,
        ),
        (
            json!({"type": "sequence", "transformations": [{"type": "projectAxis", "createdOutputs": [0]}, {"type": "translation", "translation": [30, 0, 0]}]}),
            Some(2),
        ),
    ];
    for (v, src) in cases {
        let ozm = tf(v);
        let t = ozmp_tf_to_tnd(&ozm, "", src, None, &vec![]).unwrap();
        check_equivalent(&ozm, &t);
    }
}

#[test]
fn unsupported_and_sequences() {
    let t = tf(
        json!({"type": "displacements", "input": {"name": "in_name", "path": "in_path"}, "output": {"name": "out_name", "path": "out_path"}, "path": "dpath"}),
    );
    assert!(matches!(
        setup_vectorfield_args(&t, "", &vec![]),
        Err(ome_zarr::Error::UnsupportedTransformation { .. })
    ));
    for v in [
        json!({"type": "identity"}),
        json!({"type": "sequence", "transformations": []}),
        json!({"type": "projectAxis", "droppedInputs": [0]}),
    ] {
        assert!(matches!(
            ozmp_tf_to_tnd(&tf(v), "", None, None, &vec![]),
            Err(ome_zarr::Error::UnsupportedTransformation { .. })
        ));
    }
    for n in 1..=4 {
        let seq = json!({"type": "sequence", "transformations": vec![json!({"type": "scale", "scale": [2, 3]}); n]});
        ozmp_tf_to_tnd(&tf(seq), ",", None, None, &vec![]).unwrap();
    }
}
