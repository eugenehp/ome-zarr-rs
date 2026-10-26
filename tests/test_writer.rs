//! Ports of ome-zarr-py's test_writer.py.
#![cfg(not(target_arch = "wasm32"))]

use ome_zarr::array::{DType, LazyArray, NdArray};
use ome_zarr::axes::AxesInput;
use ome_zarr::classes::{OmeZarrImage, OmeZarrMultiscale, ScaleMap, Version};
use ome_zarr::format::{CURRENT_FORMAT, Format};
use ome_zarr::scale::{Methods, ScaleFactors, Scaler};
use ome_zarr::writer::{
    self, ImageOptions, LabelsOptions, MultiscaleOptions, StorageOptions, compute_all,
};
use ome_zarr::zarr::{Compressor, Group, JsonMap, Mode, ZArray};
use serde_json::{Value, json};

fn data(shape: &[usize]) -> NdArray {
    let n: usize = shape.iter().product();
    // deterministic pseudo-random data
    let mut x: u64 = 88172645463325252;
    let v: Vec<f64> = (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % 21) as f64
        })
        .collect();
    NdArray::from_f64_vec(DType::UInt8, shape, v).unwrap()
}

fn lazy(shape: &[usize]) -> LazyArray {
    LazyArray::from_ndarray(data(shape))
}

struct Dirs {
    _tmp: tempfile::TempDir,
    path: std::path::PathBuf,
    group: Group,
    group_v3: Group,
}

fn dirs() -> Dirs {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("data");
    let root = Group::open_url(path.to_str().unwrap(), Mode::Write, Some(2)).unwrap();
    let group = root.create_group("test", false).unwrap();
    let root3 = Group::open_url(path.join("v3").to_str().unwrap(), Mode::Write, Some(3)).unwrap();
    let group_v3 = root3.create_group("test", false).unwrap();
    Dirs {
        _tmp: tmp,
        path,
        group,
        group_v3,
    }
}

fn ome(g: &Group) -> JsonMap {
    writer::get_metadata(g).unwrap()
}

fn dataset_paths(md: &JsonMap) -> Vec<String> {
    md["multiscales"][0]["datasets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d["path"].as_str().unwrap().to_string())
        .collect()
}

#[test]
fn writer_formats_shapes_and_storage_options() {
    for shape in [
        vec![1usize, 2, 1, 256, 256],
        vec![32, 256, 256],
        vec![256, 256],
    ] {
        for fmt in [Format::V04, Format::V05] {
            for per_level in [true, false] {
                let d = dirs();
                let grp_path = if fmt == Format::V05 {
                    d.path.join("v3/test")
                } else {
                    d.path.join("test")
                };
                let axes: String = "tczyx"[5 - shape.len()..].to_string();
                let mut units = JsonMap::new();
                for a in axes.chars() {
                    match a {
                        't' => units.insert("t".into(), json!("second")),
                        'z' | 'y' | 'x' => units.insert(a.to_string(), json!("micrometer")),
                        _ => None,
                    };
                }
                let chunks: Vec<Vec<u64>> = if fmt == Format::V05 {
                    vec![vec![32, 32]; 6]
                } else {
                    vec![
                        vec![128, 128],
                        vec![50, 50],
                        vec![25, 25],
                        vec![25, 25],
                        vec![25, 25],
                    ]
                };
                // as in Python, per-level options only differ for v0.4 here
                let so = if per_level && fmt != Format::V05 {
                    ome_zarr::writer::StorageOptionsArg::PerLevel(
                        chunks.iter().map(|c| StorageOptions::chunks(c)).collect(),
                    )
                } else {
                    StorageOptions::chunks(&chunks[0]).into()
                };
                let scale_vals = [1.0, 1.0, 0.5, 0.18, 0.18];
                let scale = ScaleMap(
                    axes.chars()
                        .map(|c| c.to_string())
                        .zip(scale_vals[5 - shape.len()..].iter().copied())
                        .collect(),
                );
                let opts = ImageOptions {
                    fmt: Some(fmt),
                    axes: Some(axes.as_str().into()),
                    axes_units: Some(units.clone()),
                    scale: Some(scale.clone()),
                    storage_options: Some(so),
                    ..Default::default()
                };
                let img = data(&shape);
                writer::write_image(
                    &LazyArray::from_ndarray(img.clone()),
                    grp_path.to_str().unwrap(),
                    &opts,
                )
                .unwrap();
                let out = Group::open_url(grp_path.to_str().unwrap(), Mode::Read, None).unwrap();
                let md = ome(&out);
                let paths = dataset_paths(&md);
                let arrays: Vec<ZArray> = paths.iter().map(|p| out.array(p).unwrap()).collect();
                assert_eq!(
                    arrays[0].shape(),
                    shape
                        .iter()
                        .map(|&s| s as u64)
                        .collect::<Vec<_>>()
                        .as_slice()
                );
                for ax in md["multiscales"][0]["axes"].as_array().unwrap() {
                    let n = ax["name"].as_str().unwrap();
                    if let Some(u) = units.get(n) {
                        assert_eq!(&ax["unit"], u);
                    }
                }
                // scale doubles in x/y per level, translation = s/2 - s0/2
                for (lvl, ds) in md["multiscales"][0]["datasets"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .enumerate()
                {
                    let ct = &ds["coordinateTransformations"];
                    let sc: Vec<f64> = ct[0]["scale"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap())
                        .collect();
                    let s0 = &scale.values();
                    let f = 2f64.powi(lvl as i32);
                    let n = sc.len();
                    assert!((sc[n - 1] - s0[n - 1] * f).abs() < 1e-12);
                    let tl = ct[1]["translation"][n - 1].as_f64().unwrap();
                    assert!((tl - (sc[n - 1] / 2.0 - s0[n - 1] / 2.0)).abs() < 1e-12);
                }
                for (level, a) in arrays.iter().take(2).enumerate() {
                    let expected = if per_level && fmt != Format::V05 {
                        &chunks[level]
                    } else {
                        &chunks[0]
                    };
                    assert_eq!(
                        a.chunks(),
                        ome_zarr::array::retuple(expected, a.shape()).as_slice()
                    );
                }
                assert_eq!(arrays[0].read_all().unwrap(), img);
            }
        }
    }
}

#[test]
fn writer_with_scale() {
    for shape in [
        vec![1usize, 2, 1, 256, 256],
        vec![32, 256, 256],
        vec![256, 256],
    ] {
        for fmt in [Format::V04, Format::V05] {
            let d = dirs();
            let grp_path = if fmt == Format::V05 {
                d.path.join("v3/test")
            } else {
                d.path.join("test")
            };
            let axes: String = "tczyx"[5 - shape.len()..].to_string();
            let mut scale: Vec<(String, f64)> = axes
                .chars()
                .map(|c| (c.to_string(), if c == 'y' || c == 'x' { 0.5 } else { 1.0 }))
                .collect();
            let opts = ImageOptions {
                fmt: Some(fmt),
                axes: Some(axes.as_str().into()),
                scale: Some(ScaleMap(scale.clone())),
                ..Default::default()
            };
            writer::write_image(&lazy(&shape), grp_path.to_str().unwrap(), &opts).unwrap();
            let out = Group::open_url(grp_path.to_str().unwrap(), Mode::Read, None).unwrap();
            let md = ome(&out);
            let arrays: Vec<ZArray> = dataset_paths(&md)
                .iter()
                .map(|p| out.array(p).unwrap())
                .collect();
            for (idx, ds) in md["multiscales"][0]["datasets"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
            {
                let t = &ds["coordinateTransformations"][0]["scale"];
                if idx > 0 {
                    for (i, s) in scale.iter_mut().enumerate() {
                        let rel = arrays[idx].shape()[i] as f64 / arrays[idx - 1].shape()[i] as f64;
                        s.1 /= rel;
                    }
                }
                let want: Vec<f64> = scale.iter().map(|s| s.1).collect();
                let got: Vec<f64> = t
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect();
                for (a, b) in got.iter().zip(&want) {
                    assert!((a - b).abs() < 1e-9, "{got:?} vs {want:?}");
                }
            }
        }
    }
}

#[test]
fn mix_zarr_formats() {
    let d = dirs();
    let img = lazy(&[64, 64, 64]);
    let e = writer::write_image(
        &img,
        &d.group,
        &ImageOptions {
            axes: Some("zyx".into()),
            fmt: Some(CURRENT_FORMAT),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(e.to_string().contains("Group is zarr_format: 2"));
    let mo = MultiscaleOptions {
        fmt: Some(Format::V04),
        ..MultiscaleOptions::new()
    };
    let e = writer::write_multiscale(std::slice::from_ref(&img), &d.group_v3, &mo).unwrap_err();
    assert!(e.to_string().contains("Group is zarr_format: 3"));
    let e = writer::write_plate_metadata_simple(
        &d.group_v3,
        &["A"],
        &["1"],
        &["A/1"],
        Some(Format::V04),
    )
    .unwrap_err();
    assert!(e.to_string().contains("Group is zarr_format: 3"));
    let e = writer::write_well_metadata(&d.group, &[json!({"path": "0"})], Some(CURRENT_FORMAT))
        .unwrap_err();
    assert!(e.to_string().contains("Group is zarr_format: 2"));
}

#[test]
fn group_from_path() {
    for fmt in [None, Some(Format::V04), Some(Format::V05)] {
        let d = dirs();
        let p = d.path.join("from_path");
        let opts = ImageOptions {
            axes: Some("zyx".into()),
            fmt,
            ..Default::default()
        };
        writer::write_image(&lazy(&[64, 64, 64]), p.to_str().unwrap(), &opts).unwrap();
        let g = Group::open_url(p.to_str().unwrap(), Mode::Read, None).unwrap();
        assert_eq!(g.zarr_format(), fmt.unwrap_or(CURRENT_FORMAT).zarr_format());
        assert!(ome(&g).contains_key("multiscales"));
    }
}

#[test]
fn write_image_not_computed() {
    for (fmt, v3) in [(Format::V04, false), (Format::V05, true)] {
        let d = dirs();
        let g = if v3 { &d.group_v3 } else { &d.group };
        let img = data(&[128, 200, 200]);
        let opts = ImageOptions {
            fmt: Some(fmt),
            axes: Some("zyx".into()),
            storage_options: Some(StorageOptions::chunks(&[32, 32]).into()),
            compute: false,
            name: Some("test_write_image_dask".into()),
            ..Default::default()
        };
        let jobs = writer::write_image(&LazyArray::from_ndarray(img.clone()), g, &opts).unwrap();
        assert_eq!(jobs.len(), 5);
        // nothing written yet: data reads as fill value
        let a0 = g.array("s0").unwrap();
        assert!(
            a0.read_all()
                .unwrap()
                .to_f64_vec()
                .iter()
                .all(|&v| v == 0.0)
        );
        compute_all(&jobs).unwrap();
        assert_eq!(g.array("s0").unwrap().read_all().unwrap(), img);
        let md = ome(g);
        assert_eq!(md["multiscales"][0]["name"], "test_write_image_dask");
        for p in dataset_paths(&md) {
            let a = g.array(&p).unwrap();
            assert_eq!(&a.chunks()[1..], &[32u64, 32]);
        }
    }
}

#[test]
fn write_image_scalar_chunks() {
    let d = dirs();
    let p = d.path.join("scalar");
    let opts = ImageOptions {
        axes: Some("xyz".into()),
        storage_options: Some(StorageOptions::chunks(&[32]).into()),
        ..Default::default()
    };
    // a one-element chunk list covers the last dimension; leading dims take
    // the full shape (Python's bare-int form, applied to every dim, is written
    // out per dim in Rust)
    writer::write_image(&lazy(&[64, 64, 64]), p.to_str().unwrap(), &opts).unwrap();
    let g = Group::open_url(p.to_str().unwrap(), Mode::Read, None).unwrap();
    assert_eq!(g.array("s0").unwrap().chunks(), &[64, 64, 32]);
}

#[test]
fn write_image_compressed() {
    for fmt in [Format::V04, Format::V05] {
        let d = dirs();
        let g = if fmt == Format::V05 {
            &d.group_v3
        } else {
            &d.group
        };
        let comp = Compressor::Blosc {
            cname: "zstd".into(),
            clevel: 5,
            shuffle: 1,
            blocksize: 0,
        };
        let opts = ImageOptions {
            fmt: Some(fmt),
            axes: Some("zyx".into()),
            storage_options: Some(
                StorageOptions {
                    compressor: Some(comp.clone()),
                    ..Default::default()
                }
                .into(),
            ),
            ..Default::default()
        };
        let img = data(&[16, 64, 64]);
        writer::write_image(&LazyArray::from_ndarray(img.clone()), g, &opts).unwrap();
        for p in dataset_paths(&ome(g)) {
            let a = g.array(&p).unwrap();
            assert_eq!(a.compressor(), Some(&comp));
        }
        assert_eq!(g.array("s0").unwrap().read_all().unwrap(), img);
    }
}

#[test]
fn default_compression() {
    for fmt in [Format::V04, Format::V05] {
        let d = dirs();
        let g = if fmt == Format::V05 {
            &d.group_v3
        } else {
            &d.group
        };
        let opts = ImageOptions {
            fmt: Some(fmt),
            axes: Some("tzyx".into()),
            storage_options: Some(StorageOptions::chunks(&[1, 100, 100]).into()),
            ..Default::default()
        };
        writer::write_image(&lazy(&[2, 2, 100, 100]), g, &opts).unwrap();
        let a = g.array("s0").unwrap();
        assert_eq!(
            a.compressor(),
            Some(&Compressor::Zstd {
                level: 0,
                checksum: false
            })
        );
    }
}

#[test]
fn validate_coordinate_transforms() {
    let fmt = Format::V04;
    let ct = vec![
        vec![json!({"type": "scale", "scale": [1, 1]})],
        vec![json!({"type": "scale", "scale": [1, 1]})],
    ];
    fmt.validate_coordinate_transformations(2, 2, Some(&ct))
        .unwrap();
    assert!(
        fmt.validate_coordinate_transformations(2, 1, Some(&ct))
            .unwrap_err()
            .is_value()
    );
    let bad = vec![vec![json!({"type": "scale", "scale": ["1", 1]})]];
    assert!(
        fmt.validate_coordinate_transformations(2, 1, Some(&bad))
            .unwrap_err()
            .is_type()
    );
    let bad = vec![vec![
        json!({"type": "scale", "scale": [1, 1]}),
        json!({"type": "scale", "scale": [1, 1]}),
    ]];
    assert!(
        fmt.validate_coordinate_transformations(2, 1, Some(&bad))
            .unwrap_err()
            .is_value()
    );
    let bad = vec![vec![
        json!({"type": "translation", "translation": [1, 1]}),
        json!({"type": "scale", "scale": [1, 1]}),
    ]];
    assert!(
        fmt.validate_coordinate_transformations(2, 1, Some(&bad))
            .unwrap_err()
            .is_value()
    );
    let bad = vec![vec![
        json!({"type": "scale", "scale": [1, 1]}),
        json!({"type": "translation", "translation": [1, 1, 1]}),
    ]];
    assert!(
        fmt.validate_coordinate_transformations(2, 1, Some(&bad))
            .unwrap_err()
            .is_value()
    );
    let bad = vec![vec![
        json!({"type": "scale", "scale": [1, 1]}),
        json!({"type": "translation", "translation": ["a", 1]}),
    ]];
    assert!(
        fmt.validate_coordinate_transformations(2, 1, Some(&bad))
            .unwrap_err()
            .is_type()
    );
}

#[test]
fn axes_validation() {
    let v = |a: AxesInput, fmt: Format| writer::get_valid_axes(Some(a.len()), Some(&a), None, fmt);
    let dicts: Vec<JsonMap> = ["t", "c", "z", "y", "x"]
        .iter()
        .map(|n| {
            json!({"name": n, "type": ome_zarr::axes::known_axis_type(n)})
                .as_object()
                .unwrap()
                .clone()
        })
        .collect();
    assert_eq!(
        v(AxesInput::Dicts(dicts), Format::V04)
            .unwrap()
            .unwrap()
            .as_array()
            .unwrap()
            .len(),
        5
    );
    assert!(v("yxz".into(), Format::V04).is_ok());
    assert!(v("ytx".into(), Format::V04).is_err());
    assert!(v("cytx".into(), Format::V04).is_err());
    // 3D/4D can't be guessed
    assert!(writer::get_valid_axes(Some(3), None, None, Format::V04).is_err());
    assert_eq!(
        writer::get_valid_axes(Some(2), None, None, Format::V04)
            .unwrap()
            .unwrap(),
        json!([{"name": "y", "type": "space"}, {"name": "x", "type": "space"}])
    );
    // length mismatch
    assert!(writer::get_valid_axes(Some(3), Some(&"yx".into()), None, Format::V04).is_err());
    // V0.3 names only
    assert_eq!(
        writer::get_valid_axes(Some(3), Some(&"zyx".into()), None, Format::V03)
            .unwrap()
            .unwrap(),
        json!(["z", "y", "x"])
    );
    for bad in ["yxz", "zxy", "tcyxz"] {
        assert!(
            writer::get_valid_axes(None, Some(&bad.into()), None, Format::V03).is_err(),
            "{bad}"
        );
    }
    // 0.1/0.2 ignore axes
    assert!(
        writer::get_valid_axes(Some(5), Some(&"xyz".into()), None, Format::V01)
            .unwrap()
            .is_none()
    );
}

#[test]
fn invalid_datasets() {
    let d = dirs();
    for ds in [vec![], vec![json!("0")], vec![json!({"key": 1})]] {
        assert!(
            writer::write_multiscales_metadata(
                &d.group,
                &ds,
                Some(Format::V04),
                Some(&"tczyx".into()),
                None,
                None,
                None
            )
            .is_err()
        );
    }
}

#[test]
fn valid_and_invalid_transformations() {
    let valid = vec![
        json!([{"type": "scale", "scale": [1, 1, 1, 1, 1]}]),
        json!([{"type": "scale", "scale": [1, 1, 1, 1, 1]}, {"type": "translation", "translation": [0, 0, 0, 0, 0]}]),
    ];
    for ct in valid {
        let d = dirs();
        let datasets = vec![json!({"path": "0", "coordinateTransformations": ct})];
        writer::write_multiscales_metadata(
            &d.group,
            &datasets,
            Some(Format::V04),
            Some(&"tczyx".into()),
            None,
            None,
            None,
        )
        .unwrap();
        let md = ome(&d.group);
        assert_eq!(
            md["multiscales"][0]["datasets"][0]["coordinateTransformations"],
            datasets[0]["coordinateTransformations"]
        );
    }
    let invalid = vec![
        json!([]),
        json!([{"type": "scale", "scale": [1, 1, 1, 1]}]),
        json!([{"type": "translation", "translation": [0, 0, 0, 0, 0]}]),
        json!([{"type": "scale", "scale": [1, 1, 1, 1, 1]}, {"type": "scale", "scale": [1, 1, 1, 1, 1]}]),
        json!([{"type": "scale", "scale": [1, 1, 1, 1, 1]}, {"type": "translation", "translation": [0, 0, 0, 0]}]),
        json!([{"type": "scale", "scale": [1, 1, 1, 1, "1"]}]),
        json!([{"scale": [1, 1, 1, 1, 1]}]),
    ];
    for ct in invalid {
        let d = dirs();
        let datasets = vec![json!({"path": "0", "coordinateTransformations": ct})];
        assert!(
            writer::write_multiscales_metadata(
                &d.group,
                &datasets,
                Some(Format::V04),
                Some(&"tczyx".into()),
                None,
                None,
                None
            )
            .is_err(),
            "{ct}"
        );
    }
}

#[test]
fn omero_metadata_validation() {
    let datasets: Vec<Value> = (0..5)
        .map(|l| json!({"path": l.to_string(), "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 1, 2f64.powi(l), 2f64.powi(l)]}]}))
        .collect();
    let run = |omero: Value| {
        let d = dirs();
        let mut m = JsonMap::new();
        m.insert("omero".into(), omero);
        let r = writer::write_multiscales_metadata(
            &d.group,
            &datasets,
            Some(Format::V04),
            Some(&"tczyx".into()),
            None,
            None,
            Some(&m),
        );
        (r, d)
    };
    let (r, d) = run(
        json!({"channels": [{"color": "FF0000", "window": {"start": 0, "end": 255, "min": 0, "max": 255}}]}),
    );
    r.unwrap();
    assert_eq!(ome(&d.group)["omero"]["channels"][0]["color"], "FF0000");
    assert!(
        run(json!({"channels": [{"color": "FF000"}]}))
            .0
            .unwrap_err()
            .is_type()
    );
    assert!(
        run(json!({"channels": [{"window": []}]}))
            .0
            .unwrap_err()
            .is_type()
    );
    let e = run(
        json!({"channels": [{"color": "FF0000", "window": {"start": 0, "end": 255, "min": 0}}]}),
    )
    .0
    .unwrap_err();
    assert!(e.is_key() && e.to_string().contains("window"));
    let e = run(Value::Null).0.unwrap_err();
    assert!(
        e.is_key()
            && e.to_string()
                .contains("If `'omero'` is present, value cannot be `None`.")
    );
}

fn plate_attrs(g: &Group) -> Value {
    Value::Object(ome(g))["plate"].clone()
}

#[test]
fn plates() {
    for fmt in [Format::V04, Format::V05] {
        let d = dirs();
        let g = if fmt == Format::V05 {
            &d.group_v3
        } else {
            &d.group
        };
        writer::write_plate_metadata_simple(g, &["A"], &["1"], &["A/1"], Some(fmt)).unwrap();
        let p = plate_attrs(g);
        assert_eq!(p["columns"], json!([{"name": "1"}]));
        assert_eq!(p["rows"], json!([{"name": "A"}]));
        assert_eq!(
            p["wells"],
            json!([{"path": "A/1", "rowIndex": 0, "columnIndex": 0}])
        );
        assert_eq!(p["version"], fmt.version());
        assert!(
            p.get("name").is_none()
                && p.get("field_count").is_none()
                && p.get("acquisitions").is_none()
        );
    }
    // 12 wells
    let d = dirs();
    let rows: Vec<String> = ["A", "B", "C", "D"].iter().map(|s| s.to_string()).collect();
    let cols: Vec<String> = ["1", "2", "3"].iter().map(|s| s.to_string()).collect();
    let wells: Vec<Value> = rows
        .iter()
        .flat_map(|r| cols.iter().map(move |c| json!(format!("{r}/{c}"))))
        .collect();
    writer::write_plate_metadata(
        &d.group,
        &rows,
        &cols,
        &wells,
        Some(Format::V04),
        None,
        Some(1),
        Some("test"),
    )
    .unwrap();
    let p = plate_attrs(&d.group);
    assert_eq!(p["wells"].as_array().unwrap().len(), 12);
    assert_eq!(
        p["wells"][11],
        json!({"path": "D/3", "rowIndex": 3, "columnIndex": 2})
    );
    assert_eq!(p["name"], "test");
    assert_eq!(p["field_count"], 1);
    // acquisitions
    let acq = vec![
        json!({"id": 0, "maximumfieldcount": 1, "name": "a", "description": "d", "starttime": 1, "endtime": 2}),
    ];
    writer::write_plate_metadata(
        &d.group,
        &rows,
        &cols,
        &wells[..1],
        Some(Format::V04),
        Some(&acq),
        None,
        None,
    )
    .unwrap();
    assert_eq!(plate_attrs(&d.group)["acquisitions"], json!(acq));
    for bad in [json!([0, 1]), json!([{"name": "0"}]), json!([{"id": "0"}])] {
        assert!(
            writer::write_plate_metadata(
                &d.group,
                &rows,
                &cols,
                &wells[..1],
                Some(Format::V04),
                Some(bad.as_array().unwrap()),
                None,
                None
            )
            .is_err()
        );
    }
    // invalid wells, rows and columns
    let bad_wells = [
        json!("A"),
        json!("A/1/0"),
        json!([0, 0]),
        json!({"path": "A/1", "rowIndex": 0, "columnIndex": 1}),
        json!({"path": "A/1", "rowIndex": 0}),
        json!("E/1"),
        json!("A/4"),
    ];
    for w in bad_wells {
        assert!(
            writer::write_plate_metadata(
                &d.group,
                &rows,
                &cols,
                std::slice::from_ref(&w),
                Some(Format::V04),
                None,
                None,
                None
            )
            .is_err(),
            "{w}"
        );
    }
    assert!(
        writer::write_plate_metadata(
            &d.group,
            &rows,
            &cols,
            &[],
            Some(Format::V04),
            None,
            None,
            None
        )
        .is_err()
    );
    let s = |v: &[&str]| -> Vec<String> { v.iter().map(|x| x.to_string()).collect() };
    for r in [s(&["A", "B", "B"]), s(&["A", "&"])] {
        assert!(
            writer::write_plate_metadata(
                &d.group,
                &r,
                &cols,
                &wells[..1],
                Some(Format::V04),
                None,
                None,
                None
            )
            .unwrap_err()
            .is_value()
        );
    }
    // unspecified well keys are kept
    let w = json!({"path": "A/1", "rowIndex": 0, "columnIndex": 0, "extra": "x"});
    writer::write_plate_metadata(
        &d.group,
        &rows,
        &cols,
        std::slice::from_ref(&w),
        Some(Format::V04),
        None,
        None,
        None,
    )
    .unwrap();
    assert_eq!(plate_attrs(&d.group)["wells"][0], w);
}

#[test]
fn wells() {
    for fmt in [Format::V04, Format::V05] {
        for images in [vec![json!("0")], vec![json!({"path": "0"})]] {
            let d = dirs();
            let g = if fmt == Format::V05 {
                &d.group_v3
            } else {
                &d.group
            };
            writer::write_well_metadata(g, &images, Some(fmt)).unwrap();
            let w = Value::Object(ome(g))["well"].clone();
            assert_eq!(w["images"], json!([{"path": "0"}]));
            if fmt == Format::V04 {
                assert_eq!(w["version"], "0.4");
            } else {
                assert_eq!(ome(g)["version"], "0.5");
            }
        }
    }
    let d = dirs();
    let images = vec![
        json!({"path": "0", "acquisition": 1}),
        json!({"path": "1", "acquisition": 2}),
    ];
    writer::write_well_metadata(&d.group, &images, None).unwrap();
    assert_eq!(ome(&d.group)["well"]["images"], json!(images));
    for bad in [
        json!({"acquisition": 0}),
        json!({"acquisition": "0"}),
        json!({"path": 0}),
        json!(0),
    ] {
        assert!(
            writer::write_well_metadata(&d.group, std::slice::from_ref(&bad), None).is_err(),
            "{bad}"
        );
    }
    writer::write_well_metadata(&d.group, &[json!({"path": "0", "unknown": "x"})], None).unwrap();
}

#[test]
fn write_labels_and_multiscale_labels() {
    for fmt in [Format::V04, Format::V05] {
        let d = dirs();
        let g = if fmt == Format::V05 {
            &d.group_v3
        } else {
            &d.group
        };
        let lbl = data(&[1, 64, 64]);
        let opts = ImageOptions {
            fmt: Some(fmt),
            axes: Some("cyx".into()),
            method: Some(Methods::Nearest),
            ..Default::default()
        };
        writer::write_labels(
            &LazyArray::from_ndarray(lbl.clone()),
            g,
            "test_labels",
            &opts,
        )
        .unwrap();
        let labels = g.group("labels").unwrap();
        assert_eq!(ome(&labels)["labels"], json!(["test_labels"]));
        let lg = labels.group("test_labels").unwrap();
        let md = ome(&lg);
        assert!(md.contains_key("image-label"));
        assert_eq!(lg.array("s0").unwrap().read_all().unwrap(), lbl);
        // a second label image
        let mut lopts = LabelsOptions {
            multiscale: MultiscaleOptions {
                fmt: Some(fmt),
                axes: Some("cyx".into()),
                ..MultiscaleOptions::new()
            },
            label_metadata: None,
        };
        let mut meta = JsonMap::new();
        meta.insert(
            "colors".into(),
            json!([{"label-value": 1, "rgba": [255, 0, 0, 255]}]),
        );
        meta.insert(
            "properties".into(),
            json!([{"label-value": 1, "class": "a"}]),
        );
        lopts.label_metadata = Some(meta);
        let pyramid = vec![
            LazyArray::from_ndarray(lbl.clone()),
            LazyArray::from_ndarray(lbl.slice_region(&[0..1, 0..32, 0..32])),
        ];
        writer::write_multiscale_labels(&pyramid, g, "second", &lopts).unwrap();
        assert_eq!(ome(&labels)["labels"], json!(["test_labels", "second"]));
        let il = ome(&labels.group("second").unwrap())["image-label"].clone();
        assert_eq!(
            il["colors"],
            json!([{"label-value": 1, "rgba": [255, 0, 0, 255]}])
        );
        assert_eq!(il["version"], fmt.version());
    }
}

#[test]
fn legacy_scaler() {
    for method in ["nearest", "gaussian", "local_mean", "zoom", "resize_image"] {
        for max in [1, 2, 3] {
            let d = dirs();
            let scaler = Scaler {
                downscale: 2,
                method: method.into(),
                max_layer: max,
                ..Default::default()
            };
            let opts = ImageOptions {
                axes: Some("zyx".into()),
                scaler: Some(scaler),
                ..Default::default()
            };
            writer::write_image(&lazy(&[3, 64, 64]), &d.group_v3, &opts).unwrap();
            assert_eq!(dataset_paths(&ome(&d.group_v3)).len(), max + 1);
        }
    }
}

#[test]
fn build_pyramid_shapes() {
    for method in [
        Methods::Nearest,
        Methods::Resize,
        Methods::LocalMean,
        Methods::Zoom,
    ] {
        for (shape, sf, dims) in [
            (
                vec![1usize, 2, 1, 256, 256],
                ScaleFactors::Ints(vec![2, 4, 8, 16]),
                "tczyx",
            ),
            (
                vec![1, 2, 1, 256, 256],
                ScaleFactors::per_dim(vec![
                    vec![("z", 1.0), ("y", 2.0), ("x", 2.0)],
                    vec![("z", 1.0), ("y", 4.0), ("x", 4.0)],
                ]),
                "tczyx",
            ),
            (
                vec![128, 128, 128],
                ScaleFactors::per_dim(vec![
                    vec![("z", 2.0), ("y", 2.0), ("x", 2.0)],
                    vec![("z", 4.0), ("y", 4.0), ("x", 4.0)],
                ]),
                "zyx",
            ),
            (
                vec![256, 256],
                ScaleFactors::per_dim(vec![
                    vec![("y", 2.0), ("x", 2.0)],
                    vec![("y", 4.0), ("x", 4.0)],
                ]),
                "yx",
            ),
        ] {
            let d = dirs();
            let opts = ImageOptions {
                axes: Some(dims.into()),
                method: Some(method),
                scale_factors: Some(sf.clone()),
                ..Default::default()
            };
            writer::write_image(&lazy(&shape), &d.group_v3, &opts).unwrap();
            let dimv: Vec<String> = dims.chars().map(|c| c.to_string()).collect();
            let levels = sf.resolve(&dimv);
            let arrays: Vec<ZArray> = dataset_paths(&ome(&d.group_v3))
                .iter()
                .map(|p| d.group_v3.array(p).unwrap())
                .collect();
            assert_eq!(arrays.len(), levels.len() + 1);
            for (i, a) in arrays.iter().enumerate().skip(1) {
                for (k, dname) in dims.chars().enumerate() {
                    let prev = arrays[i - 1].shape()[k];
                    let rel = if i == 1 {
                        levels[0][k]
                    } else {
                        levels[i - 1][k] / levels[i - 2][k]
                    };
                    let want = if "zyx".contains(dname) {
                        (prev as f64 / rel).ceil() as u64
                    } else {
                        prev
                    };
                    assert_eq!(a.shape()[k], want, "{method} level {i} dim {dname}");
                }
            }
        }
    }
}

#[test]
fn writer_fails_if_existing() {
    let d = dirs();
    let p = d.path.join("overwrite");
    let img = OmeZarrImage::from_axes(lazy(&[1, 1, 1, 32, 32]), "tczyx").unwrap();
    let ms = OmeZarrMultiscale::builder(img)
        .method(Some(Methods::Nearest))
        .build()
        .unwrap();
    ms.to_ome_zarr(p.as_path(), None, Version::V06, true, false)
        .unwrap();
    assert!(matches!(
        ms.to_ome_zarr(p.as_path(), None, Version::V06, true, false),
        Err(ome_zarr::Error::Io(_))
    ));
    ms.to_ome_zarr(p.as_path(), None, Version::V06, true, true)
        .unwrap();
}

#[test]
fn deprecated_formats_rejected() {
    let d = dirs();
    for fmt in [Format::V01, Format::V02, Format::V03] {
        let r = writer::write_image(
            &lazy(&[1, 1, 1, 32, 32]),
            &d.group,
            &ImageOptions {
                fmt: Some(fmt),
                ..Default::default()
            },
        );
        assert!(matches!(r, Err(ome_zarr::Error::Deprecation(_))));
    }
}
