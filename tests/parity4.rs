//! Round 4 of the differential parity tests with ome-zarr-py: the parameter
//! grids of ome-zarr-py's tests/test_scaler.py (`test_build_pyramid`,
//! `test_legacy_scaler`, `test_pyramid_args`, `test_big_dask_pyramid`), each
//! store compared with the one Python wrote.
#![cfg(not(target_arch = "wasm32"))]

mod common;

use common::*;
use ome_zarr::array::{DType, LazyArray};
use ome_zarr::scale::{Methods, ScaleFactors, Scaler};
use ome_zarr::store::Store;
use ome_zarr::writer::{self, ImageOptions};
use ome_zarr::zarr::{Group, Mode, ZArray};

fn small(shape: &[usize]) -> LazyArray {
    LazyArray::from_ndarray(gen_data(shape, DType::UInt8, 21, 0))
}

fn axes(dims: &str) -> ome_zarr::axes::AxesInput {
    dims.chars()
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .into()
}

fn write_and_compare(name: &str, data: &LazyArray, opts: &ImageOptions) {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join(name);
    let g = Group::open_url(p.to_str().unwrap(), Mode::Write, None).unwrap();
    writer::write_image(data, &g, opts).unwrap();
    compare_stores(&py_store(name), &p);
}

#[test]
fn build_pyramid_grid() {
    let one = |y: f64| vec![("t", 1.0), ("c", 1.0), ("z", 1.0), ("y", y), ("x", y)];
    let part = |y: f64| vec![("z", 1.0), ("y", y), ("x", y)];
    let zyx = |f: f64| vec![("z", f), ("y", f), ("x", f)];
    let yx = |f: f64| vec![("y", f), ("x", f)];
    let grid: Vec<(&str, Vec<usize>, ScaleFactors, &str)> = vec![
        (
            "5d_dict",
            vec![1, 2, 1, 256, 256],
            ScaleFactors::per_dim([one(2.0), one(4.0), one(8.0)]),
            "tczyx",
        ),
        (
            "5d_ints",
            vec![1, 2, 1, 256, 256],
            ScaleFactors::Ints(vec![2, 4, 8, 16]),
            "tczyx",
        ),
        (
            "5d_partial",
            vec![1, 2, 1, 256, 256],
            ScaleFactors::per_dim([part(2.0), part(4.0), part(8.0)]),
            "tczyx",
        ),
        (
            "3d_z",
            vec![128, 128, 128],
            ScaleFactors::per_dim([zyx(2.0), zyx(4.0), zyx(8.0)]),
            "zyx",
        ),
        (
            "2d",
            vec![256, 256],
            ScaleFactors::per_dim([yx(2.0), yx(4.0), yx(8.0)]),
            "yx",
        ),
    ];
    for (key, shape, sf, dims) in grid {
        for method in ["nearest", "resize", "local_mean", "zoom"] {
            let opts = ImageOptions {
                axes: Some(axes(dims)),
                method: Some(Methods::parse(method).unwrap()),
                scale_factors: Some(sf.clone()),
                ..Default::default()
            };
            write_and_compare(&format!("bp_{key}_{method}.zarr"), &small(&shape), &opts);
        }
    }
}

fn shapes() -> [(&'static str, Vec<usize>); 3] {
    [
        ("5d", vec![1, 2, 1, 256, 256]),
        ("3d", vec![3, 512, 512]),
        ("2d", vec![256, 256]),
    ]
}

fn legacy(method: &str, levels: usize) -> Scaler {
    Scaler {
        downscale: 2,
        method: method.into(),
        max_layer: levels,
        ..Default::default()
    }
}

#[test]
fn legacy_scaler_grid() {
    for (sk, shape) in shapes() {
        let dims = &"tczyx"[5 - shape.len()..];
        for method in ["nearest", "gaussian", "local_mean", "zoom", "resize_image"] {
            for levels in [1, 2, 3] {
                let opts = ImageOptions {
                    axes: Some(axes(dims)),
                    scaler: Some(legacy(method, levels)),
                    ..Default::default()
                };
                write_and_compare(
                    &format!("legacy_{sk}_{method}_{levels}.zarr"),
                    &small(&shape),
                    &opts,
                );
            }
        }
    }
}

#[test]
fn pyramid_args() {
    for (sk, shape) in shapes() {
        let dims = &"tczyx"[5 - shape.len()..];
        for method in ["gaussian", "laplacian"] {
            let opts = ImageOptions {
                axes: Some(axes(dims)),
                scaler: Some(legacy(method, 2)),
                ..Default::default()
            };
            write_and_compare(
                &format!("pyrargs_{sk}_{method}.zarr"),
                &small(&shape),
                &opts,
            );
        }
    }
}

#[test]
fn big_dask_pyramid_level() {
    let data = small(&[667, 956]).rechunk(&[100, 100]);
    let level = Scaler::default()
        .resize_image_lazy(&data)
        .unwrap()
        .compute()
        .unwrap();
    let py = ZArray::open(
        &Store::local(py_store("big_dask_level1.zarr"), true).unwrap(),
        "",
    )
    .unwrap()
    .read_all()
    .unwrap();
    assert_eq!(py.shape(), level.shape());
    assert!(
        py == level,
        "resize_image of a chunked array differs from dask's"
    );
}
