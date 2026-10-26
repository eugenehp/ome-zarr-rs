//! Round 7 of the differential parity tests with ome-zarr-py: float16 and
//! complex images through every downsampling method, and the numcodecs
//! compressors zlib / bz2 / lz4 / lzma (observations7.json).
#![cfg(not(target_arch = "wasm32"))]

mod common;

use common::*;
use ndarray::{ArrayD, IxDyn};
use num_complex::Complex;
use ome_zarr::array::{DType, LazyArray, NdArray};
use ome_zarr::format::Format;
use ome_zarr::scale::{Methods, ScaleFactors};
use ome_zarr::writer::{self, ImageOptions, StorageOptions, StorageOptionsArg};
use ome_zarr::zarr::{Compressor, Group, Mode};
use serde_json::{Value, json};

fn obs() -> Value {
    serde_json::from_str(&std::fs::read_to_string(fixtures().join("observations7.json")).unwrap())
        .unwrap()
}

/// `(np.arange(40 * 52).reshape(40, 52) % 37) / 3.0`
fn base() -> ArrayD<f64> {
    ArrayD::from_shape_fn(IxDyn(&[40, 52]), |i| ((i[0] * 52 + i[1]) % 37) as f64 / 3.0)
}

fn data(dtype: DType) -> NdArray {
    let b = base();
    let flipped = ArrayD::from_shape_fn(IxDyn(&[40, 52]), |i| b[[39 - i[0], i[1]]]);
    match dtype {
        DType::Float16 => NdArray::Float16(b.mapv(half::f16::from_f64)),
        // d = base.astype(c64) + 1j * base[::-1].astype(c64)
        DType::Complex64 => NdArray::Complex64(
            ndarray::Zip::from(&b)
                .and(&flipped)
                .map_collect(|&r, &i| Complex::new(r as f32, i as f32)),
        ),
        _ => NdArray::Complex128(
            ndarray::Zip::from(&b)
                .and(&flipped)
                .map_collect(|&r, &i| Complex::new(r, i)),
        ),
    }
}

#[test]
fn float16_and_complex_images() {
    let o = obs();
    let d = tempfile::tempdir().unwrap();
    let mut failures = vec![];
    for (dname, dtype) in [
        ("float16", DType::Float16),
        ("complex64", DType::Complex64),
        ("complex128", DType::Complex128),
    ] {
        for fmt in [Format::V04, Format::V05] {
            for m in ["resize", "nearest", "local_mean", "zoom"] {
                let name = format!("dt_{dname}_{m}_{}.zarr", fmt.version());
                let p = d.path().join(&name);
                let g = Group::open_url(p.to_str().unwrap(), Mode::Write, Some(fmt.zarr_format()))
                    .unwrap();
                let opts = ImageOptions {
                    fmt: Some(fmt),
                    axes: Some("yx".into()),
                    scale_factors: Some(ScaleFactors::Ints(vec![2, 4])),
                    method: Some(Methods::parse(m).unwrap()),
                    ..Default::default()
                };
                let r = writer::write_image(&LazyArray::from_ndarray(data(dtype)), &g, &opts);
                if let Some(e) = o["errors"].get(&name) {
                    match r {
                        Err(err)
                            if err.py_type() == e["type"]
                                && err.py_str() == e["msg"].as_str().unwrap() => {}
                        Err(err) => failures.push(format!(
                            "{name}: python {e}, rust {}: {}",
                            err.py_type(),
                            err.py_str()
                        )),
                        Ok(_) => failures.push(format!("{name}: python {e}, rust wrote it")),
                    }
                    continue;
                }
                if let Err(e) = r {
                    failures.push(format!("{name}: {e}"));
                    continue;
                }
                if let Err(e) = std::panic::catch_unwind(|| compare_stores(&py_store(&name), &p)) {
                    failures.push(format!(
                        "{name}: {}",
                        e.downcast_ref::<String>().cloned().unwrap_or_default()
                    ));
                    continue;
                }
                let mut out = String::new();
                ome_zarr::utils::info_to(py_store(&name).to_str().unwrap(), true, &mut out)
                    .unwrap();
                if out != o["info"][&name].as_str().unwrap() {
                    failures.push(format!(
                        "{name} info:\n--- python\n{}--- rust\n{out}",
                        o["info"][&name].as_str().unwrap()
                    ));
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

#[test]
fn numcodecs_compressors() {
    let d = tempfile::tempdir().unwrap();
    for (cname, id, args) in [
        ("zlib", "zlib", vec![("level", json!(3))]),
        ("lz4", "lz4", vec![("acceleration", json!(2))]),
        ("bz2", "bz2", vec![("level", json!(5))]),
        ("lzma", "lzma", vec![]),
        ("lzma_preset", "lzma", vec![("preset", json!(2))]),
    ] {
        let comp = Compressor::numcodecs(id, &args).unwrap();
        for fmt in [Format::V04, Format::V05] {
            let name = format!("codec_{cname}_{}.zarr", fmt.version());
            let p = d.path().join(&name);
            let g =
                Group::open_url(p.to_str().unwrap(), Mode::Write, Some(fmt.zarr_format())).unwrap();
            let opts = ImageOptions {
                fmt: Some(fmt),
                axes: Some("cyx".into()),
                scale_factors: Some(ScaleFactors::Ints(vec![2])),
                storage_options: Some(StorageOptionsArg::All(StorageOptions {
                    compressor: Some(comp.clone()),
                    ..Default::default()
                })),
                ..Default::default()
            };
            writer::write_image(
                &LazyArray::from_ndarray(gen_data(&[2, 30, 41], DType::UInt16, 251, 0)),
                &g,
                &opts,
            )
            .unwrap();
            // the Python store reads in Rust, the Rust store matches it
            compare_stores(&py_store(&name), &p);
        }
    }
}
