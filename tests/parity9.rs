//! Round 9 of the differential parity tests with ome-zarr-py: 60 random
//! writer configurations (dtype, axes, shape, version, method, scale factors,
//! chunks, shards, compressor, dask chunking, labels), replayed from the
//! parameters ome-zarr-py was run with (observations9.json).
#![cfg(not(target_arch = "wasm32"))]

mod common;

use common::*;
use ome_zarr::array::{DType, LazyArray};
use ome_zarr::format::Format;
use ome_zarr::scale::{Methods, ScaleFactors};
use ome_zarr::writer::{self, ImageOptions, StorageOptions, StorageOptionsArg};
use ome_zarr::zarr::{Compressor, Group, Mode};
use serde_json::{Value, json};

fn u64s(v: &Value) -> Option<Vec<u64>> {
    v.as_array()
        .map(|a| a.iter().map(|x| x.as_u64().unwrap()).collect())
}

fn scale_factors(v: &Value) -> ScaleFactors {
    let a = v.as_array().unwrap();
    if a.iter().all(|x| x.is_u64()) {
        return ScaleFactors::Ints(a.iter().map(|x| x.as_u64().unwrap()).collect());
    }
    ScaleFactors::per_dim(
        a.iter()
            .map(|d| {
                d.as_object()
                    .unwrap()
                    .iter()
                    .map(|(k, v)| (k.clone(), v.as_f64().unwrap()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
    )
}

fn compressor(c: &str) -> Option<Compressor> {
    Some(match c {
        "blosc" => Compressor::Blosc {
            cname: "zstd".into(),
            clevel: 3,
            shuffle: 1,
            blocksize: 0,
        },
        "zstd" => Compressor::Zstd {
            level: 3,
            checksum: false,
        },
        "zlib" => Compressor::numcodecs("zlib", &[("level", json!(2))]).unwrap(),
        _ => return None,
    })
}

#[test]
fn random_writer_configurations() {
    let obs: Value = serde_json::from_str(
        &std::fs::read_to_string(fixtures().join("observations9.json")).unwrap(),
    )
    .unwrap();
    let d = tempfile::tempdir().unwrap();
    let mut failures = vec![];
    for (name, o) in obs.as_object().unwrap() {
        let sc = &o["scenario"];
        let fmt = if sc["fmt"] == "0.4" {
            Format::V04
        } else {
            Format::V05
        };
        let dtype = DType::parse(sc["dtype"].as_str().unwrap()).unwrap();
        let shape: Vec<usize> = u64s(&sc["shape"])
            .unwrap()
            .iter()
            .map(|&x| x as usize)
            .collect();
        let off = if dtype == DType::Int16 { 100 } else { 0 };
        let mut data = LazyArray::from_ndarray(gen_data(&shape, dtype, 251, off));
        if let Some(c) = u64s(&sc["dask_chunks"]) {
            data = data.rechunk(&c);
        }
        let so = StorageOptions {
            chunks: u64s(&sc["chunks"]),
            shards: u64s(&sc["shards"]),
            compressor: compressor(sc["compressor"].as_str().unwrap()),
            ..Default::default()
        };
        let axes = sc["axes"].as_str().unwrap();
        let p = d.path().join(name);
        let g = Group::open_url(p.to_str().unwrap(), Mode::Write, Some(fmt.zarr_format())).unwrap();
        let sf = scale_factors(&sc["scale_factors"]);
        let opts = ImageOptions {
            fmt: Some(fmt),
            axes: Some(axes.into()),
            method: Some(Methods::parse(sc["method"].as_str().unwrap()).unwrap()),
            scale_factors: Some(sf.clone()),
            storage_options: if so == StorageOptions::default() {
                None
            } else {
                Some(StorageOptionsArg::All(so))
            },
            ..Default::default()
        };
        let r = writer::write_image(&data, &g, &opts).and_then(|_| {
            if sc["labels"] == true {
                let lab = LazyArray::from_ndarray(gen_data(&shape, DType::UInt8, 5, 0));
                let lo = ImageOptions {
                    fmt: Some(fmt),
                    axes: Some(axes.into()),
                    scale_factors: Some(sf),
                    ..Default::default()
                };
                writer::write_labels(&lab, &g, "lab", &lo)?;
            }
            Ok(())
        });
        match (&o["error"], r) {
            (Value::Null, Ok(())) => {
                if let Err(e) = std::panic::catch_unwind(|| compare_stores(&py_store(name), &p)) {
                    failures.push(format!(
                        "{name} {sc}: {}",
                        e.downcast_ref::<String>().cloned().unwrap_or_default()
                    ));
                }
            }
            (Value::Null, Err(e)) => failures.push(format!("{name} {sc}: rust error {e}")),
            // ome-zarr-py's dask `local_mean` crashes when a downsampled
            // dimension is not a multiple of its factor (blocks come out
            // larger than declared); Rust writes the level (see README)
            (want, Ok(()))
                if sc["method"] == "local_mean"
                    && want["msg"]
                        .as_str()
                        .unwrap()
                        .starts_with("could not broadcast input array") => {}
            (want, Ok(())) => failures.push(format!("{name} {sc}: python {want}, rust wrote it")),
            (want, Err(e)) => {
                if e.py_type() != want["type"] || e.py_str() != want["msg"].as_str().unwrap() {
                    failures.push(format!(
                        "{name}: python {want}, rust {}: {}",
                        e.py_type(),
                        e.py_str()
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} differ:\n{}",
        failures.len(),
        obs.as_object().unwrap().len(),
        failures.join("\n")
    );
}
