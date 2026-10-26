//! Round 6 of the differential parity tests with ome-zarr-py: the reader
//! from many starting points (legacy versions, v0.6 images and scenes, labels,
//! wells, single arrays, downloads), every array compared by the SHA-256 of
//! its bytes (observations6.json).
#![cfg(not(target_arch = "wasm32"))]

mod common;

use common::*;
use ome_zarr::array::NdArray;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn bytes(a: &NdArray) -> Vec<u8> {
    match a {
        NdArray::Bool(x) => x.iter().map(|&b| b as u8).collect(),
        NdArray::UInt8(x) => x.iter().copied().collect(),
        NdArray::Int8(x) => x.iter().map(|v| *v as u8).collect(),
        NdArray::UInt16(x) => x.iter().flat_map(|v| v.to_le_bytes()).collect(),
        NdArray::Int16(x) => x.iter().flat_map(|v| v.to_le_bytes()).collect(),
        NdArray::UInt32(x) => x.iter().flat_map(|v| v.to_le_bytes()).collect(),
        NdArray::Int32(x) => x.iter().flat_map(|v| v.to_le_bytes()).collect(),
        NdArray::UInt64(x) => x.iter().flat_map(|v| v.to_le_bytes()).collect(),
        NdArray::Int64(x) => x.iter().flat_map(|v| v.to_le_bytes()).collect(),
        NdArray::Float32(x) => x.iter().flat_map(|v| v.to_le_bytes()).collect(),
        NdArray::Float64(x) => x.iter().flat_map(|v| v.to_le_bytes()).collect(),
        NdArray::Float16(x) => x.iter().flat_map(|v| v.to_le_bytes()).collect(),
        NdArray::Complex64(x) => x
            .iter()
            .flat_map(|v| [v.re.to_le_bytes(), v.im.to_le_bytes()].concat())
            .collect(),
        NdArray::Complex128(x) => x
            .iter()
            .flat_map(|v| [v.re.to_le_bytes(), v.im.to_le_bytes()].concat())
            .collect(),
    }
}

fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

fn nodes(path: &str) -> ome_zarr::Result<Option<Value>> {
    let Some(loc) = ome_zarr::io::parse_url_r(path)? else {
        return Ok(None);
    };
    let nodes = ome_zarr::reader::Reader::new(loc)?.nodes()?;
    let mut out = vec![];
    for n in &nodes {
        let mut data = vec![];
        for d in &n.data {
            let a = d.compute()?;
            data.push(json!({
                "shape": d.shape(), "dtype": d.dtype().name(), "chunks": d.chunks(),
                "sha256": hex(&Sha256::digest(bytes(&a))),
            }));
        }
        out.push(json!({
            "repr": n.to_string(),
            "visible": n.visible(),
            "specs": n.specs.iter().map(|s| s.kind().to_string()).collect::<Vec<_>>(),
            "metadata": Value::Object(n.metadata.clone()),
            "data": data,
        }));
    }
    Ok(Some(Value::Array(out)))
}

#[test]
fn reader_from_many_starting_points() {
    let obs: Value = serde_json::from_str(
        &std::fs::read_to_string(fixtures().join("observations6.json")).unwrap(),
    )
    .unwrap();
    let mut failures = vec![];
    for (name, want) in obs.as_object().unwrap() {
        let got = nodes(py_store(name).to_str().unwrap());
        match (want.get("nodes"), want.get("error"), got) {
            (Some(w), _, Ok(Some(g))) => {
                if &g != w {
                    let (ws, gs) = (
                        serde_json::to_string_pretty(w).unwrap(),
                        serde_json::to_string_pretty(&g).unwrap(),
                    );
                    let (wl, gl): (Vec<&str>, Vec<&str>) =
                        (ws.lines().collect(), gs.lines().collect());
                    let i = wl
                        .iter()
                        .zip(&gl)
                        .position(|(a, b)| a != b)
                        .unwrap_or(wl.len().min(gl.len()));
                    failures.push(format!(
                        "{name}: line {i}\n  python: {:?}\n  rust:   {:?}",
                        wl.get(i),
                        gl.get(i)
                    ));
                }
            }
            // Python's parse_url returned None and the Reader failed on it
            (None, Some(e), Ok(None)) if e["msg"].as_str().unwrap().contains("'NoneType'") => {}
            (None, Some(e), Err(err)) => {
                let (t, m) = (e["type"].as_str().unwrap(), e["msg"].as_str().unwrap());
                if (err.py_type(), err.py_str().as_str()) != (t, m) {
                    failures.push(format!(
                        "{name}:\n  python {t}: {m}\n  rust   {}: {}",
                        err.py_type(),
                        err.py_str()
                    ));
                }
            }
            (w, e, g) => failures.push(format!(
                "{name}: python {:?} / {:?}, rust {:?}",
                w.map(|_| "nodes"),
                e,
                g.map(|o| o.map(|_| "nodes"))
            )),
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
