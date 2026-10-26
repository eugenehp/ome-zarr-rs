//! Round 10 of the differential parity tests with ome-zarr-py: 40 random
//! `OMEZarrImage` / `OMEZarrMultiscale` / `OMEZarrLabels` configurations
//! written as OME-Zarr 0.4 / 0.5 / 0.6 (observations10.json). Label colours
//! use numpy's seeded generator, as the Python run did.
#![cfg(not(target_arch = "wasm32"))]

mod common;

use std::collections::BTreeMap;

use common::*;
use ome_zarr::array::{DType, LazyArray};
use ome_zarr::classes::{
    ChannelColor, LabelsArg, OmeZarrImage, OmeZarrLabels, OmeZarrMultiscale, ScaleMap, Version,
};
use ome_zarr::scale::{Methods, ScaleFactors};
use serde_json::Value;

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

fn image(sc: &Value, data: ome_zarr::array::NdArray, name: &str, with_units: bool) -> OmeZarrImage {
    let axes: Vec<String> = sc["axes"]
        .as_str()
        .unwrap()
        .chars()
        .map(|c| c.to_string())
        .collect();
    let scale = sc["scale"].as_object().map(|m| {
        ScaleMap::from_pairs(
            m.iter()
                .map(|(k, v)| (k.clone(), v.as_f64().unwrap()))
                .collect(),
        )
    });
    let units: Option<BTreeMap<String, String>> = if !with_units {
        None
    } else {
        sc["units"].as_object().map(|m| {
            m.iter()
                .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
                .collect()
        })
    };
    OmeZarrImage::new(
        LazyArray::from_ndarray(data),
        &axes,
        scale,
        units,
        None,
        name,
    )
    .unwrap()
}

#[test]
fn random_class_api_images() {
    let obs: Value = serde_json::from_str(
        &std::fs::read_to_string(fixtures().join("observations10.json")).unwrap(),
    )
    .unwrap();
    let d = tempfile::tempdir().unwrap();
    let mut failures = vec![];
    for (i, (name, o)) in obs.as_object().unwrap().iter().enumerate() {
        let sc = &o["scenario"];
        let shape: Vec<usize> = sc["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        let dtype = DType::parse(sc["dtype"].as_str().unwrap()).unwrap();
        let img = image(
            sc,
            gen_data(&shape, dtype, 251, 0),
            sc["image_name"].as_str().unwrap(),
            true,
        );
        let sf = scale_factors(&sc["scale_factors"]);
        let mut b = OmeZarrMultiscale::builder(img)
            .scale_factors(Some(sf.clone()))
            .method(Some(
                Methods::parse(sc["method"].as_str().unwrap()).unwrap(),
            ));
        if let Some(n) = sc["channel_names"].as_array() {
            let names: Vec<&str> = n.iter().map(|x| x.as_str().unwrap()).collect();
            b = b.channel_names(&names);
        }
        if let Some(c) = sc["channel_colors"].as_array() {
            b = b.channel_colors(
                c.iter()
                    .map(|x| match x {
                        Value::String(s) => ChannelColor::Hex(s.clone()),
                        a => ChannelColor::Rgb(
                            a.as_array()
                                .unwrap()
                                .iter()
                                .map(|v| v.as_u64().unwrap() as u8)
                                .collect(),
                        ),
                    })
                    .collect(),
            );
        }
        if let Some(c) = sc["contrast_limits"].as_array() {
            b = b.contrast_limits(
                c.iter()
                    .map(|p| (p[0].as_f64().unwrap(), p[1].as_f64().unwrap()))
                    .collect(),
            );
        }
        if sc["labels"] == true {
            ome_zarr::utils::np_random_seed(i as u32);
            let li = image(sc, gen_data(&shape, DType::UInt8, 4, 0), "lab", false);
            b = b.labels(LabelsArg::One(
                OmeZarrLabels::new(li, Some(sf), Some(Methods::Nearest), true).unwrap(),
            ));
        }
        let version = Version::parse(sc["version"].as_str().unwrap()).unwrap();
        let p = d.path().join(name);
        let r = b.build().and_then(|ms| {
            ms.to_ome_zarr(p.as_path(), None, version, true, true)
                .map(|_| ())
        });
        match (&o["error"], r) {
            (Value::Null, Ok(())) => {
                if let Err(e) = std::panic::catch_unwind(|| compare_stores(&py_store(name), &p)) {
                    failures.push(format!(
                        "{name} {sc}: {}",
                        e.downcast_ref::<String>().cloned().unwrap_or_default()
                    ));
                }
                // seeded label colours: the label metadata is byte-identical too
                for f in ["labels/lab/zarr.json", "labels/lab/.zattrs"] {
                    if let Ok(want) = std::fs::read(py_store(name).join(f))
                        && std::fs::read(p.join(f)).ok() != Some(want)
                    {
                        failures.push(format!("{name}: {f} differs (label colours)"));
                    }
                }
            }
            (want, r) => failures.push(format!(
                "{name} {sc}: python {want}, rust {:?}",
                r.err().map(|e| e.to_string())
            )),
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} differ:\n{}",
        failures.len(),
        obs.as_object().unwrap().len(),
        failures.join("\n\n")
    );
}
