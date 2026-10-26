//! `ZarrLocation`'s accessors as ome-zarr-py's (fixtures/zarr_location.json,
//! recorded with relative paths from the parity stores directory). Alone in
//! this binary: it changes the working directory.
#![cfg(not(target_arch = "wasm32"))]

use ome_zarr::format::CURRENT_FORMAT;
use ome_zarr::io::{ZarrLocation, parse_url};
use serde_json::{Value, json};

#[test]
fn accessors_match_python() {
    let fx = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let obs: Value =
        serde_json::from_str(&std::fs::read_to_string(fx.join("zarr_location.json")).unwrap())
            .unwrap();
    std::env::set_current_dir(fx.join("parity/stores")).unwrap();
    let mut failures = vec![];
    for (p, want) in obs["cases"].as_object().unwrap() {
        let loc = ZarrLocation::new(p.as_str(), "r", CURRENT_FORMAT).unwrap();
        let mut keys: Vec<String> = loc.root_attrs().keys().cloned().collect();
        keys.sort();
        let mut got = json!({
            "repr": loc.to_string(), "str": loc.to_string(), "path": loc.path(), "basename": loc.basename(),
            "parts": loc.parts(), "version": loc.version(), "exists": loc.exists(), "subpath": loc.subpath("x"),
            "subpath_up": loc.subpath("../y"), "subpath_empty": loc.subpath(""), "root_attrs": keys,
        });
        if loc.exists() {
            got["create_labels"] = json!(loc.create("labels").unwrap().to_string());
        }
        got["parse_url"] = match parse_url(p.as_str(), "r", CURRENT_FORMAT).unwrap() {
            Some(l) => json!(l.to_string()),
            None => Value::Null,
        };
        for (k, v) in want.as_object().unwrap() {
            if &got[k] != v {
                failures.push(format!("{p} {k}: python {v}, rust {}", got[k]));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
