//! The `view --web` / `finder` server answers like ome-zarr-py's
//! (fixtures/server.json, recorded by scripts/gen_server_fixtures.py from its
//! CORSRequestHandler serving fixtures/server_root).
#![cfg(not(target_arch = "wasm32"))]

use std::path::Path;

use ome_zarr::server::{Body, respond_to};
use serde_json::Value;

#[test]
fn responses_match_python() {
    let fx = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let root = fx.join("server_root");
    // the recorded Last-Modified dates are of these mtimes
    for e in walk(&root) {
        let f = std::fs::File::options().write(true).open(&e).ok();
        if let Some(f) = f {
            f.set_modified(std::time::UNIX_EPOCH + std::time::Duration::from_secs(1700000000))
                .unwrap();
        }
    }
    let cases: Vec<Value> =
        serde_json::from_str(&std::fs::read_to_string(fx.join("server.json")).unwrap()).unwrap();
    let mut failures = vec![];
    for c in &cases {
        let headers: Vec<(String, String)> = c["headers"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
            .collect();
        let (method, path) = (c["method"].as_str().unwrap(), c["path"].as_str().unwrap());
        let r = respond_to(&root, method, path, &headers);
        let body = match &r.body {
            Body::Bytes(b) => b.clone(),
            Body::File { path, start, len } => {
                std::fs::read(path).unwrap()[*start as usize..(*start + *len) as usize].to_vec()
            }
        };
        let got_headers: Vec<Value> = r
            .headers
            .iter()
            .map(|(k, v)| serde_json::json!([k, v]))
            .collect();
        let want_body = c["body"]
            .as_str()
            .unwrap()
            .chars()
            .map(|ch| ch as u8)
            .collect::<Vec<u8>>();
        if r.status as i64 != c["status"].as_i64().unwrap()
            || r.reason != c["reason"].as_str().unwrap()
            || got_headers != *c["response_headers"].as_array().unwrap()
            || body != want_body
        {
            failures.push(format!(
                "{method} {path} {}\n  python {} {} {}\n  rust   {} {} {:?}  body {}",
                c["headers"],
                c["status"],
                c["reason"],
                c["response_headers"],
                r.status,
                r.reason,
                r.headers,
                if body == want_body { "same" } else { "differs" }
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} differ:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}

fn walk(p: &Path) -> Vec<std::path::PathBuf> {
    let mut out = vec![];
    for e in std::fs::read_dir(p).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            out.extend(walk(&p));
        } else {
            out.push(p);
        }
    }
    out
}
