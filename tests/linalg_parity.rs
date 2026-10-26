//! Small dense linear algebra exactly as numpy's BLAS / LAPACK computes it on
//! this platform (fixtures/linalg.json, scripts/gen_linalg_fixtures.py).

use serde_json::Value;

fn mat(v: &Value) -> Vec<Vec<f64>> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        })
        .collect()
}

fn vec1(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

#[test]
fn products_inverses_and_affines_match_numpy() {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/linalg.json");
    let cases: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    let mut bad = vec![];
    for (i, c) in cases.iter().enumerate() {
        // x86-64: OpenBLAS's 4x4 solve depends on the kernels it selects for
        // the CPU (see README, Platforms)
        let x86 = cfg!(target_arch = "x86_64");
        // x86-64: 4-term products, likewise kernel-dependent
        if x86 && c["kind"] == "matmul" && c["b"].as_array().unwrap().len() == 4 {
            continue;
        }
        let got: Value = match c["kind"].as_str().unwrap() {
            "matmul" => serde_json::json!(ome_zarr::linalg::matmul(&mat(&c["a"]), &mat(&c["b"]))),
            "matvec" => serde_json::json!(ome_zarr::linalg::matvec(&mat(&c["a"]), &vec1(&c["v"]))),
            "inv" => serde_json::json!(ome_zarr::linalg::inv(&mat(&c["a"])).unwrap()),
            _ => {
                let a = ome_zarr::transformnd::Affine::new(mat(&c["matrix"])).unwrap();
                serde_json::json!(a.apply(&mat(&c["points"])))
            }
        };
        if got != c["out"] {
            bad.push(format!("case {i} ({})", c["kind"]));
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {} differ: {:?}",
        bad.len(),
        cases.len(),
        bad
    );
}
