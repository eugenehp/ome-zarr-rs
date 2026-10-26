//! pydantic `ValidationError`s of the ome-zarr-models classes, as pydantic
//! reports them (fixtures/pydantic.json, recorded by
//! scripts/gen_pydantic_fixtures.py: valid inputs and systematic mutations).

use serde_json::Value;

/// Python `set` reprs are ordered by the hash seed: compare them unordered.
fn normalise_sets(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(i) = rest.find("{'") {
        out.push_str(&rest[..i]);
        let Some(j) = rest[i..].find('}') else { break };
        let mut items: Vec<&str> = rest[i + 1..i + j].split(", ").collect();
        items.sort();
        out.push('{');
        out.push_str(&items.join(", "));
        out.push('}');
        rest = &rest[i + j + 1..];
    }
    out.push_str(rest);
    out
}

#[test]
fn validation_errors_match_pydantic() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pydantic.json");
    let cases: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut failures = vec![];
    for c in &cases {
        let entry = c["entry"].as_str().unwrap();
        let r = ome_zarr::models::pydantic::validate_model(entry, &c["input"]);
        let ok = c["ok"].as_bool().unwrap();
        let got = match &r {
            Ok(_) => "OK".to_string(),
            Err(ome_zarr::Error::Validation(m)) => m.clone(),
            Err(e) => format!("{}: {}", e.py_type(), e.py_str()),
        };
        let want = if ok {
            "OK".to_string()
        } else if let Some(e) = c["error"].as_str() {
            e.to_string()
        } else {
            c["other"].as_str().unwrap().to_string()
        };
        if normalise_sets(&got) != normalise_sets(&want) {
            // the differing lines only (titles can be long)
            let (w, g): (Vec<&str>, Vec<&str>) = (want.lines().collect(), got.lines().collect());
            let i = w
                .iter()
                .zip(&g)
                .position(|(a, b)| a != b)
                .unwrap_or(w.len().min(g.len()));
            let ctx = |l: &[&str]| l[i.saturating_sub(1)..(i + 3).min(l.len())].join("\n");
            failures.push(format!(
                "{entry} {}\n--- python ({} lines)\n{}\n--- rust ({} lines)\n{}",
                c["input"],
                w.len(),
                ctx(&w),
                g.len(),
                ctx(&g)
            ));
        }
    }
    let n = failures.len();
    failures.truncate(12);
    assert!(
        n == 0,
        "{n} of {} differ:\n{}",
        cases.len(),
        failures.join("\n\n")
    );
}
