//! The `ome_zarr` command line parses arguments, reports errors and prints
//! help exactly as ome-zarr-py's argparse does (fixtures/cli_argparse.json,
//! written by scripts/gen_cli_fixtures.py). Compared with the parser as
//! ome-zarr-py defines it, i.e. without this crate's `render` and `view --web`.

use ome_zarr::cli::parser;
use serde_json::Value;

#[test]
fn argparse_output_matches_python() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cli_argparse.json");
    let cases: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let p = parser(false);
    let mut failures = vec![];
    for c in &cases {
        let argv: Vec<String> = c["argv"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a.as_str().unwrap().to_string())
            .collect();
        let (code, out, err) = match p.parse(&argv) {
            Ok(ns) => (0, String::new(), format!("<parsed {:?}>", ns.command)),
            Err(e) => (e.code, e.stdout, e.stderr),
        };
        let (wc, wo, we) = (
            c["code"].as_i64().unwrap() as i32,
            c["stdout"].as_str().unwrap(),
            c["stderr"].as_str().unwrap(),
        );
        if (code, out.as_str(), err.as_str()) != (wc, wo, we) {
            failures.push(format!(
                "{argv:?}\n--- python ({wc})\n{wo}{we}--- rust ({code})\n{out}{err}"
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

#[test]
fn extensions_parse() {
    let cli =
        ome_zarr::cli::parse_args(&["view".into(), "x.zarr".into(), "--web".into(), "-f".into()])
            .unwrap();
    assert!(matches!(
        cli.command,
        ome_zarr::cli::Command::View {
            web: true,
            force: true,
            ..
        }
    ));
    let cli = ome_zarr::cli::parse_args(&[
        "-vv".into(),
        "render".into(),
        "x".into(),
        "-oout.png".into(),
        "--wid".into(),
        "64".into(),
    ])
    .unwrap();
    assert_eq!(cli.verbose, 2);
    match cli.command {
        ome_zarr::cli::Command::Render {
            output,
            width,
            height,
            ..
        } => assert_eq!((output.as_str(), width, height), ("out.png", 64, 512)),
        other => panic!("{other:?}"),
    }
}
