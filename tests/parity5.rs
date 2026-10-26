//! Round 5 of the differential parity tests with ome-zarr-py: what `info` and
//! `download` print, and every `ome_zarr` log record they emit, at INFO and
//! DEBUG level (observations5.json).
#![cfg(not(target_arch = "wasm32"))]

mod common;

use std::sync::Mutex;

use common::*;
use serde_json::{Value, json};

static RECORDS: Mutex<Vec<Value>> = Mutex::new(Vec::new());

struct Capture;

impl log::Log for Capture {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.target().starts_with("ome_zarr")
    }
    fn log(&self, r: &log::Record) {
        if !self.enabled(r.metadata()) {
            return;
        }
        let level = match r.level() {
            log::Level::Error => "ERROR",
            log::Level::Warn => "WARNING",
            log::Level::Info => "INFO",
            log::Level::Debug | log::Level::Trace => "DEBUG",
        };
        RECORDS.lock().unwrap().push(json!([
            level,
            r.target().replace("::", "."),
            r.args().to_string()
        ]));
    }
    fn flush(&self) {}
}

fn run(level: &str, f: impl FnOnce() -> String) -> (String, Vec<Value>) {
    log::set_max_level(if level == "DEBUG" {
        log::LevelFilter::Debug
    } else {
        log::LevelFilter::Info
    });
    RECORDS.lock().unwrap().clear();
    let out = f();
    (out, std::mem::take(&mut *RECORDS.lock().unwrap()))
}

/// Replace dask / Rust progress bars by one marker per completed bar.
fn progress(s: &str) -> String {
    let mut out = vec![];
    for line in s.split('\n') {
        let last = line.rsplit('\r').next().unwrap_or("");
        if last.contains("% Completed |") {
            out.push("<progress>".to_string());
        } else {
            out.push(line.to_string());
        }
    }
    out.join("\n")
}

fn diff(name: &str, what: &str, want: &[String], got: &[String], failures: &mut Vec<String>) {
    if want != got {
        let i = want
            .iter()
            .zip(got)
            .position(|(a, b)| a != b)
            .unwrap_or(want.len().min(got.len()));
        failures.push(format!(
            "{name} {what}: first difference at {i} of {}/{}\n  python: {}\n  rust:   {}",
            want.len(),
            got.len(),
            want.get(i)
                .map(|s| s.chars().take(300).collect::<String>())
                .unwrap_or_default(),
            got.get(i)
                .map(|s| s.chars().take(300).collect::<String>())
                .unwrap_or_default()
        ));
    }
}

#[test]
fn info_and_download_output_and_logs() {
    log::set_boxed_logger(Box::new(Capture)).unwrap();
    let obs: Value = serde_json::from_str(
        &std::fs::read_to_string(fixtures().join("observations5.json")).unwrap(),
    )
    .unwrap();
    let mut failures = vec![];
    let lines = |v: &[Value]| -> Vec<String> { v.iter().map(|r| r.to_string()).collect() };
    for (key, want) in obs.as_object().unwrap() {
        let parts: Vec<&str> = key.split(':').collect();
        let (kind, name) = (parts[0], parts[1]);
        let (out, logs) = match kind {
            "info" => {
                let stats = parts[2] == "True";
                run(parts[3], || {
                    let mut s = String::new();
                    ome_zarr::utils::info_to(py_store(name).to_str().unwrap(), stats, &mut s)
                        .unwrap();
                    s
                })
            }
            _ => {
                let out_dir = py_store(&format!("dl5_{name}"));
                let d = tempfile::tempdir().unwrap();
                let tmp_out = d.path().join("out");

                run("INFO", || {
                    let mut s = String::new();
                    ome_zarr::utils::download_to(
                        py_store(name).to_str().unwrap(),
                        tmp_out.to_str().unwrap(),
                        &mut s,
                    )
                    .unwrap();
                    s.replace(tmp_out.to_str().unwrap(), out_dir.to_str().unwrap())
                })
            }
        };
        let want_out = progress(want["stdout"].as_str().unwrap());
        let got_out = progress(&out);
        diff(
            key,
            "stdout",
            &want_out.lines().map(String::from).collect::<Vec<_>>(),
            &got_out.lines().map(String::from).collect::<Vec<_>>(),
            &mut failures,
        );
        diff(
            key,
            "logs",
            &lines(want["logs"].as_array().unwrap()),
            &lines(&logs),
            &mut failures,
        );
    }
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
