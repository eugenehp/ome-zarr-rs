//! Opening OME-Zarr from online sources: directories and zip archives over
//! HTTP (with and without range requests), served by a small local server.
//! Tests against public cloud buckets are `#[ignore]`d (they need network).
#![cfg(not(target_arch = "wasm32"))]

mod common;

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use common::*;
use ome_zarr::store::StoreKind;

/// A static file server; `ranges` toggles HTTP Range support. Logs the
/// `Range` header of every GET.
struct Server {
    base: String,
    log: Arc<Mutex<Vec<String>>>,
}

fn serve(root: PathBuf, ranges: bool) -> Server {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let log = Arc::new(Mutex::new(vec![]));
    let log2 = log.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let root = root.clone();
            let log = log2.clone();
            std::thread::spawn(move || {
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 {
                        return;
                    }
                    let mut parts = line.split_whitespace();
                    let (method, path) = (
                        parts.next().unwrap_or("").to_string(),
                        parts.next().unwrap_or("/").to_string(),
                    );
                    let mut range = None;
                    loop {
                        let mut h = String::new();
                        reader.read_line(&mut h).unwrap();
                        let h = h.trim_end();
                        if h.is_empty() {
                            break;
                        }
                        if let Some(v) = h.strip_prefix("range: ").or(h.strip_prefix("Range: ")) {
                            range = Some(v.to_string());
                        }
                    }
                    let file = root.join(path.trim_start_matches('/'));
                    let body = std::fs::read(&file).ok().filter(|_| file.is_file());
                    let Some(body) = body else {
                        let _ = stream
                            .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
                        continue;
                    };
                    if method == "GET" {
                        log.lock().unwrap().push(range.clone().unwrap_or_default());
                    }
                    let accept = if ranges {
                        "Accept-Ranges: bytes\r\n"
                    } else {
                        ""
                    };
                    let (status, data, extra) = match (&range, ranges) {
                        (Some(r), true) => {
                            let (a, b) = r.trim_start_matches("bytes=").split_once('-').unwrap();
                            let a: usize = a.parse().unwrap();
                            let b: usize = b
                                .parse::<usize>()
                                .unwrap_or(body.len() - 1)
                                .min(body.len() - 1);
                            (
                                "206 Partial Content",
                                body[a..=b].to_vec(),
                                format!("Content-Range: bytes {a}-{b}/{}\r\n", body.len()),
                            )
                        }
                        _ => ("200 OK", body, String::new()),
                    };
                    let head = format!(
                        "HTTP/1.1 {status}\r\nContent-Length: {}\r\n{accept}{extra}\r\n",
                        data.len()
                    );
                    let _ = stream.write_all(head.as_bytes());
                    if method != "HEAD" {
                        let _ = stream.write_all(&data);
                    }
                }
            });
        }
    });
    Server { base, log }
}

fn zip_dir(src: &Path, dst: &Path) {
    let f = std::fs::File::create(dst).unwrap();
    let mut z = zip::ZipWriter::new(f);
    let mut files = vec![];
    walk(src, src, &mut files);
    files.sort();
    for rel in files {
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        z.start_file(rel.to_string_lossy(), opts).unwrap();
        let mut data = vec![];
        std::fs::File::open(src.join(&rel))
            .unwrap()
            .read_to_end(&mut data)
            .unwrap();
        z.write_all(&data).unwrap();
    }
    z.finish().unwrap();
}

fn fixture_root() -> (tempfile::TempDir, PathBuf) {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().to_path_buf();
    zip_dir(
        &py_store("coins_0.4.zarr"),
        &root.join("coins.ome.zarr.zip"),
    );
    zip_dir(
        &py_store("plate_0.5.zarr"),
        &root.join("plate.ome.zarr.zip"),
    );
    (d, root)
}

fn info(url: &str) -> String {
    let mut out = String::new();
    ome_zarr::utils::info_to(url, false, &mut out).unwrap();
    out
}

#[test]
fn directory_over_http() {
    let srv = serve(py_store(""), true);
    let out = info(&format!("{}/coins_0.4.zarr", srv.base));
    assert!(out.contains("coins_0.4.zarr/labels/coins/"), "{out}");
    let loc = ome_zarr::io::parse_url(
        format!("{}/coins_0.4.zarr", srv.base),
        "r",
        ome_zarr::format::CURRENT_FORMAT,
    )
    .unwrap()
    .unwrap();
    assert_eq!(loc.store().kind(), StoreKind::Http);
}

#[test]
fn zip_over_http_with_and_without_ranges() {
    let (_d, root) = fixture_root();
    let local = info(py_store("coins_0.4.zarr").to_str().unwrap());
    for ranges in [true, false] {
        let srv = serve(root.clone(), ranges);
        let url = format!("{}/coins.ome.zarr.zip", srv.base);
        let out = info(&url);
        // the same nodes as the directory, under the zip URL
        let names = |s: &str, base: &str| -> Vec<String> {
            s.lines()
                .filter(|l| l.contains("[zgroup]"))
                .map(|l| {
                    l.split_whitespace()
                        .next()
                        .unwrap()
                        .trim_start_matches(base)
                        .trim_matches('/')
                        .to_string()
                })
                .collect()
        };
        let base_local = py_store("coins_0.4.zarr")
            .canonicalize()
            .unwrap()
            .display()
            .to_string();
        assert_eq!(
            names(&out, &url),
            names(&local, &base_local),
            "ranges={ranges}"
        );
        let log = srv.log.lock().unwrap().clone();
        assert_eq!(
            log.iter().all(|r| r.starts_with("bytes=")),
            ranges,
            "{log:?}"
        );
        // read pixels through the zip
        let loc = ome_zarr::io::parse_url(&url, "r", ome_zarr::format::CURRENT_FORMAT)
            .unwrap()
            .unwrap();
        let a = ome_zarr::zarr::ZArray::open(loc.store(), "s0")
            .unwrap()
            .read_all()
            .unwrap();
        let b = ome_zarr::zarr::ZArray::open(
            &ome_zarr::store::Store::local(py_store("coins_0.4.zarr"), true).unwrap(),
            "s0",
        )
        .unwrap()
        .read_all()
        .unwrap();
        assert!(a == b);
    }
}

#[test]
fn download_from_a_zip_url() {
    let (_d, root) = fixture_root();
    let srv = serve(root, true);
    let out = tempfile::tempdir().unwrap();
    ome_zarr::utils::download(
        &format!("{}/coins.ome.zarr.zip", srv.base),
        out.path().to_str().unwrap(),
    )
    .unwrap();
    // `.zip` is dropped from the name; contents as a download of the directory
    let reference = tempfile::tempdir().unwrap();
    ome_zarr::utils::download(
        py_store("coins_0.4.zarr").to_str().unwrap(),
        reference.path().to_str().unwrap(),
    )
    .unwrap();
    std::fs::rename(
        reference.path().join("coins_0.4.zarr"),
        reference.path().join("coins.ome.zarr"),
    )
    .unwrap();
    compare_stores(
        &reference.path().join("coins.ome.zarr"),
        &out.path().join("coins.ome.zarr"),
    );
}

#[test]
fn plate_in_a_zip_url() {
    let (_d, root) = fixture_root();
    let srv = serve(root, true);
    let url = format!("{}/plate.ome.zarr.zip", srv.base);
    let loc = ome_zarr::io::parse_url(&url, "r", ome_zarr::format::CURRENT_FORMAT)
        .unwrap()
        .unwrap();
    let reader = ome_zarr::reader::Reader::new(loc).unwrap();
    let nodes = reader.nodes().unwrap();
    assert!(!nodes.is_empty());
    // wells and images inside the archive are reachable
    let well = ome_zarr::io::parse_url(url.as_str(), "r", ome_zarr::format::CURRENT_FORMAT)
        .unwrap()
        .unwrap()
        .create("A/1/0")
        .unwrap();
    assert!(well.exists());
}

#[test]
fn web_view_of_remote_images() {
    use ome_zarr::utils::public_http_url;
    let srv = serve(py_store(""), true);
    let url = format!("{}/coins_0.4.zarr", srv.base);
    let v = ome_zarr::utils::view(&url, 8000, true, false)
        .unwrap()
        .unwrap();
    assert_eq!(
        v,
        format!("https://ome.github.io/ome-ngff-validator/?source={url}")
    );
    assert!(
        ome_zarr::utils::view(&format!("{}/nope.zarr", srv.base), 8000, true, false)
            .unwrap()
            .is_none()
    );
    assert_eq!(
        public_http_url("gs://bkt/a/b.zarr").unwrap(),
        "https://storage.googleapis.com/bkt/a/b.zarr"
    );
    assert_eq!(public_http_url("/local/path"), None);
    assert_eq!(public_http_url("https://x/y.zip"), None);
}

#[test]
fn cloud_scheme_detection() {
    use ome_zarr::store::{cloud_scheme, is_zip_url};
    assert_eq!(cloud_scheme("s3://b/k"), Some("s3"));
    assert_eq!(cloud_scheme("gs://b/k"), Some("gs"));
    assert_eq!(
        cloud_scheme("abfs://c@a.dfs.core.windows.net/k"),
        Some("abfs")
    );
    assert_eq!(cloud_scheme("s3data/x"), None);
    assert!(is_zip_url("https://h/a.ome.zarr.zip?token=1"));
    assert!(!is_zip_url("https://h/a.ome.zarr"));
}

// one test: the endpoint is process-wide environment
#[test]
#[ignore = "network: public S3 buckets on AWS and EMBL-EBI"]
fn s3_anonymous_aws_and_custom_endpoint() {
    let out =
        info("s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8");
    assert!(out.contains("(6368, 1600, 12000)"), "{out}");
    // SAFETY: no other test in this binary reads the environment concurrently
    // in a way that depends on this variable (the other tests are local)
    unsafe { std::env::set_var("AWS_ENDPOINT_URL", "https://uk1s3.embassy.ebi.ac.uk") };
    let out = info("s3://idr/zarr/v0.4/idr0062A/6001240.zarr");
    assert!(out.contains("6001240.zarr/labels/0/"), "{out}");
}
