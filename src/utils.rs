//! Utility methods for ome_zarr access.

use serde_json::Value;

use crate::error::{Error, Result};

/// Recursively remove `None` (null) values, empty dicts and empty lists from a
/// nested JSON object (`_recursive_pop_nones`).
pub fn recursive_pop_nones(data: &Value) -> Value {
    match data {
        Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (k, v) in map {
                match v {
                    Value::Object(_) => {
                        let nested = recursive_pop_nones(v);
                        if nested.as_object().is_some_and(|m| !m.is_empty()) {
                            out.insert(k.clone(), nested);
                        }
                    }
                    Value::Array(items) => {
                        let mut list = vec![];
                        for item in items {
                            match item {
                                Value::Object(_) => {
                                    let n = recursive_pop_nones(item);
                                    if n.as_object().is_some_and(|m| !m.is_empty()) {
                                        list.push(n);
                                    }
                                }
                                Value::Null => {}
                                other => list.push(other.clone()),
                            }
                        }
                        if !list.is_empty() {
                            out.insert(k.clone(), Value::Array(list));
                        }
                    }
                    Value::Null => {}
                    other => {
                        out.insert(k.clone(), other.clone());
                    }
                }
            }
            Value::Object(out)
        }
        other => other.clone(),
    }
}

/// Find and remove the prefix common to all paths. Returns the last element of
/// the common prefix; each path keeps that element as its first element.
///
/// ```
/// let mut paths = vec![vec!["a".to_string(), "b".into()], vec!["a".into(), "b".into(), "c".into()]];
/// assert_eq!(ome_zarr::utils::strip_common_prefix(&mut paths).unwrap(), "b");
/// assert_eq!(paths, vec![vec!["b".to_string()], vec!["b".to_string(), "c".to_string()]]);
/// ```
pub fn strip_common_prefix(parts: &mut [Vec<String>]) -> Result<String> {
    let min_length = parts.iter().map(|p| p.len()).min().unwrap_or(0);
    let mut first_mismatch = 0;
    for idx in 0..min_length {
        let first = &parts[0][idx];
        if parts.iter().all(|p| &p[idx] == first) {
            first_mismatch += 1;
        } else {
            break;
        }
    }
    if first_mismatch == 0 {
        let mut msg = "No common prefix:\n".to_string();
        for p in parts.iter() {
            msg.push_str(&format!("{}\n", crate::pyjson::str_list_repr(p)));
        }
        return Err(Error::value(msg));
    }
    let common = parts[0][first_mismatch - 1].clone();
    for p in parts.iter_mut() {
        *p = p[first_mismatch - 1..].to_vec();
    }
    Ok(common)
}

/// Split a path into all of its components (`splitall`: repeated
/// `os.path.split`, so `a/b/` gives `['a', 'b', '']` and `.` parts are kept).
pub fn splitall(path: &std::path::Path) -> Vec<String> {
    let mut path = path.to_string_lossy().to_string();
    let mut parts: Vec<String> = vec![];
    loop {
        let (head, tail) = posix_split(&path);
        if head == path {
            parts.insert(0, head);
            break;
        } else if tail == path {
            parts.insert(0, tail);
            break;
        }
        path = head;
        parts.insert(0, tail);
    }
    parts
}

/// `posixpath.split`.
fn posix_split(p: &str) -> (String, String) {
    let i = p.rfind('/').map(|i| i + 1).unwrap_or(0);
    let (head, tail) = (&p[..i], &p[i..]);
    let head = if !head.is_empty() && head.chars().any(|c| c != '/') {
        head.trim_end_matches('/')
    } else {
        head
    };
    (head.to_string(), tail.to_string())
}

use std::path::{Path, PathBuf};

use log::{debug, info as log_info};
use serde_json::json;

use crate::classes::OmeZarrScene;
use crate::format::format_from_version;
use crate::io::{StoreLike, ZarrLocation, parse_url_r};
use crate::reader::{Node, Reader, Spec, py_tuple};
use crate::zarr::{ArraySpec, Group, JsonMap, Mode, store_lazy};

/// Print information about an OME-Zarr fileset, returning all nodes found.
///
/// With `stats`, the minimum/maximum of every array is computed (a full read).
pub fn info(path: &str, stats: bool) -> Result<Vec<Node>> {
    let mut out = String::new();
    let nodes = info_to(path, stats, &mut out)?;
    print!("{out}");
    Ok(nodes)
}

/// Like [`info`], writing the report into `out` instead of stdout.
pub fn info_to(path: &str, stats: bool, out: &mut String) -> Result<Vec<Node>> {
    use std::fmt::Write;
    let zarr = parse_url_r(path)?.ok_or_else(|| Error::Assertion("not a zarr: None".into()))?;
    let reader = Reader::new(zarr)?;
    let mut nodes = vec![];
    reader.visit(&mut |node: &Node| {
        let node = node.clone();
        if node.specs.is_empty() {
            let _ = writeln!(out, "not an ome-zarr node: {node}");
            return Ok(());
        }
        let _ = writeln!(out, "{node}");
        let loc = &node.zarr;
        let version = loc
            .zgroup
            .get("version")
            .and_then(|v| v.as_str())
            .map(String::from)
            .or_else(|| {
                loc.zgroup
                    .get("multiscales")?
                    .get(0)?
                    .get("version")?
                    .as_str()
                    .map(String::from)
            })
            .unwrap_or_default();
        let _ = writeln!(out, " - version: {version}");
        let _ = writeln!(out, " - metadata");
        for spec in &node.specs {
            let _ = writeln!(out, "   - {}", spec.kind());
        }
        let _ = writeln!(out, " - data");
        for array in &node.data {
            let mut minmax = String::new();
            if stats && array.dtype().is_complex() {
                if let Some((lo, hi)) = array.min_max_complex()? {
                    let d = array.dtype();
                    minmax = format!(
                        " minmax=({}, {})",
                        crate::pyjson::np_complex_repr(d, lo.0, lo.1),
                        crate::pyjson::np_complex_repr(d, hi.0, hi.1)
                    );
                }
            } else if stats && let Some((lo, hi)) = array.min_max()? {
                let d = array.dtype();
                minmax = format!(
                    " minmax=({}, {})",
                    crate::pyjson::np_scalar_repr(d, lo),
                    crate::pyjson::np_scalar_repr(d, hi)
                );
            }
            let _ = writeln!(out, "   - {}{minmax}", py_tuple(array.shape(), false));
        }
        debug!(
            "[{}]",
            node.data
                .iter()
                .map(|a| a.dask_repr())
                .collect::<Vec<_>>()
                .join(", ")
        );
        nodes.push(node);
        Ok(())
    })?;
    Ok(nodes)
}

/// Images found by [`find_multiscales`]: (path to image, name, parent dir).
pub type FoundImage = (PathBuf, String, String);

fn read_zattrs(dir: &Path) -> Option<serde_json::Value> {
    for name in [".zattrs", "zarr.json"] {
        let p = dir.join(name);
        if p.exists() {
            let text = std::fs::read_to_string(&p).ok()?;
            let mut v: serde_json::Value = serde_json::from_str(&text).ok()?;
            if let Some(ome) = v.get("attributes").and_then(|a| a.get("ome")) {
                v = ome.clone();
            }
            return Some(v);
        }
    }
    None
}

fn parent_str(p: &Path) -> String {
    p.parent()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default()
}

fn base_str(p: &Path) -> String {
    p.file_name()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default()
}

/// Return the list of images under an OME-Zarr path: each is
/// (full path to the multiscales image, display name, parent directory).
pub fn find_multiscales(path: &Path) -> Vec<FoundImage> {
    let Some(zattrs) = read_zattrs(path) else {
        println!("No .zattrs or zarr.json found in {}", path.display());
        return vec![];
    };
    if let Some(plate) = zattrs.get("plate") {
        let wells = plate
            .get("wells")
            .and_then(|w| w.as_array())
            .cloned()
            .unwrap_or_default();
        if let Some(w0) = wells.first() {
            let p = path
                .join(w0.get("path").and_then(|p| p.as_str()).unwrap_or(""))
                .join("0");
            return vec![(p, base_str(path), parent_str(path))];
        }
        log_info!("No wells found in plate{}", path.display());
        return vec![];
    }
    if let Some(scene) = zattrs.get("scene") {
        let mut p = path.to_path_buf();
        for ct in scene
            .get("coordinateTransformations")
            .and_then(|c| c.as_array())
            .cloned()
            .unwrap_or_default()
        {
            if let Some(ip) = ct
                .get("input")
                .and_then(|i| i.get("path"))
                .and_then(|p| p.as_str())
            {
                p = path.join(ip);
                break;
            }
        }
        return vec![(p, base_str(path), parent_str(path))];
    }
    if zattrs.get("bioformats2raw.layout") == Some(&json!(3)) {
        let xml = path.join("OME").join("METADATA.ome.xml");
        match std::fs::read_to_string(&xml) {
            Ok(text) => match roxmltree::Document::parse(&text) {
                Ok(doc) => {
                    let mut images = vec![];
                    let mut series = 0;
                    for child in doc.root_element().children().filter(|c| c.is_element()) {
                        if child.tag_name().name().ends_with("Image") {
                            let default = format!("{} Series:{series}", base_str(path));
                            let name = child.attribute("Name").map(String::from).unwrap_or(default);
                            images.push((path.join(series.to_string()), name, parent_str(path)));
                            series += 1;
                        }
                    }
                    return images;
                }
                Err(e) => println!("{e}"),
            },
            Err(e) => println!("{e}"),
        }
        return vec![];
    }
    if zattrs
        .get("multiscales")
        .is_some_and(crate::format::is_truthy)
    {
        return vec![(path.to_path_buf(), base_str(path), parent_str(path))];
    }
    vec![]
}

fn has_zarr_metadata(p: &Path) -> bool {
    p.join(".zattrs").exists() || p.join("zarr.json").exists()
}

/// The public `https://` URL of a remote fileset, for browser-based tools:
/// http(s) URLs as they are, `s3://bucket/key` on AWS (or `AWS_ENDPOINT_URL`,
/// path style), `gs://bucket/key` and `az://container/key` (with
/// `AZURE_STORAGE_ACCOUNT_NAME`). `None` for local paths and zip archives.
pub fn public_http_url(url: &str) -> Option<String> {
    if url.starts_with("http://") || url.starts_with("https://") {
        return (!crate::store::is_zip_url(url)).then(|| url.to_string());
    }
    let scheme = crate::store::cloud_scheme(url)?;
    let rest = &url[scheme.len() + 3..];
    let (bucket, key) = rest.split_once('/').unwrap_or((rest, ""));
    let join = |base: String| {
        if key.is_empty() {
            base
        } else {
            format!("{base}/{key}")
        }
    };
    match scheme {
        "s3" => Some(match std::env::var("AWS_ENDPOINT_URL") {
            Ok(ep) => join(format!("{}/{bucket}", ep.trim_end_matches('/'))),
            Err(_) => join(format!("https://{bucket}.s3.amazonaws.com")),
        }),
        "gs" | "gcs" => Some(join(format!("https://storage.googleapis.com/{bucket}"))),
        "az" | "azure" => {
            let account = std::env::var("AZURE_STORAGE_ACCOUNT_NAME").ok()?;
            Some(join(format!(
                "https://{account}.blob.core.windows.net/{bucket}"
            )))
        }
        _ => None,
    }
}

/// The ome-ngff-validator URL for an image served on `port`.
pub fn validator_url(input_path: &str, port: u16) -> (PathBuf, String) {
    let p = Path::new(input_path);
    let (mut parent, mut name) = (
        p.parent().map(Path::to_path_buf).unwrap_or_default(),
        base_str(p),
    );
    if name.is_empty() {
        name = base_str(&parent);
        parent = parent.parent().map(Path::to_path_buf).unwrap_or_default();
    }
    let url =
        format!("https://ome.github.io/ome-ngff-validator/?source=http://localhost:{port}/{name}");
    (parent, url)
}

/// Serve the parent directory with CORS and open the ome-ngff-validator in a
/// browser. `dry_run` skips opening the browser and serving (for testing).
/// Returns the validator URL, or `None` if nothing was found.
pub fn view(input_path: &str, port: u16, dry_run: bool, force: bool) -> Result<Option<String>> {
    // remote images are opened in the validator directly, nothing is served
    if let Some(remote) = public_http_url(input_path) {
        if !force && parse_url_r(input_path)?.is_none() {
            println!("No OME-Zarr image found at {input_path}. Use -f to force open in browser.");
            return Ok(None);
        }
        let url = format!(
            "https://ome.github.io/ome-ngff-validator/?source={}",
            remote.trim_end_matches('/')
        );
        if !dry_run {
            crate::server::open_browser(&url);
        }
        return Ok(Some(url));
    }
    if !force {
        let p = Path::new(input_path);
        let zarrs = if has_zarr_metadata(p) {
            find_multiscales(p)
        } else {
            vec![]
        };
        if zarrs.is_empty() {
            println!(
                "No OME-Zarr images found in {input_path}. Try $ ome_zarr finder {input_path} or use -f to force open in browser."
            );
            return Ok(None);
        }
    }
    let (parent, url) = validator_url(input_path, port);
    if dry_run {
        return Ok(Some(url));
    }
    crate::server::open_browser(&url);
    crate::server::serve(&parent, port)?;
    Ok(Some(url))
}

fn walk_images(path: &Path, out: &mut Vec<FoundImage>) {
    if has_zarr_metadata(path) {
        out.extend(find_multiscales(path));
        return;
    }
    let Ok(rd) = std::fs::read_dir(path) else {
        return;
    };
    // directory order, as Python's `Path.iterdir()`
    for p in rd.filter_map(|e| e.ok().map(|e| e.path())) {
        if has_zarr_metadata(&p) {
            out.extend(find_multiscales(&p));
        } else if p.is_dir() {
            walk_images(&p, out);
        }
    }
}

fn relpath(path: &Path, start: &Path) -> PathBuf {
    let a = crate::io::absolute(path);
    let b = crate::io::absolute(start);
    let ac: Vec<_> = a.components().collect();
    let bc: Vec<_> = b.components().collect();
    let common = ac.iter().zip(&bc).take_while(|(x, y)| x == y).count();
    let mut out = PathBuf::new();
    for _ in common..bc.len() {
        out.push("..");
    }
    for c in &ac[common..] {
        out.push(c.as_os_str());
    }
    if out.as_os_str().is_empty() {
        out.push(".");
    }
    out
}

/// Find all OME-Zarr images below `input_path`, write `biofile_finder.csv`
/// into it and open the directory in BioFile Finder. `dry_run` skips opening
/// the browser and serving. Returns the BioFile Finder URL.
pub fn finder(input_path: &str, port: u16, dry_run: bool) -> Result<Option<String>> {
    let p = Path::new(input_path);
    let (mut parent_path, mut server_dir) = (
        p.parent().map(Path::to_path_buf).unwrap_or_default(),
        base_str(p),
    );
    if server_dir.is_empty() {
        server_dir = base_str(&parent_path);
        parent_path = parent_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_default();
    }
    let mut zarrs = vec![];
    walk_images(p, &mut zarrs);
    if zarrs.is_empty() {
        println!("No OME-Zarr files found in {input_path}");
        return Ok(None);
    }
    let csv_path = p.join("biofile_finder.csv");
    let mut w =
        csv::Writer::from_path(&csv_path).map_err(|e| Error::Io(std::io::Error::other(e)))?;
    let werr = |e: csv::Error| Error::Io(std::io::Error::other(e));
    w.write_record(["File Path", "File Name", "Folders", "Uploaded"])
        .map_err(werr)?;
    for (img, name, dir) in &zarrs {
        let rel = relpath(img, p);
        let rel_url = splitall(&rel).join("/");
        let file_path = format!("http://localhost:{port}/{server_dir}/{rel_url}");
        let name = if name.is_empty() {
            base_str(img)
        } else {
            name.clone()
        };
        let folders = splitall(&relpath(Path::new(dir), p)).join(",");
        let timestamp = std::fs::metadata(img)
            .and_then(|m| m.modified())
            .map(format_utc_timestamp)
            .unwrap_or_default();
        w.write_record([file_path, name, folders, timestamp])
            .map_err(werr)?;
    }
    w.flush()?;
    let source = json!({
        "uri": format!("http://localhost:{port}/{server_dir}/biofile_finder.csv"),
        "type": "csv",
        "name": "biofile_finder.csv",
    });
    // urllib.parse.quote: keep alphanumerics and "_.-~/"
    const QUOTE: &percent_encoding::AsciiSet = &percent_encoding::NON_ALPHANUMERIC
        .remove(b'_')
        .remove(b'.')
        .remove(b'-')
        .remove(b'~')
        .remove(b'/');
    let s = percent_encoding::utf8_percent_encode(&py_json_dumps(&source), QUOTE).to_string();
    let url = format!("https://bff.allencell.org/app?source={s}&v=2");
    if dry_run {
        return Ok(Some(url));
    }
    crate::server::open_browser(&url);
    crate::server::serve(&parent_path, port)?;
    Ok(Some(url))
}

/// mtime as `YYYY-MM-DD HH:MM:SS.UTC` (Python `strftime("%Y-%m-%d %H:%M:%S.%Z")`).
#[cfg(not(target_arch = "wasm32"))]
fn format_utc_timestamp(t: std::time::SystemTime) -> String {
    let dt: chrono::DateTime<chrono::Utc> = t.into();
    dt.format("%Y-%m-%d %H:%M:%S.UTC").to_string()
}

#[cfg(target_arch = "wasm32")]
fn format_utc_timestamp(_t: std::time::SystemTime) -> String {
    String::new()
}

/// `json.dumps` formatting (`", "` and `": "` separators).
pub fn py_json_dumps(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Object(m) => {
            let items: Vec<String> = m
                .iter()
                .map(|(k, v)| {
                    format!(
                        "{}: {}",
                        serde_json::to_string(k).unwrap(),
                        py_json_dumps(v)
                    )
                })
                .collect();
            format!("{{{}}}", items.join(", "))
        }
        serde_json::Value::Array(a) => format!(
            "[{}]",
            a.iter().map(py_json_dumps).collect::<Vec<_>>().join(", ")
        ),
        other => other.to_string(),
    }
}

/// Download an OME-Zarr from the given path (local or remote) into
/// `output_dir`. All nodes found from the path are included.
pub fn download(input_path: &str, output_dir: &str) -> Result<()> {
    download_with(input_path, output_dir, &mut |s: &str| {
        use std::io::Write;
        print!("{s}");
        let _ = std::io::stdout().flush();
    })
}

/// Like [`download`], writing the progress report into `out` instead of stdout.
pub fn download_to(input_path: &str, output_dir: &str, out: &mut String) -> Result<()> {
    download_with(input_path, output_dir, &mut |s: &str| out.push_str(s))
}

fn download_with(input_path: &str, output_dir: &str, emit: &mut dyn FnMut(&str)) -> Result<()> {
    let location =
        parse_url_r(input_path)?.ok_or_else(|| Error::Assertion("not a zarr: None".into()))?;
    if location.root_attrs().contains_key("scene") {
        let name = Path::new(location.path())
            .file_name()
            .map(|n| n.to_string_lossy().to_string());
        let name = name
            .filter(|n| !n.is_empty())
            .unwrap_or_else(|| "scene.zarr".into());
        let group = location.group()?;
        let scene = OmeZarrScene::from_ome_zarr(crate::writer::GroupLike::Group(group))?;
        let target = Path::new(output_dir).join(name);
        scene.to_ome_zarr(&target.to_string_lossy(), true, true)?;
        return Ok(());
    }
    let is_zip = location.store().kind() == crate::store::StoreKind::Zip;
    let reader = Reader::new(location)?;
    let nodes = reader.nodes()?;
    // resolved paths: ome-zarr-py mixes the given root path with resolved
    // child paths, and fails for relative or symlinked paths
    let mut paths: Vec<Vec<String>> = nodes.iter().map(|n| n.zarr.resolved_parts()).collect();
    let mut common = strip_common_prefix(&mut paths)?;
    // a zipped fileset `image.ome.zarr.zip` is downloaded as `image.ome.zarr`
    if is_zip
        && let Some(stem) = common
            .strip_suffix(".zip")
            .or(common.strip_suffix(".ZIP"))
            .map(String::from)
    {
        for p in paths.iter_mut() {
            if p.first() == Some(&common) {
                p[0] = stem.clone();
            }
        }
        common = stem;
    }
    let output_path = Path::new(output_dir);
    let root_path = output_path.join(&common);
    if root_path.exists() {
        return Err(Error::Assertion(format!(
            "{} already exists!",
            root_path.display()
        )));
    }
    emit("downloading...\n");
    for p in &paths {
        emit(&format!("   {}\n", p.iter().collect::<PathBuf>().display()));
    }
    emit(&format!("to {output_dir}\n"));
    let mut pairs: Vec<(Vec<String>, &Node)> = paths.into_iter().zip(nodes.iter()).collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    for (path, node) in pairs {
        let target: PathBuf = output_path.join(path.iter().collect::<PathBuf>());
        std::fs::create_dir_all(&target)?;
        let fmt = format_from_version(node.zarr.version())?;
        let mut metadata = JsonMap::new();
        node.write_metadata(&mut metadata);
        if fmt.zarr_format() == 3 {
            let mut m = JsonMap::new();
            m.insert("ome".into(), serde_json::Value::Object(metadata));
            metadata = m;
        }
        let store = crate::store::Store::local(&target, false)?;
        let root = Group::open(&store, "", Mode::Write, Some(fmt.zarr_format()))?;
        root.set_attrs(&metadata)?;
        for spec in &node.specs {
            if let Spec::Multiscales(ms) = spec {
                let dim_names: Option<Vec<String>> = node
                    .metadata
                    .get("axes")
                    .and_then(|a| a.as_array())
                    .map(|a| {
                        a.iter()
                            .map(|x| {
                                x.get("name")
                                    .and_then(|n| n.as_str())
                                    .or(x.as_str())
                                    .unwrap_or("")
                                    .to_string()
                            })
                            .collect()
                    });
                for (dataset, data) in ms.datasets.iter().zip(&node.data).rev() {
                    log_info!("resolution {dataset}...");
                    let mut spec = ArraySpec::new(
                        data.shape(),
                        data.dtype(),
                        &data.chunksize(),
                        fmt.zarr_format(),
                    );
                    spec.separator = "/".into();
                    if fmt.zarr_format() != 2 {
                        spec.dimension_names = dim_names.clone();
                    }
                    let arr = root.create_array(dataset, &spec, false)?;
                    store_lazy(data, &arr)?;
                    emit(&format!(
                        "[########################################] | 100% Completed | {dataset}\n"
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Open a location (helper used by the CLI and tests).
pub fn open_location(path: impl Into<StoreLike>) -> Result<ZarrLocation> {
    ZarrLocation::open(path)
}

/// numpy's legacy `RandomState` generator: MT19937 with `np.random.seed`'s
/// seeding, so seeded label colours match ome-zarr-py's exactly.
#[derive(Clone)]
pub struct Mt19937 {
    key: [u32; 624],
    pos: usize,
}

impl Mt19937 {
    /// `np.random.seed(seed)` (`mt19937_seed`).
    pub fn new(seed: u32) -> Mt19937 {
        let mut key = [0u32; 624];
        let mut s = seed;
        for (pos, k) in key.iter_mut().enumerate() {
            *k = s;
            s = 1812433253u32
                .wrapping_mul(s ^ (s >> 30))
                .wrapping_add(pos as u32 + 1);
        }
        Mt19937 { key, pos: 624 }
    }

    fn generate(&mut self) {
        const N: usize = 624;
        const M: usize = 397;
        let mag = |y: u32| if y & 1 == 1 { 0x9908_b0df } else { 0 };
        for i in 0..N {
            let y = (self.key[i] & 0x8000_0000) | (self.key[(i + 1) % N] & 0x7fff_ffff);
            self.key[i] = self.key[(i + M) % N] ^ (y >> 1) ^ mag(y);
        }
        self.pos = 0;
    }

    /// `genrand_int32`.
    pub fn next_u32(&mut self) -> u32 {
        if self.pos >= 624 {
            self.generate();
        }
        let mut y = self.key[self.pos];
        self.pos += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `RandomState.randint(low, high)` for a range below 2**32: masked
    /// rejection sampling of 32-bit draws.
    pub fn randint(&mut self, low: i64, high: i64) -> i64 {
        let rng = (high - low - 1) as u64;
        if rng == 0 {
            return low;
        }
        let mut mask = rng;
        for s in [1, 2, 4, 8, 16, 32] {
            mask |= mask >> s;
        }
        loop {
            let v = (self.next_u32() as u64) & mask;
            if v <= rng {
                return low + v as i64;
            }
        }
    }
}

static NP_RANDOM: std::sync::Mutex<Option<Mt19937>> = std::sync::Mutex::new(None);

/// `np.random.seed(seed)`: seed the process-wide generator used for random
/// label colours, which then match ome-zarr-py's for the same seed.
pub fn np_random_seed(seed: u32) {
    *NP_RANDOM.lock().unwrap() = Some(Mt19937::new(seed));
}

/// `np.random.randint(low, high)` from the process-wide generator (seeded
/// from entropy until [`np_random_seed`] is called).
pub fn np_random_randint(low: i64, high: i64) -> i64 {
    let mut g = NP_RANDOM.lock().unwrap();
    let mt = g.get_or_insert_with(|| Mt19937::new(Rng::from_entropy().next_u64() as u32));
    mt.randint(low, high)
}

/// A small SplitMix64 random number generator (no OS dependency, works on wasm).
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    /// Seeded from the standard library's per-process hash keys and a counter.
    pub fn from_entropy() -> Rng {
        use std::hash::{BuildHasher, Hasher};
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
        Rng(h.finish())
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `0..n` (`np.random.randint(0, n)`).
    pub fn below(&mut self, n: u64) -> u64 {
        ((self.next_u64() as u128 * n as u128) >> 64) as u64
    }
}

#[cfg(test)]
mod splitall_tests {
    use super::*;

    /// ome-zarr-py's `splitall` / `strip_common_prefix` on these inputs.
    #[test]
    fn as_python() {
        for (p, want) in [
            ("a/b/c", vec!["a", "b", "c"]),
            ("/a/b", vec!["/", "a", "b"]),
            ("a", vec!["a"]),
            ("/", vec!["/"]),
            ("", vec![""]),
            ("a//b", vec!["a", "b"]),
            ("./a/b", vec![".", "a", "b"]),
            ("a/b/", vec!["a", "b", ""]),
            ("../x/y", vec!["..", "x", "y"]),
            ("/a/./b/..", vec!["/", "a", ".", "b", ".."]),
        ] {
            assert_eq!(splitall(std::path::Path::new(p)), want, "{p}");
        }
        let mut parts = vec![vec!["a".to_string()], vec!["b".to_string()]];
        assert_eq!(
            strip_common_prefix(&mut parts).unwrap_err().py_str(),
            "No common prefix:\n['a']\n['b']\n"
        );
        let mut parts: Vec<Vec<String>> =
            [vec!["a", "b", "c"], vec!["a", "b", "d"], vec!["a", "e"]]
                .iter()
                .map(|v| v.iter().map(|s| s.to_string()).collect())
                .collect();
        assert_eq!(strip_common_prefix(&mut parts).unwrap(), "a");
        assert_eq!(parts[2], vec!["a", "e"]);
    }
}
