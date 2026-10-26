//! Reading logic for ome-zarr.
//!
//! Primary entry point is the [`parse_url`] method.

use std::path::{Path, PathBuf};

use log::{debug, warn};

use crate::array::LazyArray;
use std::fmt;

use crate::error::{Error, Result};
use crate::format::{CURRENT_FORMAT, Format, detect_format};
use crate::store::{Store, StoreKind};
use crate::zarr::{Group, JsonMap, Mode, ZArray, join_path};

impl fmt::Display for StoreLike {
    /// What Python's `%s` shows for the `path` argument of `ZarrLocation`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StoreLike::Path(p) => write!(f, "{p}"),
            StoreLike::Store(s) => write!(f, "{}", s.py_repr()),
            StoreLike::StorePath(s, p) => write!(f, "{}/{p}", s.py_repr()),
        }
    }
}

/// What a [`ZarrLocation`] can be created from (zarr-python's `StoreLike`).
#[derive(Debug, Clone)]
pub enum StoreLike {
    /// A local path or a URL (`http(s)://`, `s3://`).
    Path(String),
    /// A store, the location being its root.
    Store(Store),
    /// A store plus a prefix inside it (`StorePath`).
    StorePath(Store, String),
}

impl From<&str> for StoreLike {
    fn from(s: &str) -> Self {
        StoreLike::Path(s.to_string())
    }
}

impl From<String> for StoreLike {
    fn from(s: String) -> Self {
        StoreLike::Path(s)
    }
}

impl From<&String> for StoreLike {
    fn from(s: &String) -> Self {
        StoreLike::Path(s.clone())
    }
}

impl From<&Path> for StoreLike {
    fn from(p: &Path) -> Self {
        let resolved = std::fs::canonicalize(p).unwrap_or_else(|_| absolute(p));
        StoreLike::Path(resolved.to_string_lossy().to_string())
    }
}

impl From<PathBuf> for StoreLike {
    fn from(p: PathBuf) -> Self {
        StoreLike::from(p.as_path())
    }
}

impl From<&PathBuf> for StoreLike {
    fn from(p: &PathBuf) -> Self {
        StoreLike::from(p.as_path())
    }
}

impl From<Store> for StoreLike {
    fn from(s: Store) -> Self {
        StoreLike::Store(s)
    }
}

impl From<&Store> for StoreLike {
    fn from(s: &Store) -> Self {
        StoreLike::Store(s.clone())
    }
}

/// Make a path absolute and normalise `.` / `..` components (like `Path.resolve`
/// for paths that do not exist).
pub fn absolute(p: &Path) -> PathBuf {
    let base = if p.is_absolute() {
        PathBuf::new()
    } else {
        std::env::current_dir().unwrap_or_default()
    };
    let mut out = PathBuf::new();
    for c in base.join(p).components() {
        match c {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// IO primitive for reading and writing Zarr data. Uses a store for all data
/// access.
///
/// No assumptions about the existence of the given path string are made.
/// Attempts are made to load various metadata files and cache them internally.
#[derive(Debug, Clone)]
pub struct ZarrLocation {
    fmt: Format,
    mode: String,
    /// A store passed in directly has no path of its own: the location is then
    /// the store plus a prefix inside it.
    prefix: Option<String>,
    /// The resolved path (`subpath`, the repr).
    resolved: String,
    /// The path as given (Python's `ZarrLocation.path`; `path` is resolved).
    given_path: String,
    store: Store,
    /// Path of this location inside `store` when the store is shared with a
    /// parent location (remote stores), and `prefix` is `None`.
    root_prefix: String,
    /// Group attributes (the `ome` namespace for zarr v3).
    pub zgroup: JsonMap,
    /// Array attributes (unused, kept for API compatibility).
    pub zarray: JsonMap,
    metadata: JsonMap,
    exists: bool,
}

impl ZarrLocation {
    /// Create a location in mode `"r"` with the current format.
    pub fn open(path: impl Into<StoreLike>) -> Result<ZarrLocation> {
        ZarrLocation::new(path, "r", CURRENT_FORMAT)
    }

    pub fn new(path: impl Into<StoreLike>, mode: &str, fmt: Format) -> Result<ZarrLocation> {
        let path = path.into();
        debug!("ZarrLocation.__init__ path: {path}, fmt: {}", fmt.version());
        // logged as given (Python keeps a `str` path unresolved)
        let given = path.to_string();
        let mut given_path: Option<String> = None;
        let (path_str, prefix, store) = match path {
            StoreLike::Path(p) => {
                given_path = Some(p.clone());
                let store = fmt.init_store(&p, mode)?;
                // resolve local paths so that child locations (which are
                // resolved) share their components with this one
                let p = if store.kind() == StoreKind::Local {
                    std::fs::canonicalize(&p)
                        .map(|c| c.to_string_lossy().to_string())
                        .unwrap_or(p)
                } else {
                    p
                };
                (p, None, store)
            }
            StoreLike::Store(s) => (
                s.root().map(String::from).unwrap_or_default(),
                Some(String::new()),
                s,
            ),
            StoreLike::StorePath(s, p) => {
                let p = p.trim_matches('/').to_string();
                (p.clone(), Some(p), s)
            }
        };
        // A store with a root path (Local/Fsspec in Python) is addressed by path
        let prefix = match (&prefix, store.kind()) {
            (Some(p), StoreKind::Local | StoreKind::Http | StoreKind::ObjectStore)
                if p.is_empty() =>
            {
                None
            }
            _ => prefix,
        };
        let mut loc = ZarrLocation {
            fmt,
            mode: mode.to_string(),
            prefix,
            given_path: given_path.unwrap_or_else(|| path_str.clone()),
            resolved: path_str,
            store,
            root_prefix: String::new(),
            zgroup: JsonMap::new(),
            zarray: JsonMap::new(),
            metadata: JsonMap::new(),
            exists: true,
        };
        loc.init_metadata()?;
        let detected = detect_format(&loc.metadata, fmt);
        debug!("ZarrLocation.__init__ {given} detected: {detected}");
        if detected != fmt {
            warn!("version mismatch: detected: {detected}, requested: {fmt}");
            loc.fmt = detected;
            if loc.prefix.is_none() {
                // as ome-zarr-py, which re-creates the store for the detected
                // format (the same store here)
                debug!(target: "ome_zarr::format", "Created nested FsspecStore({given}, {})", loc.mode);
            }
            loc.init_metadata()?;
        }
        Ok(loc)
    }

    fn with_shared_store(
        path: String,
        store: Store,
        root_prefix: String,
        mode: &str,
        fmt: Format,
    ) -> Result<ZarrLocation> {
        let mut loc = ZarrLocation {
            fmt,
            mode: mode.to_string(),
            prefix: None,
            given_path: path.clone(),
            resolved: path,
            store,
            root_prefix,
            zgroup: JsonMap::new(),
            zarray: JsonMap::new(),
            metadata: JsonMap::new(),
            exists: true,
        };
        loc.init_metadata()?;
        let detected = detect_format(&loc.metadata, fmt);
        if detected != fmt {
            warn!("version mismatch: detected: {detected}, requested: {fmt}");
            loc.fmt = detected;
            loc.init_metadata()?;
        }
        Ok(loc)
    }

    fn group_path(&self) -> &str {
        self.prefix.as_deref().unwrap_or(&self.root_prefix)
    }

    /// Load the Zarr metadata files for the given location.
    fn init_metadata(&mut self) -> Result<()> {
        self.zgroup = JsonMap::new();
        self.zarray = JsonMap::new();
        self.metadata = JsonMap::new();
        self.exists = true;
        match Group::open(&self.store, self.group_path(), Mode::Read, None) {
            Ok(group) => {
                let mut attrs = group.attrs()?;
                if let Some(serde_json::Value::Object(ome)) = attrs.get("ome") {
                    attrs = ome.clone();
                }
                self.zgroup = attrs.clone();
                self.metadata = attrs;
            }
            Err(_) => {
                if self.mode == "w" {
                    Group::open(
                        &self.store,
                        self.group_path(),
                        Mode::Append,
                        Some(self.fmt.zarr_format()),
                    )?;
                } else {
                    self.exists = false;
                }
            }
        }
        Ok(())
    }

    /// Return true if the group metadata exists.
    pub fn exists(&self) -> bool {
        self.exists
    }

    pub fn fmt(&self) -> Format {
        self.fmt
    }

    pub fn mode(&self) -> &str {
        &self.mode
    }

    /// Return the version of the OME-NGFF spec used for this location.
    pub fn version(&self) -> &'static str {
        self.fmt.version()
    }

    /// The path as given (`ZarrLocation.path`).
    pub fn path(&self) -> &str {
        &self.given_path
    }

    /// Return the initialized store for this location.
    pub fn store(&self) -> &Store {
        &self.store
    }

    /// The prefix of this location inside its store (if created from a store).
    pub fn prefix(&self) -> Option<&str> {
        self.prefix.as_deref()
    }

    /// Return the contents of the zattrs file (the `ome` namespace for v3).
    pub fn root_attrs(&self) -> JsonMap {
        self.metadata.clone()
    }

    /// The group at this location.
    pub fn group(&self) -> Result<Group> {
        let mode = if self.mode == "r" {
            Mode::Read
        } else {
            Mode::ReadWrite
        };
        Group::open(&self.store, self.group_path(), mode, None)
    }

    /// The path of `subpath` inside the store, below this location's prefix.
    pub fn inner(&self, subpath: &str) -> String {
        join_path(self.group_path(), subpath)
    }

    /// Lazily load the array at `subpath` (`da.from_zarr`).
    pub fn load(&self, subpath: &str) -> Result<LazyArray> {
        Ok(ZArray::open(&self.store, &self.inner(subpath))?.to_lazy())
    }

    /// Open the array at `subpath`.
    pub fn array(&self, subpath: &str) -> Result<ZArray> {
        ZArray::open(&self.store, &self.inner(subpath))
    }

    /// Return the last element of the underlying location.
    ///
    /// ```
    /// # use ome_zarr::io::ZarrLocation;
    /// assert_eq!(ZarrLocation::open("/tmp/foo").unwrap().basename(), "foo");
    /// ```
    pub fn basename(&self) -> String {
        let p = self.resolved.strip_suffix('/').unwrap_or(&self.resolved);
        p.rsplit('/').next().unwrap_or("").to_string()
    }

    /// Create a new Zarr location for the given path.
    pub fn create(&self, path: &str) -> Result<ZarrLocation> {
        if self.prefix.is_some() {
            let child = StoreLike::StorePath(self.store.clone(), self.inner(path));
            return ZarrLocation::new(child, &self.mode, self.fmt);
        }
        let sub = self.subpath(path);
        debug!("open(ZarrLocation({sub}))");
        if matches!(
            self.store.kind(),
            StoreKind::Http | StoreKind::ObjectStore | StoreKind::Zip
        ) && !path.contains("..")
        {
            // share the connection (and metadata cache) of remote stores, and
            // the open archive of zip stores
            return ZarrLocation::with_shared_store(
                sub,
                self.store.clone(),
                self.inner(path),
                &self.mode,
                self.fmt,
            );
        }
        ZarrLocation::new(StoreLike::Path(sub), &self.mode, self.fmt)
    }

    /// The path components.
    /// The components of the resolved path (what `download` needs: the
    /// given path of the root and the resolved paths of its children then
    /// share a prefix).
    pub fn resolved_parts(&self) -> Vec<String> {
        if self.is_file() {
            crate::utils::splitall(Path::new(&self.resolved))
        } else {
            self.resolved.split('/').map(String::from).collect()
        }
    }

    pub fn parts(&self) -> Vec<String> {
        if self.is_file() {
            // `pathlib.Path(path).parts`
            let p = &self.given_path;
            let mut out: Vec<String> = vec![];
            if p.starts_with('/') {
                out.push("/".into());
            }
            out.extend(
                p.split('/')
                    .filter(|c| !c.is_empty() && *c != ".")
                    .map(String::from),
            );
            out
        } else {
            self.given_path.split('/').map(String::from).collect()
        }
    }

    /// Join `subpath` onto this location's path (file paths are resolved;
    /// http URLs are joined with `urljoin` semantics).
    pub fn subpath(&self, subpath: &str) -> String {
        if self.is_file() {
            let p = Path::new(&self.resolved).join(subpath);
            let p = std::fs::canonicalize(&p).unwrap_or_else(|_| absolute(&p));
            let s = p.to_string_lossy().to_string();
            if subpath.is_empty() {
                s.trim_end_matches('/').to_string()
            } else {
                s
            }
        } else if self.is_http() {
            let base = if self.resolved.ends_with('/') {
                self.resolved.clone()
            } else {
                format!("{}/", self.resolved)
            };
            match url::Url::parse(&base).and_then(|u| u.join(subpath)) {
                Ok(u) => u.to_string(),
                Err(_) => format!("{base}{subpath}"),
            }
        } else if self.resolved.ends_with('/') {
            normalize_url_path(&format!("{}{subpath}", self.resolved))
        } else {
            normalize_url_path(&format!("{}/{subpath}", self.resolved))
        }
    }

    /// Whether the underlying store is a local filesystem.
    pub fn is_file(&self) -> bool {
        self.store.kind() == StoreKind::Local
    }

    /// Whether the store points to an http or https URL.
    pub fn is_http(&self) -> bool {
        self.store.kind() == StoreKind::Http
    }
}

/// Resolve `.` and `..` segments of a URL-like path (`s3://bucket/a/../b` ->
/// `s3://bucket/b`), never above the bucket / authority.
fn normalize_url_path(p: &str) -> String {
    let (head, rest) = match p.find("://") {
        Some(i) => {
            let after = &p[i + 3..];
            let j = after.find('/').map(|j| i + 3 + j).unwrap_or(p.len());
            (&p[..j], &p[j..])
        }
        None => ("", p),
    };
    let absolute = rest.starts_with('/') || !head.is_empty();
    let mut out: Vec<&str> = vec![];
    for seg in rest.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                if out.last().is_some_and(|s| *s != "..") {
                    out.pop();
                } else if !absolute {
                    out.push("..");
                }
            }
            s => out.push(s),
        }
    }
    let trailing = if rest.ends_with('/') && !out.is_empty() {
        "/"
    } else {
        ""
    };
    let lead = if absolute { "/" } else { "" };
    format!("{head}{lead}{}{trailing}", out.join("/"))
}

impl PartialEq for ZarrLocation {
    fn eq(&self, rhs: &Self) -> bool {
        if self.prefix.is_some() || rhs.prefix.is_some() {
            return self.store.same(&rhs.store) && self.prefix == rhs.prefix;
        }
        self.subpath("") == rhs.subpath("")
    }
}

impl std::fmt::Display for ZarrLocation {
    /// Print the path as well as whether this is a group or an array.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut suffix = String::new();
        if !self.zgroup.is_empty() {
            suffix.push_str(" [zgroup]");
        }
        if !self.zarray.is_empty() {
            suffix.push_str(" [zarray]");
        }
        let base = if self.prefix.is_some() {
            self.inner("")
        } else {
            self.subpath("")
        };
        write!(f, "{base}{suffix}")
    }
}

/// Convert a path string or URL to a [`ZarrLocation`].
///
/// If mode is `"r"` and the path does not exist returns `None`.
pub fn parse_url(
    path: impl Into<StoreLike>,
    mode: &str,
    fmt: Format,
) -> Result<Option<ZarrLocation>> {
    let loc = match ZarrLocation::new(path, mode, fmt) {
        Ok(l) => l,
        Err(e) if mode.contains('r') => {
            debug!("parse_url failed: {e}");
            return Ok(None);
        }
        Err(e) => return Err(e),
    };
    if mode.contains('r') && !loc.exists() {
        Ok(None)
    } else {
        Ok(Some(loc))
    }
}

/// [`parse_url`] in read mode with the current format.
pub fn parse_url_r(path: impl Into<StoreLike>) -> Result<Option<ZarrLocation>> {
    parse_url(path, "r", CURRENT_FORMAT)
}

/// Error helper: "not a zarr".
pub fn not_a_zarr(path: &str) -> Error {
    Error::Assertion(format!("not a zarr: {path}"))
}

#[cfg(test)]
mod normalize_tests {
    use super::normalize_url_path;

    #[test]
    fn dot_segments() {
        assert_eq!(
            normalize_url_path("s3://b/img.zarr/labels/0/../../"),
            "s3://b/img.zarr/"
        );
        assert_eq!(
            normalize_url_path("s3://b/img.zarr/labels/0/../../labels/0"),
            "s3://b/img.zarr/labels/0"
        );
        assert_eq!(normalize_url_path("gs://b/../x"), "gs://b/x");
        assert_eq!(normalize_url_path("a.zip/labels/./x/"), "a.zip/labels/x/");
        assert_eq!(normalize_url_path("../a/b/.."), "../a");
    }
}
