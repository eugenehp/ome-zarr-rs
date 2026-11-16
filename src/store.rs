//! Storage backends (filesystem, memory, http, zip, s3) behind one dynamic store type.
//!
//! The equivalent of `zarr.storage.LocalStore`, `MemoryStore`, `FsspecStore` and
//! `ZipStore` in zarr-python.

use std::collections::BTreeMap;
use std::io::Read;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use bytes::Bytes;

use crate::error::{Error, Result};

/// Part of a value: `start..end`, or the last `n` bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteRange {
    /// Bytes `start..end`.
    Range(u64, u64),
    /// The last `n` bytes.
    Suffix(u64),
}

impl ByteRange {
    /// The bytes this range covers of a value `len` long.
    pub fn within(&self, len: u64) -> Result<Range<u64>> {
        let r = match *self {
            ByteRange::Range(a, b) => a..b,
            ByteRange::Suffix(n) => len.saturating_sub(n)..len,
        };
        if r.start > r.end || r.end > len {
            return Err(Error::Storage(format!(
                "byte range {self:?} out of bounds of {len} bytes"
            )));
        }
        Ok(r)
    }

    /// This range of `value`.
    pub fn slice(&self, value: &Bytes) -> Result<Bytes> {
        let r = self.within(value.len() as u64)?;
        Ok(value.slice(r.start as usize..r.end as usize))
    }

    /// The value of an HTTP `Range` header asking for this range; `None` if
    /// it is empty (nothing to ask for).
    pub fn http_header(&self) -> Option<String> {
        match *self {
            ByteRange::Range(a, b) if b > a => Some(format!("bytes={a}-{}", b - 1)),
            ByteRange::Suffix(n) if n > 0 => Some(format!("bytes=-{n}")),
            _ => None,
        }
    }
}

/// Every range of `value`.
pub fn slice_all(value: &Bytes, ranges: &[ByteRange]) -> Result<Vec<Bytes>> {
    ranges.iter().map(|r| r.slice(value)).collect()
}

fn read_only(key: &str) -> Error {
    Error::Storage(format!("store is read-only: cannot write {key}"))
}

fn unlistable() -> Error {
    Error::Storage("listing is not supported by this store".into())
}

/// A key-value store of Zarr documents and chunks. Keys have no leading
/// slash; a prefix is `""` or ends in `/`. Only [`get`](Self::get) is
/// required: the rest default to reading whole values, refusing writes and
/// refusing to list.
pub trait Storage: crate::par::MaybeSendSync {
    /// The value at `key`, if there is one.
    fn get(&self, key: &str) -> Result<Option<Bytes>>;

    /// Parts of the value at `key`, if there is one.
    fn get_ranges(&self, key: &str, ranges: &[ByteRange]) -> Result<Option<Vec<Bytes>>> {
        let Some(value) = self.get(key)? else {
            return Ok(None);
        };
        slice_all(&value, ranges).map(Some)
    }

    /// Whether [`get_ranges`](Self::get_ranges) reads only the parts.
    fn supports_ranges(&self) -> bool {
        false
    }

    /// The length of the value at `key`, if there is one.
    fn size(&self, key: &str) -> Result<Option<u64>> {
        Ok(self.get(key)?.map(|v| v.len() as u64))
    }

    /// Store `value` at `key`.
    fn set(&self, key: &str, _value: Bytes) -> Result<()> {
        Err(read_only(key))
    }

    /// Delete `key` (nothing if it is not there).
    fn erase(&self, key: &str) -> Result<()> {
        Err(read_only(key))
    }

    /// Delete every key under `prefix`.
    fn erase_prefix(&self, prefix: &str) -> Result<()> {
        Err(read_only(prefix))
    }

    /// Every key under `prefix`, sorted.
    fn list_prefix(&self, _prefix: &str) -> Result<Vec<String>> {
        Err(unlistable())
    }

    /// The keys directly under `prefix`, and the prefixes one level down
    /// (ending in `/`), both in full and sorted.
    fn list_dir(&self, prefix: &str) -> Result<(Vec<String>, Vec<String>)> {
        let mut keys = vec![];
        let mut prefixes: Vec<String> = vec![];
        for k in self.list_prefix(prefix)? {
            match k[prefix.len()..].split_once('/') {
                None => keys.push(k),
                Some((dir, _)) => {
                    let p = format!("{prefix}{dir}/");
                    if prefixes.last() != Some(&p) && !prefixes.contains(&p) {
                        prefixes.push(p);
                    }
                }
            }
        }
        keys.sort();
        prefixes.sort();
        Ok((keys, prefixes))
    }
}

/// A shareable, dynamically typed zarr store.
pub type DynStore = Arc<dyn Storage>;

/// An in-memory store.
#[derive(Debug, Default)]
pub struct MemoryStore {
    data: RwLock<BTreeMap<String, Bytes>>,
}

impl MemoryStore {
    pub fn new() -> MemoryStore {
        MemoryStore::default()
    }
}

impl Storage for MemoryStore {
    fn get(&self, key: &str) -> Result<Option<Bytes>> {
        Ok(self.data.read().unwrap().get(key).cloned())
    }
    fn supports_ranges(&self) -> bool {
        true
    }
    fn set(&self, key: &str, value: Bytes) -> Result<()> {
        self.data.write().unwrap().insert(key.to_string(), value);
        Ok(())
    }
    fn erase(&self, key: &str) -> Result<()> {
        self.data.write().unwrap().remove(key);
        Ok(())
    }
    fn erase_prefix(&self, prefix: &str) -> Result<()> {
        self.data
            .write()
            .unwrap()
            .retain(|k, _| !k.starts_with(prefix));
        Ok(())
    }
    fn list_prefix(&self, prefix: &str) -> Result<Vec<String>> {
        Ok(self
            .data
            .read()
            .unwrap()
            .range(prefix.to_string()..)
            .take_while(|(k, _)| k.starts_with(prefix))
            .map(|(k, _)| k.clone())
            .collect())
    }
}

/// A directory on the local filesystem, each key a file.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
pub struct FilesystemStore {
    root: PathBuf,
}

#[cfg(not(target_arch = "wasm32"))]
impl FilesystemStore {
    pub fn new(root: impl AsRef<Path>) -> FilesystemStore {
        FilesystemStore {
            root: root.as_ref().to_path_buf(),
        }
    }

    fn path(&self, key: &str) -> PathBuf {
        if key.is_empty() {
            self.root.clone()
        } else {
            self.root.join(key)
        }
    }

    /// Not there: no such file, or a directory where the file would be.
    fn absent(e: &std::io::Error) -> bool {
        use std::io::ErrorKind::*;
        matches!(e.kind(), NotFound | IsADirectory | NotADirectory)
    }

    fn walk(&self, dir: &Path, key: &str, out: &mut Vec<String>) -> Result<()> {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(e) if Self::absent(&e) => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        for e in entries {
            let e = e?;
            let name = e.file_name().to_string_lossy().to_string();
            let k = format!("{key}{name}");
            if e.file_type()?.is_dir() {
                self.walk(&e.path(), &format!("{k}/"), out)?;
            } else {
                out.push(k);
            }
        }
        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Storage for FilesystemStore {
    fn get(&self, key: &str) -> Result<Option<Bytes>> {
        match std::fs::read(self.path(key)) {
            Ok(v) => Ok(Some(Bytes::from(v))),
            Err(e) if Self::absent(&e) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
    fn get_ranges(&self, key: &str, ranges: &[ByteRange]) -> Result<Option<Vec<Bytes>>> {
        use std::io::{Seek, SeekFrom};
        let mut f = match std::fs::File::open(self.path(key)) {
            Ok(f) if f.metadata()?.is_file() => f,
            Ok(_) => return Ok(None),
            Err(e) if Self::absent(&e) => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        let len = f.metadata()?.len();
        let mut out = Vec::with_capacity(ranges.len());
        for r in ranges {
            let r = r.within(len)?;
            f.seek(SeekFrom::Start(r.start))?;
            let mut buf = vec![0; (r.end - r.start) as usize];
            f.read_exact(&mut buf)?;
            out.push(Bytes::from(buf));
        }
        Ok(Some(out))
    }
    fn supports_ranges(&self) -> bool {
        true
    }
    fn size(&self, key: &str) -> Result<Option<u64>> {
        match std::fs::metadata(self.path(key)) {
            Ok(m) if m.is_file() => Ok(Some(m.len())),
            Ok(_) => Ok(None),
            Err(e) if Self::absent(&e) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
    fn set(&self, key: &str, value: Bytes) -> Result<()> {
        let path = self.path(key);
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        Ok(std::fs::write(path, &value)?)
    }
    fn erase(&self, key: &str) -> Result<()> {
        match std::fs::remove_file(self.path(key)) {
            Err(e) if !Self::absent(&e) => Err(e.into()),
            _ => Ok(()),
        }
    }
    fn erase_prefix(&self, prefix: &str) -> Result<()> {
        match std::fs::remove_dir_all(self.path(prefix.trim_end_matches('/'))) {
            Err(e) if !Self::absent(&e) => Err(e.into()),
            _ => Ok(()),
        }
    }
    fn list_prefix(&self, prefix: &str) -> Result<Vec<String>> {
        // the directory the prefix names, or the one it ends in
        let (dir, _) = prefix.rsplit_once('/').unwrap_or(("", prefix));
        let base = if dir.is_empty() {
            String::new()
        } else {
            format!("{dir}/")
        };
        let mut out = vec![];
        self.walk(&self.path(dir), &base, &mut out)?;
        out.retain(|k| k.starts_with(prefix));
        out.sort();
        Ok(out)
    }
    fn list_dir(&self, prefix: &str) -> Result<(Vec<String>, Vec<String>)> {
        let entries = match std::fs::read_dir(self.path(prefix.trim_end_matches('/'))) {
            Ok(e) => e,
            Err(e) if Self::absent(&e) => return Ok((vec![], vec![])),
            Err(e) => return Err(e.into()),
        };
        let (mut keys, mut prefixes) = (vec![], vec![]);
        for e in entries {
            let e = e?;
            let name = e.file_name().to_string_lossy().to_string();
            if e.file_type()?.is_dir() {
                prefixes.push(format!("{prefix}{name}/"));
            } else {
                keys.push(format!("{prefix}{name}"));
            }
        }
        keys.sort();
        prefixes.sort();
        Ok((keys, prefixes))
    }
}

/// What kind of backend a store is (used e.g. for path joining semantics).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreKind {
    /// Local filesystem directory.
    Local,
    /// HTTP(S) server (read-only).
    Http,
    /// Object store such as S3 (path joining by concatenation).
    ObjectStore,
    /// In-memory store.
    Memory,
    /// Zip archive (read-only).
    Zip,
    /// Something else, passed in by the user.
    Other,
}

/// A store together with metadata about where it came from.
#[derive(Clone)]
pub struct Store {
    inner: DynStore,
    kind: StoreKind,
    /// Root path / URL of the store, if it has one.
    root: Option<String>,
    read_only: bool,
}

impl std::fmt::Debug for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Store")
            .field("kind", &self.kind)
            .field("root", &self.root)
            .field("read_only", &self.read_only)
            .finish()
    }
}

impl Store {
    /// Wrap an arbitrary store.
    pub fn from_dyn(inner: DynStore, kind: StoreKind, root: Option<String>) -> Store {
        Store {
            inner,
            kind,
            root,
            read_only: false,
        }
    }

    /// This store, read only or not.
    pub fn read_only(mut self, read_only: bool) -> Store {
        self.read_only = read_only;
        self
    }

    /// A new, empty in-memory store.
    pub fn memory() -> Store {
        Store::from_dyn(Arc::new(MemoryStore::new()), StoreKind::Memory, None)
    }

    /// A local filesystem store rooted at `path` (no assumption that it exists).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn local(path: impl AsRef<Path>, read_only: bool) -> Result<Store> {
        let path = path.as_ref();
        Ok(Store {
            inner: Arc::new(FilesystemStore::new(path)),
            kind: StoreKind::Local,
            root: Some(path.to_string_lossy().to_string()),
            read_only,
        })
    }

    /// Local filesystem stores are not available on wasm.
    #[cfg(target_arch = "wasm32")]
    pub fn local(path: impl AsRef<Path>, _read_only: bool) -> Result<Store> {
        Err(Error::NotImplemented(format!(
            "filesystem stores are not available on wasm: {}",
            path.as_ref().display()
        )))
    }

    /// A read-only zip-archive store.
    pub fn zip(path: impl AsRef<Path>) -> Result<Store> {
        let path = path.as_ref();
        let zs = ZipStore::open(path)?;
        Ok(Store {
            inner: Arc::new(zs),
            kind: StoreKind::Zip,
            root: Some(path.to_string_lossy().to_string()),
            read_only: true,
        })
    }

    /// A read-only store for a zip archive held in memory (e.g. fetched by the
    /// host application on wasm). `name` is reported as the store's root.
    pub fn zip_bytes(bytes: Vec<u8>, name: &str) -> Result<Store> {
        Ok(Store {
            inner: Arc::new(ZipStore::from_bytes(bytes)?),
            kind: StoreKind::Zip,
            root: Some(name.to_string()),
            read_only: true,
        })
    }

    /// A read-only store for a zip archive on an HTTP(S) server. Only the
    /// parts of the archive that are needed are fetched (range requests);
    /// servers without range support are read in full.
    #[cfg(all(feature = "http", not(target_arch = "wasm32")))]
    pub fn zip_url(url: &str) -> Result<Store> {
        let reader = crate::net::http::RangeReader::open(url)?;
        Ok(Store {
            inner: Arc::new(ZipStore::from_source(Box::new(reader))?),
            kind: StoreKind::Zip,
            root: Some(url.to_string()),
            read_only: true,
        })
    }

    /// A read-only HTTP(S) store.
    #[cfg(all(feature = "http", not(target_arch = "wasm32")))]
    pub fn http(url: &str) -> Result<Store> {
        Ok(Store {
            inner: Arc::new(MetadataCache::new(crate::net::http::HttpStore::new(url))),
            kind: StoreKind::Http,
            root: Some(url.to_string()),
            read_only: true,
        })
    }
    /// An S3 store, e.g. `s3://bucket/prefix`. Credentials are taken from the
    /// environment; if none are found, requests are unsigned (anonymous).
    #[cfg(all(feature = "cloud", not(target_arch = "wasm32")))]
    pub fn s3(url: &str, read_only: bool) -> Result<Store> {
        crate::net::cloud::open(url, read_only)
    }

    /// A cloud object store: `s3://bucket/prefix`, `gs://bucket/prefix`,
    /// `az://container/prefix` or `abfs://container@account.dfs.core.windows.net/prefix`.
    /// Configuration and credentials come from the usual environment
    /// variables (`AWS_*`, `GOOGLE_*`, `AZURE_*`); without credentials,
    /// requests are unsigned so public buckets can be read.
    #[cfg(all(feature = "cloud", not(target_arch = "wasm32")))]
    pub fn cloud(url: &str, read_only: bool) -> Result<Store> {
        crate::net::cloud::open(url, read_only)
    }

    /// Open a store from a path or URL: `http(s)://` (a zip archive if the
    /// URL ends in `.zip`), `s3://`, `gs://`, `az://` / `abfs://`, a local
    /// `*.zip` file or a local directory.
    pub fn from_url(path: &str, read_only: bool) -> Result<Store> {
        if path.starts_with("http://") || path.starts_with("https://") {
            #[cfg(all(feature = "http", not(target_arch = "wasm32")))]
            {
                if is_zip_url(path) {
                    return Store::zip_url(path);
                }
                return Store::http(path);
            }
            #[cfg(not(all(feature = "http", not(target_arch = "wasm32"))))]
            {
                return Err(Error::NotImplemented(
                    "http support requires the `http` feature on a native target".into(),
                ));
            }
        }
        if let Some(scheme) = cloud_scheme(path) {
            #[cfg(all(feature = "cloud", not(target_arch = "wasm32")))]
            {
                let _ = scheme;
                return crate::net::cloud::open(path, read_only);
            }
            #[cfg(not(all(feature = "cloud", not(target_arch = "wasm32"))))]
            {
                return Err(Error::NotImplemented(format!(
                    "{scheme}:// support requires the `cloud` feature on a native target"
                )));
            }
        }
        let p = Path::new(path);
        if p.is_file() && path.to_lowercase().ends_with(".zip") {
            return Store::zip(p);
        }
        Store::local(p, read_only)
    }

    /// zarr-python's `str()` of the equivalent store, e.g. `file:///data/image.zarr`.
    pub fn py_str(&self) -> String {
        let root = self.root.clone().unwrap_or_default();
        match self.kind {
            StoreKind::Local => format!("file://{root}"),
            StoreKind::Memory => "memory://".into(),
            StoreKind::Zip => format!("zip://{root}"),
            _ => root,
        }
    }

    /// zarr-python's `repr()` of the equivalent store, e.g.
    /// `LocalStore('file:///data/image.zarr')`.
    pub fn py_repr(&self) -> String {
        let root = self.root.clone().unwrap_or_default();
        match self.kind {
            StoreKind::Local => format!("LocalStore('file://{root}')"),
            StoreKind::Memory => "MemoryStore('memory://')".into(),
            StoreKind::Zip => format!("ZipStore('zip://{root}')"),
            StoreKind::Http | StoreKind::ObjectStore | StoreKind::Other => {
                format!("FsspecStore({root})")
            }
        }
    }

    pub fn inner(&self) -> &DynStore {
        &self.inner
    }

    pub fn kind(&self) -> StoreKind {
        self.kind
    }

    pub fn root(&self) -> Option<&str> {
        self.root.as_deref()
    }

    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    /// Identity comparison (Python `store is other`).
    pub fn same(&self, other: &Store) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// The local filesystem root, if this is a local store.
    pub fn local_root(&self) -> Option<PathBuf> {
        if self.kind == StoreKind::Local {
            self.root.as_ref().map(PathBuf::from)
        } else {
            None
        }
    }

    fn key(k: &str) -> &str {
        k.trim_start_matches('/')
    }

    /// Read raw bytes at a key.
    pub fn get(&self, key: &str) -> Result<Option<Bytes>> {
        self.inner.get(Self::key(key))
    }

    /// Read parts of the value at a key.
    pub fn get_ranges(&self, key: &str, ranges: &[ByteRange]) -> Result<Option<Vec<Bytes>>> {
        self.inner.get_ranges(Self::key(key), ranges)
    }

    /// Whether [`get_ranges`](Self::get_ranges) reads only the parts.
    pub fn supports_ranges(&self) -> bool {
        self.inner.supports_ranges()
    }

    /// Whether a key exists.
    pub fn exists(&self, key: &str) -> bool {
        matches!(self.inner.size(Self::key(key)), Ok(Some(_)))
    }

    /// Write raw bytes at a key.
    pub fn set(&self, key: &str, value: Vec<u8>) -> Result<()> {
        if self.read_only {
            return Err(Error::value(format!(
                "store is read-only: cannot write {key}"
            )));
        }
        self.inner.set(Self::key(key), Bytes::from(value))
    }

    /// Delete a key.
    pub fn erase(&self, key: &str) -> Result<()> {
        if self.read_only {
            return Err(Error::value("store is read-only"));
        }
        self.inner.erase(Self::key(key))
    }

    /// Delete everything under a prefix (`""` deletes everything).
    pub fn erase_prefix(&self, prefix: &str) -> Result<()> {
        if self.read_only {
            return Err(Error::value("store is read-only"));
        }
        self.inner.erase_prefix(&normalize_prefix(prefix))
    }

    /// Read and parse JSON at a key.
    pub fn get_json(&self, key: &str) -> Result<Option<serde_json::Value>> {
        match self.get(key)? {
            None => Ok(None),
            Some(b) => Ok(Some(serde_json::from_slice(&b)?)),
        }
    }

    /// Serialize and write JSON at a key.
    pub fn set_json(&self, key: &str, value: &serde_json::Value) -> Result<()> {
        // byte-identical to zarr-python: json.dumps(value, indent=2)
        self.set(key, crate::pyjson::dumps(value, Some(2)).into_bytes())
    }

    /// List the direct children (keys and sub-prefixes) of a prefix.
    pub fn list_dir(&self, prefix: &str) -> Result<(Vec<String>, Vec<String>)> {
        let prefix = normalize_prefix(prefix);
        let (keys, prefixes) = self.inner.list_dir(&prefix)?;
        let strip = |s: &str| -> String {
            s.strip_prefix(prefix.as_str())
                .unwrap_or(s)
                .trim_end_matches('/')
                .to_string()
        };
        Ok((
            keys.iter().map(|k| strip(k)).collect(),
            prefixes.iter().map(|k| strip(k)).collect(),
        ))
    }
}

fn normalize_prefix(prefix: &str) -> String {
    let p = prefix.trim_matches('/');
    if p.is_empty() {
        String::new()
    } else {
        format!("{p}/")
    }
}

/// Caches reads (including misses) of metadata documents (`zarr.json`,
/// `.zgroup`, `.zattrs`, `.zarray`), which are looked up repeatedly while
/// walking a hierarchy; each lookup is a round trip on remote stores.
pub struct MetadataCache<T> {
    inner: T,
    cache: Mutex<std::collections::HashMap<String, Option<Bytes>>>,
}

impl<T> MetadataCache<T> {
    pub fn new(inner: T) -> Self {
        MetadataCache {
            inner,
            cache: Mutex::new(Default::default()),
        }
    }
}

impl<T: Storage> MetadataCache<T> {
    /// Fetch all metadata documents of the node containing `key` concurrently.
    fn prefetch(&self, key: &str) -> Result<()> {
        let dir = match key.rfind('/') {
            Some(i) => &key[..=i],
            None => "",
        };
        let keys: Vec<String> = crate::zarr::NODE_DOCUMENTS
            .iter()
            .map(|n| format!("{dir}{n}"))
            .filter(|k| !self.cache.lock().unwrap().contains_key(k))
            .collect();
        #[cfg(target_arch = "wasm32")]
        let results: Vec<(String, Result<Option<Bytes>>)> = keys
            .iter()
            .map(|k| (k.clone(), self.inner.get(k)))
            .collect();
        #[cfg(not(target_arch = "wasm32"))]
        let results: Vec<(String, Result<Option<Bytes>>)> = std::thread::scope(|scope| {
            let handles: Vec<_> = keys
                .iter()
                .map(|k| scope.spawn(move || (k.clone(), self.inner.get(k))))
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut cache = self.cache.lock().unwrap();
        for (k, r) in results {
            cache.insert(k, r?);
        }
        Ok(())
    }
}

impl<T: Storage> Storage for MetadataCache<T> {
    fn get(&self, key: &str) -> Result<Option<Bytes>> {
        if !crate::zarr::is_metadata(key) {
            return self.inner.get(key);
        }
        if let Some(v) = self.cache.lock().unwrap().get(key) {
            return Ok(v.clone());
        }
        self.prefetch(key)?;
        if let Some(v) = self.cache.lock().unwrap().get(key) {
            return Ok(v.clone());
        }
        let v = self.inner.get(key)?;
        self.cache
            .lock()
            .unwrap()
            .insert(key.to_string(), v.clone());
        Ok(v)
    }
    fn get_ranges(&self, key: &str, ranges: &[ByteRange]) -> Result<Option<Vec<Bytes>>> {
        self.inner.get_ranges(key, ranges)
    }
    fn supports_ranges(&self) -> bool {
        self.inner.supports_ranges()
    }
    fn size(&self, key: &str) -> Result<Option<u64>> {
        if crate::zarr::is_metadata(key) {
            return Ok(self.get(key)?.map(|b| b.len() as u64));
        }
        self.inner.size(key)
    }
}

/// A seekable byte source for a zip archive (a file, memory, or a remote
/// file read with HTTP range requests).
///
/// A clone reads on its own, from its own position: each read of a member has
/// one, so reads of a remote archive overlap rather than queue.
pub trait ZipSource: std::io::Read + std::io::Seek + Send {
    fn clone_source(&self) -> Box<dyn ZipSource>;
}
impl<T: std::io::Read + std::io::Seek + Send + Clone + 'static> ZipSource for T {
    fn clone_source(&self) -> Box<dyn ZipSource> {
        Box::new(self.clone())
    }
}
impl Clone for Box<dyn ZipSource> {
    fn clone(&self) -> Self {
        self.as_ref().clone_source()
    }
}

/// A file as a [`ZipSource`]: each clone opens it for itself, when first read.
struct FileSource {
    path: PathBuf,
    file: Option<std::io::BufReader<std::fs::File>>,
    pos: u64,
}

impl Clone for FileSource {
    fn clone(&self) -> Self {
        FileSource {
            path: self.path.clone(),
            file: None,
            pos: self.pos,
        }
    }
}

impl FileSource {
    fn file(&mut self) -> std::io::Result<&mut std::io::BufReader<std::fs::File>> {
        use std::io::Seek;
        if self.file.is_none() {
            let mut f = std::io::BufReader::new(std::fs::File::open(&self.path)?);
            f.seek(std::io::SeekFrom::Start(self.pos))?;
            self.file = Some(f);
        }
        Ok(self.file.as_mut().unwrap())
    }
}

impl std::io::Read for FileSource {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.file()?.read(buf)?;
        self.pos += n as u64;
        Ok(n)
    }
}

impl std::io::Seek for FileSource {
    fn seek(&mut self, to: std::io::SeekFrom) -> std::io::Result<u64> {
        self.pos = self.file()?.seek(to)?;
        Ok(self.pos)
    }
}

/// A read-only store backed by a zip archive.
pub struct ZipStore {
    archive: Mutex<zip::ZipArchive<Box<dyn ZipSource>>>,
    names: Vec<String>,
}

impl ZipStore {
    pub fn open(path: &Path) -> Result<ZipStore> {
        std::fs::metadata(path)?;
        ZipStore::from_source(Box::new(FileSource {
            path: path.to_path_buf(),
            file: None,
            pos: 0,
        }))
    }

    /// A zip archive held in memory.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<ZipStore> {
        // `Bytes`: every read clones its source, and this clones no data.
        ZipStore::from_source(Box::new(std::io::Cursor::new(Bytes::from(bytes))))
    }

    pub fn from_source(source: Box<dyn ZipSource>) -> Result<ZipStore> {
        let archive = zip::ZipArchive::new(source).map_err(|e| Error::Storage(e.to_string()))?;
        let names = archive
            .file_names()
            .map(|s| {
                let name = s.map_err(|e| Error::Storage(e.to_string()))?;
                Ok(name.trim_start_matches('/').to_string())
            })
            .collect::<Result<_>>()?;
        Ok(ZipStore {
            archive: Mutex::new(archive),
            names,
        })
    }

    /// An archive to read through, held by nobody else: the index is shared,
    /// the source is its own, and the lock lasts only as long as the clone.
    fn archive(&self) -> zip::ZipArchive<Box<dyn ZipSource>> {
        self.archive.lock().unwrap().clone()
    }

    fn read(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let mut ar = self.archive();
        let idx = match ar.index_for_name(key) {
            Some(i) => i,
            None => match ar.index_for_name(&format!("/{key}")) {
                Some(i) => i,
                None => return Ok(None),
            },
        };
        let mut file = ar
            .by_index(idx)
            .map_err(|e| Error::Storage(e.to_string()))?;
        let mut buf = Vec::with_capacity(file.size() as usize);
        file.read_to_end(&mut buf)?;
        Ok(Some(buf))
    }
}

impl Storage for ZipStore {
    fn get(&self, key: &str) -> Result<Option<Bytes>> {
        Ok(self.read(key)?.map(Bytes::from))
    }
    fn size(&self, key: &str) -> Result<Option<u64>> {
        let mut ar = self.archive();
        match ar.index_for_name(key) {
            Some(i) => Ok(Some(
                ar.by_index(i)
                    .map_err(|e| Error::Storage(e.to_string()))?
                    .size(),
            )),
            None => Ok(None),
        }
    }
    fn list_prefix(&self, prefix: &str) -> Result<Vec<String>> {
        let mut keys: Vec<String> = self
            .names
            .iter()
            .filter(|n| !n.ends_with('/') && n.starts_with(prefix))
            .cloned()
            .collect();
        keys.sort();
        Ok(keys)
    }
}

/// True for `http(s)` URLs of a zip archive (`.../image.ome.zarr.zip`).
pub fn is_zip_url(url: &str) -> bool {
    let path = url.split(['?', '#']).next().unwrap_or(url);
    path.trim_end_matches('/').to_lowercase().ends_with(".zip")
}

/// The object-store scheme of a URL, if any.
pub fn cloud_scheme(url: &str) -> Option<&'static str> {
    ["s3", "gs", "gcs", "az", "azure", "abfs", "abfss"]
        .into_iter()
        .find(|s| url.strip_prefix(s).is_some_and(|r| r.starts_with("://")))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A source that takes a moment per read, counting how many overlap.
    #[derive(Clone)]
    struct Slow {
        inner: std::io::Cursor<Bytes>,
        now: Arc<std::sync::atomic::AtomicUsize>,
        peak: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Read for Slow {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            use std::sync::atomic::Ordering::SeqCst;
            let n = self.now.fetch_add(1, SeqCst) + 1;
            self.peak.fetch_max(n, SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(2));
            let r = self.inner.read(buf);
            self.now.fetch_sub(1, SeqCst);
            r
        }
    }

    impl std::io::Seek for Slow {
        fn seek(&mut self, to: std::io::SeekFrom) -> std::io::Result<u64> {
            self.inner.seek(to)
        }
    }

    fn zipped(members: &[(&str, &[u8])]) -> Vec<u8> {
        use std::io::Write;
        let mut w = zip::ZipWriter::new(std::io::Cursor::new(vec![]));
        for (name, data) in members {
            let opts = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            w.start_file(*name, opts).unwrap();
            w.write_all(data).unwrap();
        }
        w.finish().unwrap().into_inner()
    }

    #[test]
    fn reads_of_a_zip_overlap() {
        let bytes = Bytes::from(zipped(&[("a", b"first"), ("b", b"second")]));
        let peak = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let source = Slow {
            inner: std::io::Cursor::new(bytes),
            now: Default::default(),
            peak: peak.clone(),
        };
        let store = Arc::new(ZipStore::from_source(Box::new(source)).unwrap());
        peak.store(0, std::sync::atomic::Ordering::SeqCst);
        let threads: Vec<_> = (0..8)
            .map(|i| {
                let store = store.clone();
                std::thread::spawn(move || {
                    let (key, want): (&str, &[u8]) = if i % 2 == 0 {
                        ("a", b"first")
                    } else {
                        ("b", b"second")
                    };
                    assert_eq!(store.read(key).unwrap().unwrap(), want);
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert!(
            peak.load(std::sync::atomic::Ordering::SeqCst) > 1,
            "the reads queued"
        );
    }

    #[test]
    fn a_zip_on_disk_reads_from_every_clone() {
        let dir = std::env::temp_dir().join(format!("ome-zarr-zip-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.zip");
        std::fs::write(&path, zipped(&[("a", b"first"), ("b", b"second")])).unwrap();
        let store = ZipStore::open(&path).unwrap();
        assert_eq!(store.read("b").unwrap().unwrap(), b"second");
        assert_eq!(store.read("a").unwrap().unwrap(), b"first");
        assert_eq!(store.read("missing").unwrap(), None);
        assert!(ZipStore::open(&dir.join("absent.zip")).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn memory_roundtrip() {
        let s = Store::memory();
        s.set("a/b.json", b"{\"x\": 1}".to_vec()).unwrap();
        assert!(s.exists("a/b.json"));
        assert!(!s.exists("a/c.json"));
        let v = s.get_json("a/b.json").unwrap().unwrap();
        assert_eq!(v["x"], 1);
        let (keys, prefixes) = s.list_dir("").unwrap();
        assert!(keys.is_empty());
        assert_eq!(prefixes, vec!["a".to_string()]);
        let (keys, _) = s.list_dir("a").unwrap();
        assert_eq!(keys, vec!["b.json".to_string()]);
    }
}
