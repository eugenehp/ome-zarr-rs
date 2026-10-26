//! Storage backends (filesystem, memory, http, zip, s3) behind one dynamic store type.
//!
//! The equivalent of `zarr.storage.LocalStore`, `MemoryStore`, `FsspecStore` and
//! `ZipStore` in zarr-python.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use zarrs_storage::byte_range::ByteRangeIterator;
use zarrs_storage::{
    ListableStorageTraits, MaybeBytes, MaybeBytesIterator, OffsetBytesIterator,
    ReadableStorageTraits, ReadableWritableListableStorageTraits, StorageError, StoreKey,
    StoreKeys, StoreKeysPrefixes, StorePrefix, StorePrefixes, WritableStorageTraits,
};

use crate::error::{Error, Result};

pub use zarrs_storage::store::MemoryStore;

/// A shareable, dynamically typed zarr store.
pub type DynStore = Arc<dyn ReadableWritableListableStorageTraits>;

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

    /// A new, empty in-memory store.
    pub fn memory() -> Store {
        Store::from_dyn(Arc::new(MemoryStore::new()), StoreKind::Memory, None)
    }

    /// A local filesystem store rooted at `path` (no assumption that it exists).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn local(path: impl AsRef<Path>, read_only: bool) -> Result<Store> {
        let path = path.as_ref();
        let fs = zarrs::filesystem::FilesystemStore::new(path)
            .map_err(|e| Error::Storage(e.to_string()))?;
        Ok(Store {
            inner: Arc::new(fs),
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
            inner: Arc::new(ReadOnly(zs)),
            kind: StoreKind::Zip,
            root: Some(path.to_string_lossy().to_string()),
            read_only: true,
        })
    }

    /// A read-only store for a zip archive held in memory (e.g. fetched by the
    /// host application on wasm). `name` is reported as the store's root.
    pub fn zip_bytes(bytes: Vec<u8>, name: &str) -> Result<Store> {
        Ok(Store {
            inner: Arc::new(ReadOnly(ZipStore::from_bytes(bytes)?)),
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
        let reader = http_range::RangeReader::open(url)?;
        Ok(Store {
            inner: Arc::new(ReadOnly(ZipStore::from_source(Box::new(reader))?)),
            kind: StoreKind::Zip,
            root: Some(url.to_string()),
            read_only: true,
        })
    }

    /// A read-only HTTP(S) store.
    #[cfg(all(feature = "http", not(target_arch = "wasm32")))]
    pub fn http(url: &str) -> Result<Store> {
        let hs = zarrs_http::HTTPStore::new(url.trim_end_matches('/'))
            .map_err(|e| Error::Storage(e.to_string()))?;
        Ok(Store {
            inner: Arc::new(ReadOnly(UnlistableReadable(MetadataCache::new(hs)))),
            kind: StoreKind::Http,
            root: Some(url.to_string()),
            read_only: true,
        })
    }

    /// An S3 store, e.g. `s3://bucket/prefix`. Credentials are taken from the
    /// environment; if none are found, requests are unsigned (anonymous).
    #[cfg(all(feature = "cloud", not(target_arch = "wasm32")))]
    pub fn s3(url: &str, read_only: bool) -> Result<Store> {
        cloud::open(url, read_only)
    }

    /// A cloud object store: `s3://bucket/prefix`, `gs://bucket/prefix`,
    /// `az://container/prefix` or `abfs://container@account.dfs.core.windows.net/prefix`.
    /// Configuration and credentials come from the usual environment
    /// variables (`AWS_*`, `GOOGLE_*`, `AZURE_*`); without credentials,
    /// requests are unsigned so public buckets can be read.
    #[cfg(all(feature = "cloud", not(target_arch = "wasm32")))]
    pub fn cloud(url: &str, read_only: bool) -> Result<Store> {
        cloud::open(url, read_only)
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
                return cloud::open(path, read_only);
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

    fn key(k: &str) -> Result<StoreKey> {
        StoreKey::new(k.trim_start_matches('/')).map_err(|e| Error::Storage(e.to_string()))
    }

    /// Read raw bytes at a key.
    pub fn get(&self, key: &str) -> Result<Option<Bytes>> {
        Ok(self.inner.get(&Self::key(key)?)?)
    }

    /// Whether a key exists.
    pub fn exists(&self, key: &str) -> bool {
        matches!(
            self.inner
                .size_key(&Self::key(key).unwrap_or_else(|_| StoreKey::new("x").unwrap())),
            Ok(Some(_))
        ) && Self::key(key).is_ok()
    }

    /// Write raw bytes at a key.
    pub fn set(&self, key: &str, value: Vec<u8>) -> Result<()> {
        if self.read_only {
            return Err(Error::value(format!(
                "store is read-only: cannot write {key}"
            )));
        }
        Ok(self.inner.set(&Self::key(key)?, Bytes::from(value))?)
    }

    /// Delete a key.
    pub fn erase(&self, key: &str) -> Result<()> {
        if self.read_only {
            return Err(Error::value("store is read-only"));
        }
        Ok(self.inner.erase(&Self::key(key)?)?)
    }

    /// Delete everything under a prefix (`""` deletes everything).
    pub fn erase_prefix(&self, prefix: &str) -> Result<()> {
        if self.read_only {
            return Err(Error::value("store is read-only"));
        }
        let prefix = normalize_prefix(prefix);
        let p = StorePrefix::new(prefix).map_err(|e| Error::Storage(e.to_string()))?;
        Ok(self.inner.erase_prefix(&p)?)
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
        let p = StorePrefix::new(prefix.clone()).map_err(|e| Error::Storage(e.to_string()))?;
        let kp = self.inner.list_dir(&p)?;
        let strip = |s: &str| -> String {
            s.strip_prefix(prefix.as_str())
                .unwrap_or(s)
                .trim_end_matches('/')
                .to_string()
        };
        let keys = kp.keys().iter().map(|k| strip(k.as_str())).collect();
        let prefixes = kp.prefixes().iter().map(|k| strip(k.as_str())).collect();
        Ok((keys, prefixes))
    }

    /// The store used by zarrs for array I/O.
    pub fn as_readable(&self) -> Arc<dyn ReadableStorageTraits> {
        self.inner.clone().readable_writable().readable()
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

/// Adapter making a readable+listable store usable where a writable store is
/// expected: all writes fail.
pub struct ReadOnly<T>(pub T);

impl<T: ReadableStorageTraits> ReadableStorageTraits for ReadOnly<T> {
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        self.0.get(key)
    }
    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        self.0.get_partial_many(key, byte_ranges)
    }
    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.0.size_key(key)
    }
    fn supports_get_partial(&self) -> bool {
        self.0.supports_get_partial()
    }
}

impl<T: ListableStorageTraits> ListableStorageTraits for ReadOnly<T> {
    fn list(&self) -> Result<StoreKeys, StorageError> {
        self.0.list()
    }
    fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        self.0.list_prefix(prefix)
    }
    fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        self.0.list_dir(prefix)
    }
    fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        self.0.size_prefix(prefix)
    }
}

impl<T: crate::par::MaybeSendSync> WritableStorageTraits for ReadOnly<T> {
    fn set(&self, _key: &StoreKey, _value: Bytes) -> Result<(), StorageError> {
        Err(StorageError::ReadOnly)
    }
    fn set_partial_many(
        &self,
        _key: &StoreKey,
        _offset_values: OffsetBytesIterator,
    ) -> Result<(), StorageError> {
        Err(StorageError::ReadOnly)
    }
    fn erase(&self, _key: &StoreKey) -> Result<(), StorageError> {
        Err(StorageError::ReadOnly)
    }
    fn erase_prefix(&self, _prefix: &StorePrefix) -> Result<(), StorageError> {
        Err(StorageError::ReadOnly)
    }
    fn supports_set_partial(&self) -> bool {
        false
    }
}

/// Adapter for readable stores that cannot list (e.g. plain HTTP).
pub struct UnlistableReadable<T>(pub T);

impl<T: ReadableStorageTraits> ReadableStorageTraits for UnlistableReadable<T> {
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        self.0.get(key)
    }
    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        self.0.get_partial_many(key, byte_ranges)
    }
    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        // Some servers do not support HEAD; fall back to a GET.
        match self.0.size_key(key) {
            Ok(Some(s)) => Ok(Some(s)),
            _ => Ok(self.0.get(key)?.map(|b| b.len() as u64)),
        }
    }
    fn supports_get_partial(&self) -> bool {
        self.0.supports_get_partial()
    }
}

impl<T: crate::par::MaybeSendSync> ListableStorageTraits for UnlistableReadable<T> {
    fn list(&self) -> Result<StoreKeys, StorageError> {
        Err(StorageError::Unsupported(
            "listing is not supported by this store".into(),
        ))
    }
    fn list_prefix(&self, _prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        Err(StorageError::Unsupported(
            "listing is not supported by this store".into(),
        ))
    }
    fn list_dir(&self, _prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        Err(StorageError::Unsupported(
            "listing is not supported by this store".into(),
        ))
    }
    fn size_prefix(&self, _prefix: &StorePrefix) -> Result<u64, StorageError> {
        Err(StorageError::Unsupported(
            "listing is not supported by this store".into(),
        ))
    }
}

/// Caches reads (including misses) of metadata documents (`zarr.json`,
/// `.zgroup`, `.zattrs`, `.zarray`), which are looked up repeatedly while
/// walking a hierarchy; each lookup is a round trip on remote stores.
pub struct MetadataCache<T> {
    inner: T,
    cache: Mutex<std::collections::HashMap<String, MaybeBytes>>,
}

impl<T> MetadataCache<T> {
    pub fn new(inner: T) -> Self {
        MetadataCache {
            inner,
            cache: Mutex::new(Default::default()),
        }
    }

    fn is_metadata(key: &StoreKey) -> bool {
        let k = key.as_str();
        let name = k.rsplit('/').next().unwrap_or(k);
        matches!(name, "zarr.json" | ".zgroup" | ".zattrs" | ".zarray")
    }
}

impl<T: ReadableStorageTraits> MetadataCache<T> {
    /// Fetch all metadata documents of the node containing `key` concurrently.
    fn prefetch(&self, key: &StoreKey) -> Result<(), StorageError> {
        let k = key.as_str();
        let dir = match k.rfind('/') {
            Some(i) => &k[..=i],
            None => "",
        };
        let keys: Vec<String> = ["zarr.json", ".zgroup", ".zattrs", ".zarray"]
            .iter()
            .map(|n| format!("{dir}{n}"))
            .filter(|k| !self.cache.lock().unwrap().contains_key(k))
            .collect();
        #[cfg(target_arch = "wasm32")]
        let results: Vec<(String, Result<MaybeBytes, StorageError>)> = keys
            .iter()
            .map(|k| {
                let r = StoreKey::new(k.clone())
                    .map_err(|e| StorageError::Other(e.to_string()))
                    .and_then(|sk| self.inner.get(&sk));
                (k.clone(), r)
            })
            .collect();
        #[cfg(not(target_arch = "wasm32"))]
        let results: Vec<(String, Result<MaybeBytes, StorageError>)> =
            std::thread::scope(|scope| {
                let handles: Vec<_> = keys
                    .iter()
                    .map(|k| {
                        let k = k.clone();
                        scope.spawn(move || {
                            let r = StoreKey::new(k.clone())
                                .map_err(|e| StorageError::Other(e.to_string()))
                                .and_then(|sk| self.inner.get(&sk));
                            (k, r)
                        })
                    })
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

impl<T: ReadableStorageTraits> ReadableStorageTraits for MetadataCache<T> {
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        if !Self::is_metadata(key) {
            return self.inner.get(key);
        }
        if let Some(v) = self.cache.lock().unwrap().get(key.as_str()) {
            return Ok(v.clone());
        }
        self.prefetch(key)?;
        if let Some(v) = self.cache.lock().unwrap().get(key.as_str()) {
            return Ok(v.clone());
        }
        let v = self.inner.get(key)?;
        self.cache
            .lock()
            .unwrap()
            .insert(key.as_str().to_string(), v.clone());
        Ok(v)
    }
    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        self.inner.get_partial_many(key, byte_ranges)
    }
    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        if Self::is_metadata(key) {
            return Ok(self.get(key)?.map(|b| b.len() as u64));
        }
        self.inner.size_key(key)
    }
    fn supports_get_partial(&self) -> bool {
        self.inner.supports_get_partial()
    }
}

/// A seekable byte source for a zip archive (a file, memory, or a remote
/// file read with HTTP range requests).
pub trait ZipSource: std::io::Read + std::io::Seek + Send {}
impl<T: std::io::Read + std::io::Seek + Send> ZipSource for T {}

/// A read-only store backed by a zip archive.
pub struct ZipStore {
    archive: Mutex<zip::ZipArchive<Box<dyn ZipSource>>>,
    names: Vec<String>,
}

impl ZipStore {
    pub fn open(path: &Path) -> Result<ZipStore> {
        let f = std::fs::File::open(path)?;
        ZipStore::from_source(Box::new(std::io::BufReader::new(f)))
    }

    /// A zip archive held in memory.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<ZipStore> {
        ZipStore::from_source(Box::new(std::io::Cursor::new(bytes)))
    }

    pub fn from_source(source: Box<dyn ZipSource>) -> Result<ZipStore> {
        let archive = zip::ZipArchive::new(source).map_err(|e| Error::Storage(e.to_string()))?;
        let names = archive
            .file_names()
            .map(|s| s.trim_start_matches('/').to_string())
            .collect();
        Ok(ZipStore {
            archive: Mutex::new(archive),
            names,
        })
    }

    fn read(&self, key: &str) -> Result<Option<Vec<u8>>, StorageError> {
        let mut ar = self.archive.lock().unwrap();
        let idx = match ar.index_for_name(key) {
            Some(i) => i,
            None => match ar.index_for_name(&format!("/{key}")) {
                Some(i) => i,
                None => return Ok(None),
            },
        };
        let mut file = ar
            .by_index(idx)
            .map_err(|e| StorageError::Other(e.to_string()))?;
        let mut buf = Vec::with_capacity(file.size() as usize);
        file.read_to_end(&mut buf)?;
        Ok(Some(buf))
    }
}

impl ReadableStorageTraits for ZipStore {
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        Ok(self.read(key.as_str())?.map(Bytes::from))
    }
    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        let Some(data) = self.read(key.as_str())? else {
            return Ok(None);
        };
        let data = Bytes::from(data);
        let out = Box::new(byte_ranges.map(move |br| {
            let len = data.len() as u64;
            let start = br.start(len) as usize;
            let end = br.end(len) as usize;
            if end > data.len() {
                Err(StorageError::Other("byte range out of bounds".into()))
            } else {
                Ok(data.slice(start..end))
            }
        }));
        Ok(Some(out))
    }
    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        let mut ar = self.archive.lock().unwrap();
        match ar.index_for_name(key.as_str()) {
            Some(i) => Ok(Some(
                ar.by_index(i)
                    .map_err(|e| StorageError::Other(e.to_string()))?
                    .size(),
            )),
            None => Ok(None),
        }
    }
    fn supports_get_partial(&self) -> bool {
        true
    }
}

impl ListableStorageTraits for ZipStore {
    fn list(&self) -> Result<StoreKeys, StorageError> {
        Ok(self
            .names
            .iter()
            .filter(|n| !n.ends_with('/'))
            .filter_map(|n| StoreKey::new(n.clone()).ok())
            .collect())
    }
    fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        Ok(self
            .list()?
            .into_iter()
            .filter(|k| k.as_str().starts_with(prefix.as_str()))
            .collect())
    }
    fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        let mut keys = vec![];
        let mut prefixes: StorePrefixes = vec![];
        for k in self.list_prefix(prefix)? {
            let rest = &k.as_str()[prefix.as_str().len()..];
            match rest.split_once('/') {
                None => keys.push(k.clone()),
                Some((dir, _)) => {
                    let p = StorePrefix::new(format!("{}{}/", prefix.as_str(), dir))
                        .map_err(|e| StorageError::Other(e.to_string()))?;
                    if !prefixes.contains(&p) {
                        prefixes.push(p);
                    }
                }
            }
        }
        Ok(StoreKeysPrefixes::new(keys, prefixes))
    }
    fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        let mut total = 0;
        for k in self.list_prefix(prefix)? {
            total += self.size_key(&k)?.unwrap_or(0);
        }
        Ok(total)
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

#[cfg(all(feature = "cloud", not(target_arch = "wasm32")))]
mod cloud {
    use super::*;
    use object_store::ObjectStore;
    use zarrs_object_store::AsyncObjectStore;
    use zarrs_storage::storage_adapter::async_to_sync::{
        AsyncToSyncBlockOn, AsyncToSyncStorageAdapter,
    };

    struct TokioBlockOn(tokio::runtime::Runtime);

    impl AsyncToSyncBlockOn for TokioBlockOn {
        fn block_on<F: core::future::Future>(&self, future: F) -> F::Output {
            self.0.block_on(future)
        }
    }

    fn has_env(vars: &[&str]) -> bool {
        vars.iter().any(|v| std::env::var_os(v).is_some())
    }

    fn build(url: &str) -> Result<Arc<dyn ObjectStore>> {
        let err = |e: object_store::Error| Error::Storage(e.to_string());
        let scheme = cloud_scheme(url).unwrap_or("");
        Ok(match scheme {
            "s3" => {
                let mut b = object_store::aws::AmazonS3Builder::from_env().with_url(url);
                if !has_env(&[
                    "AWS_ACCESS_KEY_ID",
                    "AWS_PROFILE",
                    "AWS_WEB_IDENTITY_TOKEN_FILE",
                ]) {
                    b = b.with_skip_signature(true);
                }
                if let Ok(ep) = std::env::var("AWS_ENDPOINT_URL") {
                    b = b.with_endpoint(ep);
                }
                Arc::new(b.build().map_err(err)?)
            }
            "gs" | "gcs" => {
                let gs_url = format!("gs://{}", &url[scheme.len() + 3..]);
                let mut b =
                    object_store::gcp::GoogleCloudStorageBuilder::from_env().with_url(gs_url);
                if !has_env(&[
                    "GOOGLE_APPLICATION_CREDENTIALS",
                    "GOOGLE_SERVICE_ACCOUNT",
                    "GOOGLE_SERVICE_ACCOUNT_PATH",
                    "GOOGLE_SERVICE_ACCOUNT_KEY",
                ]) {
                    b = b.with_skip_signature(true);
                }
                Arc::new(b.build().map_err(err)?)
            }
            _ => {
                let az_url = if scheme == "azure" {
                    format!("az://{}", &url[8..])
                } else {
                    url.to_string()
                };
                let mut b = object_store::azure::MicrosoftAzureBuilder::from_env().with_url(az_url);
                if !has_env(&[
                    "AZURE_STORAGE_ACCOUNT_KEY",
                    "AZURE_STORAGE_ACCESS_KEY",
                    "AZURE_STORAGE_SAS_KEY",
                    "AZURE_STORAGE_TOKEN",
                    "AZURE_CLIENT_ID",
                    "AZURE_STORAGE_USE_EMULATOR",
                ]) {
                    b = b.with_skip_signature(true);
                }
                Arc::new(b.build().map_err(err)?)
            }
        })
    }

    pub(super) fn open(url: &str, read_only: bool) -> Result<Store> {
        let parsed =
            url::Url::parse(url).map_err(|e| Error::value(format!("invalid url {url}: {e}")))?;
        let prefix = percent_encoding::percent_decode_str(parsed.path().trim_matches('/'))
            .decode_utf8_lossy()
            .to_string();
        let os = build(url)?;
        let os: Arc<dyn ObjectStore> = if prefix.is_empty() {
            os
        } else {
            Arc::new(object_store::prefix::PrefixStore::new(os, prefix.as_str()))
        };
        let store = AsyncObjectStore::new(os);
        let rt = tokio::runtime::Runtime::new().map_err(|e| Error::Storage(e.to_string()))?;
        let sync = AsyncToSyncStorageAdapter::new(Arc::new(store), TokioBlockOn(rt));
        Ok(Store {
            inner: Arc::new(sync),
            kind: StoreKind::ObjectStore,
            root: Some(url.to_string()),
            read_only,
        })
    }
}

/// `Read + Seek` over a remote file using HTTP range requests, with a small
/// block cache (zip readers seek back and forth around the central directory).
#[cfg(all(feature = "http", not(target_arch = "wasm32")))]
mod http_range {
    use super::*;
    use std::collections::HashMap;
    use std::io::{Read, Seek, SeekFrom};

    const BLOCK: u64 = 1 << 20;
    const MAX_BLOCKS: usize = 64;

    pub struct RangeReader {
        client: reqwest::blocking::Client,
        url: String,
        len: u64,
        pos: u64,
        blocks: HashMap<u64, Vec<u8>>,
        order: Vec<u64>,
        /// The whole file, if the server ignores range requests.
        whole: Option<Vec<u8>>,
    }

    fn err(e: impl std::fmt::Display) -> Error {
        Error::Storage(e.to_string())
    }

    impl RangeReader {
        pub fn open(url: &str) -> Result<RangeReader> {
            let client = reqwest::blocking::Client::new();
            let resp = client.head(url).send().map_err(err)?;
            if !resp.status().is_success() {
                return Err(Error::Storage(format!("{url}: HTTP {}", resp.status())));
            }
            let ranges = resp
                .headers()
                .get(reqwest::header::ACCEPT_RANGES)
                .is_some_and(|v| v.as_bytes() == b"bytes");
            let len = resp
                .headers()
                .get(reqwest::header::CONTENT_LENGTH)
                .and_then(|v| v.to_str().ok()?.parse::<u64>().ok());
            let mut r = RangeReader {
                client,
                url: url.to_string(),
                len: len.unwrap_or(0),
                pos: 0,
                blocks: HashMap::new(),
                order: vec![],
                whole: None,
            };
            if !ranges || len.is_none() {
                let body = r
                    .client
                    .get(url)
                    .send()
                    .and_then(|r| r.error_for_status())
                    .map_err(err)?;
                let bytes = body.bytes().map_err(err)?.to_vec();
                r.len = bytes.len() as u64;
                r.whole = Some(bytes);
            }
            Ok(r)
        }

        fn block(&mut self, idx: u64) -> std::io::Result<&[u8]> {
            if !self.blocks.contains_key(&idx) {
                let start = idx * BLOCK;
                let end = (start + BLOCK).min(self.len) - 1;
                let resp = self
                    .client
                    .get(&self.url)
                    .header(reqwest::header::RANGE, format!("bytes={start}-{end}"))
                    .send()
                    .and_then(|r| r.error_for_status())
                    .map_err(std::io::Error::other)?;
                let partial = resp.status() == reqwest::StatusCode::PARTIAL_CONTENT;
                let bytes = resp.bytes().map_err(std::io::Error::other)?.to_vec();
                let data = if partial {
                    bytes
                } else {
                    bytes[start as usize..=end as usize].to_vec()
                };
                if self.order.len() >= MAX_BLOCKS {
                    let old = self.order.remove(0);
                    self.blocks.remove(&old);
                }
                self.order.push(idx);
                self.blocks.insert(idx, data);
            }
            Ok(&self.blocks[&idx])
        }
    }

    impl Read for RangeReader {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.pos >= self.len || buf.is_empty() {
                return Ok(0);
            }
            let pos = self.pos;
            let n = if let Some(w) = &self.whole {
                let n = buf.len().min((self.len - pos) as usize);
                buf[..n].copy_from_slice(&w[pos as usize..pos as usize + n]);
                n
            } else {
                let b = self.block(pos / BLOCK)?;
                let off = (pos % BLOCK) as usize;
                let n = buf.len().min(b.len() - off);
                buf[..n].copy_from_slice(&b[off..off + n]);
                n
            };
            self.pos += n as u64;
            Ok(n)
        }
    }

    impl Seek for RangeReader {
        fn seek(&mut self, to: SeekFrom) -> std::io::Result<u64> {
            let p = match to {
                SeekFrom::Start(p) => p as i128,
                SeekFrom::End(d) => self.len as i128 + d as i128,
                SeekFrom::Current(d) => self.pos as i128 + d as i128,
            };
            if p < 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "seek before start",
                ));
            }
            self.pos = p as u64;
            Ok(self.pos)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
