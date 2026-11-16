//! Stores fetched as they are read, for hosts that cannot block on I/O: the
//! browser above all, where a [`Store`]'s synchronous reads cannot wait on a
//! network.
//!
//! A [`LazyStore`] starts empty and records every key it is asked for and does
//! not have. The host fetches those keys however it can (`fetch`, a `File`, a
//! message from a worker), hands them over, and runs the read again;
//! [`settle`] drives that loop with any async fetcher, several keys at a time.
//! Opening an image this way fetches only the metadata documents the reader
//! looks at, and reading a region only the chunks under it: a remote store is
//! streamed, never listed or downloaded whole.
//!
//! ```no_run
//! # async fn demo() -> ome_zarr::Result<()> {
//! use std::sync::Arc;
//! use ome_zarr::classes::OmeZarrMultiscale;
//! use ome_zarr::lazy::{settle, LazyStore, Progress};
//! use ome_zarr::writer::GroupLike;
//! use ome_zarr::zarr::{Group, Mode};
//!
//! let lazy = Arc::new(LazyStore::new());
//! let store = lazy.store();
//! // the host's fetch: here a closure that would issue an HTTP request
//! let fetch = |key: String| async move { Ok::<_, String>(None::<Vec<u8>>) };
//! let mut progress = Progress::default();
//! let ms = settle(&lazy, || {
//!     let g = Group::open(&store, "", Mode::Read, None)?;
//!     OmeZarrMultiscale::from_ome_zarr(GroupLike::Group(g))
//! }, &fetch, 8, &mut |_| {}, &mut progress).await?;
//! let level = &ms.images()[ms.images().len() - 1].data;
//! let voxels = settle(&lazy, || level.compute(), &fetch, 8, &mut |_| {}, &mut progress).await?;
//! # Ok(()) }
//! ```

use std::collections::HashSet;
use std::future::Future;
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use futures::stream::{self, StreamExt};

use crate::error::{Error, Result};
use crate::store::{ByteRange, MemoryStore, Storage, Store, StoreKind};

pub use crate::zarr::is_metadata;
use crate::zarr::{ZARR_JSON, ZARRAY, ZATTRS, ZGROUP, document_name, is_v2_document};

/// Which Zarr format the store turned out to be, once a metadata document
/// was found: the other format's documents are then not asked for.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Seen {
    Unknown,
    V2,
    V3,
}

/// A store filled as it is read: keys handed over with [`put`](Self::put),
/// keys asked for and not there listed by [`missing`](Self::missing) (unless
/// known [`absent`](Self::absent)). With a `fallback` (a zip in memory, say),
/// keys are read from it instead, and nothing is ever missing.
pub struct LazyStore {
    inner: MemoryStore,
    fallback: Option<Arc<dyn Storage>>,
    absent: Mutex<HashSet<String>>,
    missed: Mutex<Vec<String>>,
    seen: Mutex<Seen>,
}

impl Default for LazyStore {
    fn default() -> Self {
        Self::new()
    }
}

impl LazyStore {
    /// An empty store whose keys arrive as [`missing`](Self::missing) asks.
    pub fn new() -> LazyStore {
        LazyStore {
            inner: MemoryStore::new(),
            fallback: None,
            absent: Mutex::new(HashSet::new()),
            missed: Mutex::new(vec![]),
            seen: Mutex::new(Seen::Unknown),
        }
    }

    /// Reads served by `fallback` (e.g. a [`ZipStore`](crate::store::ZipStore)),
    /// each key kept once read.
    pub fn over(fallback: Arc<dyn Storage>) -> LazyStore {
        LazyStore {
            fallback: Some(fallback),
            ..LazyStore::new()
        }
    }

    /// This store as a [`Store`] for groups and arrays.
    pub fn store(self: &Arc<Self>) -> Store {
        Store::from_dyn(self.clone(), StoreKind::Other, None)
    }

    /// A key fetched: its bytes.
    pub fn put(&self, key: &str, bytes: Vec<u8>) -> Result<()> {
        self.inner
            .set(key.trim_start_matches('/'), Bytes::from(bytes))
    }

    /// A key fetched and not there (an HTTP 404).
    pub fn absent(&self, key: &str) {
        self.absent
            .lock()
            .unwrap()
            .insert(key.trim_start_matches('/').to_string());
    }

    /// The keys asked for since the last call and not there: fetch each,
    /// [`put`](Self::put) or [`absent`](Self::absent) it, and read again.
    pub fn missing(&self) -> Vec<String> {
        std::mem::take(&mut *self.missed.lock().unwrap())
    }

    /// Ask for `key` along with this round's misses, unless it is here or
    /// known absent: a prefetch, so keys a reader would only reach one round
    /// trip at a time arrive together.
    pub fn want(&self, key: &str) {
        let k = key.trim_start_matches('/');
        let here = self.inner.get(k).ok().flatten().is_some();
        if here || self.fallback.is_some() || self.absent.lock().unwrap().contains(k) {
            return;
        }
        let mut m = self.missed.lock().unwrap();
        if !m.iter().any(|x| x == k) {
            m.push(k.to_string());
        }
    }

    /// Prefetch what a multiscale image at `path` (its group attributes
    /// `attrs`, `zarr_format` 2 or 3) will be opened with: every level's
    /// array metadata and the labels group's, in the round that fetches them
    /// rather than one level per round.
    pub fn prefetch_multiscales(
        &self,
        path: &str,
        attrs: &serde_json::Map<String, serde_json::Value>,
        zarr_format: u8,
    ) {
        let ome = attrs.get("ome").and_then(|o| o.as_object());
        let multiscales = ome
            .and_then(|o| o.get("multiscales"))
            .or_else(|| attrs.get("multiscales"))
            .and_then(|m| m.as_array());
        let at = |k: &str| {
            if path.is_empty() {
                k.to_string()
            } else {
                format!("{}/{k}", path.trim_end_matches('/'))
            }
        };
        let (array, group) = if zarr_format == 2 {
            (&[ZARRAY, ZATTRS][..], &[ZGROUP, ZATTRS][..])
        } else {
            (&[ZARR_JSON][..], &[ZARR_JSON][..])
        };
        for ms in multiscales.into_iter().flatten() {
            for ds in ms
                .get("datasets")
                .and_then(|d| d.as_array())
                .into_iter()
                .flatten()
            {
                if let Some(p) = ds.get("path").and_then(|p| p.as_str()) {
                    for doc in array {
                        self.want(&at(&format!("{p}/{doc}")));
                    }
                }
            }
        }
        for doc in group {
            self.want(&at(&format!("labels/{doc}")));
        }
    }

    /// Drop every key but the metadata documents: the chunks of a region
    /// once it has been read (fetched again if read again).
    pub fn evict_chunks(&self) {
        self.evict_chunks_under("");
    }

    /// [`evict_chunks`](Self::evict_chunks) for the keys under `prefix`
    /// only (one array's): others fetched ahead stay.
    pub fn evict_chunks_under(&self, prefix: &str) {
        if let Ok(keys) = self.inner.list_prefix(prefix) {
            for k in keys.iter().filter(|k| !is_metadata(k)) {
                let _ = self.inner.erase(k);
            }
        }
    }

    /// Were keys asked for that have not been handed over?
    pub fn has_missing(&self) -> bool {
        !self.missed.lock().unwrap().is_empty()
    }

    /// `key`'s bytes if here (or in the fallback); else noted as missing,
    /// unless known absent or a document of the other Zarr format.
    fn find(&self, k: &str) -> Result<Option<Bytes>> {
        let name = document_name(k);
        let v2 = is_v2_document(name);
        let found = match self.inner.get(k)? {
            Some(v) => Some(v),
            None => match &self.fallback {
                Some(f) => {
                    let v = f.get(k)?;
                    if let Some(v) = &v {
                        self.inner.set(k, v.clone())?;
                    }
                    v
                }
                None => None,
            },
        };
        let mut seen = self.seen.lock().unwrap();
        if found.is_some() {
            if name == ZARR_JSON {
                *seen = Seen::V3;
            } else if v2 {
                *seen = Seen::V2;
            }
            return Ok(found);
        }
        let other_format = (*seen == Seen::V3 && v2) || (*seen == Seen::V2 && name == ZARR_JSON);
        if self.fallback.is_none() && !other_format && !self.absent.lock().unwrap().contains(k) {
            {
                let mut m = self.missed.lock().unwrap();
                if !m.iter().any(|x| x == k) {
                    m.push(k.to_string());
                }
            }
            // a v2 node's attributes are read right after it is found: ask
            // for them in the same round
            if name == ZGROUP || name == ZARRAY {
                drop(seen);
                self.want(&format!("{}{ZATTRS}", &k[..k.len() - name.len()]));
            }
        }
        Ok(None)
    }
}

/// Listing covers only the keys fetched so far: a remote store cannot be
/// listed, so readers that walk a hierarchy see what they asked for.
impl Storage for LazyStore {
    fn get(&self, key: &str) -> Result<Option<Bytes>> {
        self.find(key)
    }
    fn get_ranges(&self, key: &str, ranges: &[ByteRange]) -> Result<Option<Vec<Bytes>>> {
        if self.find(key)?.is_none() {
            return Ok(None);
        }
        self.inner.get_ranges(key, ranges)
    }
    fn supports_ranges(&self) -> bool {
        true
    }
    fn size(&self, key: &str) -> Result<Option<u64>> {
        Ok(self.find(key)?.map(|v| v.len() as u64))
    }
    fn set(&self, key: &str, value: Bytes) -> Result<()> {
        self.inner.set(key, value)
    }
    fn erase(&self, key: &str) -> Result<()> {
        self.inner.erase(key)
    }
    fn erase_prefix(&self, prefix: &str) -> Result<()> {
        self.inner.erase_prefix(prefix)
    }
    fn list_prefix(&self, prefix: &str) -> Result<Vec<String>> {
        self.inner.list_prefix(prefix)
    }
}

/// How far a [`settle`] is: keys fetched (or found absent), still in flight,
/// bytes received, and rounds of fetches (each one a network round trip).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Progress {
    pub files: usize,
    pub pending: usize,
    pub bytes: usize,
    pub rounds: usize,
}

/// One key fetched by the host: its bytes, `None` where it does not exist, or
/// why it could not be fetched.
pub type Fetched = std::result::Result<Option<Vec<u8>>, String>;

/// Run `step` until it no longer misses keys: after each run, the keys it
/// missed are fetched through `fetch`, `parallel` at a time, and handed to
/// `store`. `on_progress` hears of each key as it arrives. Gives up after 64
/// rounds (a reader that keeps finding new keys to ask for).
pub async fn settle<T, F, Fut>(
    store: &LazyStore,
    mut step: impl FnMut() -> Result<T>,
    fetch: &F,
    parallel: usize,
    on_progress: &mut dyn FnMut(Progress),
    progress: &mut Progress,
) -> Result<T>
where
    F: Fn(String) -> Fut,
    Fut: Future<Output = Fetched>,
{
    for _ in 0..64 {
        let out = step();
        let missing = store.missing();
        if missing.is_empty() {
            return out;
        }
        fetch_keys(store, missing, fetch, parallel, on_progress, progress).await?;
    }
    Err(Error::Storage("the store kept asking for more keys".into()))
}

/// Fetch `keys` into `store`, `parallel` at a time: one round. With a key
/// list found ahead (a read's misses, recorded by running it once), the
/// fetching can run alongside other work — another level read meanwhile.
///
/// A key that fails is asked for once more. One that fails again does not
/// end the round: the rest arrive and are kept, so running the read again
/// fetches only what is still missing, and its error is returned at the end.
pub async fn fetch_keys<F, Fut>(
    store: &LazyStore,
    keys: Vec<String>,
    fetch: &F,
    parallel: usize,
    on_progress: &mut dyn FnMut(Progress),
    progress: &mut Progress,
) -> Result<()>
where
    F: Fn(String) -> Fut,
    Fut: Future<Output = Fetched>,
{
    if keys.is_empty() {
        return Ok(());
    }
    progress.pending += keys.len();
    progress.rounds += 1;
    let mut arriving = stream::iter(keys.into_iter().map(|k| async move {
        let r = match fetch(k.clone()).await {
            Err(_) => fetch(k.clone()).await,
            r => r,
        };
        (k, r)
    }))
    .buffer_unordered(parallel.max(1));
    let mut failed = None;
    while let Some((k, r)) = arriving.next().await {
        match r {
            Ok(Some(b)) => {
                progress.bytes += b.len();
                store.put(&k, b)?;
                progress.files += 1;
            }
            Ok(None) => {
                store.absent(&k);
                progress.files += 1;
            }
            Err(e) => {
                failed.get_or_insert(e);
            }
        }
        progress.pending -= 1;
        on_progress(*progress);
    }
    failed.map_or(Ok(()), |e| Err(Error::Storage(e)))
}

/// The folder a tree of files' store is rooted in (`""`, or ending in `/`):
/// the shallowest one holding a metadata document.
pub fn root_of<'a>(keys: impl IntoIterator<Item = &'a str>) -> Result<String> {
    let key = keys
        .into_iter()
        .map(|k| k.replace('\\', "/").trim_start_matches('/').to_string())
        .filter(|k| is_metadata(k))
        .min_by_key(|k| (k.matches('/').count(), k.clone()))
        .ok_or_else(|| Error::value("no zarr.json, .zattrs or .zgroup: not a Zarr store"))?;
    Ok(key
        .rsplit_once('/')
        .map(|(dir, _)| format!("{dir}/"))
        .unwrap_or_default())
}

/// The https URL a store's keys are fetched under: `s3://bucket/path` and
/// `gs://bucket/path` as their public endpoints; no trailing slash.
pub fn https_url(url: &str) -> Result<String> {
    let u = url.trim().trim_end_matches('/');
    let split = |rest: &str| -> (String, String) {
        let (b, p) = rest.split_once('/').unwrap_or((rest, ""));
        (b.to_string(), p.to_string())
    };
    let out = if let Some(rest) = u.strip_prefix("s3://") {
        let (b, p) = split(rest);
        format!("https://{b}.s3.amazonaws.com/{p}")
    } else if let Some(rest) = u.strip_prefix("gs://").or_else(|| u.strip_prefix("gcs://")) {
        let (b, p) = split(rest);
        format!("https://storage.googleapis.com/{b}/{p}")
    } else if u.starts_with("https://") || u.starts_with("http://") {
        u.to_string()
    } else {
        return Err(Error::value(
            "a store URL starts with https://, s3:// or gs://",
        ));
    };
    Ok(out.trim_end_matches('/').to_string())
}
