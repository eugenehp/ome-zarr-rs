//! Zarr groups and arrays (v2 and v3) on top of a [`Store`].
//!
//! Metadata documents (`zarr.json`, `.zgroup`, `.zarray`, `.zattrs`) are read and
//! written directly so that the output matches zarr-python; chunk encoding and
//! decoding is delegated to `zarrs`.

use std::ops::Range;
use std::sync::Arc;

use serde_json::{Map, Value, json};
use zarrs::array::{Array as ZarrsArray, ArraySubset};
use zarrs_storage::ReadableWritableListableStorageTraits;

use crate::array::{ChunkedSource, Element};
use crate::array::{Chunks, DType, LazyArray, NdArray, Region, chunks::blockdims_from_blockshape};
use crate::error::{Error, Result};
use crate::store::Store;
use crate::{dtype_dispatch, nd_dispatch};

/// JSON object type used for attributes.
pub type JsonMap = Map<String, Value>;

/// Mode in which to open a group (as in `zarr.open_group`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Read only; must exist.
    Read,
    /// Read/write; must exist.
    ReadWrite,
    /// Read/write; create if it does not exist (default).
    Append,
    /// Create, deleting any existing content.
    Write,
    /// Create; fail if it exists.
    WriteNew,
}

impl Mode {
    /// Parse a Python-style mode string (`"r"`, `"r+"`, `"a"`, `"w"`, `"w-"`).
    pub fn parse(s: &str) -> Result<Mode> {
        Ok(match s {
            "r" => Mode::Read,
            "r+" => Mode::ReadWrite,
            "a" => Mode::Append,
            "w" => Mode::Write,
            "w-" | "x" => Mode::WriteNew,
            other => return Err(Error::value(format!("invalid mode: {other}"))),
        })
    }

    pub fn is_read_only(&self) -> bool {
        matches!(self, Mode::Read)
    }
}

/// The kind of node stored at a path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    Group,
    Array,
}

/// Join zarr paths, ignoring empty components.
pub fn join_path(a: &str, b: &str) -> String {
    let a = a.trim_matches('/');
    let b = b.trim_matches('/');
    match (a.is_empty(), b.is_empty()) {
        (true, _) => b.to_string(),
        (_, true) => a.to_string(),
        _ => format!("{a}/{b}"),
    }
}

fn key(path: &str, name: &str) -> String {
    join_path(path, name)
}

/// Determine what (if anything) exists at `path`, and in which zarr format.
pub fn node_at(store: &Store, path: &str) -> Result<Option<(NodeKind, u8)>> {
    if let Some(v) = store.get_json(&key(path, "zarr.json"))? {
        let kind = match v.get("node_type").and_then(|v| v.as_str()) {
            Some("array") => NodeKind::Array,
            _ => NodeKind::Group,
        };
        return Ok(Some((kind, 3)));
    }
    if store.exists(&key(path, ".zarray")) {
        return Ok(Some((NodeKind::Array, 2)));
    }
    if store.exists(&key(path, ".zgroup")) {
        return Ok(Some((NodeKind::Group, 2)));
    }
    Ok(None)
}

/// Read the user attributes of the node at `path`.
pub fn read_attrs(store: &Store, path: &str, zarr_format: u8) -> Result<JsonMap> {
    if zarr_format == 3 {
        let v = store
            .get_json(&key(path, "zarr.json"))?
            .unwrap_or(Value::Null);
        Ok(v.get("attributes")
            .and_then(|a| a.as_object())
            .cloned()
            .unwrap_or_default())
    } else {
        let v = store.get_json(&key(path, ".zattrs"))?;
        Ok(v.and_then(|v| v.as_object().cloned()).unwrap_or_default())
    }
}

/// Replace the user attributes of the node at `path`.
pub fn write_attrs(store: &Store, path: &str, zarr_format: u8, attrs: &JsonMap) -> Result<()> {
    if zarr_format == 3 {
        let k = key(path, "zarr.json");
        let mut v = store
            .get_json(&k)?
            .ok_or_else(|| Error::value(format!("no zarr.json at {path}")))?;
        v.as_object_mut()
            .ok_or_else(|| Error::value("zarr.json is not an object"))?
            .insert("attributes".into(), Value::Object(attrs.clone()));
        store.set_json(&k, &v)
    } else {
        store.set_json(&key(path, ".zattrs"), &Value::Object(attrs.clone()))
    }
}

/// A zarr group.
#[derive(Debug, Clone)]
pub struct Group {
    store: Store,
    path: String,
    zarr_format: u8,
    read_only: bool,
}

impl Group {
    /// Open or create a group at `path` inside `store` according to `mode`.
    ///
    /// `zarr_format` is required to create a group in a specific format (default 3);
    /// when opening an existing group it is checked if given.
    pub fn open(store: &Store, path: &str, mode: Mode, zarr_format: Option<u8>) -> Result<Group> {
        let path = path.trim_matches('/').to_string();
        let existing = node_at(store, &path)?;
        let read_only = mode.is_read_only() || store.is_read_only();
        match mode {
            Mode::Read | Mode::ReadWrite => match existing {
                Some((NodeKind::Group, fmt)) => {
                    if let Some(z) = zarr_format
                        && z != fmt
                    {
                        return Err(Error::value(format!(
                            "group at '{path}' is zarr_format {fmt}, not {z}"
                        )));
                    }
                    Ok(Group {
                        store: store.clone(),
                        path,
                        zarr_format: fmt,
                        read_only,
                    })
                }
                Some((NodeKind::Array, _)) => Err(Error::value(format!(
                    "node at '{path}' is an array, not a group"
                ))),
                None => Err(Error::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("No group found in store at path '{path}'"),
                ))),
            },
            Mode::Append => match existing {
                Some((NodeKind::Group, fmt)) => {
                    if let Some(z) = zarr_format
                        && z != fmt
                    {
                        return Err(Error::value(format!(
                            "group at '{path}' is zarr_format {fmt}, not {z}"
                        )));
                    }
                    Ok(Group {
                        store: store.clone(),
                        path,
                        zarr_format: fmt,
                        read_only,
                    })
                }
                Some((NodeKind::Array, _)) => Err(Error::value(format!(
                    "node at '{path}' is an array, not a group"
                ))),
                None => Group::create(store, &path, zarr_format.unwrap_or(3), None),
            },
            Mode::Write => {
                if existing.is_some() || !path.is_empty() {
                    store.erase_prefix(&path)?;
                } else {
                    store.erase_prefix("")?;
                }
                Group::create(store, &path, zarr_format.unwrap_or(3), None)
            }
            Mode::WriteNew => {
                if existing.is_some() {
                    return Err(Error::value(format!("a node already exists at '{path}'")));
                }
                Group::create(store, &path, zarr_format.unwrap_or(3), None)
            }
        }
    }

    /// `zarr.open_group(path_or_url, mode=...)`.
    pub fn open_url(path: &str, mode: Mode, zarr_format: Option<u8>) -> Result<Group> {
        let store = Store::from_url(path, mode.is_read_only())?;
        Group::open(&store, "", mode, zarr_format)
    }

    /// Write group metadata at `path`.
    pub fn create(
        store: &Store,
        path: &str,
        zarr_format: u8,
        attrs: Option<&JsonMap>,
    ) -> Result<Group> {
        let path = path.trim_matches('/').to_string();
        let attrs = attrs.cloned().unwrap_or_default();
        match zarr_format {
            3 => {
                let meta = json!({
                    "attributes": Value::Object(attrs),
                    "zarr_format": 3,
                    "node_type": "group",
                });
                store.set_json(&key(&path, "zarr.json"), &meta)?;
            }
            2 => {
                store.set_json(&key(&path, ".zgroup"), &json!({"zarr_format": 2}))?;
                store.set_json(&key(&path, ".zattrs"), &Value::Object(attrs))?;
            }
            f => return Err(Error::value(format!("unsupported zarr_format: {f}"))),
        }
        Ok(Group {
            store: store.clone(),
            path,
            zarr_format,
            read_only: false,
        })
    }

    pub fn store(&self) -> &Store {
        &self.store
    }

    /// Path of the group inside the store (no leading slash).
    pub fn path(&self) -> &str {
        &self.path
    }

    /// `"/" + path` (zarr-python `Group.name`).
    pub fn name(&self) -> String {
        format!("/{}", self.path)
    }

    /// Last path component (zarr-python `Group.basename`).
    pub fn basename(&self) -> String {
        self.path.rsplit('/').next().unwrap_or("").to_string()
    }

    pub fn zarr_format(&self) -> u8 {
        self.zarr_format
    }

    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    /// User attributes.
    pub fn attrs(&self) -> Result<JsonMap> {
        read_attrs(&self.store, &self.path, self.zarr_format)
    }

    /// Replace all user attributes.
    pub fn set_attrs(&self, attrs: &JsonMap) -> Result<()> {
        if self.read_only {
            return Err(Error::value("group is read-only"));
        }
        write_attrs(&self.store, &self.path, self.zarr_format, attrs)
    }

    /// Set a single attribute (`group.attrs[key] = value`).
    pub fn set_attr(&self, k: &str, v: Value) -> Result<()> {
        let mut a = self.attrs()?;
        a.insert(k.to_string(), v);
        self.set_attrs(&a)
    }

    /// Update attributes (`group.attrs.update(...)`).
    pub fn update_attrs(&self, other: &JsonMap) -> Result<()> {
        let mut a = self.attrs()?;
        for (k, v) in other {
            a.insert(k.clone(), v.clone());
        }
        self.set_attrs(&a)
    }

    /// Whether a child node named `name` exists (`name in group`).
    pub fn contains(&self, name: &str) -> bool {
        matches!(node_at(&self.store, &key(&self.path, name)), Ok(Some(_)))
    }

    /// What kind of node `name` is, if it exists.
    pub fn child_kind(&self, name: &str) -> Option<NodeKind> {
        node_at(&self.store, &key(&self.path, name))
            .ok()
            .flatten()
            .map(|(k, _)| k)
    }

    /// Open a child group (`group[name]`).
    pub fn group(&self, name: &str) -> Result<Group> {
        let p = key(&self.path, name);
        match node_at(&self.store, &p)? {
            Some((NodeKind::Group, fmt)) => Ok(Group {
                store: self.store.clone(),
                path: p,
                zarr_format: fmt,
                read_only: self.read_only,
            }),
            Some((NodeKind::Array, _)) => {
                Err(Error::type_err(format!("'{p}' is an array, not a group")))
            }
            None => Err(Error::key(format!(
                "'{name}' not found in group '{}'",
                self.name()
            ))),
        }
    }

    /// Open a child array (`group[name]`).
    pub fn array(&self, name: &str) -> Result<ZArray> {
        ZArray::open(&self.store, &key(&self.path, name))
    }

    /// Get or create a child group (creating intermediate groups).
    pub fn require_group(&self, name: &str) -> Result<Group> {
        let mut current = self.clone();
        for part in name.split('/').filter(|p| !p.is_empty()) {
            current = match current.child_kind(part) {
                Some(NodeKind::Group) => current.group(part)?,
                Some(NodeKind::Array) => {
                    return Err(Error::type_err(format!(
                        "'{part}' is an array, not a group"
                    )));
                }
                None => Group::create(
                    &current.store,
                    &key(&current.path, part),
                    current.zarr_format,
                    None,
                )?,
            };
        }
        Ok(current)
    }

    /// Create a child group; with `overwrite`, deletes any existing node.
    pub fn create_group(&self, name: &str, overwrite: bool) -> Result<Group> {
        let p = key(&self.path, name);
        if node_at(&self.store, &p)?.is_some() {
            if !overwrite {
                return Err(Error::value(format!("a node already exists at '{p}'")));
            }
            self.store.erase_prefix(&p)?;
        }
        // create intermediate groups
        if let Some((parent, _)) = name.trim_matches('/').rsplit_once('/') {
            self.require_group(parent)?;
        }
        Group::create(&self.store, &p, self.zarr_format, None)
    }

    fn children(&self) -> Result<Vec<(String, NodeKind)>> {
        let (_, prefixes) = self.store.list_dir(&self.path)?;
        let mut out = vec![];
        for p in prefixes {
            if let Some((k, _)) = node_at(&self.store, &key(&self.path, &p))? {
                out.push((p, k));
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(out)
    }

    /// Names of child groups.
    pub fn group_keys(&self) -> Result<Vec<String>> {
        Ok(self
            .children()?
            .into_iter()
            .filter(|(_, k)| *k == NodeKind::Group)
            .map(|(n, _)| n)
            .collect())
    }

    /// Names of child arrays.
    pub fn array_keys(&self) -> Result<Vec<String>> {
        Ok(self
            .children()?
            .into_iter()
            .filter(|(_, k)| *k == NodeKind::Array)
            .map(|(n, _)| n)
            .collect())
    }

    /// Create a child array.
    pub fn create_array(&self, name: &str, spec: &ArraySpec, overwrite: bool) -> Result<ZArray> {
        let p = key(&self.path, name);
        if node_at(&self.store, &p)?.is_some() {
            if !overwrite {
                return Err(Error::value(format!("a node already exists at '{p}'")));
            }
            self.store.erase_prefix(&p)?;
        }
        ZArray::create(&self.store, &p, spec)
    }
}

/// Byte-to-byte compression for array chunks.
#[derive(Debug, Clone, PartialEq)]
pub enum Compressor {
    /// No compression.
    None,
    /// Zstandard (zarr-python's default).
    Zstd { level: i32, checksum: bool },
    /// Blosc meta-compressor.
    Blosc {
        cname: String,
        clevel: u8,
        /// 0 = none, 1 = byte shuffle, 2 = bit shuffle
        shuffle: u8,
        blocksize: usize,
    },
    /// gzip.
    Gzip { level: u32 },
    /// Another numcodecs compressor (`zlib`, `bz2`, `lz4`, `lzma`): `config`
    /// is numcodecs' full configuration (as zarr v2 stores it), `explicit`
    /// the arguments given (as zarr-python's v3 numcodecs codecs store them).
    Numcodecs {
        id: String,
        config: JsonMap,
        explicit: JsonMap,
    },
}

impl Default for Compressor {
    fn default() -> Self {
        Compressor::Zstd {
            level: 0,
            checksum: false,
        }
    }
}

/// numcodecs' constructor defaults, in declaration order.
fn numcodecs_defaults(id: &str) -> Option<Vec<(&'static str, Value)>> {
    Some(match id {
        "zlib" | "bz2" => vec![("level", json!(1))],
        "lz4" => vec![("acceleration", json!(1))],
        "lzma" => vec![
            ("format", json!(1)),
            ("check", json!(-1)),
            ("preset", Value::Null),
            ("filters", Value::Null),
        ],
        _ => return None,
    })
}

impl Compressor {
    /// numcodecs `Zlib(level)`, `BZ2(level)`, `LZ4(acceleration)` or `LZMA(...)`
    /// with the given keyword arguments (others take numcodecs' defaults).
    pub fn numcodecs(id: &str, args: &[(&str, Value)]) -> Result<Compressor> {
        let defaults = numcodecs_defaults(id)
            .ok_or_else(|| Error::value(format!("unsupported compressor: {id}")))?;
        let mut config = JsonMap::new();
        let mut explicit = JsonMap::new();
        for (k, v) in &defaults {
            config.insert((*k).into(), v.clone());
        }
        for (k, v) in args {
            if !defaults.iter().any(|(d, _)| d == k) {
                return Err(Error::type_err(format!(
                    "{id}: unexpected keyword argument '{k}'"
                )));
            }
            config.insert((*k).into(), v.clone());
            explicit.insert((*k).into(), v.clone());
        }
        Ok(Compressor::Numcodecs {
            id: id.into(),
            config,
            explicit,
        })
    }

    /// The `Blosc(cname="zstd", clevel=5, shuffle=SHUFFLE)` compressor used by the writer.
    pub fn blosc_zstd() -> Compressor {
        Compressor::Blosc {
            cname: "zstd".into(),
            clevel: 5,
            shuffle: 1,
            blocksize: 0,
        }
    }

    /// Zarr v2 `compressor` JSON.
    pub fn to_v2(&self) -> Value {
        match self {
            Compressor::None => Value::Null,
            Compressor::Zstd { level, checksum } => {
                if *checksum {
                    json!({"id": "zstd", "level": level, "checksum": true})
                } else {
                    json!({"id": "zstd", "level": level})
                }
            }
            Compressor::Blosc {
                cname,
                clevel,
                shuffle,
                blocksize,
            } => json!({
                "id": "blosc", "cname": cname, "clevel": clevel, "shuffle": shuffle, "blocksize": blocksize
            }),
            Compressor::Gzip { level } => json!({"id": "gzip", "level": level}),
            Compressor::Numcodecs { id, config, .. } => {
                let mut m = JsonMap::new();
                m.insert("id".into(), json!(id));
                m.extend(config.clone());
                Value::Object(m)
            }
        }
    }

    /// Zarr v3 codec JSON (if any).
    pub fn to_v3(&self, typesize: usize) -> Option<Value> {
        match self {
            Compressor::None => None,
            Compressor::Zstd { level, checksum } => Some(
                json!({"name": "zstd", "configuration": {"level": level, "checksum": checksum}}),
            ),
            Compressor::Blosc {
                cname,
                clevel,
                shuffle,
                blocksize,
            } => {
                let shuffle = match shuffle {
                    0 => "noshuffle",
                    2 => "bitshuffle",
                    _ => "shuffle",
                };
                // key order as zarr-python's BloscCodec.to_dict()
                Some(json!({"name": "blosc", "configuration": {
                    "typesize": typesize, "cname": cname, "clevel": clevel,
                    "shuffle": shuffle, "blocksize": blocksize
                }}))
            }
            Compressor::Gzip { level } => {
                Some(json!({"name": "gzip", "configuration": {"level": level}}))
            }
            Compressor::Numcodecs { id, explicit, .. } => {
                Some(json!({"name": format!("numcodecs.{id}"), "configuration": explicit}))
            }
        }
    }

    /// Parse from a zarr v2 compressor JSON or a v3 codec JSON.
    pub fn from_json(v: &Value) -> Result<Compressor> {
        if v.is_null() {
            return Ok(Compressor::None);
        }
        let id = v
            .get("id")
            .or_else(|| v.get("name"))
            .and_then(|x| x.as_str())
            .ok_or_else(|| Error::value(format!("invalid compressor: {v}")))?;
        let cfg = v.get("configuration").unwrap_or(v);
        Ok(match id {
            "zstd" => Compressor::Zstd {
                level: cfg.get("level").and_then(|x| x.as_i64()).unwrap_or(0) as i32,
                checksum: cfg
                    .get("checksum")
                    .and_then(|x| x.as_bool())
                    .unwrap_or(false),
            },
            "blosc" => Compressor::Blosc {
                cname: cfg
                    .get("cname")
                    .and_then(|x| x.as_str())
                    .unwrap_or("lz4")
                    .to_string(),
                clevel: cfg.get("clevel").and_then(|x| x.as_u64()).unwrap_or(5) as u8,
                shuffle: match cfg.get("shuffle") {
                    Some(Value::String(s)) => match s.as_str() {
                        "noshuffle" => 0,
                        "bitshuffle" => 2,
                        _ => 1,
                    },
                    Some(x) => x.as_u64().unwrap_or(1) as u8,
                    None => 1,
                },
                blocksize: cfg.get("blocksize").and_then(|x| x.as_u64()).unwrap_or(0) as usize,
            },
            "gzip" => Compressor::Gzip {
                level: cfg.get("level").and_then(|x| x.as_u64()).unwrap_or(1) as u32,
            },
            other if numcodecs_defaults(other.trim_start_matches("numcodecs.")).is_some() => {
                let id = other.trim_start_matches("numcodecs.");
                let args: Vec<(&str, Value)> = cfg
                    .as_object()
                    .map(|m| {
                        m.iter()
                            .filter(|(k, _)| *k != "id")
                            .map(|(k, v)| (k.as_str(), v.clone()))
                            .collect()
                    })
                    .unwrap_or_default();
                Compressor::numcodecs(id, &args)?
            }
            other => return Err(Error::value(format!("unsupported compressor: {other}"))),
        })
    }
}

/// Everything needed to create an array.
#[derive(Debug, Clone)]
pub struct ArraySpec {
    pub shape: Vec<u64>,
    pub dtype: DType,
    /// Chunk shape (the inner chunk shape if sharded).
    pub chunks: Vec<u64>,
    /// Shard shape (zarr v3 only).
    pub shards: Option<Vec<u64>>,
    pub zarr_format: u8,
    pub compressor: Compressor,
    pub fill_value: f64,
    pub dimension_names: Option<Vec<String>>,
    /// Chunk key separator (`"/"` or `"."`).
    pub separator: String,
    pub attributes: JsonMap,
}

impl ArraySpec {
    pub fn new(shape: &[u64], dtype: DType, chunks: &[u64], zarr_format: u8) -> ArraySpec {
        ArraySpec {
            shape: shape.to_vec(),
            dtype,
            chunks: chunks.to_vec(),
            shards: None,
            zarr_format,
            compressor: Compressor::default(),
            fill_value: 0.0,
            dimension_names: None,
            separator: "/".into(),
            attributes: JsonMap::new(),
        }
    }

    fn fill_json(&self) -> Value {
        match self.dtype {
            DType::Bool => json!(self.fill_value != 0.0),
            DType::Float16 | DType::Float32 | DType::Float64 => {
                if self.fill_value.is_nan() {
                    json!("NaN")
                } else {
                    json!(self.fill_value)
                }
            }
            // zarr-python writes complex fill values as [real, imag]
            DType::Complex64 | DType::Complex128 => {
                let f = |v: f64| if v.is_nan() { json!("NaN") } else { json!(v) };
                json!([f(self.fill_value), f(0.0)])
            }
            _ => json!(self.fill_value as i64),
        }
    }

    /// Metadata documents for this spec: `(key, json)` pairs.
    pub fn metadata(&self) -> Result<Vec<(&'static str, Value)>> {
        if self.chunks.len() != self.shape.len() {
            return Err(Error::value(format!(
                "chunks {:?} must have the same number of dimensions as shape {:?}",
                self.chunks, self.shape
            )));
        }
        if self.zarr_format == 2 {
            if self.shards.is_some() {
                return Err(Error::value("sharding is only supported for zarr_format 3"));
            }
            let zarray = json!({
                "shape": self.shape,
                "chunks": self.chunks,
                "dtype": self.dtype.zarr_v2(),
                "fill_value": self.fill_json(),
                "order": "C",
                "filters": null,
                "dimension_separator": self.separator,
                "compressor": self.compressor.to_v2(),
                "zarr_format": 2,
            });
            return Ok(vec![
                (".zarray", zarray),
                (".zattrs", Value::Object(self.attributes.clone())),
            ]);
        }
        let itemsize = self.dtype.itemsize();
        let bytes_codec = if itemsize > 1 {
            json!({"name": "bytes", "configuration": {"endian": "little"}})
        } else {
            json!({"name": "bytes"})
        };
        let mut inner_codecs = vec![bytes_codec.clone()];
        if let Some(c) = self.compressor.to_v3(itemsize) {
            inner_codecs.push(c);
        }
        let (grid_shape, codecs) = match &self.shards {
            Some(shards) => {
                if shards.len() != self.shape.len() {
                    return Err(Error::value(
                        "shards must have the same number of dimensions as shape",
                    ));
                }
                for (s, c) in shards.iter().zip(&self.chunks) {
                    if *c == 0 || s % c != 0 {
                        return Err(Error::value(format!(
                            "shard shape {shards:?} must be divisible by chunk shape {:?}",
                            self.chunks
                        )));
                    }
                }
                let sharding = json!({"name": "sharding_indexed", "configuration": {
                    "chunk_shape": self.chunks,
                    "codecs": inner_codecs,
                    "index_codecs": [
                        {"name": "bytes", "configuration": {"endian": "little"}},
                        {"name": "crc32c"}
                    ],
                    "index_location": "end",
                }});
                (shards.clone(), vec![sharding])
            }
            None => (self.chunks.clone(), inner_codecs),
        };
        // key order as zarr-python's ArrayV3Metadata.to_dict()
        let mut meta = JsonMap::new();
        meta.insert("shape".into(), json!(self.shape));
        meta.insert("data_type".into(), json!(self.dtype.name()));
        meta.insert(
            "chunk_grid".into(),
            json!({"name": "regular", "configuration": {"chunk_shape": grid_shape}}),
        );
        meta.insert(
            "chunk_key_encoding".into(),
            json!({"name": "default", "configuration": {"separator": self.separator}}),
        );
        meta.insert("fill_value".into(), self.fill_json());
        meta.insert("codecs".into(), json!(codecs));
        meta.insert("attributes".into(), Value::Object(self.attributes.clone()));
        if let Some(names) = &self.dimension_names {
            meta.insert("dimension_names".into(), json!(names));
        }
        meta.insert("zarr_format".into(), json!(3));
        meta.insert("node_type".into(), json!("array"));
        meta.insert("storage_transformers".into(), json!([]));
        let meta = Value::Object(meta);
        Ok(vec![("zarr.json", meta)])
    }
}

type DynArray = ZarrsArray<dyn ReadableWritableListableStorageTraits>;

/// A zarr array.
#[derive(Clone)]
pub struct ZArray {
    store: Store,
    path: String,
    zarr_format: u8,
    shape: Vec<u64>,
    dtype: DType,
    /// Read/write chunk shape (inner chunks if sharded).
    chunks: Vec<u64>,
    /// Shard shape if sharded.
    shards: Option<Vec<u64>>,
    dimension_names: Option<Vec<String>>,
    compressor: Option<Compressor>,
    inner: Arc<DynArray>,
}

impl std::fmt::Debug for ZArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZArray")
            .field("path", &self.path)
            .field("zarr_format", &self.zarr_format)
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("chunks", &self.chunks)
            .field("shards", &self.shards)
            .finish()
    }
}

impl ZArray {
    /// Write the metadata for a new array and open it.
    pub fn create(store: &Store, path: &str, spec: &ArraySpec) -> Result<ZArray> {
        let path = path.trim_matches('/');
        for (name, doc) in spec.metadata()? {
            store.set_json(&key(path, name), &doc)?;
        }
        ZArray::open(store, path)
    }

    /// Open an existing array.
    pub fn open(store: &Store, path: &str) -> Result<ZArray> {
        let path = path.trim_matches('/').to_string();
        let (zarr_format, shape, dtype, chunks, shards, dimension_names, compressor) =
            if let Some(v) = store.get_json(&key(&path, "zarr.json"))? {
                if v.get("node_type").and_then(|x| x.as_str()) != Some("array") {
                    return Err(Error::type_err(format!("'{path}' is not an array")));
                }
                let shape = json_u64_vec(&v["shape"])?;
                let dtype = DType::parse(v["data_type"].as_str().unwrap_or(""))?;
                let grid = json_u64_vec(&v["chunk_grid"]["configuration"]["chunk_shape"])?;
                let codecs = v["codecs"].as_array().cloned().unwrap_or_default();
                let mut chunks = grid.clone();
                let mut shards = None;
                let mut compressor = None;
                let mut find_compressor = |codecs: &[Value]| {
                    for c in codecs {
                        if let Some(n) = c.get("name").and_then(|x| x.as_str())
                            && matches!(n, "zstd" | "blosc" | "gzip")
                        {
                            compressor = Compressor::from_json(c).ok();
                        }
                    }
                };
                find_compressor(&codecs);
                if let Some(sh) = codecs
                    .iter()
                    .find(|c| c.get("name").and_then(|x| x.as_str()) == Some("sharding_indexed"))
                {
                    chunks = json_u64_vec(&sh["configuration"]["chunk_shape"])?;
                    shards = Some(grid);
                    let inner = sh["configuration"]["codecs"]
                        .as_array()
                        .cloned()
                        .unwrap_or_default();
                    find_compressor(&inner);
                }
                let names = v
                    .get("dimension_names")
                    .and_then(|n| n.as_array())
                    .map(|a| {
                        a.iter()
                            .map(|x| x.as_str().unwrap_or("").to_string())
                            .collect()
                    });
                (3u8, shape, dtype, chunks, shards, names, compressor)
            } else if let Some(v) = store.get_json(&key(&path, ".zarray"))? {
                let shape = json_u64_vec(&v["shape"])?;
                let dtype = DType::parse(v["dtype"].as_str().unwrap_or(""))?;
                let chunks = json_u64_vec(&v["chunks"])?;
                let compressor = Compressor::from_json(&v["compressor"]).ok();
                (2u8, shape, dtype, chunks, None, None, compressor)
            } else {
                return Err(Error::ArrayNotFound(format!(
                    "No array found in store {} at path {path}",
                    store.py_str()
                )));
            };
        let inner = ZarrsArray::open(store.inner().clone(), &format!("/{path}"))?;
        Ok(ZArray {
            store: store.clone(),
            path,
            zarr_format,
            shape,
            dtype,
            chunks,
            shards,
            dimension_names,
            compressor,
            inner: Arc::new(inner),
        })
    }

    pub fn store(&self) -> &Store {
        &self.store
    }
    pub fn path(&self) -> &str {
        &self.path
    }
    pub fn zarr_format(&self) -> u8 {
        self.zarr_format
    }
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    /// The (inner) chunk shape.
    pub fn chunks(&self) -> &[u64] {
        &self.chunks
    }
    /// The shard shape, if sharded.
    pub fn shards(&self) -> Option<&[u64]> {
        self.shards.as_deref()
    }
    pub fn dimension_names(&self) -> Option<&[String]> {
        self.dimension_names.as_deref()
    }
    pub fn compressor(&self) -> Option<&Compressor> {
        self.compressor.as_ref()
    }

    /// User attributes.
    pub fn attrs(&self) -> Result<JsonMap> {
        read_attrs(&self.store, &self.path, self.zarr_format)
    }

    /// Replace all user attributes.
    pub fn set_attrs(&self, attrs: &JsonMap) -> Result<()> {
        write_attrs(&self.store, &self.path, self.zarr_format, attrs)
    }

    /// Shape of a unit of writing: the shard if sharded, otherwise the chunk.
    pub fn write_block_shape(&self) -> &[u64] {
        self.shards.as_deref().unwrap_or(&self.chunks)
    }

    /// Read a region.
    pub fn read(&self, region: &[Range<u64>]) -> Result<NdArray> {
        let subset = ArraySubset::new_with_ranges(region);
        Ok(dtype_dispatch!(self.dtype, T => {
            let a: ndarray::ArrayD<T> = self.inner.retrieve_array_subset(&subset)?;
            crate::array::IntoNdArray::into_nd(a)
        }))
    }

    /// Read the whole array.
    pub fn read_all(&self) -> Result<NdArray> {
        let r: Vec<Range<u64>> = self.shape.iter().map(|&s| 0..s).collect();
        self.read(&r)
    }

    /// Write one block (chunk, or shard if sharded) at grid indices `idx`.
    /// `data` must have the block's shape (cropped at the array edge).
    pub fn write_block(&self, idx: &[u64], data: &NdArray) -> Result<()> {
        let block = self.write_block_shape();
        let full: Vec<usize> = block.iter().map(|&b| b as usize).collect();
        let data = if data.dtype() != self.dtype {
            data.astype(self.dtype)
        } else {
            data.clone()
        };
        // pad edge blocks to the full block shape
        let data = if data.shape() != full {
            let mut padded = NdArray::zeros(self.dtype, &full);
            let r: Vec<Range<usize>> = data.shape().iter().map(|&s| 0..s).collect();
            padded.assign_region(&r, &data)?;
            padded
        } else {
            data
        };
        nd_dispatch!(&data, a => {
            let v: Vec<_> = a.iter().cloned().collect();
            self.inner.store_chunk(idx, &v)?;
        });
        Ok(())
    }

    /// Write a region (aligned to the write grid or not).
    pub fn write_region(&self, region: &[Range<u64>], data: &NdArray) -> Result<()> {
        let subset = ArraySubset::new_with_ranges(region);
        nd_dispatch!(data.astype(self.dtype), a => {
            let a = a.as_standard_layout().into_owned();
            let v: Vec<_> = a.iter().cloned().collect();
            self.inner.store_array_subset(&subset, &v)?;
        });
        Ok(())
    }

    /// Regions of all write blocks, with their grid indices.
    pub fn write_blocks(&self) -> Vec<(Vec<u64>, Region)> {
        let block = self.write_block_shape();
        let counts: Vec<usize> = self
            .shape
            .iter()
            .zip(block)
            .map(|(&s, &b)| if s == 0 { 0 } else { s.div_ceil(b) as usize })
            .collect();
        crate::array::lazy::grid_indices(&counts)
            .into_iter()
            .map(|idx| {
                let region = idx
                    .iter()
                    .enumerate()
                    .map(|(d, &i)| {
                        let s = i as u64 * block[d];
                        s..(s + block[d]).min(self.shape[d])
                    })
                    .collect();
                (idx.iter().map(|&i| i as u64).collect(), region)
            })
            .collect()
    }

    /// A lazy view (`da.from_zarr`), chunked like the array's chunks.
    pub fn to_lazy(&self) -> LazyArray {
        let chunks = blockdims_from_blockshape(&self.shape, &self.chunks);
        LazyArray::from_source(Arc::new(ZarrSource {
            array: self.clone(),
            chunks,
        }))
    }
}

fn json_u64_vec(v: &Value) -> Result<Vec<u64>> {
    v.as_array()
        .ok_or_else(|| Error::value(format!("expected a list of integers, got {v}")))?
        .iter()
        .map(|x| {
            x.as_u64()
                .ok_or_else(|| Error::value(format!("expected integer, got {x}")))
        })
        .collect()
}

struct ZarrSource {
    array: ZArray,
    chunks: Chunks,
}

impl ChunkedSource for ZarrSource {
    fn shape(&self) -> &[u64] {
        &self.array.shape
    }
    fn dtype(&self) -> DType {
        self.array.dtype
    }
    fn chunks(&self) -> &Chunks {
        &self.chunks
    }
    fn read(&self, region: &[Range<u64>]) -> Result<NdArray> {
        self.array.read(region)
    }
    fn describe(&self) -> String {
        format!("from-zarr({})", self.array.path)
    }
}

/// Write a lazy array into a new zarr array (`da.to_zarr`), block by block in parallel.
pub fn store_lazy(data: &LazyArray, array: &ZArray) -> Result<()> {
    use crate::par::*;
    if data.shape() != array.shape() {
        return Err(Error::value(format!(
            "shape mismatch: data {:?} vs array {:?}",
            data.shape(),
            array.shape()
        )));
    }
    array
        .write_blocks()
        .into_par_iter()
        .try_for_each(|(idx, region)| {
            let block = data.read(&region)?;
            array.write_block(&idx, &block)
        })
}

/// Convenience to keep the `Element` import used in generic helpers.
#[allow(dead_code)]
fn _element_marker<T: Element>() {}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(zarr_format: u8, shards: Option<Vec<u64>>) {
        let store = Store::memory();
        let root = Group::open(&store, "", Mode::Write, Some(zarr_format)).unwrap();
        let mut spec = ArraySpec::new(&[5, 7], DType::UInt16, &[2, 3], zarr_format);
        spec.shards = shards;
        let arr = root.create_array("s0", &spec, false).unwrap();
        let data: Vec<f64> = (0..35).map(|v| v as f64).collect();
        let nd = NdArray::from_f64_vec(DType::UInt16, &[5, 7], data).unwrap();
        store_lazy(&LazyArray::from_ndarray(nd.clone()), &arr).unwrap();
        let arr = root.array("s0").unwrap();
        assert_eq!(arr.read_all().unwrap(), nd);
        assert_eq!(arr.chunks(), &[2, 3]);
        assert!(root.contains("s0"));
        assert_eq!(root.array_keys().unwrap(), vec!["s0".to_string()]);
    }

    #[test]
    fn v2_roundtrip() {
        roundtrip(2, None);
    }

    #[test]
    fn v3_roundtrip() {
        roundtrip(3, None);
    }

    #[test]
    fn v3_sharded_roundtrip() {
        roundtrip(3, Some(vec![4, 6]));
    }

    #[test]
    fn attrs() {
        let store = Store::memory();
        let root = Group::open(&store, "", Mode::Write, Some(2)).unwrap();
        root.set_attr("a", json!(1)).unwrap();
        let g = root.require_group("x/y").unwrap();
        assert_eq!(g.name(), "/x/y");
        assert_eq!(root.attrs().unwrap()["a"], 1);
        assert_eq!(root.group_keys().unwrap(), vec!["x".to_string()]);
    }
}
