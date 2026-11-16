//! Zarr groups and arrays (v2 and v3) on top of a [`Store`].
//!
//! Metadata documents (`zarr.json`, `.zgroup`, `.zarray`, `.zattrs`) are read and
//! written directly so that the output matches zarr-python; chunks are
//! encoded and decoded by [`crate::codecs`]. As zarr-python, a chunk holding
//! only the fill value is not stored, nor is such an inner chunk of a shard.

use std::ops::Range;
use std::sync::Arc;

use serde_json::{Value, json};

use crate::array::ChunkedSource;
use crate::array::chunks::{block_region, blockdims_from_blockshape, blocks_under, overlap};
use crate::array::{Chunks, DType, LazyArray, NdArray, Region};
use crate::codecs::Chain;
use crate::dtype_dispatch;
use crate::error::{Error, Result};
use crate::store::{ByteRange, Store, slice_all};

/// JSON object type used for attributes.
pub use crate::pyjson::JsonMap;

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

/// Zarr v3's metadata document: a node's type, its array's description and
/// its attributes.
pub const ZARR_JSON: &str = "zarr.json";
/// Zarr v2's group marker.
pub const ZGROUP: &str = ".zgroup";
/// Zarr v2's array description.
pub const ZARRAY: &str = ".zarray";
/// Zarr v2's attributes.
pub const ZATTRS: &str = ".zattrs";
/// Zarr v2's consolidated metadata (every node's, at the root).
pub const ZMETADATA: &str = ".zmetadata";

/// The metadata documents a node can have, v3's then v2's.
pub const NODE_DOCUMENTS: [&str; 4] = [ZARR_JSON, ZGROUP, ZATTRS, ZARRAY];

/// The last component of a key: the document's name.
pub fn document_name(key: &str) -> &str {
    key.rsplit('/').next().unwrap_or(key)
}

/// Is `name` one of Zarr v2's metadata documents (consolidated included)?
pub fn is_v2_document(name: &str) -> bool {
    matches!(name, ZGROUP | ZARRAY | ZATTRS | ZMETADATA)
}

/// Is `key` a metadata document of either format (not a chunk)?
pub fn is_metadata(key: &str) -> bool {
    let name = document_name(key);
    name == ZARR_JSON || is_v2_document(name)
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
    if let Some(v) = store.get_json(&key(path, ZARR_JSON))? {
        let kind = match v.get("node_type").and_then(|v| v.as_str()) {
            Some("array") => NodeKind::Array,
            _ => NodeKind::Group,
        };
        return Ok(Some((kind, 3)));
    }
    if store.exists(&key(path, ZARRAY)) {
        return Ok(Some((NodeKind::Array, 2)));
    }
    if store.exists(&key(path, ZGROUP)) {
        return Ok(Some((NodeKind::Group, 2)));
    }
    Ok(None)
}

/// Read the user attributes of the node at `path`.
pub fn read_attrs(store: &Store, path: &str, zarr_format: u8) -> Result<JsonMap> {
    if zarr_format == 3 {
        let v = store
            .get_json(&key(path, ZARR_JSON))?
            .unwrap_or(Value::Null);
        Ok(v.get("attributes")
            .and_then(|a| a.as_object())
            .cloned()
            .unwrap_or_default())
    } else {
        let v = store.get_json(&key(path, ZATTRS))?;
        Ok(v.and_then(|v| v.as_object().cloned()).unwrap_or_default())
    }
}

/// Replace the user attributes of the node at `path`.
pub fn write_attrs(store: &Store, path: &str, zarr_format: u8, attrs: &JsonMap) -> Result<()> {
    if zarr_format == 3 {
        let k = key(path, ZARR_JSON);
        let mut v = store
            .get_json(&k)?
            .ok_or_else(|| Error::value(format!("no zarr.json at {path}")))?;
        v.as_object_mut()
            .ok_or_else(|| Error::value("zarr.json is not an object"))?
            .insert("attributes".into(), Value::Object(attrs.clone()));
        store.set_json(&k, &v)
    } else {
        store.set_json(&key(path, ZATTRS), &Value::Object(attrs.clone()))
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
                store.set_json(&key(&path, ZARR_JSON), &meta)?;
            }
            2 => {
                store.set_json(&key(&path, ZGROUP), &json!({"zarr_format": 2}))?;
                store.set_json(&key(&path, ZATTRS), &Value::Object(attrs))?;
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
                (ZARRAY, zarray),
                (ZATTRS, Value::Object(self.attributes.clone())),
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
        Ok(vec![(ZARR_JSON, meta)])
    }
}

/// How an array's chunks are named and stored.
#[derive(Debug)]
struct Layout {
    /// The chunk grid's chunk shape (the shard shape if sharded).
    grid: Vec<u64>,
    /// `0.0` keys (zarr v2, or v3's `v2` encoding) rather than `c/0/0`.
    v2_keys: bool,
    separator: String,
    chain: Chain,
    /// The fill value's little-endian bytes.
    fill: Vec<u8>,
}

impl Layout {
    fn v3(v: &Value, ndim: usize, dtype: DType) -> Result<Layout> {
        let grid = &v["chunk_grid"];
        if grid["name"].as_str() != Some("regular") {
            return Err(Error::Zarr(format!("unsupported chunk grid: {grid}")));
        }
        if v.get("storage_transformers")
            .and_then(Value::as_array)
            .is_some_and(|t| !t.is_empty())
        {
            return Err(Error::Zarr("storage transformers are not supported".into()));
        }
        let encoding = &v["chunk_key_encoding"];
        let v2_keys = match encoding["name"].as_str() {
            Some("default") | None => false,
            Some("v2") => true,
            Some(other) => {
                return Err(Error::Zarr(format!(
                    "unsupported chunk key encoding: {other}"
                )));
            }
        };
        let separator = encoding["configuration"]["separator"]
            .as_str()
            .unwrap_or(if v2_keys { "." } else { "/" })
            .to_string();
        let codecs = v["codecs"].as_array().cloned().unwrap_or_default();
        Ok(Layout {
            grid: json_u64_vec(&grid["configuration"]["chunk_shape"])?,
            v2_keys,
            separator,
            chain: Chain::from_v3(&codecs, ndim)?,
            fill: fill_bytes(dtype, &v["fill_value"])?,
        })
    }

    fn v2(v: &Value, ndim: usize, dtype: DType) -> Result<Layout> {
        let big_endian = v["dtype"].as_str().is_some_and(|d| d.starts_with('>'));
        Ok(Layout {
            grid: json_u64_vec(&v["chunks"])?,
            v2_keys: true,
            separator: v["dimension_separator"].as_str().unwrap_or(".").to_string(),
            chain: Chain::from_v2(
                ndim,
                v["order"].as_str() == Some("F"),
                big_endian,
                &v["filters"],
                &v["compressor"],
            )?,
            fill: fill_bytes(dtype, &v["fill_value"])?,
        })
    }

    /// The key of chunk `idx` of the array at `path`.
    fn key(&self, path: &str, idx: &[u64]) -> String {
        let parts: Vec<String> = idx.iter().map(u64::to_string).collect();
        let name = if self.v2_keys {
            if parts.is_empty() {
                "0".to_string()
            } else {
                parts.join(&self.separator)
            }
        } else {
            std::iter::once("c".to_string())
                .chain(parts)
                .collect::<Vec<_>>()
                .join(&self.separator)
        };
        join_path(path, &name)
    }

    fn filled(&self, dtype: DType, shape: &[u64]) -> NdArray {
        let s: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
        NdArray::filled_le(dtype, &s, &self.fill)
    }
}

/// A fill value's little-endian bytes: a number, `"NaN"`, `"Infinity"`,
/// `"-Infinity"`, a `"0x…"` bit pattern, a bool, `[real, imaginary]` for a
/// complex type, or `null` (zero).
fn fill_bytes(dtype: DType, v: &Value) -> Result<Vec<u8>> {
    let size = dtype.itemsize();
    let bad = || Error::Zarr(format!("invalid fill value for {dtype}: {v}"));
    if v.is_null() {
        return Ok(vec![0; size]);
    }
    if dtype.is_complex() {
        let half = size / 2;
        let part = |x: &Value| -> Result<Vec<u8>> {
            let float = if half == 4 {
                DType::Float32
            } else {
                DType::Float64
            };
            fill_bytes(float, x)
        };
        let (re, im) = match v {
            Value::Array(a) if a.len() == 2 => (part(&a[0])?, part(&a[1])?),
            _ => (part(v)?, vec![0; half]),
        };
        return Ok([re, im].concat());
    }
    let scalar = |x: f64| -> Vec<u8> {
        let mut out = vec![0; size];
        dtype_dispatch!(dtype, T => {
            use crate::array::Element;
            T::from_f64(x).write_le(&mut out)
        });
        out
    };
    match v {
        Value::Bool(b) => Ok(scalar(*b as u8 as f64)),
        Value::Number(n) => Ok(match dtype {
            // exactly, beyond what an f64 holds
            DType::Int64 => n.as_i64().ok_or_else(bad)?.to_le_bytes().to_vec(),
            DType::UInt64 => n.as_u64().ok_or_else(bad)?.to_le_bytes().to_vec(),
            _ => scalar(n.as_f64().ok_or_else(bad)?),
        }),
        Value::String(s) => {
            if let Some(hex) = s.strip_prefix("0x") {
                let bits = u64::from_str_radix(hex, 16).map_err(|_| bad())?;
                return Ok(bits.to_le_bytes()[..size].to_vec());
            }
            let x = match s.as_str() {
                "NaN" => f64::NAN,
                "Infinity" => f64::INFINITY,
                "-Infinity" => f64::NEG_INFINITY,
                _ => s.parse::<f64>().map_err(|_| bad())?,
            };
            Ok(scalar(x))
        }
        _ => Err(bad()),
    }
}

/// The first error, in order, of results computed in any order.
fn first_error<T>(results: Vec<Result<T>>) -> Result<Vec<T>> {
    results.into_iter().collect()
}

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
    layout: Arc<Layout>,
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
        let (zarr_format, shape, dtype, chunks, shards, dimension_names, compressor, layout) =
            if let Some(v) = store.get_json(&key(&path, ZARR_JSON))? {
                if v.get("node_type").and_then(|x| x.as_str()) != Some("array") {
                    return Err(Error::type_err(format!("'{path}' is not an array")));
                }
                let shape = json_u64_vec(&v["shape"])?;
                let dtype = DType::parse(v["data_type"].as_str().unwrap_or(""))?;
                let layout = Layout::v3(&v, shape.len(), dtype)?;
                let grid = layout.grid.clone();
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
                (3u8, shape, dtype, chunks, shards, names, compressor, layout)
            } else if let Some(v) = store.get_json(&key(&path, ZARRAY))? {
                let shape = json_u64_vec(&v["shape"])?;
                let dtype = DType::parse(v["dtype"].as_str().unwrap_or(""))?;
                let layout = Layout::v2(&v, shape.len(), dtype)?;
                let chunks = layout.grid.clone();
                let compressor = Compressor::from_json(&v["compressor"]).ok();
                (2u8, shape, dtype, chunks, None, None, compressor, layout)
            } else {
                return Err(Error::ArrayNotFound(format!(
                    "No array found in store {} at path {path}",
                    store.py_str()
                )));
            };
        if layout.grid.len() != shape.len() || layout.grid.contains(&0) {
            return Err(Error::Zarr(format!(
                "chunk shape {:?} for an array of shape {shape:?}",
                layout.grid
            )));
        }
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
            layout: Arc::new(layout),
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
        use crate::par::*;
        self.check_region(region)?;
        let mut out = self.layout.filled(self.dtype, &region_shape(region));
        if out.is_empty() {
            return Ok(out);
        }
        let pieces = first_error(
            self.blocks_under(region)
                .into_par_iter()
                .map(|idx| self.read_block_part(&idx, region))
                .collect(),
        )?;
        for (dst, part) in pieces.into_iter().flatten() {
            out.assign_region(&dst, &part)?;
        }
        Ok(out)
    }

    /// Read the whole array.
    pub fn read_all(&self) -> Result<NdArray> {
        let r: Vec<Range<u64>> = self.shape.iter().map(|&s| 0..s).collect();
        self.read(&r)
    }

    fn check_region(&self, region: &[Range<u64>]) -> Result<()> {
        if region.len() != self.shape.len()
            || region
                .iter()
                .zip(&self.shape)
                .any(|(r, &s)| r.start > r.end || r.end > s)
        {
            return Err(Error::Zarr(format!(
                "region {region:?} is not within an array of shape {:?}",
                self.shape
            )));
        }
        Ok(())
    }

    /// Grid indices of the blocks (chunks, or shards) under a region.
    fn blocks_under(&self, region: &[Range<u64>]) -> Vec<Vec<u64>> {
        blocks_under(region, &self.layout.grid)
    }

    /// The region of block `idx`, whole (past the array's edge too).
    fn block_region(&self, idx: &[u64]) -> Vec<Range<u64>> {
        block_region(idx, &self.layout.grid)
    }

    /// The part of block `idx` under `region`, and where in the region it
    /// goes; `None` if the block is not stored (the fill value).
    fn read_block_part(
        &self,
        idx: &[u64],
        region: &[Range<u64>],
    ) -> Result<Option<(Vec<Range<usize>>, NdArray)>> {
        let layout = &self.layout;
        let block = self.block_region(idx);
        let (within, dst) = overlap(&block, region);
        let key = layout.key(&self.path, idx);
        if let Some(sharding) = layout.chain.partial_sharding() {
            let within: Vec<Range<u64>> = within
                .iter()
                .map(|r| r.start as u64..r.end as u64)
                .collect();
            let part = if self.store.supports_ranges() {
                let fetch = |ranges: &[ByteRange]| self.store.get_ranges(&key, ranges);
                sharding.read_region(&fetch, &layout.grid, &within, self.dtype, &layout.fill)?
            } else {
                let Some(whole) = self.store.get(&key)? else {
                    return Ok(None);
                };
                let fetch = |ranges: &[ByteRange]| slice_all(&whole, ranges).map(Some);
                sharding.read_region(&fetch, &layout.grid, &within, self.dtype, &layout.fill)?
            };
            return Ok(part.map(|p| (dst, p)));
        }
        let Some(bytes) = self.store.get(&key)? else {
            return Ok(None);
        };
        let chunk = layout
            .chain
            .decode(&bytes, &layout.grid, self.dtype, &layout.fill)
            .map_err(|e| Error::Zarr(format!("{key}: {e}")))?;
        Ok(Some((dst, chunk.slice_region(&within))))
    }

    /// Store block `idx`, whole: not at all if it is only the fill value.
    fn store_block(&self, idx: &[u64], block: &NdArray) -> Result<()> {
        let layout = &self.layout;
        let key = layout.key(&self.path, idx);
        let encoded = if block.all_le(&layout.fill) {
            None
        } else {
            layout.chain.encode(block, &layout.fill)?
        };
        match encoded {
            Some(b) => self.store.set(&key, b),
            None => self.store.erase(&key),
        }
    }

    /// Write one block (chunk, or shard if sharded) at grid indices `idx`.
    /// `data` must have the block's shape (cropped at the array edge).
    pub fn write_block(&self, idx: &[u64], data: &NdArray) -> Result<()> {
        let data = if data.dtype() != self.dtype {
            data.astype(self.dtype)
        } else {
            data.clone()
        };
        // edge blocks are stored whole, the fill value past the edge
        let block = if data.shape_u64() != self.layout.grid {
            let mut padded = self.layout.filled(self.dtype, &self.layout.grid);
            let r: Vec<Range<usize>> = data.shape().iter().map(|&s| 0..s).collect();
            padded.assign_region(&r, &data)?;
            padded
        } else {
            data
        };
        self.store_block(idx, &block)
    }

    /// Write a region (aligned to the write grid or not).
    pub fn write_region(&self, region: &[Range<u64>], data: &NdArray) -> Result<()> {
        use crate::par::*;
        self.check_region(region)?;
        if data.shape_u64() != region_shape(region) {
            return Err(Error::value(format!(
                "data of shape {:?} for region {region:?}",
                data.shape()
            )));
        }
        let data = data.astype(self.dtype);
        first_error(
            self.blocks_under(region)
                .into_par_iter()
                .map(|idx| {
                    let block = self.block_region(&idx);
                    let (within, src) = overlap(&block, region);
                    // what of the block lies inside the array is all written
                    let whole = block
                        .iter()
                        .zip(&within)
                        .zip(&self.shape)
                        .all(|((b, w), &s)| w.start == 0 && w.end as u64 >= s.min(b.end) - b.start);
                    let mut base = if whole {
                        self.layout.filled(self.dtype, &self.layout.grid)
                    } else {
                        self.read_block(&idx)?
                    };
                    base.assign_region(&within, &data.slice_region(&src))?;
                    self.store_block(&idx, &base)
                })
                .collect(),
        )?;
        Ok(())
    }

    /// Block `idx` as stored, whole (the fill value if it is not).
    fn read_block(&self, idx: &[u64]) -> Result<NdArray> {
        let layout = &self.layout;
        match self.store.get(&layout.key(&self.path, idx))? {
            Some(bytes) => layout
                .chain
                .decode(&bytes, &layout.grid, self.dtype, &layout.fill),
            None => Ok(layout.filled(self.dtype, &layout.grid)),
        }
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

fn region_shape(region: &[Range<u64>]) -> Vec<u64> {
    region.iter().map(|r| r.end - r.start).collect()
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

    /// Unaligned writes keep what the blocks held; a block back to the fill
    /// value is erased.
    fn partial_writes(zarr_format: u8, shards: Option<Vec<u64>>) {
        let store = Store::memory();
        let mut spec = ArraySpec::new(&[5, 7], DType::Int32, &[2, 3], zarr_format);
        spec.shards = shards;
        let arr = ZArray::create(&store, "a", &spec).unwrap();
        let ones = NdArray::full(DType::Int32, &[5, 7], 1.0);
        arr.write_region(&[0..5, 0..7], &ones).unwrap();
        let twos = NdArray::full(DType::Int32, &[2, 4], 2.0);
        arr.write_region(&[1..3, 2..6], &twos).unwrap();
        let mut want = ones.clone();
        want.assign_region(&[1..3, 2..6], &twos).unwrap();
        assert_eq!(arr.read_all().unwrap(), want);
        assert_eq!(
            arr.read(&[2..4, 1..6]).unwrap(),
            want.slice_region(&[2..4, 1..6])
        );
        let chunks = |s: &Store| {
            s.inner()
                .list_prefix("a/")
                .unwrap()
                .into_iter()
                .filter(|k| !is_metadata(k))
                .count()
        };
        assert!(chunks(&store) > 0);
        arr.write_region(&[0..5, 0..7], &NdArray::zeros(DType::Int32, &[5, 7]))
            .unwrap();
        assert_eq!(chunks(&store), 0, "every block back to the fill value");
        assert_eq!(
            arr.read_all().unwrap(),
            NdArray::zeros(DType::Int32, &[5, 7])
        );
    }

    #[test]
    fn partial_writes_v2() {
        partial_writes(2, None);
    }

    #[test]
    fn partial_writes_v3() {
        partial_writes(3, None);
    }

    #[test]
    fn partial_writes_v3_sharded() {
        partial_writes(3, Some(vec![4, 6]));
    }

    #[test]
    fn chunk_keys() {
        let layout = |v2_keys: bool, separator: &str| Layout {
            grid: vec![1, 1],
            v2_keys,
            separator: separator.into(),
            chain: Chain::from_v2(2, false, false, &Value::Null, &Value::Null).unwrap(),
            fill: vec![0],
        };
        assert_eq!(layout(false, "/").key("a/0", &[1, 2]), "a/0/c/1/2");
        assert_eq!(layout(false, ".").key("", &[1, 2]), "c.1.2");
        assert_eq!(layout(true, ".").key("a", &[1, 2]), "a/1.2");
        assert_eq!(layout(true, "/").key("a", &[1, 2]), "a/1/2");
        assert_eq!(layout(false, "/").key("a", &[]), "a/c");
        assert_eq!(layout(true, ".").key("a", &[]), "a/0");
    }

    #[test]
    fn fill_values() {
        let f32s = |v: Value| {
            f32::from_le_bytes(fill_bytes(DType::Float32, &v).unwrap().try_into().unwrap())
        };
        assert!(f32s(json!("NaN")).is_nan());
        assert_eq!(f32s(json!("-Infinity")), f32::NEG_INFINITY);
        assert_eq!(f32s(json!("0x3f800000")), 1.0);
        assert_eq!(f32s(json!(2.5)), 2.5);
        assert_eq!(
            fill_bytes(DType::UInt64, &json!(u64::MAX)).unwrap(),
            u64::MAX.to_le_bytes()
        );
        assert_eq!(fill_bytes(DType::Bool, &json!(true)).unwrap(), [1]);
        assert_eq!(fill_bytes(DType::Int16, &Value::Null).unwrap(), [0, 0]);
        let c = fill_bytes(DType::Complex64, &json!([1.0, "NaN"])).unwrap();
        assert_eq!(f32::from_le_bytes(c[..4].try_into().unwrap()), 1.0);
        assert!(f32::from_le_bytes(c[4..].try_into().unwrap()).is_nan());
        assert!(fill_bytes(DType::UInt8, &json!("abc")).is_err());
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
