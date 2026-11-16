//! The Zarr codec pipeline, in pure Rust: how a chunk's elements become the
//! bytes stored, and back.
//!
//! A chain of codecs is a v3 `codecs` list, or a v2 array's `order`, byte order and
//! `compressor`, in the same form: array-to-array codecs (`transpose`), one
//! array-to-bytes codec (`bytes`, or `sharding_indexed` with a chain of its
//! own per inner chunk), then bytes-to-bytes codecs: `blosc` ([`crate::blosc`]),
//! `zstd` (`zstd-rs`), `gzip`, `zlib`, `bz2`, `lz4` (numcodecs' block format: a
//! little-endian `u32` size, then an LZ4 block), `lzma` (xz or "alone" streams)
//! and `crc32c`. Zarr v2 names them by `id`; zarr v3 by `name`, the
//! numcodecs ones as `numcodecs.<id>`.

use bytes::Bytes;
use serde_json::{Value, json};

use crate::array::chunks::{block_region, blocks_under, overlap};
use crate::array::{DType, NdArray};
use crate::error::{Error, Result};
use crate::pyjson::JsonMap;
use crate::store::{ByteRange, slice_all};

fn codec_err(e: impl std::fmt::Display) -> Error {
    Error::Zarr(e.to_string())
}

fn unsupported(what: impl std::fmt::Display) -> Error {
    Error::Zarr(format!("unsupported codec: {what}"))
}

// ---------------------------------------------------------- bytes to bytes

/// Blosc's `shuffle`: numcodecs' integer (-1 auto, 0 none, 1 byte, 2 bit) or
/// zarr v3's name.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum BloscShuffle {
    Int(i64),
    Name(String),
}

/// A bytes-to-bytes codec.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum BytesCodec {
    Blosc {
        compressor: crate::blosc::Compressor,
        clevel: i64,
        shuffle: BloscShuffle,
        typesize: Option<usize>,
        blocksize: usize,
    },
    Zstd {
        level: i64,
        checksum: bool,
    },
    Gzip {
        level: u32,
    },
    Zlib {
        level: u32,
    },
    Bz2 {
        level: u32,
    },
    Lz4,
    /// 1 = xz, 2 = "alone" (.lzma).
    Lzma {
        format: i64,
    },
    Crc32c,
}

fn int(cfg: &JsonMap, key: &str, default: i64) -> i64 {
    cfg.get(key).and_then(Value::as_i64).unwrap_or(default)
}

impl BytesCodec {
    /// From a codec's name (v3, `numcodecs.` or not) or id (v2) and its
    /// configuration.
    pub(crate) fn parse(name: &str, cfg: &JsonMap) -> Result<BytesCodec> {
        let level = |default| int(cfg, "level", default).clamp(0, 9) as u32;
        Ok(match name.trim_start_matches("numcodecs.") {
            "blosc" => {
                let cname = cfg.get("cname").and_then(Value::as_str).unwrap_or("lz4");
                BytesCodec::Blosc {
                    compressor: crate::blosc::Compressor::from_name(cname).ok_or_else(|| {
                        Error::Zarr(format!("blosc compressor `{cname}` is not known"))
                    })?,
                    clevel: int(cfg, "clevel", 5),
                    shuffle: match cfg.get("shuffle") {
                        Some(Value::String(s)) => BloscShuffle::Name(s.clone()),
                        Some(v) => BloscShuffle::Int(v.as_i64().unwrap_or(1)),
                        None => BloscShuffle::Int(1),
                    },
                    typesize: cfg
                        .get("typesize")
                        .and_then(Value::as_u64)
                        .map(|t| t as usize),
                    blocksize: int(cfg, "blocksize", 0) as usize,
                }
            }
            "zstd" => BytesCodec::Zstd {
                level: int(cfg, "level", 0),
                checksum: cfg
                    .get("checksum")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            },
            "gzip" => BytesCodec::Gzip { level: level(1) },
            "zlib" => BytesCodec::Zlib { level: level(1) },
            "bz2" => BytesCodec::Bz2 {
                level: level(1).max(1),
            },
            "lz4" => BytesCodec::Lz4,
            "lzma" => {
                if cfg.get("filters").is_some_and(|f| !f.is_null()) {
                    return Err(unsupported("lzma filters"));
                }
                match int(cfg, "format", 1) {
                    f @ (1 | 2) => BytesCodec::Lzma { format: f },
                    f => return Err(unsupported(format!("lzma format {f}"))),
                }
            }
            "crc32c" => BytesCodec::Crc32c,
            other => return Err(unsupported(other)),
        })
    }

    /// A v2 `compressor` (`null` for none).
    fn from_v2(v: &Value) -> Result<Option<BytesCodec>> {
        if v.is_null() {
            return Ok(None);
        }
        let cfg = v
            .as_object()
            .ok_or_else(|| Error::Zarr(format!("invalid compressor: {v}")))?;
        let id = cfg
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::Zarr(format!("invalid compressor: {v}")))?;
        BytesCodec::parse(id, cfg).map(Some)
    }

    fn blosc_shuffle(shuffle: &BloscShuffle, typesize: usize) -> crate::blosc::Shuffle {
        use crate::blosc::Shuffle;
        match shuffle {
            BloscShuffle::Int(0) => Shuffle::None,
            BloscShuffle::Int(2) => Shuffle::Bit,
            // numcodecs' AUTOSHUFFLE: bits for single bytes, else bytes
            BloscShuffle::Int(-1) if typesize == 1 => Shuffle::Bit,
            BloscShuffle::Int(_) => Shuffle::Byte,
            BloscShuffle::Name(n) if n == "noshuffle" => Shuffle::None,
            BloscShuffle::Name(n) if n == "bitshuffle" => Shuffle::Bit,
            BloscShuffle::Name(_) => Shuffle::Byte,
        }
    }

    fn encode(&self, data: Vec<u8>) -> Result<Vec<u8>> {
        use std::io::Write;
        Ok(match self {
            BytesCodec::Blosc {
                compressor,
                clevel,
                shuffle,
                typesize,
                blocksize,
            } => {
                // zarr v3 names the element size; a v2 compressor is not told
                // the array's: whole bytes then
                let typesize = typesize.unwrap_or(1);
                crate::blosc::compress(
                    &data,
                    *compressor,
                    *clevel as i32,
                    Self::blosc_shuffle(shuffle, typesize),
                    typesize,
                    *blocksize,
                )
                .map_err(codec_err)?
            }
            BytesCodec::Zstd { level, checksum } => {
                // level 0 is zstd's default, 3
                let level = if *level == 0 {
                    3
                } else {
                    (*level).clamp(1, zstd_rs::MAX_LEVEL as i64) as i32
                };
                let cfg = zstd_rs::CompressionConfig {
                    level,
                    checksum: *checksum,
                    ..zstd_rs::CompressionConfig::DEFAULT
                };
                let mut out = Vec::new();
                zstd_rs::Compressor::new(cfg)
                    .and_then(|mut c| c.compress(&data, None, &mut out))
                    .map_err(codec_err)?;
                out
            }
            BytesCodec::Gzip { level } => {
                let mut e =
                    flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(*level));
                e.write_all(&data)?;
                e.finish()?
            }
            BytesCodec::Zlib { level } => {
                let mut e =
                    flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(*level));
                e.write_all(&data)?;
                e.finish()?
            }
            BytesCodec::Bz2 { level } => {
                let mut e =
                    bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::new(*level));
                e.write_all(&data)?;
                e.finish()?
            }
            BytesCodec::Lz4 => lz4_flex::block::compress_prepend_size(&data),
            BytesCodec::Lzma { format } => {
                let mut input = std::io::Cursor::new(&data[..]);
                let mut out = vec![];
                match format {
                    2 => lzma_rs::lzma_compress(&mut input, &mut out),
                    _ => lzma_rs::xz_compress(&mut input, &mut out),
                }
                .map_err(codec_err)?;
                out
            }
            BytesCodec::Crc32c => {
                let crc = crc32c::crc32c(&data);
                let mut out = data;
                out.extend_from_slice(&crc.to_le_bytes());
                out
            }
        })
    }

    /// `size`: the decoded length, if known.
    fn decode(&self, data: &[u8], size: Option<usize>) -> Result<Vec<u8>> {
        use std::io::Read;
        let mut out = Vec::with_capacity(size.unwrap_or(0));
        match self {
            BytesCodec::Blosc { .. } => return crate::blosc::decompress(data).map_err(codec_err),
            BytesCodec::Zstd { .. } => {
                // the decoded size when known; else the frame's own, up to 4 GiB
                let limit = size.unwrap_or(u32::MAX as usize);
                zstd_rs::Decompressor::new()
                    .decompress(data, None, limit, &mut out)
                    .map_err(codec_err)?;
            }
            BytesCodec::Gzip { .. } => {
                flate2::read::MultiGzDecoder::new(data).read_to_end(&mut out)?;
            }
            BytesCodec::Zlib { .. } => {
                flate2::read::ZlibDecoder::new(data).read_to_end(&mut out)?;
            }
            BytesCodec::Bz2 { .. } => {
                bzip2::read::MultiBzDecoder::new(data).read_to_end(&mut out)?;
            }
            BytesCodec::Lz4 => {
                return lz4_flex::block::decompress_size_prepended(data).map_err(codec_err);
            }
            BytesCodec::Lzma { format } => {
                let mut input = std::io::BufReader::new(data);
                match format {
                    2 => lzma_rs::lzma_decompress(&mut input, &mut out),
                    _ => lzma_rs::xz_decompress(&mut input, &mut out),
                }
                .map_err(codec_err)?;
            }
            BytesCodec::Crc32c => {
                let Some(n) = data.len().checked_sub(4) else {
                    return Err(Error::Zarr("crc32c: shorter than its checksum".into()));
                };
                let want = u32::from_le_bytes(data[n..].try_into().unwrap());
                if crc32c::crc32c(&data[..n]) != want {
                    return Err(Error::Zarr("crc32c checksum mismatch".into()));
                }
                out.extend_from_slice(&data[..n]);
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------- the chain

/// The array-to-bytes codec.
#[derive(Debug, Clone, PartialEq)]
enum ArrayToBytes {
    Bytes { big_endian: bool },
    Sharding(Box<Sharding>),
}

/// A chunk's codecs, in order.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Chain {
    /// `transpose` orders, applied first.
    transposes: Vec<Vec<usize>>,
    a2b: ArrayToBytes,
    b2b: Vec<BytesCodec>,
}

fn json_usizes(v: &Value) -> Option<Vec<usize>> {
    v.as_array()?
        .iter()
        .map(|x| x.as_u64().map(|x| x as usize))
        .collect()
}

/// An element's bytes swapped, unit by unit (each part of a complex value).
fn swap(data: &mut [u8], unit: usize) {
    if unit > 1 {
        data.chunks_exact_mut(unit).for_each(<[u8]>::reverse);
    }
}

impl Chain {
    /// A v3 `codecs` list for an array of `ndim` dimensions.
    pub(crate) fn from_v3(codecs: &[Value], ndim: usize) -> Result<Chain> {
        let mut transposes = vec![];
        let mut a2b = None;
        let mut b2b = vec![];
        for c in codecs {
            let (name, cfg) = match c {
                Value::String(n) => (n.as_str(), JsonMap::new()),
                _ => (
                    c.get("name")
                        .and_then(Value::as_str)
                        .ok_or_else(|| Error::Zarr(format!("invalid codec: {c}")))?,
                    c.get("configuration")
                        .and_then(Value::as_object)
                        .cloned()
                        .unwrap_or_default(),
                ),
            };
            match name {
                "transpose" => {
                    if a2b.is_some() {
                        return Err(unsupported("transpose after an array-to-bytes codec"));
                    }
                    let order = match cfg.get("order") {
                        Some(Value::String(s)) if s == "C" => (0..ndim).collect(),
                        Some(Value::String(s)) if s == "F" => (0..ndim).rev().collect(),
                        Some(v) => json_usizes(v)
                            .ok_or_else(|| Error::Zarr(format!("invalid transpose order: {v}")))?,
                        None => return Err(Error::Zarr("transpose without an order".into())),
                    };
                    let mut sorted = order.clone();
                    sorted.sort();
                    if sorted != (0..ndim).collect::<Vec<_>>() {
                        return Err(Error::Zarr(format!(
                            "transpose order {order:?} is not a permutation of {ndim} axes"
                        )));
                    }
                    transposes.push(order);
                }
                "bytes" | "sharding_indexed" => {
                    if a2b.is_some() {
                        return Err(unsupported("two array-to-bytes codecs"));
                    }
                    a2b = Some(if name == "bytes" {
                        ArrayToBytes::Bytes {
                            big_endian: cfg.get("endian").and_then(Value::as_str) == Some("big"),
                        }
                    } else {
                        ArrayToBytes::Sharding(Box::new(Sharding::parse(&cfg, ndim)?))
                    });
                }
                _ => {
                    if a2b.is_none() {
                        return Err(unsupported(format!(
                            "{name} before an array-to-bytes codec"
                        )));
                    }
                    b2b.push(BytesCodec::parse(name, &cfg)?);
                }
            }
        }
        Ok(Chain {
            transposes,
            a2b: a2b.ok_or_else(|| Error::Zarr("no array-to-bytes codec".into()))?,
            b2b,
        })
    }

    /// A v2 array's: `order` `"F"`, a big-endian `dtype`, its `filters` (none
    /// supported) and `compressor`.
    pub(crate) fn from_v2(
        ndim: usize,
        fortran: bool,
        big_endian: bool,
        filters: &Value,
        compressor: &Value,
    ) -> Result<Chain> {
        if filters.as_array().is_some_and(|f| !f.is_empty()) {
            return Err(unsupported(format!("filters {filters}")));
        }
        Ok(Chain {
            transposes: if fortran && ndim > 1 {
                vec![(0..ndim).rev().collect()]
            } else {
                vec![]
            },
            a2b: ArrayToBytes::Bytes { big_endian },
            b2b: BytesCodec::from_v2(compressor)?.into_iter().collect(),
        })
    }

    /// The shape the array-to-bytes codec is given.
    fn encoded_shape(&self, shape: &[u64]) -> Vec<u64> {
        self.transposes.iter().fold(shape.to_vec(), |s, order| {
            order.iter().map(|&i| s[i]).collect()
        })
    }

    /// Sharding, if a shard's chunks can be read one by one: nothing before
    /// it or after it.
    pub(crate) fn partial_sharding(&self) -> Option<&Sharding> {
        match &self.a2b {
            ArrayToBytes::Sharding(s) if self.transposes.is_empty() && self.b2b.is_empty() => {
                Some(s)
            }
            _ => None,
        }
    }

    /// A chunk of `shape` from its stored bytes.
    pub(crate) fn decode(
        &self,
        data: &[u8],
        shape: &[u64],
        dtype: DType,
        fill: &[u8],
    ) -> Result<NdArray> {
        let inner = self.encoded_shape(shape);
        let n: u64 = inner.iter().product();
        let size = n as usize * dtype.itemsize();
        let mut bytes: Option<Vec<u8>> = None;
        for (i, c) in self.b2b.iter().enumerate().rev() {
            let src = bytes.as_deref().unwrap_or(data);
            // the codec next to `bytes` gives back the elements
            let hint = (i == 0 && matches!(self.a2b, ArrayToBytes::Bytes { .. })).then_some(size);
            bytes = Some(c.decode(src, hint)?);
        }
        let data = bytes.as_deref().unwrap_or(data);
        let usize_shape: Vec<usize> = inner.iter().map(|&s| s as usize).collect();
        let mut a = match &self.a2b {
            ArrayToBytes::Bytes { big_endian } => {
                if *big_endian && dtype.swap_size() > 1 {
                    let mut owned = data.to_vec();
                    swap(&mut owned, dtype.swap_size());
                    NdArray::from_le_bytes(dtype, &usize_shape, &owned)?
                } else {
                    NdArray::from_le_bytes(dtype, &usize_shape, data)?
                }
            }
            ArrayToBytes::Sharding(s) => s.decode_all(data, &inner, dtype, fill)?,
        };
        for order in self.transposes.iter().rev() {
            let mut inverse = vec![0; order.len()];
            for (i, &o) in order.iter().enumerate() {
                inverse[o] = i;
            }
            a = a.permuted(&inverse);
        }
        Ok(a)
    }

    /// A chunk's bytes to store; `None` if a shard holds only `fill`.
    pub(crate) fn encode(&self, chunk: &NdArray, fill: &[u8]) -> Result<Option<Vec<u8>>> {
        let mut a = std::borrow::Cow::Borrowed(chunk);
        for order in &self.transposes {
            a = std::borrow::Cow::Owned(a.permuted(order));
        }
        let mut data = match &self.a2b {
            ArrayToBytes::Bytes { big_endian } => {
                let mut b = a.to_le_bytes();
                if *big_endian {
                    swap(&mut b, a.dtype().swap_size());
                }
                b
            }
            ArrayToBytes::Sharding(s) => match s.encode(&a, fill)? {
                Some(b) => b,
                None => return Ok(None),
            },
        };
        for c in &self.b2b {
            data = c.encode(data)?;
        }
        Ok(Some(data))
    }
}

// ---------------------------------------------------------- sharding

/// `sharding_indexed`: a shard's inner chunks, each encoded with `inner`,
/// one after another, and an index of their offsets and lengths
/// (`u64::MAX` twice for a chunk not stored) encoded with `index`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Sharding {
    pub(crate) chunk_shape: Vec<u64>,
    inner: Chain,
    index: Chain,
    index_at_end: bool,
}

const EMPTY: u64 = u64::MAX;

/// Reads parts of a shard: `None` if it is not there.
pub(crate) type Fetch<'a> = dyn Fn(&[ByteRange]) -> Result<Option<Vec<Bytes>>> + 'a;

impl Sharding {
    fn parse(cfg: &JsonMap, ndim: usize) -> Result<Sharding> {
        let chunk_shape: Vec<u64> = cfg
            .get("chunk_shape")
            .and_then(json_usizes)
            .ok_or_else(|| Error::Zarr("sharding without a chunk_shape".into()))?
            .into_iter()
            .map(|c| c as u64)
            .collect();
        if chunk_shape.len() != ndim || chunk_shape.contains(&0) {
            return Err(Error::Zarr(format!(
                "sharding chunk_shape {chunk_shape:?} for {ndim} dimensions"
            )));
        }
        let list = |key: &str, default: Value| -> Vec<Value> {
            cfg.get(key)
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_else(|| default.as_array().cloned().unwrap_or_default())
        };
        let index_codecs = list(
            "index_codecs",
            json!([{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c"}]),
        );
        let index = Chain::from_v3(&index_codecs, ndim + 1)?;
        if !matches!(index.a2b, ArrayToBytes::Bytes { .. })
            || index.b2b.iter().any(|c| *c != BytesCodec::Crc32c)
        {
            return Err(unsupported(format!(
                "index_codecs {}",
                Value::Array(index_codecs)
            )));
        }
        Ok(Sharding {
            chunk_shape,
            inner: Chain::from_v3(&list("codecs", json!([{"name": "bytes"}])), ndim)?,
            index,
            index_at_end: cfg.get("index_location").and_then(Value::as_str) != Some("start"),
        })
    }

    /// Inner chunks per dimension of a shard of `shape`.
    fn grid(&self, shape: &[u64]) -> Result<Vec<u64>> {
        shape
            .iter()
            .zip(&self.chunk_shape)
            .map(|(&s, &c)| {
                if s % c == 0 {
                    Ok(s / c)
                } else {
                    Err(Error::Zarr(format!(
                        "shard shape {shape:?} is not divisible by chunk shape {:?}",
                        self.chunk_shape
                    )))
                }
            })
            .collect()
    }

    /// The index's stored length, for `n` chunks.
    fn index_size(&self, n: u64) -> usize {
        (n * 16) as usize + 4 * self.index.b2b.len()
    }

    /// The index's place in a shard `len` bytes long.
    fn index_range(&self, n: u64) -> ByteRange {
        let size = self.index_size(n) as u64;
        if self.index_at_end {
            ByteRange::Suffix(size)
        } else {
            ByteRange::Range(0, size)
        }
    }

    /// `(offset, length)` of each inner chunk, in C order.
    fn decode_index(&self, data: &[u8], grid: &[u64]) -> Result<Vec<(u64, u64)>> {
        let mut shape = grid.to_vec();
        shape.push(2);
        let index = self.index.decode(data, &shape, DType::UInt64, &[0; 8])?;
        let flat = index.to_le_bytes();
        Ok(flat
            .as_chunks::<16>()
            .0
            .iter()
            .map(|e| {
                (
                    u64::from_le_bytes(e[..8].try_into().unwrap()),
                    u64::from_le_bytes(e[8..].try_into().unwrap()),
                )
            })
            .collect())
    }

    /// A whole shard of `shape` from its bytes.
    fn decode_all(&self, data: &[u8], shape: &[u64], dtype: DType, fill: &[u8]) -> Result<NdArray> {
        let whole: Vec<std::ops::Range<u64>> = shape.iter().map(|&s| 0..s).collect();
        let data = Bytes::copy_from_slice(data);
        let fetch = |ranges: &[ByteRange]| slice_all(&data, ranges).map(Some);
        Ok(self
            .read_region(&fetch, shape, &whole, dtype, fill)?
            .unwrap_or_else(|| {
                let s: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
                NdArray::filled_le(dtype, &s, fill)
            }))
    }

    /// `region` of a shard of `shape`, reading only the index and the chunks
    /// under it through `fetch`; `None` if the shard is not there.
    pub(crate) fn read_region(
        &self,
        fetch: &Fetch,
        shape: &[u64],
        region: &[std::ops::Range<u64>],
        dtype: DType,
        fill: &[u8],
    ) -> Result<Option<NdArray>> {
        let grid = self.grid(shape)?;
        let n: u64 = grid.iter().product();
        let Some(index) = fetch(&[self.index_range(n)])? else {
            return Ok(None);
        };
        let index = self.decode_index(&index[0], &grid)?;
        // the inner chunks under the region, and where their bytes are
        let mut wanted = vec![];
        for idx in blocks_under(region, &self.chunk_shape) {
            let flat = idx.iter().zip(&grid).fold(0, |acc, (&i, &g)| acc * g + i);
            let (offset, len) = index[flat as usize];
            if offset != EMPTY || len != EMPTY {
                wanted.push((idx, ByteRange::Range(offset, offset + len)));
            }
        }
        let out_shape: Vec<usize> = region.iter().map(|r| (r.end - r.start) as usize).collect();
        let mut out = NdArray::filled_le(dtype, &out_shape, fill);
        if wanted.is_empty() {
            return Ok(Some(out));
        }
        let ranges: Vec<ByteRange> = wanted.iter().map(|(_, r)| *r).collect();
        let Some(pieces) = fetch(&ranges)? else {
            return Ok(None);
        };
        for ((idx, _), piece) in wanted.iter().zip(pieces) {
            let chunk = self.inner.decode(&piece, &self.chunk_shape, dtype, fill)?;
            let (src, dst) = overlap(&block_region(idx, &self.chunk_shape), region);
            out.assign_region(&dst, &chunk.slice_region(&src))?;
        }
        Ok(Some(out))
    }

    /// A shard's bytes; `None` if every inner chunk is `fill`.
    fn encode(&self, shard: &NdArray, fill: &[u8]) -> Result<Option<Vec<u8>>> {
        let shape = shard.shape_u64();
        let grid = self.grid(&shape)?;
        let n: u64 = grid.iter().product();
        let index_size = self.index_size(n) as u64;
        let base = if self.index_at_end { 0 } else { index_size };
        let counts: Vec<usize> = grid.iter().map(|&g| g as usize).collect();
        let mut data = vec![];
        let mut index = Vec::with_capacity(n as usize * 16);
        let mut stored = false;
        for idx in crate::array::lazy::grid_indices(&counts) {
            let idx: Vec<u64> = idx.iter().map(|&i| i as u64).collect();
            let region: Vec<std::ops::Range<usize>> = block_region(&idx, &self.chunk_shape)
                .iter()
                .map(|r| r.start as usize..r.end as usize)
                .collect();
            let chunk = shard.slice_region(&region);
            let encoded = if chunk.all_le(fill) {
                None
            } else {
                self.inner.encode(&chunk, fill)?
            };
            let (offset, len) = match encoded {
                Some(b) => {
                    let at = (base + data.len() as u64, b.len() as u64);
                    data.extend_from_slice(&b);
                    stored = true;
                    at
                }
                None => (EMPTY, EMPTY),
            };
            index.extend_from_slice(&offset.to_le_bytes());
            index.extend_from_slice(&len.to_le_bytes());
        }
        if !stored {
            return Ok(None);
        }
        let mut index_shape: Vec<usize> = counts;
        index_shape.push(2);
        let index = NdArray::from_le_bytes(DType::UInt64, &index_shape, &index)?;
        let index = self
            .index
            .encode(&index, &[0; 8])?
            .expect("an index is always stored");
        Ok(Some(if self.index_at_end {
            data.extend_from_slice(&index);
            data
        } else {
            let mut out = index;
            out.extend_from_slice(&data);
            out
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(shape: &[usize]) -> NdArray {
        let n: usize = shape.iter().product();
        NdArray::from_f64_vec(DType::UInt16, shape, (0..n).map(|v| v as f64).collect()).unwrap()
    }

    #[test]
    fn every_bytes_codec_round_trips() {
        let data: Vec<u8> = (0..4000u32).flat_map(|v| (v % 251).to_le_bytes()).collect();
        for c in [
            json!({"id": "blosc", "cname": "zstd", "clevel": 5, "shuffle": 1}),
            json!({"id": "zstd", "level": 3}),
            json!({"id": "gzip", "level": 5}),
            json!({"id": "zlib", "level": 1}),
            json!({"id": "bz2", "level": 9}),
            json!({"id": "lz4", "acceleration": 1}),
            json!({"id": "lzma", "format": 1}),
            json!({"id": "lzma", "format": 2}),
            json!({"id": "crc32c"}),
        ] {
            let codec = BytesCodec::from_v2(&c).unwrap().unwrap();
            let encoded = codec.encode(data.clone()).unwrap();
            assert_eq!(codec.decode(&encoded, None).unwrap(), data, "{c}");
        }
    }

    #[test]
    fn a_corrupt_crc32c_is_caught() {
        let codec = BytesCodec::Crc32c;
        let mut encoded = codec.encode(b"chunk".to_vec()).unwrap();
        encoded[0] ^= 1;
        assert!(codec.decode(&encoded, None).is_err());
    }

    #[test]
    fn transpose_and_big_endian_round_trip() {
        let codecs = json!([
            {"name": "transpose", "configuration": {"order": [1, 0, 2]}},
            {"name": "bytes", "configuration": {"endian": "big"}},
            {"name": "gzip", "configuration": {"level": 1}}
        ]);
        let chain = Chain::from_v3(codecs.as_array().unwrap(), 3).unwrap();
        let a = arr(&[2, 3, 4]);
        let encoded = chain.encode(&a, &[0, 0]).unwrap().unwrap();
        assert_eq!(
            chain
                .decode(&encoded, &[2, 3, 4], DType::UInt16, &[0, 0])
                .unwrap(),
            a
        );
        // big endian, the transposed array's C order: t[0, 1, 0] = a[1, 0, 0]
        let raw = flate2::read::GzDecoder::new(&encoded[..]);
        let raw: Vec<u8> = std::io::Read::bytes(raw).map(|b| b.unwrap()).collect();
        assert_eq!(&raw[..4], &[0, 0, 0, 1]);
        assert_eq!(&raw[8..10], &[0, 12]);
    }

    #[test]
    fn a_v2_fortran_order_is_a_transpose() {
        let chain = Chain::from_v2(2, true, false, &Value::Null, &Value::Null).unwrap();
        let a = arr(&[2, 3]);
        let encoded = chain.encode(&a, &[0, 0]).unwrap().unwrap();
        let firsts: Vec<u16> = encoded
            .as_chunks::<2>()
            .0
            .iter()
            .take(3)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        assert_eq!(firsts, [0, 3, 1]);
        assert_eq!(
            chain
                .decode(&encoded, &[2, 3], DType::UInt16, &[0, 0])
                .unwrap(),
            a
        );
    }

    fn sharding(location: &str) -> Chain {
        let codecs = json!([{"name": "sharding_indexed", "configuration": {
            "chunk_shape": [2, 2],
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "zstd", "configuration": {"level": 0, "checksum": false}}],
            "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c"}],
            "index_location": location,
        }}]);
        Chain::from_v3(codecs.as_array().unwrap(), 2).unwrap()
    }

    #[test]
    fn shards_round_trip_and_leave_out_empty_chunks() {
        for location in ["end", "start"] {
            let chain = sharding(location);
            let mut a = NdArray::zeros(DType::UInt16, &[4, 4]);
            a.assign_region(&[2..4, 0..2], &arr(&[2, 2])).unwrap();
            let encoded = chain.encode(&a, &[0, 0]).unwrap().unwrap();
            assert_eq!(
                chain
                    .decode(&encoded, &[4, 4], DType::UInt16, &[0, 0])
                    .unwrap(),
                a
            );
            let s = chain.partial_sharding().unwrap();
            let index = match location {
                "end" => &encoded[encoded.len() - 68..],
                _ => &encoded[..68],
            };
            let entries = s.decode_index(index, &[2, 2]).unwrap();
            let stored: Vec<bool> = entries.iter().map(|e| e.0 != EMPTY).collect();
            assert_eq!(stored, [false, false, true, false], "{location}");
            assert!(
                chain
                    .encode(&NdArray::zeros(DType::UInt16, &[4, 4]), &[0, 0])
                    .unwrap()
                    .is_none()
            );
        }
    }

    #[test]
    fn a_shard_region_reads_only_its_chunks() {
        let chain = sharding("end");
        let a = arr(&[4, 4]);
        let encoded = Bytes::from(chain.encode(&a, &[0, 0]).unwrap().unwrap());
        let asked = std::sync::Mutex::new(vec![]);
        let fetch = |ranges: &[ByteRange]| -> Result<Option<Vec<Bytes>>> {
            asked.lock().unwrap().push(ranges.len());
            slice_all(&encoded, ranges).map(Some)
        };
        let s = chain.partial_sharding().unwrap();
        let got = s
            .read_region(&fetch, &[4, 4], &[1..2, 1..3], DType::UInt16, &[0, 0])
            .unwrap()
            .unwrap();
        assert_eq!(got, a.slice_region(&[1..2, 1..3]));
        assert_eq!(*asked.lock().unwrap(), [1, 2], "the index, then two chunks");
    }
}
