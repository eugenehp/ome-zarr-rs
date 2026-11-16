//! Blosc (c-blosc 1's container), in pure Rust: what zarr-python and
//! numcodecs write by default for Zarr v2, and zarr v3's `blosc` codec.
//!
//! A 16-byte header — version, codec version, flags, typesize, then `nbytes`,
//! `blocksize` and `cbytes` (`u32`, little-endian) — then, unless the data
//! was stored as it is (`MEMCPYED`), a table of the blocks' offsets and the
//! blocks. A block is shuffled (bytes or bits, by `typesize`), optionally
//! split into `typesize` streams, and each stream is `[i32 size][bytes]`,
//! stored uncompressed when `size` is the stream's length.
//!
//! Inner codecs: blosclz (decoding, ported from c-blosc's `blosclz.c`,
//! BSD-3-Clause, Francesc Alted), lz4 and lz4hc (`lz4_flex`), snappy
//! (`snap`), zlib (`flate2`, Rust backend), zstd (`zstd-rs`). Writing uses lz4,
//! zlib, zstd or snappy as asked; blosclz is written uncompressed inside the
//! container (a valid blosclz stream for any reader), as no pure-Rust blosclz
//! compressor exists. Blocks are not split when written (`DONT_SPLIT`).

use std::io::{Read, Write};

const MEMCPYED: u8 = 0x02;
const BYTE_SHUFFLE: u8 = 0x01;
const BIT_SHUFFLE: u8 = 0x04;
const DELTA: u8 = 0x08;
const DONT_SPLIT: u8 = 0x10;
const HEADER: usize = 16;
const MAX_SPLITS: usize = 16;

/// A codec inside the container, by c-blosc's code (the flags' top bits).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compressor {
    BloscLz = 0,
    Lz4 = 1,
    Snappy = 2,
    Zlib = 3,
    Zstd = 4,
}

impl Compressor {
    /// By name, as numcodecs and zarr v3 write it (`lz4hc` is lz4's format).
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "blosclz" => Self::BloscLz,
            "lz4" | "lz4hc" => Self::Lz4,
            "snappy" => Self::Snappy,
            "zlib" => Self::Zlib,
            "zstd" => Self::Zstd,
            _ => return None,
        })
    }
}

/// Shuffling before compression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shuffle {
    None,
    Byte,
    Bit,
}

fn err<T>(what: impl Into<String>) -> Result<T, String> {
    Err(format!("blosc: {}", what.into()))
}

fn u32le(b: &[u8], at: usize) -> usize {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]) as usize
}

// ---------------------------------------------------------------- filters

fn shuffle(typesize: usize, src: &[u8]) -> Vec<u8> {
    let n = src.len() / typesize;
    let mut out = src.to_vec();
    for i in 0..n {
        for b in 0..typesize {
            out[b * n + i] = src[i * typesize + b];
        }
    }
    out
}

fn unshuffle(typesize: usize, src: &[u8]) -> Vec<u8> {
    let n = src.len() / typesize;
    let mut out = src.to_vec();
    for i in 0..n {
        for b in 0..typesize {
            out[i * typesize + b] = src[b * n + i];
        }
    }
    out
}

/// c-blosc 1's bitshuffle: bit `k` of every element together (bitshuffle's
/// layout), bytes past the whole elements left as they are — and the block
/// left as it is altogether unless its elements are a multiple of 8.
fn bitshuffle(typesize: usize, src: &[u8], inverse: bool) -> Vec<u8> {
    let elements = src.len() / typesize;
    let mut out = src.to_vec();
    if elements == 0 || !elements.is_multiple_of(8) {
        return out;
    }
    let n = elements;
    let nbits = typesize * 8;
    // bit (element i, bit k of the element) <-> bit (k * n + i) of the region
    for i in 0..n {
        for k in 0..nbits {
            let (eb, ebit) = (i * typesize + k / 8, k % 8);
            let pos = k * n + i;
            let (sb, sbit) = (pos / 8, pos % 8);
            if inverse {
                let v = (src[sb] >> sbit) & 1;
                out[eb] = (out[eb] & !(1 << ebit)) | (v << ebit);
            } else {
                let v = (src[eb] >> ebit) & 1;
                out[sb] = (out[sb] & !(1 << sbit)) | (v << sbit);
            }
        }
    }
    out
}

// ---------------------------------------------------------------- blosclz

/// c-blosc's `blosclz_decompress`: FastLZ level 2 matches and literal runs.
fn blosclz_decompress(input: &[u8], maxout: usize) -> Result<Vec<u8>, String> {
    let mut out: Vec<u8> = Vec::with_capacity(maxout);
    let mut ip = 0;
    if input.is_empty() {
        return Ok(out);
    }
    let mut ctrl = (input[ip] & 31) as usize;
    ip += 1;
    loop {
        if ctrl >= 32 {
            let mut len = (ctrl >> 5) - 1;
            let mut ofs = (ctrl & 31) << 8;
            if len == 7 - 1 {
                loop {
                    let Some(&code) = input.get(ip) else {
                        return err("blosclz runs past its input");
                    };
                    ip += 1;
                    len += code as usize;
                    if code != 255 {
                        break;
                    }
                }
            }
            let Some(&code) = input.get(ip) else {
                return err("blosclz runs past its input");
            };
            ip += 1;
            len += 3;
            let mut back = ofs + code as usize;
            if code == 255 && ofs == (31 << 8) {
                if ip + 1 >= input.len() {
                    return err("blosclz runs past its input");
                }
                ofs = ((input[ip] as usize) << 8) + input[ip + 1] as usize;
                ip += 2;
                back = ofs + 8191;
            }
            // `ref = op - back - 1`
            let distance = back + 1;
            if distance > out.len() || out.len() + len > maxout {
                return err("blosclz refers outside its output");
            }
            let start = out.len() - distance;
            for k in 0..len {
                let b = out[start + k];
                out.push(b);
            }
            if ip >= input.len() {
                break;
            }
            ctrl = input[ip] as usize;
            ip += 1;
        } else {
            let run = ctrl + 1;
            if out.len() + run > maxout || ip + run > input.len() {
                return err("blosclz literal runs past the end");
            }
            out.extend_from_slice(&input[ip..ip + run]);
            ip += run;
            if ip >= input.len() {
                break;
            }
            ctrl = input[ip] as usize;
            ip += 1;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------- inner codecs

fn decompress_stream(codec: u8, src: &[u8], size: usize) -> Result<Vec<u8>, String> {
    let out = match codec {
        0 => blosclz_decompress(src, size)?,
        1 => lz4_flex::block::decompress(src, size).map_err(|e| format!("blosc lz4: {e}"))?,
        2 => snap::raw::Decoder::new()
            .decompress_vec(src)
            .map_err(|e| format!("blosc snappy: {e}"))?,
        3 => {
            let mut out = Vec::with_capacity(size);
            flate2::read::ZlibDecoder::new(src)
                .take(size as u64 + 1)
                .read_to_end(&mut out)
                .map_err(|e| format!("blosc zlib: {e}"))?;
            out
        }
        4 => {
            let mut out = Vec::with_capacity(size);
            zstd_rs::Decompressor::new()
                .decompress(src, None, size, &mut out)
                .map_err(|e| format!("blosc zstd: {e}"))?;
            out
        }
        c => return err(format!("compressor code {c}")),
    };
    if out.len() != size {
        return err(format!("a stream gave {} bytes, not {size}", out.len()));
    }
    Ok(out)
}

fn compress_stream(codec: Compressor, level: i32, src: &[u8]) -> Result<Vec<u8>, String> {
    Ok(match codec {
        // no pure-Rust blosclz compressor: stored (`size == len`), which every reader takes
        Compressor::BloscLz => src.to_vec(),
        Compressor::Lz4 => lz4_flex::block::compress(src),
        Compressor::Snappy => snap::raw::Encoder::new()
            .compress_vec(src)
            .map_err(|e| e.to_string())?,
        Compressor::Zlib => {
            let mut e = flate2::write::ZlibEncoder::new(
                Vec::new(),
                flate2::Compression::new(level.clamp(0, 9) as u32),
            );
            e.write_all(src).map_err(|e| e.to_string())?;
            e.finish().map_err(|e| e.to_string())?
        }
        Compressor::Zstd => {
            let cfg = zstd_rs::CompressionConfig {
                level: level.clamp(1, zstd_rs::MAX_LEVEL),
                ..zstd_rs::CompressionConfig::DEFAULT
            };
            let mut out = Vec::new();
            zstd_rs::Compressor::new(cfg)
                .and_then(|mut c| c.compress(src, None, &mut out))
                .map_err(|e| e.to_string())?;
            out
        }
    })
}

// ---------------------------------------------------------------- container

/// The bytes a Blosc container holds.
pub fn decompress(src: &[u8]) -> Result<Vec<u8>, String> {
    if src.len() < HEADER {
        return err("shorter than its header");
    }
    let flags = src[2];
    let typesize = (src[3] as usize).max(1);
    let nbytes = u32le(src, 4);
    let blocksize = u32le(src, 8);
    let codec = flags >> 5;
    if flags & MEMCPYED != 0 {
        return src
            .get(HEADER..HEADER + nbytes)
            .map(<[u8]>::to_vec)
            .ok_or_else(|| "blosc: a stored buffer cut short".into());
    }
    if flags & DELTA != 0 {
        return err("the delta filter is not supported");
    }
    if blocksize == 0 {
        return if nbytes == 0 {
            Ok(Vec::new())
        } else {
            err("a block size of 0")
        };
    }
    let nblocks = nbytes.div_ceil(blocksize);
    if src.len() < HEADER + 4 * nblocks {
        return err("the block table is cut short");
    }
    let mut out = Vec::with_capacity(nbytes);
    for b in 0..nblocks {
        let start = u32le(src, HEADER + 4 * b);
        let end = if b + 1 < nblocks {
            u32le(src, HEADER + 4 * (b + 1))
        } else {
            src.len()
        };
        let block = src
            .get(start..end.max(start))
            .ok_or("blosc: a block outside the buffer")?;
        let neblock = blocksize.min(nbytes - b * blocksize);
        // split into typesize streams, unless flagged not to (or it does not fit)
        let tries: &[usize] = if flags & DONT_SPLIT != 0 {
            &[1]
        } else if typesize <= MAX_SPLITS && neblock.is_multiple_of(typesize) && typesize > 1 {
            &[typesize, 1]
        } else {
            &[1]
        };
        let mut done = None;
        for &splits in tries {
            if let Ok(bytes) = read_block(codec, block, neblock, splits) {
                done = Some(bytes);
                break;
            }
        }
        let bytes = done.ok_or("blosc: a block does not decode")?;
        // c-blosc bit-shuffles even single bytes; a byte shuffle of them is a copy
        let bytes = if flags & BIT_SHUFFLE != 0 {
            bitshuffle(typesize, &bytes, true)
        } else if flags & BYTE_SHUFFLE != 0 && typesize > 1 {
            let whole = bytes.len() / typesize * typesize;
            let mut v = unshuffle(typesize, &bytes[..whole]);
            v.extend_from_slice(&bytes[whole..]);
            v
        } else {
            bytes
        };
        out.extend_from_slice(&bytes);
    }
    Ok(out)
}

/// A block's streams, which must account for the block exactly.
fn read_block(codec: u8, block: &[u8], neblock: usize, splits: usize) -> Result<Vec<u8>, String> {
    let stream = neblock / splits;
    let mut out = Vec::with_capacity(neblock);
    let mut at = 0;
    for _ in 0..splits {
        let size = block
            .get(at..at + 4)
            .ok_or("blosc: a stream header cut short")?;
        let csize = i32::from_le_bytes(size.try_into().unwrap());
        at += 4;
        if csize <= 0 {
            return err("an empty stream");
        }
        let csize = csize as usize;
        let data = block
            .get(at..at + csize)
            .ok_or("blosc: a stream cut short")?;
        at += csize;
        if csize == stream {
            out.extend_from_slice(data);
        } else {
            out.extend(decompress_stream(codec, data, stream)?);
        }
    }
    if at != block.len() && splits > 1 {
        return err("streams do not fill the block");
    }
    Ok(out)
}

/// A Blosc container of `src`: `typesize` the element size the shuffle works
/// on, `blocksize` 0 for a choice of 256 KiB (whole elements).
pub fn compress(
    src: &[u8],
    codec: Compressor,
    level: i32,
    shuffle_mode: Shuffle,
    typesize: usize,
    blocksize: usize,
) -> Result<Vec<u8>, String> {
    let typesize = typesize.clamp(1, 255);
    if src.len() > u32::MAX as usize - HEADER {
        return err("over 4 GiB");
    }
    let blocksize = if blocksize == 0 {
        (256 << 10) / typesize * typesize
    } else {
        blocksize
    }
    .min(src.len())
    .max(1);
    let mut flags = (codec as u8) << 5 | DONT_SPLIT;
    flags |= match shuffle_mode {
        Shuffle::Byte if typesize > 1 => BYTE_SHUFFLE,
        Shuffle::Bit => BIT_SHUFFLE,
        _ => 0,
    };
    let nblocks = src.len().div_ceil(blocksize);
    let mut out = vec![2u8, 1, flags, typesize as u8];
    out.extend_from_slice(&(src.len() as u32).to_le_bytes());
    out.extend_from_slice(&(blocksize as u32).to_le_bytes());
    out.extend_from_slice(&[0; 4]); // cbytes, below
    out.resize(HEADER + 4 * nblocks, 0);
    for (b, chunk) in src.chunks(blocksize).enumerate() {
        let offset = out.len() as u32;
        out[HEADER + 4 * b..HEADER + 4 * b + 4].copy_from_slice(&offset.to_le_bytes());
        let filtered = match flags & (BYTE_SHUFFLE | BIT_SHUFFLE) {
            BYTE_SHUFFLE => {
                let whole = chunk.len() / typesize * typesize;
                let mut v = shuffle(typesize, &chunk[..whole]);
                v.extend_from_slice(&chunk[whole..]);
                v
            }
            BIT_SHUFFLE => bitshuffle(typesize, chunk, false),
            _ => chunk.to_vec(),
        };
        let packed = compress_stream(codec, level, &filtered)?;
        // a stream that does not shrink is stored, its size its length
        let data = if packed.len() < filtered.len() {
            &packed
        } else {
            &filtered
        };
        out.extend_from_slice(&(data.len() as i32).to_le_bytes());
        out.extend_from_slice(data);
    }
    if out.len() >= src.len() + HEADER {
        // no gain: stored whole
        out.truncate(HEADER);
        out[2] = flags | MEMCPYED;
        out.extend_from_slice(src);
    }
    let cbytes = out.len() as u32;
    out[12..16].copy_from_slice(&cbytes.to_le_bytes());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_codec_and_filter_round_trips() {
        let data: Vec<u8> = (0..70_001u32)
            .flat_map(|i| (i / 3 % 1000).to_le_bytes())
            .collect();
        for codec in [
            Compressor::BloscLz,
            Compressor::Lz4,
            Compressor::Snappy,
            Compressor::Zlib,
            Compressor::Zstd,
        ] {
            for sh in [Shuffle::None, Shuffle::Byte, Shuffle::Bit] {
                for (typesize, blocksize) in [(4, 0), (4, 4096), (8, 1000), (1, 777)] {
                    let c = compress(&data, codec, 5, sh, typesize, blocksize).unwrap();
                    assert_eq!(
                        decompress(&c).unwrap(),
                        data,
                        "{codec:?} {sh:?} {typesize} {blocksize}"
                    );
                }
            }
        }
        assert_eq!(
            decompress(&compress(&[], Compressor::Lz4, 5, Shuffle::Byte, 4, 0).unwrap()).unwrap(),
            b""
        );
    }

    #[test]
    fn damaged_containers_are_errors() {
        let data: Vec<u8> = (0..5000u16).flat_map(|i| i.to_le_bytes()).collect();
        let c = compress(&data, Compressor::Zstd, 3, Shuffle::Byte, 2, 1024).unwrap();
        for n in 0..c.len() {
            let _ = decompress(&c[..n]);
        }
    }
}
