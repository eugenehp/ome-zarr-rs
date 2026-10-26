//! A zip archive read in byte ranges: a remote `.ome.zarr.zip` opened from
//! its tail and read file by file, never downloaded whole.
//!
//! The host supplies the ranged reads (on wasm, `fetch` with a `Range`
//! header). [`RemoteZip::open`] reads the end of central directory from the
//! archive's last bytes (one request), and the central directory if those
//! did not hold it (a second); [`RemoteZip::get`] then reads one member: its
//! local header and data in one range (a second only when its extra field
//! is unusually long), inflated if it was deflated. With the keys a
//! [`LazyStore`](crate::lazy::LazyStore) misses fetched through `get`, an
//! image streams out of a remote zip as out of a remote folder.
//!
//! The archive's length is not needed up front: the end record says where
//! the central directory sits and so where the file ends. A server that
//! ignores `Range` and sends the whole file is answered with
//! [`Opened::Whole`]: read it in memory instead.

use std::collections::HashMap;
use std::future::Future;
use std::io::Read;

use crate::error::{Error, Result};

const EOCD: u32 = 0x0605_4b50;
const ZIP64_LOCATOR: u32 = 0x0706_4b50;
const ZIP64_EOCD: u32 = 0x0606_4b50;
const CENTRAL: u32 = 0x0201_4b50;
const LOCAL: u32 = 0x0403_4b50;
/// The end record (22 bytes) and the longest comment it may carry.
pub const TAIL: u64 = 22 + 0xFFFF;
/// Room fetched past a member's name for its local extra field.
const EXTRA_SLACK: u64 = 256;

fn u16_at(b: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(at..at + 2)?.try_into().ok()?))
}
fn u32_at(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?))
}
fn u64_at(b: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_le_bytes(b.get(at..at + 8)?.try_into().ok()?))
}
fn bad(what: &str) -> Error {
    Error::Storage(format!("zip: {what}"))
}

/// One member: where its local header starts, its sizes and how it is
/// stored (0 stored, 8 deflated).
#[derive(Debug, Clone, PartialEq)]
pub struct Member {
    pub header_offset: u64,
    pub compressed: u64,
    pub uncompressed: u64,
    pub method: u16,
    name_len: u64,
}

/// What [`RemoteZip::open`] found.
pub enum Opened {
    /// The archive's index: read members with [`RemoteZip::get`].
    Index(RemoteZip),
    /// The server sent the whole archive: read it in memory.
    Whole(Vec<u8>),
}

/// A remote zip's central directory.
#[derive(Debug, Clone, Default)]
pub struct RemoteZip {
    members: HashMap<String, Member>,
}

/// The tail of the archive as the host fetched it.
pub enum Tail {
    /// the last bytes asked for (or fewer, when the file is shorter)
    Suffix(Vec<u8>),
    /// the server ignored the range: the whole file
    Whole(Vec<u8>),
}

impl RemoteZip {
    /// Read the archive's index: `tail(n)` fetches its last `n` bytes,
    /// `range(start, len)` any span.
    pub async fn open<T, TF, R, RF>(tail: &T, range: &R) -> Result<Opened>
    where
        T: Fn(u64) -> TF,
        TF: Future<Output = std::result::Result<Tail, String>>,
        R: Fn(u64, u64) -> RF,
        RF: Future<Output = std::result::Result<Vec<u8>, String>>,
    {
        let t = match tail(TAIL).await.map_err(Error::Storage)? {
            Tail::Whole(b) => return Ok(Opened::Whole(b)),
            Tail::Suffix(t) => t,
        };
        let at = (0..t.len().saturating_sub(21))
            .rev()
            .find(|&i| u32_at(&t, i) == Some(EOCD))
            .ok_or_else(|| bad("no end of central directory record (not a zip?)"))?;
        let mut count = u16_at(&t, at + 10).ok_or_else(|| bad("short end record"))? as u64;
        let mut size = u32_at(&t, at + 12).ok_or_else(|| bad("short end record"))? as u64;
        let mut offset = u32_at(&t, at + 16).ok_or_else(|| bad("short end record"))? as u64;
        // where the tail starts in the file: the end record follows the
        // central directory (and, in zip64, its own end record and locator)
        let mut tail_start = (offset + size).checked_sub(at as u64);
        if (count == 0xFFFF || size == 0xFFFF_FFFF || offset == 0xFFFF_FFFF)
            && at >= 20
            && u32_at(&t, at - 20) == Some(ZIP64_LOCATOR)
        {
            let record = u64_at(&t, at - 12).ok_or_else(|| bad("short zip64 locator"))?;
            // the zip64 end record (56 bytes) and the locator (20) come just
            // before the end record: where the tail starts in the file
            let ts = (record + 76).checked_sub(at as u64);
            let r = match ts {
                Some(ts) if record >= ts => {
                    t[(record - ts) as usize..(record - ts + 56) as usize].to_vec()
                }
                _ => range(record, 56).await.map_err(Error::Storage)?,
            };
            if u32_at(&r, 0) != Some(ZIP64_EOCD) {
                return Err(bad("bad zip64 end record"));
            }
            count = u64_at(&r, 32).ok_or_else(|| bad("short zip64 end record"))?;
            size = u64_at(&r, 40).ok_or_else(|| bad("short zip64 end record"))?;
            offset = u64_at(&r, 48).ok_or_else(|| bad("short zip64 end record"))?;
            tail_start = ts;
        }
        let cd = match tail_start {
            Some(ts) if offset >= ts && (offset - ts + size) as usize <= t.len() => {
                t[(offset - ts) as usize..(offset - ts + size) as usize].to_vec()
            }
            _ => range(offset, size).await.map_err(Error::Storage)?,
        };
        let mut members = HashMap::with_capacity(count as usize);
        let mut p = 0usize;
        for _ in 0..count {
            if u32_at(&cd, p) != Some(CENTRAL) {
                return Err(bad("bad central directory entry"));
            }
            let field = |o: usize| {
                u32_at(&cd, p + o)
                    .map(u64::from)
                    .ok_or_else(|| bad("short central directory entry"))
            };
            let method = u16_at(&cd, p + 10).ok_or_else(|| bad("short entry"))?;
            let (mut compressed, mut uncompressed) = (field(20)?, field(24)?);
            let name_len = u16_at(&cd, p + 28).ok_or_else(|| bad("short entry"))? as usize;
            let extra_len = u16_at(&cd, p + 30).ok_or_else(|| bad("short entry"))? as usize;
            let comment_len = u16_at(&cd, p + 32).ok_or_else(|| bad("short entry"))? as usize;
            let mut header_offset = field(42)?;
            let name = String::from_utf8_lossy(
                cd.get(p + 46..p + 46 + name_len)
                    .ok_or_else(|| bad("short name"))?,
            )
            .into_owned();
            // zip64: the sizes and offset that overflowed, in this order
            let extra = cd
                .get(p + 46 + name_len..p + 46 + name_len + extra_len)
                .unwrap_or(&[]);
            let mut e = 0;
            while e + 4 <= extra.len() {
                let (id, len) = (
                    u16_at(extra, e).unwrap_or(0),
                    u16_at(extra, e + 2).unwrap_or(0) as usize,
                );
                if id == 0x0001 {
                    let mut q = e + 4;
                    for v in [&mut uncompressed, &mut compressed, &mut header_offset] {
                        if *v == 0xFFFF_FFFF {
                            *v = u64_at(extra, q).unwrap_or(*v);
                            q += 8;
                        }
                    }
                }
                e += 4 + len;
            }
            p += 46 + name_len + extra_len + comment_len;
            if !name.ends_with('/') {
                let key = name.replace('\\', "/").trim_start_matches('/').to_string();
                members.insert(
                    key,
                    Member {
                        header_offset,
                        compressed,
                        uncompressed,
                        method,
                        name_len: name_len as u64,
                    },
                );
            }
        }
        Ok(Opened::Index(RemoteZip { members }))
    }

    /// The members' names (keys).
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.members.keys().map(String::as_str)
    }

    pub fn member(&self, key: &str) -> Option<&Member> {
        self.members.get(key)
    }

    /// Member `key`'s contents, `None` if there is no such member: its local
    /// header and data in one range, inflated if deflated.
    pub async fn get<R, RF>(
        &self,
        key: &str,
        range: &R,
    ) -> std::result::Result<Option<Vec<u8>>, String>
    where
        R: Fn(u64, u64) -> RF,
        RF: Future<Output = std::result::Result<Vec<u8>, String>>,
    {
        let Some(m) = self.members.get(key) else {
            return Ok(None);
        };
        let guess = 30 + m.name_len + EXTRA_SLACK;
        let mut b = range(m.header_offset, guess + m.compressed).await?;
        if u32_at(&b, 0) != Some(LOCAL) {
            return Err(format!("zip: no local header for {key}"));
        }
        let start = 30 + u16_at(&b, 26).unwrap_or(0) as u64 + u16_at(&b, 28).unwrap_or(0) as u64;
        let end = start + m.compressed;
        if (b.len() as u64) < end {
            // a long extra field: the data from where it starts
            b = range(m.header_offset + start, m.compressed).await?;
        } else {
            b = b[start as usize..end as usize].to_vec();
        }
        match m.method {
            0 => Ok(Some(b)),
            8 => {
                let mut out = Vec::with_capacity(m.uncompressed as usize);
                flate2::read::DeflateDecoder::new(&b[..])
                    .read_to_end(&mut out)
                    .map_err(|e| format!("zip: inflating {key}: {e}"))?;
                Ok(Some(out))
            }
            other => Err(format!(
                "zip: {key} is stored with method {other}, not stored or deflated"
            )),
        }
    }
}
