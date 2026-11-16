//! http(s) reads: a store of keys under a URL, and `Read + Seek` over one
//! remote file with range requests and a small block cache (zip readers
//! seek back and forth around the central directory).
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::sync::{Arc, Mutex};

use bytes::Bytes;

use super::{Response, request};
use crate::error::{Error, Result};
use crate::retry::Policy;
use crate::store::{ByteRange, Storage};

const BLOCK: u64 = 1 << 20;
const MAX_BLOCKS: usize = 64;

#[derive(Clone)]
pub struct RangeReader {
    url: String,
    len: u64,
    pos: u64,
    /// Shared by clones, each of which keeps its own position.
    cache: Arc<Mutex<Blocks>>,
    policy: Policy,
    /// The whole file, if the server ignores range requests.
    whole: Option<Arc<Vec<u8>>>,
}

#[derive(Default)]
struct Blocks {
    blocks: HashMap<u64, Arc<Vec<u8>>>,
    /// Blocks held, the least recently read first.
    order: Vec<u64>,
}

impl Blocks {
    fn get(&mut self, idx: u64) -> Option<Arc<Vec<u8>>> {
        let block = self.blocks.get(&idx)?.clone();
        // Zip readers keep coming back to the central directory: what
        // is read goes last, not what was fetched.
        if let Some(at) = self.order.iter().position(|&b| b == idx) {
            self.order.remove(at);
            self.order.push(idx);
        }
        Some(block)
    }

    fn insert(&mut self, idx: u64, block: Arc<Vec<u8>>) {
        if self.blocks.insert(idx, block).is_some() {
            return;
        }
        if self.order.len() >= MAX_BLOCKS {
            let old = self.order.remove(0);
            self.blocks.remove(&old);
        }
        self.order.push(idx);
    }
}

/// A GET, retried; `None` for a 404, an error for any other status past 2xx.
fn get(url: &str, range: Option<&str>, policy: Policy) -> Result<Option<Response>> {
    let headers: Vec<(String, String)> = range
        .map(|r| vec![("Range".to_string(), r.to_string())])
        .unwrap_or_default();
    let resp = request(policy, "GET", url, &headers, &[])?;
    match resp.status {
        404 => Ok(None),
        _ if resp.ok() => Ok(Some(resp)),
        _ => Err(resp.error(url)),
    }
}

/// The keys of a store under a URL, each fetched with a GET: read only,
/// and not listable.
pub struct HttpStore {
    base: String,
    policy: Policy,
}

impl HttpStore {
    pub fn new(url: &str) -> HttpStore {
        HttpStore {
            base: url.trim_end_matches('/').to_string(),
            policy: Policy::default(),
        }
    }

    fn url(&self, key: &str) -> String {
        format!("{}/{key}", self.base)
    }
}

impl Storage for HttpStore {
    fn get(&self, key: &str) -> Result<Option<Bytes>> {
        Ok(get(&self.url(key), None, self.policy)?.map(|r| r.body))
    }
    fn get_ranges(&self, key: &str, ranges: &[ByteRange]) -> Result<Option<Vec<Bytes>>> {
        let url = self.url(key);
        let mut out = Vec::with_capacity(ranges.len());
        for r in ranges {
            let Some(header) = r.http_header() else {
                out.push(Bytes::new());
                continue;
            };
            let Some(resp) = get(&url, Some(&header), self.policy)? else {
                return Ok(None);
            };
            out.push(if resp.status == 206 {
                resp.body
            } else {
                // the server sent it all
                r.slice(&resp.body)?
            });
        }
        Ok(Some(out))
    }
    fn supports_ranges(&self) -> bool {
        true
    }
    fn size(&self, key: &str) -> Result<Option<u64>> {
        let url = self.url(key);
        let head = request(self.policy, "HEAD", &url, &[], &[]);
        match head {
            Ok(r) if r.status == 404 => Ok(None),
            Ok(r) if r.ok() => match r.header("content-length").and_then(|v| v.parse().ok()) {
                Some(n) => Ok(Some(n)),
                // no length: a GET then
                None => Ok(self.get(key)?.map(|b| b.len() as u64)),
            },
            // some servers do not answer HEAD: a GET then
            _ => Ok(self.get(key)?.map(|b| b.len() as u64)),
        }
    }
}

impl RangeReader {
    pub fn open(url: &str) -> Result<RangeReader> {
        Self::open_with(url, Policy::default())
    }

    fn open_with(url: &str, policy: Policy) -> Result<RangeReader> {
        let resp = request(policy, "HEAD", url, &[], &[])?;
        if !resp.ok() {
            return Err(resp.error(url));
        }
        let ranges = resp.header("accept-ranges") == Some("bytes");
        let len = resp
            .header("content-length")
            .and_then(|v| v.parse::<u64>().ok());
        let mut r = RangeReader {
            url: url.to_string(),
            len: len.unwrap_or(0),
            pos: 0,
            cache: Default::default(),
            policy,
            whole: None,
        };
        if !ranges || len.is_none() {
            let bytes = get(url, None, policy)?
                .ok_or_else(|| Error::Storage(format!("{url}: HTTP 404")))?
                .body;
            r.len = bytes.len() as u64;
            r.whole = Some(Arc::new(bytes.to_vec()));
        }
        Ok(r)
    }

    /// A block, from the cache or the network; the cache is not held
    /// meanwhile, so clones fetch side by side.
    fn block(&self, idx: u64) -> std::io::Result<Arc<Vec<u8>>> {
        if let Some(block) = self.cache.lock().unwrap().get(idx) {
            return Ok(block);
        }
        let start = idx * BLOCK;
        let end = (start + BLOCK).min(self.len) - 1;
        let range = format!("bytes={start}-{end}");
        let resp = get(&self.url, Some(&range), self.policy)
            .map_err(std::io::Error::other)?
            .ok_or_else(|| std::io::Error::other(format!("{}: HTTP 404", self.url)))?;
        let data = Arc::new(if resp.status == 206 {
            resp.body.to_vec()
        } else {
            resp.body[start as usize..=end as usize].to_vec()
        });
        self.cache.lock().unwrap().insert(idx, data.clone());
        Ok(data)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    const QUICK: Policy = Policy {
        refused: 6,
        lost: 2,
        base: Duration::from_millis(1),
    };
    const SIZE: usize = 3 * BLOCK as usize;

    /// Serves a file of `SIZE` bytes by range, refusing the first
    /// `refuse` requests; `gets` counts the ranged reads.
    fn serve(refuse: usize, gets: Arc<AtomicUsize>) -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let mut seen = 0;
            for stream in listener.incoming() {
                let mut stream = stream.unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let (mut first, mut range) = (String::new(), None);
                reader.read_line(&mut first).unwrap();
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" || line.is_empty() {
                        break;
                    }
                    if let Some(v) = line.to_ascii_lowercase().strip_prefix("range: bytes=") {
                        let (a, b) = v.trim().split_once('-').unwrap();
                        range = Some((a.parse::<usize>().unwrap(), b.parse::<usize>().unwrap()));
                    }
                }
                seen += 1;
                let reply = if seen <= refuse {
                    "HTTP/1.1 503 Unavailable\r\nretry-after: 0\r\ncontent-length: 0\r\nconnection: close\r\n\r\n".to_string()
                } else if first.starts_with("HEAD") {
                    format!(
                        "HTTP/1.1 200 OK\r\naccept-ranges: bytes\r\ncontent-length: {SIZE}\r\nconnection: close\r\n\r\n"
                    )
                } else {
                    let (a, b) = range.unwrap();
                    gets.fetch_add(1, Ordering::SeqCst);
                    let _ = write!(
                        stream,
                        "HTTP/1.1 206 Partial\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                        b - a + 1
                    );
                    let _ = stream.write_all(&vec![(a / BLOCK as usize) as u8; b - a + 1]);
                    continue;
                };
                let _ = stream.write_all(reply.as_bytes());
            }
        });
        format!("http://{addr}/file")
    }

    fn byte_at(r: &mut RangeReader, block: u64) -> u8 {
        r.seek(SeekFrom::Start(block * BLOCK)).unwrap();
        let mut b = [0];
        r.read_exact(&mut b).unwrap();
        b[0]
    }

    #[test]
    fn clones_read_on_their_own_and_share_the_cache() {
        let gets = Arc::new(AtomicUsize::new(0));
        let mut a = RangeReader::open_with(&serve(0, gets.clone()), QUICK).unwrap();
        assert_eq!(byte_at(&mut a, 1), 1);
        let mut b = a.clone();
        assert_eq!(byte_at(&mut b, 1), 1);
        assert_eq!(gets.load(Ordering::SeqCst), 1, "fetched once");
        assert_eq!(byte_at(&mut b, 2), 2);
        let mut next = [0];
        a.read_exact(&mut next).unwrap();
        assert_eq!(
            (a.pos, next[0]),
            (BLOCK + 2, 1),
            "a kept its place while b moved"
        );
    }

    #[test]
    fn a_server_refusing_for_now_is_asked_again() {
        let url = serve(3, Default::default());
        let mut r = RangeReader::open_with(&url, QUICK).unwrap();
        assert_eq!(byte_at(&mut r, 2), 2);
    }

    #[test]
    fn the_block_read_most_recently_stays() {
        let gets = Arc::new(AtomicUsize::new(0));
        let mut r = RangeReader::open_with(&serve(0, gets.clone()), QUICK).unwrap();
        // A full cache, as a long read leaves it, block 0 oldest of all.
        byte_at(&mut r, 0);
        for b in 100..100 + MAX_BLOCKS as u64 - 1 {
            r.cache.lock().unwrap().insert(b, Default::default());
        }
        byte_at(&mut r, 0);
        byte_at(&mut r, 1);
        assert_eq!(gets.load(Ordering::SeqCst), 2);
        assert_eq!(byte_at(&mut r, 0), 0);
        assert_eq!(
            gets.load(Ordering::SeqCst),
            2,
            "read again, it was not the one to go"
        );
    }
}
