//! Cloud object stores over [`super::request`], in pure Rust: Amazon S3 and
//! S3-compatible services ([`s3`], SigV4), Google Cloud Storage ([`gcs`],
//! OAuth 2 tokens from a service account or a user's credentials) and Azure
//! Blob Storage ([`azure`], Shared Key, SAS, or OAuth 2 tokens).
//!
//! Each reads its configuration and credentials from the environment, as
//! the vendors' own tools do (`AWS_*`, `GOOGLE_*`, `AZURE_*`); without
//! credentials requests go unsigned, so public buckets can be read.

pub(crate) mod azure;
pub(crate) mod crypto;
pub(crate) mod gcs;
pub(crate) mod s3;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;

use super::{Response, query_string, request};
use crate::error::{Error, Result};
use crate::retry::Policy;
use crate::store::{ByteRange, Storage, Store, StoreKind, cloud_scheme};

/// A request for a provider to sign.
#[derive(Debug, Clone)]
pub(crate) struct Req {
    pub method: &'static str,
    /// Scheme and authority: `https://bucket.s3.us-east-1.amazonaws.com`.
    pub origin: String,
    /// The path, encoded, from `/`.
    pub path: String,
    /// The query, not encoded.
    pub query: Vec<(String, String)>,
    pub headers: Vec<(String, String)>,
    pub body: Bytes,
}

impl Req {
    pub(crate) fn new(method: &'static str, origin: &str, path: String) -> Req {
        Req {
            method,
            origin: origin.to_string(),
            path,
            query: vec![],
            headers: vec![],
            body: Bytes::new(),
        }
    }

    pub(crate) fn query(mut self, k: &str, v: &str) -> Req {
        self.query.push((k.to_string(), v.to_string()));
        self
    }

    pub(crate) fn header(mut self, k: &str, v: &str) -> Req {
        self.headers.push((k.to_string(), v.to_string()));
        self
    }

    /// The `Host` header ureq will send: the authority.
    pub(crate) fn host(&self) -> &str {
        self.origin
            .split_once("://")
            .map_or(self.origin.as_str(), |(_, h)| h)
    }

    pub(crate) fn url(&self) -> String {
        if self.query.is_empty() {
            format!("{}{}", self.origin, self.path)
        } else {
            format!("{}{}?{}", self.origin, self.path, query_string(&self.query))
        }
    }
}

/// One page of a listing.
#[derive(Debug, Default)]
pub(crate) struct Page {
    pub keys: Vec<String>,
    pub prefixes: Vec<String>,
    /// Where the next page starts, if there is one.
    pub next: Option<String>,
}

/// What differs between the services.
pub(crate) trait Provider: Send + Sync {
    /// A request for object `key` (its whole name in the bucket).
    fn object(&self, method: &'static str, key: &str) -> Req;
    /// Add credentials to `req`, as the last thing before it is sent.
    fn authorize(&self, req: &mut Req) -> Result<()>;
    /// One page of the objects under `prefix`; with `delimiter`, those one
    /// level down only, and the prefixes under it.
    fn list_page(&self, prefix: &str, delimiter: bool, next: Option<&str>) -> Result<Page>;
    /// The header a byte range goes in.
    fn range_header(&self) -> &'static str {
        "Range"
    }
    /// Whether `bytes=-n` (the last `n` bytes) is understood.
    fn suffix_ranges(&self) -> bool {
        true
    }
    /// What a PUT needs besides the body.
    fn put_headers(&self, _req: &mut Req) {}
}

/// Sign and send `req`.
pub(crate) fn send(provider: &dyn Provider, mut req: Req) -> Result<Response> {
    provider.authorize(&mut req)?;
    let url = req.url();
    request(Policy::default(), req.method, &url, &req.headers, &req.body)
}

/// A credential that expires (a bearer token, a temporary key), fetched
/// again a minute before it does.
pub(crate) struct Token<T = String> {
    cached: Mutex<Option<(T, Instant)>>,
}

impl<T> Default for Token<T> {
    fn default() -> Self {
        Token {
            cached: Mutex::new(None),
        }
    }
}

impl<T: Clone> Token<T> {
    /// The credential, from the cache or from `fetch` (it and how many
    /// seconds it lasts). Threads asking meanwhile wait for the one fetch.
    pub(crate) fn get(&self, fetch: impl FnOnce() -> Result<(T, u64)>) -> Result<T> {
        let mut cached = self.cached.lock().unwrap();
        if let Some((token, until)) = cached.as_ref()
            && Instant::now() + Duration::from_secs(60) < *until
        {
            return Ok(token.clone());
        }
        let (token, secs) = fetch()?;
        *cached = Some((token.clone(), Instant::now() + Duration::from_secs(secs)));
        Ok(token)
    }
}

/// An OAuth 2 token endpoint's answer: `access_token` and `expires_in`.
pub(crate) fn oauth_token(url: &str, form: &[(&str, &str)]) -> Result<(String, u64)> {
    let headers = [(
        "Content-Type".to_string(),
        "application/x-www-form-urlencoded".to_string(),
    )];
    let resp = request(
        Policy::default(),
        "POST",
        url,
        &headers,
        query_string(form).as_bytes(),
    )?;
    if !resp.ok() {
        return Err(resp.error(url));
    }
    let v: serde_json::Value = serde_json::from_slice(&resp.body)?;
    let token = v["access_token"]
        .as_str()
        .ok_or_else(|| Error::Storage(format!("{url}: no access_token in the answer")))?;
    let secs = v["expires_in"]
        .as_u64()
        .or_else(|| v["expires_in"].as_str()?.parse().ok())
        .unwrap_or(3600);
    Ok((token.to_string(), secs))
}

pub(crate) fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

pub(crate) fn env_any(names: &[&str]) -> Option<String> {
    names.iter().find_map(|n| env(n))
}

pub(crate) fn env_true(name: &str) -> bool {
    env(name).is_some_and(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
}

/// The text of the first descendant of `node` named `name`.
pub(crate) fn xml_text<'a>(node: roxmltree::Node<'a, 'a>, name: &str) -> Option<&'a str> {
    node.descendants()
        .find(|n| n.tag_name().name() == name)
        .and_then(|n| n.text())
}

/// The texts of the `child` elements of every `parent` element.
pub(crate) fn xml_texts(doc: &roxmltree::Document, parent: &str, child: &str) -> Vec<String> {
    doc.descendants()
        .filter(|n| n.tag_name().name() == parent)
        .filter_map(|n| {
            n.children()
                .find(|c| c.tag_name().name() == child)
                .and_then(|c| c.text())
        })
        .map(str::to_string)
        .collect()
}

/// The objects of a bucket under `root`, as a [`Storage`].
pub(crate) struct CloudStore {
    provider: Box<dyn Provider>,
    /// The store's place in the bucket: `""` or ending in `/`.
    root: String,
}

impl CloudStore {
    fn full(&self, key: &str) -> String {
        format!("{}{key}", self.root)
    }

    fn list(&self, prefix: &str, delimiter: bool) -> Result<Page> {
        let full = self.full(prefix);
        let mut all = Page::default();
        let mut next = None;
        loop {
            let page = self.provider.list_page(&full, delimiter, next.as_deref())?;
            all.keys.extend(page.keys);
            all.prefixes.extend(page.prefixes);
            match page.next {
                Some(n) if next.as_ref() != Some(&n) => next = Some(n),
                _ => break,
            }
        }
        let strip = |k: String| k.strip_prefix(&self.root).map(str::to_string);
        all.keys = all.keys.into_iter().filter_map(strip).collect();
        all.prefixes = all.prefixes.into_iter().filter_map(strip).collect();
        all.keys.sort();
        all.prefixes.sort();
        all.prefixes.dedup();
        Ok(all)
    }
}

impl Storage for CloudStore {
    fn get(&self, key: &str) -> Result<Option<Bytes>> {
        let resp = send(
            &*self.provider,
            self.provider.object("GET", &self.full(key)),
        )?;
        match resp.status {
            404 => Ok(None),
            _ if resp.ok() => Ok(Some(resp.body)),
            _ => Err(resp.error(key)),
        }
    }

    fn get_ranges(&self, key: &str, ranges: &[ByteRange]) -> Result<Option<Vec<Bytes>>> {
        let mut len = None;
        let mut out = Vec::with_capacity(ranges.len());
        for r in ranges {
            let r = match *r {
                // a bounded range where suffixes are not understood
                ByteRange::Suffix(n) if !self.provider.suffix_ranges() => {
                    if len.is_none() {
                        len = self.size(key)?;
                    }
                    let Some(l) = len else {
                        return Ok(None);
                    };
                    ByteRange::Range(l.saturating_sub(n), l)
                }
                r => r,
            };
            let Some(header) = r.http_header() else {
                out.push(Bytes::new());
                continue;
            };
            let req = self
                .provider
                .object("GET", &self.full(key))
                .header(self.provider.range_header(), &header);
            let resp = send(&*self.provider, req)?;
            match resp.status {
                404 => return Ok(None),
                206 => out.push(resp.body),
                // the whole object
                _ if resp.ok() => out.push(r.slice(&resp.body)?),
                _ => return Err(resp.error(key)),
            }
        }
        Ok(Some(out))
    }

    fn supports_ranges(&self) -> bool {
        true
    }

    fn size(&self, key: &str) -> Result<Option<u64>> {
        let resp = send(
            &*self.provider,
            self.provider.object("HEAD", &self.full(key)),
        )?;
        match resp.status {
            404 => Ok(None),
            _ if resp.ok() => Ok(resp.header("content-length").and_then(|v| v.parse().ok())),
            _ => Err(resp.error(key)),
        }
    }

    fn set(&self, key: &str, value: Bytes) -> Result<()> {
        let mut req = self.provider.object("PUT", &self.full(key));
        req.body = value;
        self.provider.put_headers(&mut req);
        let resp = send(&*self.provider, req)?;
        if !resp.ok() {
            return Err(resp.error(key));
        }
        Ok(())
    }

    fn erase(&self, key: &str) -> Result<()> {
        let resp = send(
            &*self.provider,
            self.provider.object("DELETE", &self.full(key)),
        )?;
        if !resp.ok() && resp.status != 404 {
            return Err(resp.error(key));
        }
        Ok(())
    }

    fn erase_prefix(&self, prefix: &str) -> Result<()> {
        use crate::par::*;
        let keys = self.list_prefix(prefix)?;
        let results: Vec<Result<()>> = keys.par_iter().map(|k| self.erase(k)).collect();
        results.into_iter().collect()
    }

    fn list_prefix(&self, prefix: &str) -> Result<Vec<String>> {
        Ok(self.list(prefix, false)?.keys)
    }

    fn list_dir(&self, prefix: &str) -> Result<(Vec<String>, Vec<String>)> {
        let page = self.list(prefix, true)?;
        Ok((page.keys, page.prefixes))
    }
}

/// A cloud store from its URL: `s3://bucket/prefix`, `gs://bucket/prefix`,
/// `az://container/prefix` (the account from `AZURE_STORAGE_ACCOUNT_NAME`)
/// or `abfs[s]://container@account.dfs.core.windows.net/prefix`.
pub(crate) fn open(url: &str, read_only: bool) -> Result<Store> {
    let scheme =
        cloud_scheme(url).ok_or_else(|| Error::value(format!("not a cloud URL: {url}")))?;
    let rest = &url[scheme.len() + 3..];
    let rest = rest.split(['?', '#']).next().unwrap_or(rest);
    let (authority, path) = rest.split_once('/').unwrap_or((rest, ""));
    if authority.is_empty() {
        return Err(Error::value(format!("no bucket in {url}")));
    }
    let path = percent_encoding::percent_decode_str(path.trim_matches('/'))
        .decode_utf8_lossy()
        .to_string();
    let provider: Box<dyn Provider> = match scheme {
        "s3" => Box::new(s3::S3::from_env(authority)?),
        "gs" | "gcs" => Box::new(gcs::Gcs::from_env(authority)?),
        _ => Box::new(azure::Azure::from_env(scheme, authority)?),
    };
    let root = if path.is_empty() {
        String::new()
    } else {
        format!("{path}/")
    };
    Ok(Store::from_dyn(
        Arc::new(CloudStore { provider, root }),
        StoreKind::ObjectStore,
        Some(url.to_string()),
    )
    .read_only(read_only))
}
