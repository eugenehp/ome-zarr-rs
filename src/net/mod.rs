//! Network I/O, in pure Rust: http(s) with ureq and rustls, whose
//! cryptography is graviola's (`tls-graviola`, x86-64 and aarch64) or
//! RustCrypto's (`tls-rustcrypto`, any target); no C compiler, no ring, no
//! aws-lc. On top of it, a store of keys under a URL and `Read + Seek` over
//! one remote file ([`http`]), and cloud object stores ([`cloud`]).
//!
//! Every request goes through [`request`], which asks again when a server
//! refuses for now ([`crate::retry`]).

pub(crate) mod http;

#[cfg(feature = "cloud")]
pub(crate) mod cloud;

use std::io::Read;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use bytes::Bytes;

use crate::retry::{Failure, Policy, retry};

#[cfg(not(any(
    feature = "tls-rustcrypto",
    all(
        feature = "tls-graviola",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
)))]
compile_error!(
    "the `http` feature needs https's cryptography: enable `tls-graviola` (x86-64, aarch64) or `tls-rustcrypto` (any target)"
);

/// The agent every request goes through: rustls with webpki's roots and
/// graviola's cryptography where it runs and is enabled, else RustCrypto's.
/// A status past 2xx is a response, not an error, so callers can tell a
/// missing object (404) from a failure.
pub(crate) fn agent() -> &'static ureq::Agent {
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    AGENT.get_or_init(|| {
        #[cfg(all(
            feature = "tls-graviola",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ))]
        let crypto = rustls_graviola::default_provider();
        #[cfg(all(
            feature = "tls-rustcrypto",
            not(all(
                feature = "tls-graviola",
                any(target_arch = "x86_64", target_arch = "aarch64")
            ))
        ))]
        let crypto = rustls_rustcrypto::provider();
        let tls = ureq::tls::TlsConfig::builder()
            .provider(ureq::tls::TlsProvider::Rustls)
            .root_certs(ureq::tls::RootCerts::WebPki)
            .unversioned_rustls_crypto_provider(Arc::new(crypto))
            .build();
        ureq::Agent::config_builder()
            .tls_config(tls)
            .http_status_as_error(false)
            .user_agent(concat!("ome-zarr-rs/", env!("CARGO_PKG_VERSION")))
            .build()
            .into()
    })
}

/// A response, read whole.
#[derive(Debug)]
pub(crate) struct Response {
    pub status: u16,
    pub headers: ureq::http::HeaderMap,
    pub body: Bytes,
}

impl Response {
    pub(crate) fn ok(&self) -> bool {
        (200..300).contains(&self.status)
    }

    pub(crate) fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name)?.to_str().ok()
    }

    /// The error for an unexpected status, with what the server said.
    pub(crate) fn error(&self, what: &str) -> crate::error::Error {
        let said = String::from_utf8_lossy(&self.body[..self.body.len().min(400)]);
        crate::error::Error::Storage(format!("{what}: HTTP {} {}", self.status, said.trim()))
    }
}

/// One request: `headers` as given, `body` sent if not empty (or for PUT
/// and POST). Statuses that mean "not now" (429, 500, 502, 503, 504) and lost
/// connections are failures [`retry`] tries again; any other status is a
/// response.
fn send_once(
    method: &str,
    url: &str,
    headers: &[(String, String)],
    body: &[u8],
) -> Result<Response, Failure<std::io::Error>> {
    let mut builder = ureq::http::Request::builder().method(method).uri(url);
    for (k, v) in headers {
        builder = builder.header(k.as_str(), v.as_str());
    }
    let sent = if body.is_empty() && !matches!(method, "PUT" | "POST") {
        builder
            .body(())
            .map_err(|e| Failure::Fatal(std::io::Error::other(e)))
            .and_then(|r| agent().run(r).map_err(lost))
    } else {
        builder
            .body(body)
            .map_err(|e| Failure::Fatal(std::io::Error::other(e)))
            .and_then(|r| agent().run(r).map_err(lost))
    };
    let mut resp = sent?;
    let status = resp.status().as_u16();
    let headers = resp.headers().clone();
    let mut data = Vec::new();
    resp.body_mut()
        .as_reader()
        .read_to_end(&mut data)
        .map_err(|e| Failure::Lost(std::io::Error::other(format!("{url}: {e}"))))?;
    if matches!(status, 429 | 500 | 502 | 503 | 504) {
        let after = headers
            .get("retry-after")
            .and_then(|v| v.to_str().ok()?.trim().parse::<u64>().ok())
            .map(|s| Duration::from_secs(s.min(30)));
        return Err(Failure::Refused(
            after,
            std::io::Error::other(format!("{url}: HTTP {status}")),
        ));
    }
    Ok(Response {
        status,
        headers,
        body: Bytes::from(data),
    })
}

fn lost(e: ureq::Error) -> Failure<std::io::Error> {
    match e {
        // the request itself was wrong: asking again would not help
        ureq::Error::BadUri(_) | ureq::Error::Http(_) | ureq::Error::InvalidProxyUrl => {
            Failure::Fatal(std::io::Error::other(e))
        }
        _ => Failure::Lost(std::io::Error::other(e)),
    }
}

/// A request, asked again while the server refuses for now or the
/// connection is lost.
pub(crate) fn request(
    policy: Policy,
    method: &str,
    url: &str,
    headers: &[(String, String)],
    body: &[u8],
) -> crate::error::Result<Response> {
    retry(policy, || send_once(method, url, headers, body))
        .map_err(|e| crate::error::Error::Storage(e.to_string()))
}

/// RFC 3986 unreserved characters stay; everything else is percent-encoded
/// (`/` too, unless `keep_slash`): the encoding SigV4 and the cloud APIs ask
/// for.
#[cfg(feature = "cloud")]
pub(crate) fn uri_encode(s: &str, keep_slash: bool) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            b'/' if keep_slash => out.push('/'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// `k=v&…`, both encoded, in the order given.
#[cfg(feature = "cloud")]
pub(crate) fn query_string(query: &[(impl AsRef<str>, impl AsRef<str>)]) -> String {
    query
        .iter()
        .map(|(k, v)| {
            format!(
                "{}={}",
                uri_encode(k.as_ref(), false),
                uri_encode(v.as_ref(), false)
            )
        })
        .collect::<Vec<_>>()
        .join("&")
}

#[cfg(all(test, feature = "cloud"))]
mod tests {
    use super::*;

    #[test]
    fn encoding() {
        assert_eq!(uri_encode("a b/c~d+é", true), "a%20b/c~d%2B%C3%A9");
        assert_eq!(uri_encode("a/b", false), "a%2Fb");
        let q = [("prefix".to_string(), "x/y z".to_string())];
        assert_eq!(query_string(&q), "prefix=x%2Fy%20z");
    }
}
