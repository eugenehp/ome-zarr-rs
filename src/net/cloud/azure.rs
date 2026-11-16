//! Azure Blob Storage.
//!
//! The account from an `abfs[s]://container@account.dfs.core.windows.net`
//! URL, or `AZURE_STORAGE_ACCOUNT_NAME` (`AZURE_STORAGE_ACCOUNT`) for
//! `az://container`. Credentials from the environment: an account key
//! (`AZURE_STORAGE_ACCOUNT_KEY`, `AZURE_STORAGE_ACCESS_KEY`; Shared Key
//! signing), a SAS token (`AZURE_STORAGE_SAS_KEY`, `AZURE_STORAGE_SAS_TOKEN`),
//! a bearer token (`AZURE_STORAGE_TOKEN`), or a service principal's secret
//! (`AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_TENANT_ID`; traded for a
//! token). `AZURE_STORAGE_USE_EMULATOR`: Azurite, at `127.0.0.1:10000` with
//! its well-known account. None: unsigned.

use super::crypto::{base64, base64_decode, hmac_sha256};
use super::{
    Page, Provider, Req, Token, env, env_any, env_true, oauth_token, send, xml_text, xml_texts,
};
use crate::error::{Error, Result};
use crate::net::uri_encode;

const VERSION: &str = "2023-11-03";
const EMULATOR_ACCOUNT: &str = "devstoreaccount1";
const EMULATOR_KEY: &str =
    "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==";

enum Credentials {
    Anonymous,
    SharedKey(Vec<u8>),
    /// A SAS token's query parameters, decoded.
    Sas(Vec<(String, String)>),
    Bearer(String),
    ClientSecret {
        tenant: String,
        client_id: String,
        secret: String,
        token: Token,
    },
}

pub(crate) struct Azure {
    account: String,
    container: String,
    origin: String,
    /// `/devstoreaccount1` for the emulator (path-style), else `""`.
    base: String,
    credentials: Credentials,
}

fn decode(s: &str) -> String {
    percent_encoding::percent_decode_str(s)
        .decode_utf8_lossy()
        .to_string()
}

/// The `Authorization` header's signature for a request (Shared Key):
/// `path` as sent (encoded), `query` and `headers` not encoded.
pub(crate) fn shared_key(
    account: &str,
    key: &[u8],
    method: &str,
    path: &str,
    query: &[(String, String)],
    headers: &[(String, String)],
    content_length: usize,
) -> String {
    let header = |name: &str| -> String {
        headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.clone())
            .unwrap_or_default()
    };
    let mut ms: Vec<(String, String)> = headers
        .iter()
        .filter(|(k, _)| k.to_ascii_lowercase().starts_with("x-ms-"))
        .map(|(k, v)| (k.to_ascii_lowercase(), v.trim().to_string()))
        .collect();
    ms.sort();
    let canonical_headers: String = ms.iter().map(|(k, v)| format!("{k}:{v}\n")).collect();
    let mut q: Vec<(String, String)> = query
        .iter()
        .map(|(k, v)| (k.to_ascii_lowercase(), v.clone()))
        .collect();
    q.sort();
    let mut resource = format!("/{account}{path}");
    for (k, v) in &q {
        resource.push_str(&format!("\n{k}:{v}"));
    }
    let length = if content_length > 0 {
        content_length.to_string()
    } else {
        String::new()
    };
    let to_sign = [
        method.to_string(),
        header("Content-Encoding"),
        header("Content-Language"),
        length,
        header("Content-MD5"),
        header("Content-Type"),
        header("Date"),
        header("If-Modified-Since"),
        header("If-Match"),
        header("If-None-Match"),
        header("If-Unmodified-Since"),
        header("Range"),
    ]
    .join("\n")
        + "\n"
        + &canonical_headers
        + &resource;
    base64(&hmac_sha256(key, to_sign.as_bytes()))
}

impl Azure {
    /// From a URL's scheme and authority (`container`, or
    /// `container@account.dfs.core.windows.net` for abfs).
    pub(crate) fn from_env(scheme: &str, authority: &str) -> Result<Azure> {
        let emulator = env_true("AZURE_STORAGE_USE_EMULATOR");
        let (container, account) = match authority.split_once('@') {
            Some((c, host)) => (c.to_string(), host.split('.').next().map(str::to_string)),
            None => (authority.to_string(), None),
        };
        let account = if emulator {
            EMULATOR_ACCOUNT.to_string()
        } else {
            account
                .or_else(|| env_any(&["AZURE_STORAGE_ACCOUNT_NAME", "AZURE_STORAGE_ACCOUNT"]))
                .ok_or_else(|| {
                    Error::value(format!(
                        "{scheme}://{authority}: no account (use container@account.dfs.core.windows.net, or set AZURE_STORAGE_ACCOUNT_NAME)"
                    ))
                })?
        };
        let credentials = if emulator {
            Credentials::SharedKey(base64_decode(EMULATOR_KEY)?)
        } else if let Some(key) = env_any(&[
            "AZURE_STORAGE_ACCOUNT_KEY",
            "AZURE_STORAGE_ACCESS_KEY",
            "AZURE_STORAGE_MASTER_KEY",
        ]) {
            Credentials::SharedKey(base64_decode(&key)?)
        } else if let Some(sas) = env_any(&["AZURE_STORAGE_SAS_KEY", "AZURE_STORAGE_SAS_TOKEN"]) {
            Credentials::Sas(
                sas.trim_start_matches('?')
                    .split('&')
                    .filter(|p| !p.is_empty())
                    .map(|p| {
                        let (k, v) = p.split_once('=').unwrap_or((p, ""));
                        (decode(k), decode(v))
                    })
                    .collect(),
            )
        } else if let Some(token) = env("AZURE_STORAGE_TOKEN") {
            Credentials::Bearer(token)
        } else if let (Some(client_id), Some(secret), Some(tenant)) = (
            env("AZURE_CLIENT_ID"),
            env("AZURE_CLIENT_SECRET"),
            env_any(&["AZURE_TENANT_ID", "AZURE_AUTHORITY_ID"]),
        ) {
            Credentials::ClientSecret {
                tenant,
                client_id,
                secret,
                token: Token::default(),
            }
        } else {
            Credentials::Anonymous
        };
        let (origin, base) = if emulator {
            let host =
                env("AZURITE_BLOB_STORAGE_URL").unwrap_or_else(|| "http://127.0.0.1:10000".into());
            (
                host.trim_end_matches('/').to_string(),
                format!("/{EMULATOR_ACCOUNT}"),
            )
        } else {
            (
                format!("https://{account}.blob.core.windows.net"),
                String::new(),
            )
        };
        Ok(Azure {
            account,
            container,
            origin,
            base,
            credentials,
        })
    }

    fn container_path(&self) -> String {
        format!("{}/{}", self.base, uri_encode(&self.container, false))
    }
}

impl Provider for Azure {
    fn object(&self, method: &'static str, key: &str) -> Req {
        Req::new(
            method,
            &self.origin,
            format!("{}/{}", self.container_path(), uri_encode(key, true)),
        )
    }

    fn authorize(&self, req: &mut Req) -> Result<()> {
        req.headers.push(("x-ms-version".into(), VERSION.into()));
        let date = chrono::Utc::now()
            .format("%a, %d %b %Y %H:%M:%S GMT")
            .to_string();
        req.headers.push(("x-ms-date".into(), date));
        match &self.credentials {
            Credentials::Anonymous => {}
            Credentials::SharedKey(key) => {
                let signature = shared_key(
                    &self.account,
                    key,
                    req.method,
                    &req.path,
                    &req.query,
                    &req.headers,
                    req.body.len(),
                );
                req.headers.push((
                    "Authorization".into(),
                    format!("SharedKey {}:{signature}", self.account),
                ));
            }
            Credentials::Sas(pairs) => req.query.extend(pairs.iter().cloned()),
            Credentials::Bearer(token) => req
                .headers
                .push(("Authorization".into(), format!("Bearer {token}"))),
            Credentials::ClientSecret {
                tenant,
                client_id,
                secret,
                token,
            } => {
                let t = token.get(|| {
                    oauth_token(
                        &format!("https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"),
                        &[
                            ("grant_type", "client_credentials"),
                            ("client_id", client_id),
                            ("client_secret", secret),
                            ("scope", "https://storage.azure.com/.default"),
                        ],
                    )
                })?;
                req.headers
                    .push(("Authorization".into(), format!("Bearer {t}")));
            }
        }
        Ok(())
    }

    fn range_header(&self) -> &'static str {
        "x-ms-range"
    }

    fn suffix_ranges(&self) -> bool {
        false
    }

    fn put_headers(&self, req: &mut Req) {
        req.headers
            .push(("x-ms-blob-type".into(), "BlockBlob".into()));
        req.headers
            .push(("Content-Type".into(), "application/octet-stream".into()));
    }

    fn list_page(&self, prefix: &str, delimiter: bool, next: Option<&str>) -> Result<Page> {
        let mut req = Req::new("GET", &self.origin, self.container_path())
            .query("restype", "container")
            .query("comp", "list")
            .query("prefix", prefix);
        if delimiter {
            req = req.query("delimiter", "/");
        }
        if let Some(n) = next {
            req = req.query("marker", n);
        }
        let resp = send(self, req)?;
        if !resp.ok() {
            return Err(resp.error(&format!(
                "listing {}/{}/{prefix}",
                self.account, self.container
            )));
        }
        let text = String::from_utf8_lossy(&resp.body);
        // a byte-order mark before the declaration, sometimes
        let doc = roxmltree::Document::parse(text.trim_start_matches('\u{feff}'))
            .map_err(|e| Error::Storage(format!("Azure listing: {e}")))?;
        Ok(Page {
            keys: xml_texts(&doc, "Blob", "Name"),
            prefixes: xml_texts(&doc, "BlobPrefix", "Name"),
            next: xml_text(doc.root(), "NextMarker").map(str::to_string),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    /// The string to sign as Azure's "Authorize with Shared Key" lays it
    /// out, signed here independently of `shared_key`'s own assembly.
    #[test]
    fn shared_key_string_to_sign() {
        let key = b"secret-key";
        let headers = h(&[
            ("x-ms-version", VERSION),
            ("x-ms-date", "Sun, 11 Oct 2009 21:49:13 GMT"),
            ("x-ms-range", "bytes=0-9"),
            ("Content-Type", "application/octet-stream"),
        ]);
        let query = h(&[("restype", "container"), ("comp", "list")]);
        let got = shared_key("acct", key, "PUT", "/cont/a%20b", &query, &headers, 5);
        let want = "PUT\n\n\n5\n\napplication/octet-stream\n\n\n\n\n\n\n\
            x-ms-date:Sun, 11 Oct 2009 21:49:13 GMT\nx-ms-range:bytes=0-9\nx-ms-version:2023-11-03\n\
            /acct/cont/a%20b\ncomp:list\nrestype:container";
        assert_eq!(got, base64(&hmac_sha256(key, want.as_bytes())));
    }

    #[test]
    fn listing_pages_parse() {
        let xml = "\u{feff}<?xml version=\"1.0\" encoding=\"utf-8\"?>\
<EnumerationResults ServiceEndpoint=\"https://a.blob.core.windows.net/\" ContainerName=\"c\">\
<Prefix>img/</Prefix><Blobs><Blob><Name>img/zarr.json</Name><Properties/></Blob>\
<BlobPrefix><Name>img/0/</Name></BlobPrefix></Blobs><NextMarker>2!80!MDAw</NextMarker></EnumerationResults>";
        let doc = roxmltree::Document::parse(xml.trim_start_matches('\u{feff}')).unwrap();
        assert_eq!(xml_texts(&doc, "Blob", "Name"), ["img/zarr.json"]);
        assert_eq!(xml_texts(&doc, "BlobPrefix", "Name"), ["img/0/"]);
        assert_eq!(xml_text(doc.root(), "NextMarker"), Some("2!80!MDAw"));
    }
}
