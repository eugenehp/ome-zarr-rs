//! Amazon S3 and S3-compatible services, signed with AWS Signature Version 4.
//!
//! Configuration from the environment: `AWS_ENDPOINT_URL` (or
//! `AWS_ENDPOINT_URL_S3`) for another service, reached path-style unless
//! `AWS_VIRTUAL_HOSTED_STYLE_REQUEST`; `AWS_REGION` / `AWS_DEFAULT_REGION`
//! (else the profile's, else the bucket's own, asked of AWS); credentials
//! from `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_SESSION_TOKEN`,
//! from the `AWS_PROFILE` profile of `~/.aws/credentials`, or by web identity
//! (`AWS_WEB_IDENTITY_TOKEN_FILE` and `AWS_ROLE_ARN`, through STS). None of
//! those, or `AWS_SKIP_SIGNATURE`: unsigned.

use std::collections::HashMap;
use std::path::PathBuf;

use super::crypto::{hex, hmac_sha256, sha256};
use super::{Page, Provider, Req, Token, env, env_any, env_true, send, xml_text, xml_texts};
use crate::error::{Error, Result};
use crate::net::uri_encode;

/// A key: its id, secret and session token.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Key {
    pub id: String,
    pub secret: String,
    pub token: Option<String>,
}

enum Credentials {
    Anonymous,
    Static(Key),
    /// STS's AssumeRoleWithWebIdentity.
    WebIdentity {
        role_arn: String,
        session: String,
        token_file: String,
        key: Token<Key>,
    },
}

pub(crate) struct S3 {
    bucket: String,
    region: String,
    /// `https://host[:port]` of another service, if not AWS.
    endpoint: Option<String>,
    path_style: bool,
    credentials: Credentials,
}

/// `[name]` sections of an AWS config or credentials file.
fn ini(path: &PathBuf) -> HashMap<String, HashMap<String, String>> {
    let mut out: HashMap<String, HashMap<String, String>> = HashMap::new();
    let Ok(text) = std::fs::read_to_string(path) else {
        return out;
    };
    let mut section = String::new();
    for line in text.lines().map(str::trim) {
        if line.starts_with('#') || line.starts_with(';') || line.is_empty() {
            continue;
        }
        if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            section = name.trim().to_string();
        } else if let Some((k, v)) = line.split_once('=') {
            out.entry(section.clone())
                .or_default()
                .insert(k.trim().to_string(), v.trim().to_string());
        }
    }
    out
}

fn aws_file(var: &str, name: &str) -> Option<PathBuf> {
    env(var).map(PathBuf::from).or_else(|| {
        let home = env("HOME").or_else(|| env("USERPROFILE"))?;
        Some(PathBuf::from(home).join(".aws").join(name))
    })
}

impl S3 {
    pub(crate) fn from_env(bucket: &str) -> Result<S3> {
        let profile = env("AWS_PROFILE");
        let config = aws_file("AWS_CONFIG_FILE", "config").map(|p| ini(&p));
        let profile_config = |key: &str| -> Option<String> {
            let name = profile.as_deref().unwrap_or("default");
            let section = if name == "default" {
                "default".to_string()
            } else {
                format!("profile {name}")
            };
            config.as_ref()?.get(&section)?.get(key).cloned()
        };
        let endpoint = env_any(&["AWS_ENDPOINT_URL_S3", "AWS_ENDPOINT_URL"])
            .or_else(|| profile_config("endpoint_url"))
            .map(|e| e.trim_end_matches('/').to_string());
        let credentials = if env_true("AWS_SKIP_SIGNATURE") {
            Credentials::Anonymous
        } else if let (Some(id), Some(secret)) =
            (env("AWS_ACCESS_KEY_ID"), env("AWS_SECRET_ACCESS_KEY"))
        {
            Credentials::Static(Key {
                id,
                secret,
                token: env("AWS_SESSION_TOKEN"),
            })
        } else if let (Some(token_file), Some(role_arn)) =
            (env("AWS_WEB_IDENTITY_TOKEN_FILE"), env("AWS_ROLE_ARN"))
        {
            Credentials::WebIdentity {
                role_arn,
                session: env("AWS_ROLE_SESSION_NAME").unwrap_or_else(|| "ome-zarr-rs".into()),
                token_file,
                key: Token::default(),
            }
        } else if let Some(name) = &profile {
            let file = aws_file("AWS_SHARED_CREDENTIALS_FILE", "credentials")
                .map(|p| ini(&p))
                .unwrap_or_default();
            let section = file.get(name).ok_or_else(|| {
                Error::Storage(format!("AWS profile `{name}` not in the credentials file"))
            })?;
            let get = |k: &str| section.get(k).cloned();
            match (get("aws_access_key_id"), get("aws_secret_access_key")) {
                (Some(id), Some(secret)) => Credentials::Static(Key {
                    id,
                    secret,
                    token: get("aws_session_token"),
                }),
                _ => {
                    return Err(Error::Storage(format!(
                        "AWS profile `{name}` has no aws_access_key_id and aws_secret_access_key"
                    )));
                }
            }
        } else {
            Credentials::Anonymous
        };
        let path_style = match &endpoint {
            Some(_) => !env_true("AWS_VIRTUAL_HOSTED_STYLE_REQUEST"),
            // a dotted name does not fit the wildcard certificate
            None => bucket.contains('.'),
        };
        let mut s3 = S3 {
            bucket: bucket.to_string(),
            region: String::new(),
            endpoint,
            path_style,
            credentials,
        };
        s3.region = match env_any(&["AWS_REGION", "AWS_DEFAULT_REGION"])
            .or_else(|| profile_config("region"))
        {
            Some(r) => r,
            None if s3.endpoint.is_none() => {
                s3.bucket_region().unwrap_or_else(|| "us-east-1".into())
            }
            None => "us-east-1".into(),
        };
        Ok(s3)
    }

    /// The bucket's region, as AWS tells anyone who asks.
    fn bucket_region(&self) -> Option<String> {
        let url = if self.path_style {
            format!("https://s3.amazonaws.com/{}", self.bucket)
        } else {
            format!("https://{}.s3.amazonaws.com/", self.bucket)
        };
        let resp =
            crate::net::request(crate::retry::Policy::default(), "HEAD", &url, &[], &[]).ok()?;
        resp.header("x-amz-bucket-region").map(str::to_string)
    }

    fn origin(&self) -> String {
        match (&self.endpoint, self.path_style) {
            (Some(e), true) => e.clone(),
            (Some(e), false) => match e.split_once("://") {
                Some((scheme, host)) => format!("{scheme}://{}.{host}", self.bucket),
                None => format!("https://{}.{e}", self.bucket),
            },
            (None, true) => format!("https://s3.{}.amazonaws.com", self.region),
            (None, false) => format!("https://{}.s3.{}.amazonaws.com", self.bucket, self.region),
        }
    }

    /// The path of `key` (`""`: the bucket).
    fn path(&self, key: &str) -> String {
        let key = uri_encode(key, true);
        if self.path_style {
            format!("/{}/{key}", self.bucket)
        } else {
            format!("/{key}")
        }
    }

    fn key(&self) -> Result<Option<Key>> {
        match &self.credentials {
            Credentials::Anonymous => Ok(None),
            Credentials::Static(k) => Ok(Some(k.clone())),
            Credentials::WebIdentity {
                role_arn,
                session,
                token_file,
                key,
            } => key
                // STS keys last an hour
                .get(|| Ok((self.assume_role(role_arn, session, token_file)?, 3600)))
                .map(Some),
        }
    }

    fn assume_role(&self, role_arn: &str, session: &str, token_file: &str) -> Result<Key> {
        let token = std::fs::read_to_string(token_file)?;
        let req = Req::new(
            "GET",
            &format!("https://sts.{}.amazonaws.com", self.region),
            "/".into(),
        )
        .query("Action", "AssumeRoleWithWebIdentity")
        .query("Version", "2011-06-15")
        .query("DurationSeconds", "3600")
        .query("RoleArn", role_arn)
        .query("RoleSessionName", session)
        .query("WebIdentityToken", token.trim());
        let url = req.url();
        let resp = crate::net::request(crate::retry::Policy::default(), "GET", &url, &[], &[])?;
        if !resp.ok() {
            return Err(resp.error("AssumeRoleWithWebIdentity"));
        }
        let text = String::from_utf8_lossy(&resp.body);
        let doc = roxmltree::Document::parse(&text)
            .map_err(|e| Error::Storage(format!("STS answer: {e}")))?;
        let root = doc.root();
        let get = |n: &str| {
            xml_text(root, n)
                .map(str::to_string)
                .ok_or_else(|| Error::Storage(format!("STS answer without {n}")))
        };
        Ok(Key {
            id: get("AccessKeyId")?,
            secret: get("SecretAccessKey")?,
            token: Some(get("SessionToken")?),
        })
    }
}

/// The `Authorization` header for a request, signed with SigV4 for
/// `service` in `region` at `amz_date` (`YYYYMMDDTHHMMSSZ`). `headers` are
/// every header sent but `Host` (which is `host`), names in any case.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sigv4(
    method: &str,
    host: &str,
    path: &str,
    query: &[(String, String)],
    headers: &[(String, String)],
    payload_hash: &str,
    key: &Key,
    region: &str,
    service: &str,
    amz_date: &str,
) -> String {
    let date = &amz_date[..8];
    let mut q: Vec<(String, String)> = query
        .iter()
        .map(|(k, v)| (uri_encode(k, false), uri_encode(v, false)))
        .collect();
    q.sort();
    let canonical_query = q
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join("&");
    let mut h: Vec<(String, String)> = headers
        .iter()
        .map(|(k, v)| {
            let v = v.split_whitespace().collect::<Vec<_>>().join(" ");
            (k.to_ascii_lowercase(), v)
        })
        .collect();
    h.push(("host".into(), host.to_string()));
    h.sort();
    let canonical_headers: String = h.iter().map(|(k, v)| format!("{k}:{v}\n")).collect();
    let signed_headers = h
        .iter()
        .map(|(k, _)| k.as_str())
        .collect::<Vec<_>>()
        .join(";");
    let canonical = format!(
        "{method}\n{path}\n{canonical_query}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
    );
    let scope = format!("{date}/{region}/{service}/aws4_request");
    let to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}",
        hex(&sha256(canonical.as_bytes()))
    );
    let k = hmac_sha256(format!("AWS4{}", key.secret).as_bytes(), date.as_bytes());
    let k = hmac_sha256(&k, region.as_bytes());
    let k = hmac_sha256(&k, service.as_bytes());
    let k = hmac_sha256(&k, b"aws4_request");
    let signature = hex(&hmac_sha256(&k, to_sign.as_bytes()));
    format!(
        "AWS4-HMAC-SHA256 Credential={}/{scope},SignedHeaders={signed_headers},Signature={signature}",
        key.id
    )
}

impl Provider for S3 {
    fn object(&self, method: &'static str, key: &str) -> Req {
        Req::new(method, &self.origin(), self.path(key))
    }

    fn authorize(&self, req: &mut Req) -> Result<()> {
        let Some(key) = self.key()? else {
            return Ok(());
        };
        let amz_date = chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
        let payload_hash = hex(&sha256(&req.body));
        req.headers.push(("x-amz-date".into(), amz_date.clone()));
        req.headers
            .push(("x-amz-content-sha256".into(), payload_hash.clone()));
        if let Some(t) = &key.token {
            req.headers.push(("x-amz-security-token".into(), t.clone()));
        }
        let auth = sigv4(
            req.method,
            req.host(),
            &req.path,
            &req.query,
            &req.headers,
            &payload_hash,
            &key,
            &self.region,
            "s3",
            &amz_date,
        );
        req.headers.push(("Authorization".into(), auth));
        Ok(())
    }

    fn list_page(&self, prefix: &str, delimiter: bool, next: Option<&str>) -> Result<Page> {
        let mut req = Req::new("GET", &self.origin(), self.path(""))
            .query("list-type", "2")
            .query("prefix", prefix);
        if delimiter {
            req = req.query("delimiter", "/");
        }
        if let Some(n) = next {
            req = req.query("continuation-token", n);
        }
        let resp = send(self, req)?;
        if !resp.ok() {
            return Err(resp.error(&format!("listing s3://{}/{prefix}", self.bucket)));
        }
        let text = String::from_utf8_lossy(&resp.body);
        let doc = roxmltree::Document::parse(&text)
            .map_err(|e| Error::Storage(format!("S3 listing: {e}")))?;
        let truncated = xml_text(doc.root(), "IsTruncated") == Some("true");
        Ok(Page {
            keys: xml_texts(&doc, "Contents", "Key"),
            prefixes: xml_texts(&doc, "CommonPrefixes", "Prefix"),
            next: truncated
                .then(|| xml_text(doc.root(), "NextContinuationToken").map(str::to_string))
                .flatten(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The examples of AWS's "Signature Calculations for the Authorization
    /// Header" (S3 API reference), with their published signatures.
    fn example() -> Key {
        Key {
            id: "AKIAIOSFODNN7EXAMPLE".into(),
            secret: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".into(),
            token: None,
        }
    }

    const EMPTY: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

    fn signature(auth: &str) -> &str {
        auth.rsplit("Signature=").next().unwrap()
    }

    fn h(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn get_object() {
        let auth = sigv4(
            "GET",
            "examplebucket.s3.amazonaws.com",
            "/test.txt",
            &[],
            &h(&[
                ("Range", "bytes=0-9"),
                ("x-amz-content-sha256", EMPTY),
                ("x-amz-date", "20130524T000000Z"),
            ]),
            EMPTY,
            &example(),
            "us-east-1",
            "s3",
            "20130524T000000Z",
        );
        assert!(auth.contains(
            "Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request,SignedHeaders=host;range;x-amz-content-sha256;x-amz-date,"
        ));
        assert_eq!(
            signature(&auth),
            "f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41"
        );
    }

    #[test]
    fn get_bucket_lifecycle() {
        let auth = sigv4(
            "GET",
            "examplebucket.s3.amazonaws.com",
            "/",
            &[("lifecycle".into(), String::new())],
            &h(&[
                ("x-amz-content-sha256", EMPTY),
                ("x-amz-date", "20130524T000000Z"),
            ]),
            EMPTY,
            &example(),
            "us-east-1",
            "s3",
            "20130524T000000Z",
        );
        assert_eq!(
            signature(&auth),
            "fea454ca298b7da1c68078a5d1bdbfbbe0d65c699e0f91ac7a200a0136783543"
        );
    }

    #[test]
    fn list_objects() {
        let auth = sigv4(
            "GET",
            "examplebucket.s3.amazonaws.com",
            "/",
            &[
                ("max-keys".into(), "2".into()),
                ("prefix".into(), "J".into()),
            ],
            &h(&[
                ("x-amz-content-sha256", EMPTY),
                ("x-amz-date", "20130524T000000Z"),
            ]),
            EMPTY,
            &example(),
            "us-east-1",
            "s3",
            "20130524T000000Z",
        );
        assert_eq!(
            signature(&auth),
            "34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7"
        );
    }

    #[test]
    fn listing_pages_parse() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>b</Name><Prefix>img/</Prefix><KeyCount>3</KeyCount><IsTruncated>true</IsTruncated>
  <NextContinuationToken>1ueGcxLPRx1Tr/XYExHnhbYLgveDs2J/wm36Hy4vbOwM=</NextContinuationToken>
  <Contents><Key>img/zarr.json</Key><Size>10</Size></Contents>
  <Contents><Key>img/a &amp; b</Key><Size>1</Size></Contents>
  <CommonPrefixes><Prefix>img/0/</Prefix></CommonPrefixes>
</ListBucketResult>"#;
        let doc = roxmltree::Document::parse(xml).unwrap();
        assert_eq!(
            xml_texts(&doc, "Contents", "Key"),
            ["img/zarr.json", "img/a & b"]
        );
        assert_eq!(xml_texts(&doc, "CommonPrefixes", "Prefix"), ["img/0/"]);
        assert_eq!(xml_text(doc.root(), "IsTruncated"), Some("true"));
    }

    #[test]
    fn profiles_parse() {
        let dir = std::env::temp_dir().join(format!("ome-zarr-aws-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("credentials");
        std::fs::write(
            &p,
            "[default]\naws_access_key_id = A\n\n# comment\n[work]\naws_access_key_id=B\naws_secret_access_key = s=x\n",
        )
        .unwrap();
        let f = ini(&p);
        assert_eq!(f["default"]["aws_access_key_id"], "A");
        assert_eq!(f["work"]["aws_secret_access_key"], "s=x");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
