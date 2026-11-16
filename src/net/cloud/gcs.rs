//! Google Cloud Storage: objects through its XML API, listings through its
//! JSON API, both at `storage.googleapis.com`.
//!
//! Credentials from the environment: a service account's key file at
//! `GOOGLE_SERVICE_ACCOUNT`, `GOOGLE_SERVICE_ACCOUNT_PATH` or
//! `GOOGLE_APPLICATION_CREDENTIALS`, or the key itself in
//! `GOOGLE_SERVICE_ACCOUNT_KEY` (a signed JWT traded for a token, RS256), or
//! a user's (`"type": "authorized_user"`, as `gcloud auth
//! application-default login` writes; its refresh token traded for one).
//! None: unsigned.

use super::crypto::{base64url, pem_der, rsa_sha256};
use super::{Page, Provider, Req, Token, env_any, oauth_token, send};
use crate::error::{Error, Result};
use crate::net::uri_encode;

const ORIGIN: &str = "https://storage.googleapis.com";
const TOKEN_URI: &str = "https://oauth2.googleapis.com/token";
const SCOPE: &str = "https://www.googleapis.com/auth/devstorage.read_write";

enum Credentials {
    Anonymous,
    ServiceAccount {
        email: String,
        key_id: Option<String>,
        /// The PKCS#8 private key.
        der: Vec<u8>,
        token_uri: String,
    },
    User {
        client_id: String,
        client_secret: String,
        refresh_token: String,
    },
}

pub(crate) struct Gcs {
    bucket: String,
    credentials: Credentials,
    token: Token,
}

/// The JWT a service account signs to ask for a token, issued at `now`
/// (seconds since the epoch).
pub(crate) fn service_account_jwt(
    email: &str,
    key_id: Option<&str>,
    der: &[u8],
    token_uri: &str,
    now: i64,
) -> Result<String> {
    let mut header = serde_json::json!({"alg": "RS256", "typ": "JWT"});
    if let Some(k) = key_id {
        header["kid"] = k.into();
    }
    let claims = serde_json::json!({
        "iss": email, "scope": SCOPE, "aud": token_uri, "iat": now, "exp": now + 3600,
    });
    let signed = format!(
        "{}.{}",
        base64url(header.to_string().as_bytes()),
        base64url(claims.to_string().as_bytes())
    );
    let signature = rsa_sha256(der, signed.as_bytes())?;
    Ok(format!("{signed}.{}", base64url(&signature)))
}

impl Gcs {
    pub(crate) fn from_env(bucket: &str) -> Result<Gcs> {
        let json =
            if let Some(key) = env_any(&["GOOGLE_SERVICE_ACCOUNT_KEY"]) {
                Some(key)
            } else if let Some(path) = env_any(&[
                "GOOGLE_SERVICE_ACCOUNT",
                "GOOGLE_SERVICE_ACCOUNT_PATH",
                "GOOGLE_APPLICATION_CREDENTIALS",
            ]) {
                Some(std::fs::read_to_string(&path).map_err(|e| {
                    Error::Storage(format!("reading Google credentials {path}: {e}"))
                })?)
            } else {
                None
            };
        let credentials = match json {
            None => Credentials::Anonymous,
            Some(json) => Self::credentials(&json)?,
        };
        Ok(Gcs {
            bucket: bucket.to_string(),
            credentials,
            token: Token::default(),
        })
    }

    fn credentials(json: &str) -> Result<Credentials> {
        let v: serde_json::Value = serde_json::from_str(json)?;
        let field = |k: &str| -> Result<String> {
            v[k].as_str()
                .map(str::to_string)
                .ok_or_else(|| Error::Storage(format!("Google credentials without {k}")))
        };
        match v["type"].as_str() {
            Some("service_account") => Ok(Credentials::ServiceAccount {
                email: field("client_email")?,
                key_id: v["private_key_id"].as_str().map(str::to_string),
                der: pem_der(&field("private_key")?)?,
                token_uri: v["token_uri"].as_str().unwrap_or(TOKEN_URI).to_string(),
            }),
            Some("authorized_user") => Ok(Credentials::User {
                client_id: field("client_id")?,
                client_secret: field("client_secret")?,
                refresh_token: field("refresh_token")?,
            }),
            other => Err(Error::Storage(format!(
                "unsupported Google credentials type: {other:?}"
            ))),
        }
    }

    fn bearer(&self) -> Result<Option<String>> {
        match &self.credentials {
            Credentials::Anonymous => Ok(None),
            Credentials::ServiceAccount {
                email,
                key_id,
                der,
                token_uri,
            } => self
                .token
                .get(|| {
                    let now = chrono::Utc::now().timestamp();
                    let jwt = service_account_jwt(email, key_id.as_deref(), der, token_uri, now)?;
                    oauth_token(
                        token_uri,
                        &[
                            ("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer"),
                            ("assertion", &jwt),
                        ],
                    )
                })
                .map(Some),
            Credentials::User {
                client_id,
                client_secret,
                refresh_token,
            } => self
                .token
                .get(|| {
                    oauth_token(
                        TOKEN_URI,
                        &[
                            ("grant_type", "refresh_token"),
                            ("client_id", client_id),
                            ("client_secret", client_secret),
                            ("refresh_token", refresh_token),
                        ],
                    )
                })
                .map(Some),
        }
    }
}

impl Provider for Gcs {
    fn object(&self, method: &'static str, key: &str) -> Req {
        Req::new(
            method,
            ORIGIN,
            format!("/{}/{}", self.bucket, uri_encode(key, true)),
        )
    }

    fn authorize(&self, req: &mut Req) -> Result<()> {
        if let Some(token) = self.bearer()? {
            req.headers
                .push(("Authorization".into(), format!("Bearer {token}")));
        }
        Ok(())
    }

    fn list_page(&self, prefix: &str, delimiter: bool, next: Option<&str>) -> Result<Page> {
        let mut req = Req::new(
            "GET",
            ORIGIN,
            format!("/storage/v1/b/{}/o", uri_encode(&self.bucket, false)),
        )
        .query("prefix", prefix)
        .query("fields", "items(name),prefixes,nextPageToken");
        if delimiter {
            req = req.query("delimiter", "/");
        }
        if let Some(n) = next {
            req = req.query("pageToken", n);
        }
        let resp = send(self, req)?;
        if !resp.ok() {
            return Err(resp.error(&format!("listing gs://{}/{prefix}", self.bucket)));
        }
        let v: serde_json::Value = serde_json::from_slice(&resp.body)?;
        let strings = |a: &serde_json::Value| -> Vec<String> {
            a.as_array()
                .into_iter()
                .flatten()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect()
        };
        Ok(Page {
            keys: v["items"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|i| i["name"].as_str().map(str::to_string))
                .collect(),
            prefixes: strings(&v["prefixes"]),
            next: v["nextPageToken"].as_str().map(str::to_string),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;

    #[test]
    fn a_service_account_jwt_is_signed_rs256() {
        let pem = include_str!("../../../tests/fixtures/cloud/rsa-test-key.pem");
        let key = serde_json::json!({
            "type": "service_account", "client_email": "x@p.iam.gserviceaccount.com",
            "private_key_id": "k1", "private_key": pem,
        })
        .to_string();
        let Credentials::ServiceAccount {
            email,
            key_id,
            der,
            token_uri,
        } = Gcs::credentials(&key).unwrap()
        else {
            panic!("not a service account");
        };
        assert_eq!(token_uri, TOKEN_URI);
        let jwt = service_account_jwt(&email, key_id.as_deref(), &der, &token_uri, 1_700_000_000)
            .unwrap();
        let parts: Vec<&str> = jwt.split('.').collect();
        assert_eq!(parts.len(), 3);
        let decode = |s: &str| {
            base64::engine::general_purpose::URL_SAFE_NO_PAD
                .decode(s)
                .unwrap()
        };
        let header: serde_json::Value = serde_json::from_slice(&decode(parts[0])).unwrap();
        let claims: serde_json::Value = serde_json::from_slice(&decode(parts[1])).unwrap();
        assert_eq!(header["alg"], "RS256");
        assert_eq!(header["kid"], "k1");
        assert_eq!(claims["iss"], "x@p.iam.gserviceaccount.com");
        assert_eq!(claims["exp"], 1_700_003_600);
        assert_eq!(decode(parts[2]).len(), 256, "a 2048-bit signature");
    }

    #[test]
    fn user_credentials_parse() {
        let json = r#"{"type": "authorized_user", "client_id": "i", "client_secret": "s", "refresh_token": "r"}"#;
        assert!(matches!(
            Gcs::credentials(json).unwrap(),
            Credentials::User { .. }
        ));
        assert!(Gcs::credentials(r#"{"type": "external_account"}"#).is_err());
    }
}
