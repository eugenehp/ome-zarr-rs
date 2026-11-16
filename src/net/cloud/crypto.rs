//! What signing requests takes: SHA-256 and HMAC-SHA-256 (RustCrypto's), and
//! RSASSA-PKCS1-v1_5 with SHA-256 for Google's service-account tokens:
//! graviola's (constant time) where `tls-graviola` runs, else RustCrypto's.

use base64::Engine;

use crate::error::{Error, Result};

pub(crate) fn sha256(data: &[u8]) -> [u8; 32] {
    use sha2::Digest;
    sha2::Sha256::digest(data).into()
}

pub(crate) fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    use hmac::{KeyInit, Mac};
    let mut mac = hmac::Hmac::<sha2::Sha256>::new_from_slice(key).expect("HMAC takes any key");
    mac.update(data);
    mac.finalize().into_bytes().into()
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub(crate) fn base64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

pub(crate) fn base64_decode(s: &str) -> Result<Vec<u8>> {
    base64::engine::general_purpose::STANDARD
        .decode(s.trim())
        .map_err(|e| Error::Storage(format!("invalid base64: {e}")))
}

/// Base64url without padding, as JWTs are.
pub(crate) fn base64url(bytes: &[u8]) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

/// The DER inside a PEM document (`-----BEGIN …-----`).
pub(crate) fn pem_der(pem: &str) -> Result<Vec<u8>> {
    let body: String = pem
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with("-----"))
        .collect();
    base64_decode(&body)
}

/// `message` signed with RSASSA-PKCS1-v1_5 and SHA-256 (RS256), by the
/// PKCS#8 private key `der`.
pub(crate) fn rsa_sha256(der: &[u8], message: &[u8]) -> Result<Vec<u8>> {
    #[cfg(all(
        feature = "tls-graviola",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    {
        let key = graviola::signing::rsa::SigningKey::from_pkcs8_der(der)
            .map_err(|e| Error::Storage(format!("invalid RSA private key: {e:?}")))?;
        let mut signature = vec![0; key.modulus_len_bytes()];
        let n = key
            .sign_pkcs1_sha256(&mut signature, message)
            .map_err(|e| Error::Storage(format!("RSA signing failed: {e:?}")))?
            .len();
        signature.truncate(n);
        Ok(signature)
    }
    #[cfg(not(all(
        feature = "tls-graviola",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )))]
    {
        use rsa::pkcs8::DecodePrivateKey;
        use rsa::signature::{SignatureEncoding, Signer};
        let key = rsa::PrivateKey::from_pkcs8_der(der)
            .map_err(|e| Error::Storage(format!("invalid RSA private key: {e}")))?;
        let signer = rsa::pkcs1v15::SigningKey::<rsa::sha2::Sha256>::new(key);
        Ok(signer.sign(message).to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hmac_matches_rfc_4231() {
        // test case 2
        let mac = hmac_sha256(b"Jefe", b"what do ya want for nothing?");
        assert_eq!(
            hex(&mac),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
        assert_eq!(
            hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn rs256_matches_openssl() {
        // tests/fixtures/cloud/README.md
        let key = pem_der(include_str!(
            "../../../tests/fixtures/cloud/rsa-test-key.pem"
        ))
        .unwrap();
        let want = include_str!("../../../tests/fixtures/cloud/rsa-test-signature.b64");
        let got = rsa_sha256(&key, b"header.claims").unwrap();
        assert_eq!(base64(&got), want.trim());
    }
}
