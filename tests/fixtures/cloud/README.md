A throwaway 2048-bit RSA key (PKCS#8) for tests only, made with
`openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048`, and
`openssl dgst -sha256 -sign` of the bytes `header.claims` (base64): an
independent RS256 signature to check src/net/cloud/crypto.rs against.
