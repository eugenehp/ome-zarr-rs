#!/usr/bin/env sh
# Run the wasm test suite (tests/wasm.rs) in node with wasm-bindgen-test-runner
# (`cargo install wasm-bindgen-cli` at the version pinned in Cargo.toml).
set -e
export CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner
cargo test --target wasm32-unknown-unknown --no-default-features --features io --test wasm "$@"
