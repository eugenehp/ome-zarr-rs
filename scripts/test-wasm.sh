#!/usr/bin/env sh
# Run the wasm test suite (tests/wasm.rs) in node with wasm-bindgen-test-runner
# (`cargo install wasm-bindgen-cli` at the version pinned in Cargo.toml).
set -e
LLVM="${LLVM:-$(brew --prefix llvm 2>/dev/null || echo /usr)}"
export CC_wasm32_unknown_unknown="${WASM_CLANG:-$LLVM/bin/clang}"
export AR_wasm32_unknown_unknown="${WASM_AR:-$LLVM/bin/llvm-ar}"
export CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner
cargo test --target wasm32-unknown-unknown --no-default-features --test wasm "$@"
