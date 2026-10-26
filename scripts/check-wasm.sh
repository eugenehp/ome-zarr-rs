#!/usr/bin/env sh
# Build the library for wasm32-unknown-unknown.
#
# The zstd and blosc codecs are C; they need a clang with the WebAssembly
# backend (Apple's clang has none: `brew install llvm`). Override WASM_CLANG
# / WASM_AR to point elsewhere.
set -e
LLVM="${LLVM:-$(brew --prefix llvm 2>/dev/null || echo /usr)}"
export CC_wasm32_unknown_unknown="${WASM_CLANG:-$LLVM/bin/clang}"
export AR_wasm32_unknown_unknown="${WASM_AR:-$LLVM/bin/llvm-ar}"
cargo build --lib --target wasm32-unknown-unknown --no-default-features "$@"
