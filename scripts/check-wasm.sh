#!/usr/bin/env sh
# Build the library for wasm32-unknown-unknown. Every codec is pure Rust:
# no C compiler is needed.
set -e
cargo build --lib --target wasm32-unknown-unknown --no-default-features --features io "$@"
