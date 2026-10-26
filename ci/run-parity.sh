#!/usr/bin/env bash
# Regenerate the Python fixtures inside a Linux container for the given
# platform (default: linux/amd64) and run the Rust tests against them.
#   ci/run-parity.sh [linux/amd64|linux/arm64] [path to ome-zarr-py]
set -euo pipefail
platform="${1:-linux/amd64}"
here="$(cd "$(dirname "$0")/.." && pwd)"
py_src="$(cd "${2:-$here/../ome-zarr-py}" && pwd)"
tag="ome-zarr-rs-parity:${platform//\//-}"
docker build --platform "$platform" -t "$tag" -f "$here/ci/Dockerfile" "$here/ci"
# TEST_CMD: run this instead of the full test suite (after the fixtures)
docker run --rm --platform "$platform" -e TEST_CMD="${TEST_CMD:-}" \
    -v "$here":/src/ome-zarr-rs:ro -v "$py_src":/src/ome-zarr-py:ro \
    -v "ome-zarr-rs-cargo-${platform//\//-}":/root/.cargo/registry \
    -v "ome-zarr-rs-target-${platform//\//-}":/work/target \
    "$tag" bash -euo pipefail -c '
        cp -r /src/ome-zarr-py /tmp/ome-zarr-py && pip install -q --no-deps -e /tmp/ome-zarr-py
        tar -C /src/ome-zarr-rs --exclude=./target -cf - . | tar -C /work -xf -
        cd /work
        for s in gen_parity_fixtures gen_ndimage_fixtures gen_zarr_fixtures gen_cli_fixtures gen_server_fixtures gen_linalg_fixtures gen_pydantic_fixtures gen_location_fixtures; do
            echo "== $s"; (cd /tmp && python -I /work/scripts/$s.py > /tmp/$s.log 2>&1) || { tail -30 /tmp/$s.log; exit 1; }
        done
        uname -m; python -c "import numpy, scipy; numpy.show_config()" 2>/dev/null | grep -A3 -i "blas\b" | head -8 || true
        if [ -n "$TEST_CMD" ]; then eval "$TEST_CMD"; exit $?; fi
        set +e
        cargo test --release --no-fail-fast > /tmp/test.log 2>&1
        code=$?
        grep -E "^test result|FAILED|^error" /tmp/test.log | sort | uniq -c
        grep -B2 -A30 "panicked at" /tmp/test.log | cut -c1-400 | head -400
        exit $code
    '
