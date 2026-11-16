"""Has zarr-python and numcodecs (c-blosc, libzstd) read what tests/codecs.rs
wrote with the crate's pure-Rust codecs ($CARGO_TARGET_TMPDIR/written/codecs,
target/tmp by default): arrays (v2 and v3, blosc and zstd) and raw blosc
chunks of every compressor and shuffle.
    cargo test --test codecs && python3 scripts/check_codecs.py
"""
import os
import pathlib

import numcodecs
import numpy as np
import zarr

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(os.environ.get("CARGO_TARGET_TMPDIR", ROOT / "target/tmp")) / "written" / "codecs"
want = np.load(ROOT / "tests/fixtures/codecs/array.npy")
n = 0
for path in sorted(OUT.glob("*.zarr")):
    got = zarr.open_array(str(path), mode="r")[:]
    assert got.dtype == want.dtype and np.array_equal(got, want), path
    n += 1
raw = (OUT / "raw.bin").read_bytes()
blosc = numcodecs.Blosc()
for path in sorted(OUT.glob("blosc_*.bin")):
    assert blosc.decode(path.read_bytes()) == raw, path
    n += 1
print(f"zarr-python and numcodecs read all {n} written arrays and chunks")
