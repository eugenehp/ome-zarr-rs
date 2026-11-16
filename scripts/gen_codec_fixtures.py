"""Writes tests/fixtures/codecs/: chunks numcodecs encodes (c-blosc and
libzstd), which the crate's pure-Rust blosc and zstd must decode, and a
zarr-python array with each, which it must read. tests/codecs.rs; the chunks
the tests encode are read back by scripts/check_codecs.py.
    python3 scripts/gen_codec_fixtures.py   (numcodecs, zarr, numpy)
"""
import json
import pathlib

import numpy as np
import numcodecs
import zarr

OUT = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "codecs"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(11)
cases = []
arrays = {
    "u1": (np.arange(20_000) % 251).astype("u1"),
    "i2": (np.sin(np.arange(12_000) / 50) * 3000).astype("<i2"),
    "f4": rng.standard_normal(8_000).astype("<f4").cumsum().astype("<f4"),
    "f8": np.repeat(rng.standard_normal(100), 37).astype("<f8"),
}
for dt, a in arrays.items():
    raw = a.tobytes()
    (OUT / f"{dt}.raw").write_bytes(raw)
    for cname in ["blosclz", "lz4", "lz4hc", "zlib", "zstd"]:
        for shuffle in [0, 1, 2, -1]:
            for blocksize in [0, 4096]:
                c = numcodecs.Blosc(cname=cname, clevel=5, shuffle=shuffle, blocksize=blocksize)
                name = f"blosc_{dt}_{cname}_{shuffle}_{blocksize}.bin"
                (OUT / name).write_bytes(c.encode(a))
                cases.append({"file": name, "raw": f"{dt}.raw"})
    for level in [1, 3, 9]:
        for checksum in [False, True]:
            c = numcodecs.Zstd(level=level, checksum=checksum)
            name = f"zstd_{dt}_{level}_{int(checksum)}.bin"
            (OUT / name).write_bytes(c.encode(raw))
            cases.append({"file": name, "raw": f"{dt}.raw"})
(OUT / "cases.json").write_text(json.dumps(cases, indent=1) + "\n")

# arrays zarr-python writes with each codec, v2 and v3
data = np.arange(64 * 48, dtype="<u2").reshape(64, 48) % 977
np.save(OUT / "array.npy", data)
for fmt in [2, 3]:
    for codec in ["blosc", "zstd"]:
        path = OUT / f"v{fmt}_{codec}.zarr"
        if fmt == 2:
            comp = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=2) if codec == "blosc" else numcodecs.Zstd(level=5)
            z = zarr.create_array(store=str(path), shape=data.shape, chunks=(20, 16), dtype="<u2", zarr_format=2, compressors=comp, overwrite=True)
        else:
            from zarr.codecs import BloscCodec, ZstdCodec
            comp = BloscCodec(cname="lz4", clevel=5, shuffle="shuffle", typesize=2) if codec == "blosc" else ZstdCodec(level=4, checksum=True)
            z = zarr.create_array(store=str(path), shape=data.shape, chunks=(20, 16), dtype="<u2", zarr_format=3, compressors=comp, overwrite=True)
        z[:] = data
print(len(cases), "chunks;", "arrays v2/v3 blosc/zstd")
