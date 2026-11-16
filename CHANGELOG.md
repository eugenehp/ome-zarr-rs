# Changelog

## 0.3.0

0.2.0 was not released: its changes are here.

- Zarr I/O is the crate's own, in pure Rust, and zarrs is gone: stores behind
  `store::Storage` (memory, filesystem, zip, http, the clouds), chunk keys
  and fill values in `zarr`, and the codec chain in `codecs`: `bytes`,
  `transpose`, `sharding_indexed` (only the inner chunks a region needs read,
  by byte range), blosc (`blosc`), zstd (zstd-rs), gzip, zlib, bz2, lz4, lzma
  and crc32c. numcodecs' and zarr-python's chunks and arrays read, and what
  is written read by them; as zarr-python, a chunk holding only the fill
  value is not stored. With `io` alone no C is compiled, on any target.
- The network is pure Rust too: https through ureq and rustls, with
  graviola's cryptography (`tls-graviola`, default) or RustCrypto's
  (`tls-rustcrypto`), and S3, Google Cloud Storage and Azure Blob Storage
  clients of the crate's own (`net::cloud`) in place of reqwest, object_store
  and tokio: SigV4 (checked against AWS's published examples), service
  account and user tokens for Google (RS256), Shared Key, SAS and OAuth 2 for
  Azure, from the same environment variables. OpenSSL and aws-lc are gone:
  with the default features nothing is compiled from C, on any target.
  **Breaking:** `http` needs a `tls-*` feature (`tls-graviola` is in the
  defaults).
- **Breaking:** `store::DynStore` is `Arc<dyn store::Storage>`, a trait of
  this crate's; `store::ReadOnly`, `store::UnlistableReadable` and
  `Store::as_readable` are gone (a store that does not write or list leaves
  those methods out), and zarrs' error types no longer convert into `Error`.
  `array::Element` reads and writes its own bytes (`read_le`, `write_le`).
- The `io` feature, on by default: arrays, stores, codecs, pyramids, reading
  and writing. Without it the crate is the OME-NGFF metadata alone (`models`,
  `linalg`, `pyjson`, `error`), with no Zarr I/O and no C, for an app with its
  own Zarr reader.
- **Breaking:** `default-features = false` now leaves `io` out too; add
  `features = ["io"]` to keep the reader.
- `JsonMap` and `recursive_pop_nones` live in `pyjson`; the old paths still work.
- http(s) stores and zipped filesets ask again when a server refuses for now
  (429, 500, 502, 503, 504) or a connection is lost, with `Retry-After` honoured
  where the response is at hand.
- A remote zip's block cache drops the least recently read block, so the
  central directory stays.
- Members of a zip archive read in parallel, each read with a source of its
  own; a remote archive's blocks are fetched side by side. **Breaking:** a
  `ZipSource` must now be `Clone`, its clones reading independently.
- `lazy::fetch_keys` asks for a failed key once more, and a key that still
  fails no longer loses the rest of its round.
- **Breaking:** `rust-version` is 1.99.
- An image missing several levels, or an array with several unreadable
  blocks, reports the first in order. Which one was reported depended on the
  threads' timing.
- threers 0.0.12, winit 0.30 and zip 9. zarrs, and with it the zstd,
  zstd-safe, zstd-sys, blosc-src, lz4-sys, libz-sys and paste crates (and
  their advisories), are gone from the tree.

## 0.1.0

First release: a Rust port of [ome-zarr-py](https://github.com/ome/ome-zarr-py)
(at commit `dfd7273e6`), verified against it output for output.

- Read OME-NGFF 0.1 – 0.6; write 0.4 (Zarr v2), 0.5 and 0.6 (Zarr v3), with
  sharding and the numcodecs compressors (blosc, zstd, gzip, zlib, bz2, lz4,
  lzma) and every numpy numeric type including float16 and complex.
- Multiscale pyramids with `resize`, `nearest`, `local_mean` and `zoom`
  (ports of scikit-image / scipy, bit-exact including dask's block-wise
  processing), the legacy `Scaler`, labels, plates, wells and scenes with
  coordinate transformation graphs.
- The `ome_zarr` command line (`info`, `download`, `view`, `finder`, `create`,
  `scale`, `csv_to_labels`, plus `render`), with argparse-identical
  arguments, help and errors, and Python-style logging.
- Local, zip, http(s) (including zipped filesets with range requests), S3,
  GCS and Azure sources.
- pydantic-identical validation errors for the ome-zarr-models metadata.
- GPU rendering and a native viewer with threers (feature `render`).
- Builds for `wasm32-unknown-unknown`.

See the README for the differences from ome-zarr-py and the platforms parity
is verified on.
