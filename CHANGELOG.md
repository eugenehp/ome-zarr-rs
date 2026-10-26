# Changelog

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
