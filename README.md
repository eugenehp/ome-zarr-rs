# ome-zarr (Rust)

[![crates.io](https://img.shields.io/crates/v/ome-zarr.svg)](https://crates.io/crates/ome-zarr)
[![docs.rs](https://docs.rs/ome-zarr/badge.svg)](https://docs.rs/ome-zarr)
[![CI](https://github.com/eugenehp/ome-zarr-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/eugenehp/ome-zarr-rs/actions/workflows/ci.yml)

A Rust implementation of [ome-zarr-py](https://github.com/ome/ome-zarr-py):
read and write [OME-Zarr](https://ngff.openmicroscopy.org/) (OME-NGFF) images,
labels, plates, wells and scenes, build multiscale pyramids, and look at them —
natively, on the GPU, with [threers](https://github.com/eugenehp/threers).

* **Every OME-NGFF version**: reads 0.1 – 0.6, writes 0.4 (Zarr v2), 0.5 and
  0.6 (Zarr v3), with sharding and the numcodecs compressors zarr-python
  writes (blosc, zstd, gzip, zlib, bz2, lz4, lzma). Every numpy numeric type:
  bool, (u)int8 – 64, float16 / 32 / 64 and complex64 / 128.
* **Byte-for-byte parity with ome-zarr-py**: for the same input, every metadata
  document and every pyramid level is identical to what the Python package
  writes. The downsampling kernels are ports of scikit-image's `resize` /
  `downscale_local_mean` and scipy's `gaussian_filter`, `zoom` and
  `map_coordinates`, down to their floating point evaluation order — including
  dask's block-wise processing of chunked arrays.
* **Local and online sources**: directories, zip archives, http(s) (including
  zipped filesets, read with range requests), S3, Google Cloud Storage and
  Azure, lazily read with a small dask-like chunked array engine; parallel
  writes with rayon.
* **GPU rendering with threers**: channel composites with the `omero` colours
  and windows, label overlays, headless PNG output and an interactive viewer.

## Command line

```bash
cargo install --git https://github.com/eugenehp/ome-zarr-rs   # or: cargo install --path .

ome_zarr create --method=coins coins.zarr            # sample data (also: astronaut, --format 0.4)
ome_zarr info coins.zarr                             # --stats for min/max
ome_zarr info https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr
ome_zarr view coins.zarr                             # native viewer (threers)
ome_zarr view --web coins.zarr                       # serve + open ome-ngff-validator
ome_zarr render coins.zarr -o coins.png              # headless GPU render
ome_zarr download https://…/6001240.zarr --output=.  # copy a (remote) image locally
ome_zarr finder images/                              # BioFile Finder CSV + browser
ome_zarr scale array.zarr out.zarr zyx --method nearest
ome_zarr csv_to_labels props.csv id "area#d,name" image.zarr label-value
```

Every command that reads an image accepts a local path or an online source:

| source | example |
| --- | --- |
| http(s) | `https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr` |
| zipped fileset (local or http) | `https://example.org/image.ome.zarr.zip` |
| S3 | `s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8` |
| S3-compatible | `AWS_ENDPOINT_URL=https://uk1s3.embassy.ebi.ac.uk ome_zarr info s3://idr/zarr/v0.4/idr0062A/6001240.zarr` |
| Google Cloud Storage | `gs://bucket/image.zarr` |
| Azure Blob Storage | `az://container/image.zarr` (with `AZURE_STORAGE_ACCOUNT_NAME`) |

Cloud credentials and settings come from the usual `AWS_*`, `GOOGLE_*` and
`AZURE_*` environment variables; with none set, requests are unsigned, so
public buckets work without an account. Remote zip archives are read with
HTTP range requests (only the parts needed), or downloaded once if the server
does not support them. `view --web` of a remote image opens
ome-ngff-validator on its public URL directly.

The commands and options are those of the Python `ome_zarr` tool, parsed the
way its `argparse` parser does (abbreviated options such as `--out`, `--x=v`,
the same usage, help and error messages and exit codes), and logging as
Python's `logging` prints it (`WARNING:ome_zarr.io:...`, `-v` / `-q` steps of
one level). `view --web` and `finder` serve the files as ome-zarr-py's
server does (HTTP/1.0, range requests, directory listings, error pages and
the request log). Two additions: `render`, and `view` now opens a native window (the
browser-based behaviour is `view --web`).

Viewer controls: drag to pan, scroll to zoom (the pyramid level follows the
zoom), `↑`/`↓` step through z, `[`/`]` through time, `1`–`9` toggle channels,
`L` toggles labels, `R` resets, `S` saves a screenshot, `Q` quits.

## Library

```rust
use ome_zarr::array::{DType, LazyArray, NdArray};
use ome_zarr::classes::{OmeZarrImage, OmeZarrLabels, OmeZarrMultiscale, LabelsArg, ScaleMap, Version};
use ome_zarr::scale::{Methods, ScaleFactors};

// a 2-channel z-stack
let data = NdArray::zeros(DType::UInt16, &[2, 64, 512, 512]);
let image = OmeZarrImage::from_axes(LazyArray::from_ndarray(data), "czyx")?
    .with_scale(ScaleMap::from([("z", 2.0), ("y", 0.3), ("x", 0.3)]))?
    .with_units(&[("z", "micrometer"), ("y", "micrometer"), ("x", "micrometer")]);

let labels = OmeZarrLabels::from_image(/* an OmeZarrImage of label values */ label_image)?;

let multiscale = OmeZarrMultiscale::builder(image)
    .scale_factors(Some(ScaleFactors::Ints(vec![2, 4, 8])))
    .method(Some(Methods::Resize))
    .channel_names(&["DAPI", "GFP"])
    .labels(LabelsArg::One(labels))
    .build()?;
multiscale.to_ome_zarr("image.zarr", None, Version::V05, true, true)?;

// read it back (any version, local or remote)
let ms = OmeZarrMultiscale::from_ome_zarr("image.zarr")?;
let level1 = ms.images()[1].data.compute()?;
```

The functional API mirrors `ome_zarr.writer` / `ome_zarr.reader`:

```rust
use ome_zarr::{io::parse_url_r, reader::Reader, writer::{self, ImageOptions}};

writer::write_image(&data, "out.zarr", &ImageOptions { axes: Some("zyx".into()), ..Default::default() })?;
for node in Reader::new(parse_url_r("out.zarr")?.unwrap())?.nodes()? {
    println!("{node}: {:?}", node.data.iter().map(|d| d.shape().to_vec()).collect::<Vec<_>>());
}
```

| Python | Rust |
|---|---|
| `ome_zarr.io.parse_url`, `ZarrLocation` | [`io`](src/io.rs) |
| `ome_zarr.reader.Reader`, `Node`, specs | [`reader`](src/reader.rs) |
| `ome_zarr.writer.*` | [`writer`](src/writer.rs) |
| `OMEZarrImage`, `OMEZarrMultiscale`, `OMEZarrLabels` | [`classes::image`](src/classes/image.rs) |
| `OMEZarrScene`, `tnd_utils` | [`classes::scene`](src/classes/scene.rs), [`classes::tnd_utils`](src/classes/tnd_utils.rs) |
| `ome_zarr.scale`, `ome_zarr.dask_utils` | [`scale`](src/scale.rs) |
| `ome_zarr.format`, `ome_zarr.axes` | [`format`](src/format.rs), [`axes`](src/axes.rs) |
| `ome_zarr.utils` (`info`, `download`, `view`, `finder`) | [`utils`](src/utils.rs), [`server`](src/server.rs) |
| `ome_zarr.csv`, `ome_zarr.data`, `ome_zarr.conversions` | [`csv`](src/csv.rs), [`data`](src/data.rs), [`conversions`](src/conversions.rs) |
| `ome_zarr.cli` | [`cli`](src/cli.rs) |
| ome-zarr-models (v0.4 / v0.5 / v0.6) | [`models`](src/models) |
| transformnd | [`transformnd`](src/transformnd.rs) |
| numpy / dask arrays | [`array`](src/array) (`NdArray`, `LazyArray`) |
| scipy.ndimage, scikit-image | [`ndimage`](src/ndimage.rs), [`skimage`](src/skimage.rs) |
| zarr-python | [`zarr`](src/zarr.rs), [`store`](src/store.rs) (on [zarrs](https://zarrs.dev)) |
| — | [`render`](src/render) (threers) |

## Tests

```bash
cargo test                      # unit, parity and ported ome-zarr-py tests
cargo test -- --ignored         # also the tests that read remote IDR data
```

`tests/fixtures` holds reference outputs written by the Python packages, by
scripts run in an environment with ome-zarr-py installed:

* `scripts/gen_parity_fixtures.py` runs the same scenarios as
  `tests/parity.rs` to `tests/parity10.rs` and writes `fixtures/parity`. The
  scenarios cover every dtype and method, chunked and delayed writes, plates,
  scenes, downloads, finder, CLI output, the parameter grids of ome-zarr-py's
  `test_scaler.py`, what `info` / `download` print and log (every
  `ome_zarr.*` record, INFO and DEBUG), the reader from many starting points
  (array bytes compared by SHA-256), 100 seeded random writer / class-API
  configurations (with numpy-seeded label colours compared exactly), and the
  exception type and message of
  each error case. The Rust outputs must be byte-identical (metadata) and equal
  (array data). Random label colours are masked.
* `scripts/gen_ndimage_fixtures.py` and `scripts/gen_zarr_fixtures.py` cover
  the scipy / scikit-image ports and the zarr codecs;
  `scripts/gen_linalg_fixtures.py` numpy's small matrix products, inverses
  and transformnd's affines.
* `scripts/gen_pydantic_fixtures.py` exports ome-zarr-models' pydantic core
  schemas (`src/models/pydantic_schemas.json`, interpreted by
  `models::pydantic`) and records pydantic's `ValidationError` for ~2,500
  valid and systematically broken inputs: Rust reports the same errors,
  character for character.
* `scripts/gen_cli_fixtures.py` and `scripts/gen_server_fixtures.py` record
  the CLI's argparse output and the `view` / `finder` server's responses.

### Platforms

Floating point results match numpy / scipy bit for bit on each platform,
which differ among themselves in the last bit: scipy's macOS arm64 wheels
fuse multiply-adds (clang), its Linux wheels never do; numpy's BLAS
(Accelerate, OpenBLAS) sums small matrix products in a kernel-specific order.
`ci/run-parity.sh linux/amd64` (or `linux/arm64`) regenerates every fixture
with the pinned Python environment of `ci/requirements-parity.txt` in a
Docker container and runs the test suite against it; macOS arm64, Linux
arm64 and Linux x86-64 pass. numpy's BLAS / LAPACK results are reproduced per
platform (`src/linalg.rs`): `np.linalg.inv` up to 4x4 (the largest matrix a
3-D affine needs) and small matrix products. On macOS, `inv` is Accelerate's
`dgesv`, linked and called directly; elsewhere a model of OpenBLAS's `getf2` /
`getrs`. OpenBLAS selects x86-64 kernels by CPU model: the generic kernels (as
under emulation) are matched, except for the tail rows of 4-term products;
on CPUs where it picks FMA kernels, transform outputs can differ from numpy
in the last bit.

### WebAssembly

The library builds for `wasm32-unknown-unknown`, with every data type and
compressor (zstd, blosc and bz2 are compiled from C with LLVM clang; lz4,
lzma, zlib and gzip are pure Rust), pyramids, transforms, pydantic-identical
validation and the CLI's argument parser (`cli::parse_args`). The
filesystem, HTTP and cloud stores, the file server, the viewer and
multi-threading are native-only; memory and zip stores work everywhere. To read online data in a web app, fetch it with
the browser and hand it over: `Store::zip_bytes(bytes, name)` for a zipped
fileset, or `Store::from_dyn` with a storage adapter of your own.

```bash
scripts/check-wasm.sh           # cargo check for wasm32 (zstd/blosc need LLVM clang)
scripts/test-wasm.sh            # run tests/wasm.rs under node (wasm-bindgen-test-runner)
```

## Differences from ome-zarr-py

* `view` opens the native threers viewer; `view --web` is the Python behaviour.
* Label colours are random as in ome-zarr-py; `utils::np_random_seed(seed)`
  is `np.random.seed(seed)` (numpy's MT19937), after which they are the
  colours ome-zarr-py chooses for the same seed.
* Python `warnings.warn` messages are logged (`WARNING:ome_zarr.<module>:...`)
  rather than printed with a Python source location.
* `compute=False` returns [`DelayedWrite`](src/writer.rs) jobs instead of
  dask delayed objects; run them with `writer::compute_all`.
* Additional sources: zipped filesets (`*.zip`, local or over http, opened
  read-only; `download` names the copy without `.zip`), `gs://` and `az://`.
  `view --web` of a remote image opens the validator on its URL, where
  ome-zarr-py reports "No OME-Zarr images found".
* `download` works for relative and symlinked paths (ome-zarr-py mixes the
  given root path with resolved child paths and fails with "No common
  prefix" or "/ already exists!").
* `local_mean` of a chunked (dask) array whose downsampled dimension is not a
  multiple of the factor writes the level; in ome-zarr-py, dask fails with
  "could not broadcast input array" (its blocks come out larger than
  declared).
* `Scaler.scale()` writes its pyramid (in ome-zarr-py it fails with zarr 3,
  which has no `Group.create_dataset`).
* Python's keyword arguments are option structs (`ImageOptions`,
  `MultiscaleOptions`, `StorageOptions`) and builders.

## License

BSD-2-Clause, as ome-zarr-py. The sample images in `assets/` come from
scikit-image's data directory (see `assets/README.md`).
