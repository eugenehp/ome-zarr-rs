//! # ome-zarr
//!
//! A Rust implementation of [ome-zarr-py](https://github.com/ome/ome-zarr-py):
//! reading and writing OME-Zarr (OME-NGFF 0.1 – 0.6) images, labels,
//! plates, wells and scenes, multiscale pyramid generation, and GPU rendering
//! through [threers](https://github.com/eugenehp/threers).
//!
//! | module | Python counterpart |
//! |---|---|
//! | [`remote_zip`] | — a zip read in byte ranges (a remote `.ome.zarr.zip`, never downloaded whole) |
//! | [`lazy`] | — stores fetched as they are read (wasm: the host's `fetch`) |
//! | [`io`] | `ome_zarr.io` — [`ZarrLocation`](io::ZarrLocation), [`parse_url`](io::parse_url) |
//! | [`reader`] | `ome_zarr.reader` — [`Reader`](reader::Reader), [`Node`](reader::Node), specs |
//! | [`writer`] | `ome_zarr.writer` — `write_image`, `write_labels`, plate / well metadata |
//! | [`classes`] | `ome_zarr.classes` — `OMEZarrImage`, `OMEZarrMultiscale`, `OMEZarrLabels`, `OMEZarrScene` |
//! | [`scale`] | `ome_zarr.scale` / `dask_utils` — [`Methods`](scale::Methods), pyramids, `Scaler` |
//! | [`format`](mod@format), [`axes`] | `ome_zarr.format`, `ome_zarr.axes` |
//! | [`utils`] | `ome_zarr.utils` — `info`, `download`, `view`, `finder` |
//! | [`csv`], [`data`], [`conversions`] | `ome_zarr.csv`, `ome_zarr.data`, `ome_zarr.conversions` |
//! | [`models`] | `ome-zarr-models` (0.4 / 0.5 / 0.6 metadata) |
//! | [`transformnd`] | `transformnd` (functional transforms and transform graphs) |
//! | [`ndimage`], [`skimage`] | the `scipy.ndimage` / `scikit-image` functions used for downsampling |
//! | [`array`](mod@array) | numpy / dask: [`NdArray`](array::NdArray), [`LazyArray`](array::LazyArray) |
//! | [`zarr`], [`store`] | zarr-python groups, arrays and stores (on top of `zarrs`) |
//! | [`render`] | GPU rendering and the native viewer (threers) |
//!
//! ```no_run
//! use ome_zarr::array::{DType, LazyArray, NdArray};
//! use ome_zarr::classes::{OmeZarrImage, OmeZarrMultiscale, Version};
//!
//! let data = NdArray::zeros(DType::UInt8, &[2, 64, 256, 256]);
//! let image = OmeZarrImage::from_axes(LazyArray::from_ndarray(data), "czyx")?;
//! let multiscale = OmeZarrMultiscale::builder(image).channel_names(&["DAPI", "GFP"]).build()?;
//! multiscale.to_ome_zarr("image.zarr", None, Version::V05, true, true)?;
//! # Ok::<(), ome_zarr::Error>(())
//! ```

// Index loops mirror the reference (scipy C / Python) implementations.
#![allow(clippy::needless_range_loop)]

pub mod argparse;
pub mod array;
pub mod axes;
pub mod classes;
pub mod cli;
pub mod codecs;
pub mod conversions;
pub mod csv;
pub mod data;
pub mod error;
pub mod format;
pub mod io;
pub mod lazy;
pub mod linalg;
pub mod models;
pub mod ndimage;
pub mod par;
pub mod pyjson;
pub mod reader;
pub mod remote_zip;
#[cfg(feature = "render")]
pub mod render;
pub mod scale;
#[cfg(not(target_arch = "wasm32"))]
pub mod server;
/// No sockets or browser on wasm: `view --web` / `finder` cannot serve.
#[cfg(target_arch = "wasm32")]
pub mod server {
    pub fn open_browser(_url: &str) {}
    pub fn serve(_root: &std::path::Path, _port: u16) -> crate::Result<()> {
        Err(crate::Error::NotImplemented(
            "serving files is not available on wasm".into(),
        ))
    }
}
pub mod skimage;
pub mod store;
pub mod transformnd;
pub mod utils;
pub mod writer;
pub mod zarr;

pub use classes::{OmeZarrImage, OmeZarrLabels, OmeZarrMultiscale, OmeZarrScene};
pub use error::{Error, Result};

/// The crate version (`ome_zarr.__version__`).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
