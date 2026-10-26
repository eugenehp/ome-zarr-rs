//! Write a 2 x 16 x 1024 x 1024 uint16 image as an OME-Zarr 0.5 pyramid.
//!
//! cargo run --release --example bench_write -- /tmp/bench.zarr

use ome_zarr::array::{DType, LazyArray, NdArray};
use ome_zarr::classes::{OmeZarrImage, OmeZarrMultiscale, Version};

fn main() -> ome_zarr::Result<()> {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "bench.zarr".into());
    let shape = [2usize, 16, 1024, 1024];
    let n: usize = shape.iter().product();
    let data = NdArray::from_f64_vec(
        DType::UInt16,
        &shape,
        (0..n).map(|i| ((i * 2654435761) % 4096) as f64).collect(),
    )?;
    let t = std::time::Instant::now();
    let image = OmeZarrImage::from_axes(LazyArray::from_ndarray(data), "czyx")?;
    let ms = OmeZarrMultiscale::builder(image).build()?;
    ms.to_ome_zarr(out.as_str(), None, Version::V05, true, true)?;
    println!("rust: {:.2}s", t.elapsed().as_secs_f64());
    Ok(())
}
