//! Downsampling of arrays into multiscale pyramids (`ome_zarr.scale` and
//! `ome_zarr.dask_utils`).
//!
//! Block-wise lazy downsampling mirrors the dask implementation exactly,
//! including how blocks are re-chunked, so pyramids match ome-zarr-py.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use log::warn;

use crate::array::{BlockFn, DType, LazyArray, NdArray};
use crate::error::{Error, Result};
use crate::ndimage::{self, Mode};
use crate::skimage::{self, ResizeOptions};
use crate::zarr::{ArraySpec, Group, Mode as ZMode, ZArray, store_lazy};

/// Spatial dimensions that are downsampled.
pub const SPATIAL_DIMS: [&str; 3] = ["z", "y", "x"];

/// Downsampling methods for multi-scale image generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Methods {
    /// Bilinear interpolation downsampling (default): `skimage.transform.resize`
    /// with order=1, mode='reflect', anti_aliasing=True, preserve_range=True.
    #[default]
    Resize,
    /// Nearest-neighbor downsampling: `skimage.transform.resize` with order=0,
    /// mode='reflect', anti_aliasing=False, preserve_range=True.
    Nearest,
    /// Local mean downsampling: `skimage.transform.downscale_local_mean`.
    LocalMean,
    /// Zoom-based downsampling: `scipy.ndimage.zoom` (order 1).
    Zoom,
}

impl Methods {
    /// The method's string value, e.g. `"resize"`.
    pub fn value(&self) -> &'static str {
        match self {
            Methods::Resize => "resize",
            Methods::Nearest => "nearest",
            Methods::LocalMean => "local_mean",
            Methods::Zoom => "zoom",
        }
    }

    /// Parse a method name.
    pub fn parse(s: &str) -> Result<Methods> {
        Ok(match s {
            "resize" => Methods::Resize,
            "nearest" => Methods::Nearest,
            "local_mean" => Methods::LocalMean,
            "zoom" => Methods::Zoom,
            other => return Err(Error::value(format!("'{other}' is not a valid Methods"))),
        })
    }

    /// The function this method dispatches to (for provenance metadata).
    pub fn used_function(&self) -> &'static str {
        match self {
            Methods::Resize | Methods::Nearest => "skimage.transform.resize",
            Methods::LocalMean => "skimage.transform.downscale_local_mean",
            Methods::Zoom => "scipy.ndimage.zoom",
        }
    }

    /// Apply the method to a lazy array.
    pub fn apply(&self, image: &LazyArray, output_shape: &[u64]) -> Result<LazyArray> {
        match self {
            Methods::Resize => resize(
                image,
                output_shape,
                ResizeOptions {
                    order: Some(1),
                    mode: "reflect".into(),
                    anti_aliasing: Some(true),
                    preserve_range: true,
                    ..Default::default()
                },
            ),
            Methods::Nearest => resize(
                image,
                output_shape,
                ResizeOptions {
                    order: Some(0),
                    mode: "reflect".into(),
                    anti_aliasing: Some(false),
                    preserve_range: true,
                    ..Default::default()
                },
            ),
            Methods::LocalMean => local_mean(image, output_shape),
            Methods::Zoom => zoom(image, output_shape),
        }
    }
}

impl fmt::Display for Methods {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.value())
    }
}

impl std::str::FromStr for Methods {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        Methods::parse(s)
    }
}

/// `dask_utils._better_chunksize`.
fn better_chunksize(chunksize: &[u64], factors: &[f64]) -> (Vec<u64>, Vec<u64>) {
    let better: Vec<u64> = chunksize
        .iter()
        .zip(factors)
        .map(|(&c, &f)| {
            let v = ((c as f64 * f).round_ties_even() / f).max(1.0);
            v as u64
        })
        .collect();
    let block_out: Vec<u64> = better
        .iter()
        .zip(factors)
        .map(|(&b, &f)| (b as f64 * f).ceil() as u64)
        .collect();
    (better, block_out)
}

/// Shared tail of the dask_utils functions: map blocks, crop, rechunk, cast.
fn blockwise(
    image: &LazyArray,
    better: &[u64],
    block_out: &[u64],
    output_shape: &[u64],
    func: BlockFn,
) -> Result<LazyArray> {
    let dtype = image.dtype();
    let prepared = image.rechunk(better);
    let mapped = prepared.map_blocks(func, block_out, dtype)?;
    let region: Vec<std::ops::Range<u64>> = output_shape.iter().map(|&d| 0..d).collect();
    let cropped = mapped.slice(&region)?;
    Ok(cropped.rechunk(&image.chunksize()).astype(dtype))
}

/// Block-wise `skimage.transform.resize` (`ome_zarr.dask_utils.resize`).
pub fn resize(image: &LazyArray, output_shape: &[u64], opts: ResizeOptions) -> Result<LazyArray> {
    if output_shape.len() != image.ndim() {
        return Err(Error::value("output_shape must match image dimensionality"));
    }
    let factors: Vec<f64> = output_shape
        .iter()
        .zip(image.shape())
        .map(|(&o, &i)| o as f64 / i as f64)
        .collect();
    let (better, block_out) = better_chunksize(&image.chunksize(), &factors);
    let f2 = factors.clone();
    let func: BlockFn = Arc::new(move |block: &NdArray| {
        let chunk_out: Vec<usize> = block
            .shape()
            .iter()
            .zip(&f2)
            .map(|(&s, &f)| (s as f64 * f).ceil() as usize)
            .collect();
        let dtype = block.dtype();
        Ok(skimage::resize(block, &chunk_out, &opts)?.astype(dtype))
    });
    blockwise(image, &better, &block_out, output_shape, func)
}

/// Block-wise `skimage.transform.downscale_local_mean` (`dask_utils.local_mean`).
pub fn local_mean(image: &LazyArray, output_shape: &[u64]) -> Result<LazyArray> {
    let factors: Vec<f64> = image
        .shape()
        .iter()
        .zip(output_shape)
        .map(|(&i, &o)| i as f64 / o as f64)
        .collect();
    let inv: Vec<f64> = factors.iter().map(|f| 1.0 / f).collect();
    let (better, block_out) = better_chunksize(&image.chunksize(), &inv);
    let int_factors: Vec<usize> = factors.iter().map(|&f| f as usize).collect();
    let func: BlockFn = Arc::new(move |block: &NdArray| {
        let dtype = block.dtype();
        Ok(skimage::downscale_local_mean(block, &int_factors, 0.0)?.astype(dtype))
    });
    blockwise(image, &better, &block_out, output_shape, func)
}

/// `scipy.ndimage.zoom(input, zoom, order, mode="constant")` with scipy's
/// dtype handling: float16 is rejected, complex data is zoomed per component.
pub fn ndi_zoom_typed(a: &NdArray, zoom: &[f64], order: usize, prefilter: bool) -> Result<NdArray> {
    let dtype = a.dtype();
    if dtype == DType::Float16 {
        return Err(Error::Runtime("data type not supported".into()));
    }
    if dtype.is_complex() {
        let (re, im) = skimage::complex_parts(a);
        let re = ndimage::zoom(&re, zoom, order, Mode::Constant, 0.0, prefilter, false)?;
        let im = ndimage::zoom(&im, zoom, order, Mode::Constant, 0.0, prefilter, false)?;
        return Ok(skimage::complex_from_parts(&re, &im, dtype));
    }
    let raw = ndimage::zoom(
        &a.to_f64(),
        zoom,
        order,
        Mode::Constant,
        0.0,
        prefilter,
        false,
    )?;
    Ok(skimage::ndi_cast(&raw, dtype))
}

/// Block-wise `scipy.ndimage.zoom(order=1)` (`dask_utils.zoom`).
pub fn zoom(image: &LazyArray, output_shape: &[u64]) -> Result<LazyArray> {
    let factors: Vec<f64> = image
        .shape()
        .iter()
        .zip(output_shape)
        .map(|(&i, &o)| i as f64 / o as f64)
        .collect();
    let inv: Vec<f64> = factors.iter().map(|f| 1.0 / f).collect();
    let (better, block_out) = better_chunksize(&image.chunksize(), &inv);
    let zf = inv.clone();
    let func: BlockFn = Arc::new(move |block: &NdArray| ndi_zoom_typed(block, &zf, 1, true));
    // zoom crops to floor(d / f) rather than the requested output_shape
    let crop: Vec<u64> = image
        .shape()
        .iter()
        .zip(&factors)
        .map(|(&d, &f)| crate::pyjson::py_floordiv(d as f64, f) as u64)
        .collect();
    blockwise(image, &better, &block_out, &crop, func)
}

/// Primitive downscaling by integer factors using stepped slicing
/// (`dask_utils.downscale_nearest`).
pub fn downscale_nearest(image: &LazyArray, factors: &[u64]) -> Result<LazyArray> {
    if factors.len() != image.ndim() {
        return Err(Error::value(format!(
            "Dimension mismatch: {} image dimensions, {} scale factors",
            image.ndim(),
            factors.len()
        )));
    }
    if !factors
        .iter()
        .zip(image.shape())
        .all(|(&f, &d)| f > 0 && f <= d)
    {
        return Err(Error::value(format!(
            "All scale factors must not be greater than the dimension length: ({}) <= ({})",
            crate::pyjson::shape_repr(&factors.iter().map(|&f| f as usize).collect::<Vec<_>>()),
            crate::pyjson::shape_repr(
                &image
                    .shape()
                    .iter()
                    .map(|&d| d as usize)
                    .collect::<Vec<_>>()
            )
        )));
    }
    let data = image.compute()?;
    let out_shape: Vec<usize> = image
        .shape()
        .iter()
        .zip(factors)
        .map(|(&d, &f)| d.div_ceil(f) as usize)
        .collect();
    let f: Vec<usize> = factors.iter().map(|&x| x as usize).collect();
    let out = crate::nd_map!(&data, a => {
        ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&out_shape), |idx| {
            let src: Vec<usize> = (0..f.len()).map(|d| idx[d] * f[d]).collect();
            a[ndarray::IxDyn(&src)]
        })
    });
    Ok(LazyArray::from_ndarray(out))
}

/// Per-level scale factors: either integers (applied to y/x, not z) or
/// explicit per-dimension factors.
#[derive(Debug, Clone, PartialEq)]
pub enum ScaleFactors {
    /// `[2, 4, 8, 16]`: level `i` is downsampled by `2**i` in x and y (not z).
    Ints(Vec<u64>),
    /// `[{"z": 2, "y": 2, "x": 2}, ...]`; missing dims default to 1.
    PerDim(Vec<BTreeMap<String, f64>>),
}

impl Default for ScaleFactors {
    fn default() -> Self {
        ScaleFactors::Ints(vec![2, 4, 8, 16])
    }
}

impl ScaleFactors {
    pub fn len(&self) -> usize {
        match self {
            ScaleFactors::Ints(v) => v.len(),
            ScaleFactors::PerDim(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Normalise to per-dim dicts ordered like `dims`, missing dims = 1.
    pub fn resolve(&self, dims: &[String]) -> Vec<Vec<f64>> {
        match self {
            ScaleFactors::Ints(v) => (1..=v.len())
                .map(|i| {
                    dims.iter()
                        .map(|d| {
                            if d == "z" {
                                1.0
                            } else if SPATIAL_DIMS.contains(&d.as_str()) {
                                2f64.powi(i as i32)
                            } else {
                                1.0
                            }
                        })
                        .collect()
                })
                .collect(),
            ScaleFactors::PerDim(levels) => levels
                .iter()
                .map(|l| {
                    dims.iter()
                        .map(|d| l.get(d).copied().unwrap_or(1.0))
                        .collect()
                })
                .collect(),
        }
    }

    /// Build per-dim factors from a list of (dim, factor) maps.
    pub fn per_dim<I, K>(levels: I) -> ScaleFactors
    where
        I: IntoIterator<Item = Vec<(K, f64)>>,
        K: Into<String>,
    {
        ScaleFactors::PerDim(
            levels
                .into_iter()
                .map(|l| l.into_iter().map(|(k, v)| (k.into(), v)).collect())
                .collect(),
        )
    }
}

/// Build a pyramid of downscaled images (`ome_zarr.scale._build_pyramid`).
///
/// The first element is the input image itself.
pub fn build_pyramid(
    image: &LazyArray,
    scale_factors: &ScaleFactors,
    dims: &[String],
    method: Methods,
) -> Result<Vec<LazyArray>> {
    let levels = scale_factors.resolve(dims);
    let mut images = vec![image.clone()];
    for (idx, factor) in levels.iter().enumerate() {
        let relative: Vec<f64> = if idx == 0 {
            factor.clone()
        } else {
            factor
                .iter()
                .zip(&levels[idx - 1])
                .map(|(a, b)| a / b)
                .collect()
        };
        let prev = images.last().unwrap();
        let mut target = vec![];
        for ((&s, d), &f) in prev.shape().iter().zip(dims).zip(&relative) {
            if SPATIAL_DIMS.contains(&d.as_str()) {
                let q = crate::pyjson::py_floordiv(s as f64, f);
                if q == 0.0 {
                    target.push(1);
                    warn!("Dimension {d} is too small to downsample further.");
                } else {
                    target.push(q as u64);
                }
            } else {
                target.push(s);
            }
        }
        let new_image = method.apply(prev, &target)?;
        images.push(new_image);
    }
    Ok(images)
}

/// Helper for performing various types of downsampling.
///
/// **Deprecated** (0.14.0): use scale factors and [`Methods`] instead.
#[derive(Debug, Clone, PartialEq)]
pub struct Scaler {
    /// If `true`, copy Zarr attributes from the input array to the new group.
    pub copy_metadata: bool,
    /// Downscaling factor.
    pub downscale: u64,
    /// Does not do anything.
    pub in_place: bool,
    /// If `true`, check that the values in the downsampled levels are a subset
    /// of the values found in the input array.
    pub labeled: bool,
    /// The maximum number of downsampled layers to create.
    pub max_layer: usize,
    /// Downsampling method name (see [`Scaler::methods`]).
    pub method: String,
    /// 0: nearest-neighbour, 1: bi-linear (default). Only used for resize.
    pub order: usize,
}

impl Default for Scaler {
    fn default() -> Self {
        Scaler {
            copy_metadata: false,
            downscale: 2,
            in_place: false,
            labeled: false,
            max_layer: 4,
            method: "nearest".into(),
            order: 1,
        }
    }
}

impl Scaler {
    /// The names of all downsampling methods.
    pub fn methods() -> Vec<&'static str> {
        vec![
            "gaussian",
            "laplacian",
            "local_mean",
            "nearest",
            "resize_image",
            "zoom",
        ]
    }

    /// Run the configured method.
    pub fn func(&self, base: &NdArray) -> Result<Vec<NdArray>> {
        match self.method.as_str() {
            "nearest" => self.nearest(base),
            "gaussian" => self.gaussian(base),
            "laplacian" => self.laplacian(base),
            "local_mean" => self.local_mean(base),
            "zoom" => self.zoom(base),
            "resize_image" => Ok(vec![self.resize_image(base)?]),
            other => Err(Error::value(format!("Unknown method: {other}"))),
        }
    }

    /// Resize an array to a smaller array (not pyramid), downsampling x and y.
    pub fn resize_image(&self, image: &NdArray) -> Result<NdArray> {
        let mut out_shape = image.shape();
        let n = out_shape.len();
        out_shape[n - 1] = image.shape()[n - 1] / self.downscale as usize;
        out_shape[n - 2] = image.shape()[n - 2] / self.downscale as usize;
        let dtype = image.dtype();
        let opts = ResizeOptions {
            order: Some(self.order),
            mode: "reflect".into(),
            anti_aliasing: Some(false),
            ..Default::default()
        };
        Ok(skimage::resize(&image.astype(DType::Float64), &out_shape, &opts)?.astype(dtype))
    }

    /// `resize_image` on a lazy array (dask path).
    pub fn resize_image_lazy(&self, image: &LazyArray) -> Result<LazyArray> {
        let mut out_shape = image.shape().to_vec();
        let n = out_shape.len();
        out_shape[n - 1] /= self.downscale;
        out_shape[n - 2] /= self.downscale;
        let dtype = image.dtype();
        let opts = ResizeOptions {
            order: Some(self.order),
            mode: "reflect".into(),
            anti_aliasing: Some(false),
            ..Default::default()
        };
        Ok(resize(&image.astype(DType::Float64), &out_shape, opts)?.astype(dtype))
    }

    /// Downsample using nearest neighbour, plane by plane.
    pub fn nearest(&self, base: &NdArray) -> Result<Vec<NdArray>> {
        let ds = self.downscale as usize;
        self.by_plane(base, |plane, y, x| {
            let opts = ResizeOptions {
                order: Some(0),
                preserve_range: true,
                anti_aliasing: Some(false),
                ..Default::default()
            };
            Ok(skimage::resize(plane, &[y / ds, x / ds], &opts)?.astype(plane.dtype()))
        })
    }

    /// Downsample using `skimage.transform.pyramid_gaussian`.
    pub fn gaussian(&self, base: &NdArray) -> Result<Vec<NdArray>> {
        let dtype = base.dtype();
        Ok(
            skimage::pyramid_gaussian(base, self.max_layer as i64, self.downscale as f64)?
                .into_iter()
                .map(|l| l.astype(dtype))
                .collect(),
        )
    }

    /// Downsample using `skimage.transform.pyramid_laplacian`.
    pub fn laplacian(&self, base: &NdArray) -> Result<Vec<NdArray>> {
        let dtype = base.dtype();
        Ok(
            skimage::pyramid_laplacian(base, self.max_layer as i64, self.downscale as f64)?
                .into_iter()
                .map(|l| l.astype(dtype))
                .collect(),
        )
    }

    /// Downsample using `skimage.transform.downscale_local_mean`.
    pub fn local_mean(&self, base: &NdArray) -> Result<Vec<NdArray>> {
        let mut rv = vec![base.clone()];
        let stack_dims = base.ndim() - 2;
        let mut factors = vec![1usize; stack_dims];
        factors.extend([self.downscale as usize, self.downscale as usize]);
        for _ in 0..self.max_layer {
            let next = skimage::downscale_local_mean(rv.last().unwrap(), &factors, 0.0)?
                .astype(base.dtype());
            rv.push(next);
        }
        Ok(rv)
    }

    /// Downsample using `scipy.ndimage.zoom` (legacy behaviour: zooms by
    /// `downscale**i` and returns the levels in reverse order).
    pub fn zoom(&self, base: &NdArray) -> Result<Vec<NdArray>> {
        let mut rv = vec![base.clone()];
        for i in 0..self.max_layer {
            let z = (self.downscale as f64).powi(i as i32);
            let zf = vec![z; base.ndim()];
            rv.push(ndi_zoom_typed(base, &zf, 3, true)?);
        }
        rv.reverse();
        Ok(rv)
    }

    /// Loop over the leading dimensions and apply the 2D transform to each plane.
    fn by_plane(
        &self,
        base: &NdArray,
        func: impl Fn(&NdArray, usize, usize) -> Result<NdArray>,
    ) -> Result<Vec<NdArray>> {
        let mut rv = vec![base.clone()];
        for _ in 0..self.max_layer {
            let stack = rv.last().unwrap().clone();
            let shape = stack.shape();
            let nd = shape.len();
            let (y, x) = (shape[nd - 2], shape[nd - 1]);
            if nd == 2 {
                rv.push(func(&stack, y, x)?);
                continue;
            }
            let lead: Vec<usize> = shape[..nd - 2].to_vec();
            let mut new_stack: Option<NdArray> = None;
            for idx in crate::array::lazy::grid_indices(&lead) {
                let mut region: Vec<std::ops::Range<usize>> =
                    idx.iter().map(|&i| i..i + 1).collect();
                region.push(0..y);
                region.push(0..x);
                let plane = stack.slice_region(&region).reshape(&[y, x])?;
                let out = func(&plane, y, x)?;
                let os = out.shape();
                let ns = new_stack.get_or_insert_with(|| {
                    let mut s = lead.clone();
                    s.extend(&os);
                    NdArray::zeros(base.dtype(), &s)
                });
                let mut dst: Vec<std::ops::Range<usize>> = idx.iter().map(|&i| i..i + 1).collect();
                dst.push(0..os[0]);
                dst.push(0..os[1]);
                let mut s = vec![1usize; lead.len()];
                s.extend(&os);
                ns.assign_region(&dst, &out.reshape(&s)?)?;
            }
            rv.push(new_stack.unwrap());
        }
        Ok(rv)
    }

    /// Perform downsampling of a zarr array to disk.
    pub fn scale(&self, input_array: &str, output_directory: &str) -> Result<()> {
        let store = crate::store::Store::from_url(input_array, true)?;
        let base_arr = ZArray::open(&store, "")?;
        let base = base_arr.read_all()?;
        let pyramid = self.func(&base)?;
        if self.labeled {
            self.assert_values(&pyramid)?;
        }
        let grp = self.create_group(output_directory, &base, &pyramid)?;
        if self.copy_metadata {
            let attrs = base_arr.attrs()?;
            let keys: Vec<String> = attrs.keys().cloned().collect();
            println!(
                "copying attribute keys: {}",
                crate::pyjson::str_list_repr(&keys)
            );
            grp.update_attrs(&attrs)?;
        }
        Ok(())
    }

    fn assert_values(&self, pyramid: &[NdArray]) -> Result<()> {
        let expected = pyramid[0].unique_f64();
        println!(
            "level 0 {} = {} labels",
            crate::pyjson::shape_repr(&pyramid[0].shape()),
            expected.len()
        );
        for (i, level) in pyramid.iter().enumerate().skip(1) {
            println!(
                "level {i} {} {}",
                crate::pyjson::shape_repr(&level.shape()),
                expected.len()
            );
            let found = level.unique_f64();
            if !found.iter().all(|v| expected.contains(v)) {
                return Err(Error::value(format!(
                    "{} found values are not a subset of {} values",
                    found.len(),
                    expected.len()
                )));
            }
        }
        Ok(())
    }

    fn create_group(&self, dir: &str, base: &NdArray, pyramid: &[NdArray]) -> Result<Group> {
        let grp = Group::open_url(dir, ZMode::Write, None)?;
        let write = |name: &str, data: &NdArray| -> Result<()> {
            let shape = data.shape_u64();
            let spec = ArraySpec::new(&shape, data.dtype(), &shape, grp.zarr_format());
            let arr = grp.create_array(name, &spec, true)?;
            store_lazy(&LazyArray::from_ndarray(data.clone()), &arr)
        };
        write("base", base)?;
        for (i, level) in pyramid.iter().enumerate().skip(1) {
            write(&i.to_string(), level)?;
        }
        Ok(grp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(shape: &[usize]) -> NdArray {
        let n: usize = shape.iter().product();
        NdArray::from_f64_vec(
            DType::UInt8,
            shape,
            (0..n).map(|v| (v % 251) as f64).collect(),
        )
        .unwrap()
    }

    #[test]
    fn scaler_nearest_shapes() {
        let d = data(&[1, 1, 1, 64, 64]);
        let levels = Scaler::default().nearest(&d).unwrap();
        let shapes: Vec<Vec<usize>> = levels.iter().map(|l| l.shape()).collect();
        assert_eq!(shapes[4], vec![1, 1, 1, 4, 4]);
        assert_eq!(shapes.len(), 5);
    }

    #[test]
    fn pyramid_shapes() {
        let dims: Vec<String> = "czyx".chars().map(|c| c.to_string()).collect();
        let img = LazyArray::from_ndarray(data(&[2, 5, 64, 48]));
        for m in [
            Methods::Resize,
            Methods::Nearest,
            Methods::LocalMean,
            Methods::Zoom,
        ] {
            let p = build_pyramid(&img, &ScaleFactors::Ints(vec![2, 4]), &dims, m).unwrap();
            assert_eq!(p[1].shape(), &[2, 5, 32, 24], "{m}");
            assert_eq!(p[2].shape(), &[2, 5, 16, 12], "{m}");
            let c = p[2].compute().unwrap();
            assert_eq!(c.dtype(), DType::UInt8);
        }
    }
}
