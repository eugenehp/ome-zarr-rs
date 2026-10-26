//! Ports of the scikit-image functions used for downsampling:
//! `transform.resize`, `transform.downscale_local_mean`, `filters.gaussian`,
//! `transform.pyramid_reduce`, `pyramid_gaussian`, `pyramid_laplacian` and
//! `util.img_as_float`.

use ndarray::{ArrayD, IxDyn};

use crate::array::{DType, Element, IntoNdArray, NdArray};
use crate::dtype_dispatch;
use crate::error::{Error, Result};
use crate::ndimage::{self, Mode};

/// `skimage.util.img_as_float` (float64 output for integer input,
/// float types unchanged). Returns the values and whether they are float32.
pub fn img_as_float(image: &NdArray) -> (ArrayD<f64>, bool) {
    match image {
        NdArray::Float32(a) => (a.mapv(|v| v as f64), true),
        NdArray::Float64(a) => (a.clone(), false),
        NdArray::Bool(a) => (a.mapv(|v| if v { 1.0 } else { 0.0 }), false),
        other => {
            let d = other.dtype();
            let imax = d.max_value();
            let scale = 1.0 / imax;
            let mut v = other.to_f64().mapv(|x| x * scale);
            if d.is_signed() {
                v.mapv_inplace(|x| x.max(-1.0));
            }
            (v, false)
        }
    }
}

/// `skimage._shared.utils.convert_to_float`.
pub fn convert_to_float(image: &NdArray, preserve_range: bool) -> (ArrayD<f64>, bool) {
    if image.dtype() == DType::Float16 {
        return (image.to_f64(), true);
    }
    if preserve_range {
        match image {
            NdArray::Float32(a) => (a.mapv(|v| v as f64), true),
            other => (other.to_f64(), false),
        }
    } else {
        img_as_float(image)
    }
}

fn wrap_float(a: ArrayD<f64>, f32_out: bool) -> NdArray {
    if f32_out {
        NdArray::Float32(a.mapv(|v| v as f32))
    } else {
        NdArray::Float64(a)
    }
}

/// Convert raw ndimage outputs to `dtype` with scipy's output casting.
pub fn ndi_cast(a: &ArrayD<f64>, dtype: DType) -> NdArray {
    dtype_dispatch!(dtype, T => a.mapv(<T as Element>::from_f64_ndi).into_nd())
}

/// Options for [`resize`], mirroring `skimage.transform.resize`'s keywords.
#[derive(Debug, Clone)]
pub struct ResizeOptions {
    pub order: Option<usize>,
    /// numpy.pad-style mode: 'reflect' (default), 'constant', 'edge', 'symmetric', 'wrap'.
    pub mode: String,
    pub cval: f64,
    pub clip: bool,
    pub preserve_range: bool,
    pub anti_aliasing: Option<bool>,
    pub anti_aliasing_sigma: Option<Vec<f64>>,
}

impl Default for ResizeOptions {
    fn default() -> Self {
        ResizeOptions {
            order: None,
            mode: "reflect".into(),
            cval: 0.0,
            clip: true,
            preserve_range: false,
            anti_aliasing: None,
            anti_aliasing_sigma: None,
        }
    }
}

fn validate_interpolation_order(dtype: DType, order: Option<usize>) -> Result<usize> {
    let order = match order {
        None => {
            if dtype == DType::Bool {
                0
            } else {
                1
            }
        }
        Some(o) => o,
    };
    if order > 5 {
        return Err(Error::value(
            "Spline interpolation order has to be in the range 0-5.",
        ));
    }
    if dtype == DType::Bool && order != 0 {
        return Err(Error::value(
            "Input image dtype is bool. Interpolation is not defined with bool data type. \
             Please set order to 0 or explicitly cast input image to another data type.",
        ));
    }
    Ok(order)
}

/// `skimage.transform.resize`.
pub fn resize(image: &NdArray, output_shape: &[usize], opts: &ResizeOptions) -> Result<NdArray> {
    // _preprocess_resize_output_shape
    let mut image = image.clone();
    let mut output_shape = output_shape.to_vec();
    let ndim = image.ndim();
    if output_shape.len() > ndim {
        let mut s = image.shape();
        s.extend(std::iter::repeat_n(1, output_shape.len() - ndim));
        image = image.reshape(&s)?;
    } else if output_shape.len() + 1 == ndim {
        output_shape.push(image.shape()[ndim - 1]);
    } else if output_shape.len() < ndim {
        return Err(Error::value(
            "output_shape length cannot be smaller than the image number of dimensions",
        ));
    }
    let input_shape = image.shape();
    let input_type = image.dtype();
    if input_type == DType::Float16 {
        image = image.astype(DType::Float32);
    }
    let anti_aliasing = opts.anti_aliasing.unwrap_or_else(|| {
        input_type != DType::Bool
            && !(input_type.is_integer() && opts.order == Some(0))
            && output_shape.iter().zip(&input_shape).any(|(o, i)| o < i)
    });
    if input_type == DType::Bool && anti_aliasing {
        return Err(Error::value(
            "anti_aliasing must be False for boolean images",
        ));
    }
    let factors: Vec<f64> = input_shape
        .iter()
        .zip(&output_shape)
        .map(|(&i, &o)| i as f64 / o as f64)
        .collect();
    let order = validate_interpolation_order(input_type, opts.order)?;
    if order == 0 && input_type.is_complex() {
        return resize_complex_nearest(&image, &factors, anti_aliasing, opts);
    }
    // working values and output dtype
    let (values, out_dtype) = if order > 0 {
        let (v, f32_) = convert_to_float(&image, opts.preserve_range);
        (v, if f32_ { DType::Float32 } else { DType::Float64 })
    } else {
        (image.to_f64(), input_type)
    };
    let ndi_mode = Mode::from_skimage(&opts.mode)?;
    let filtered = if anti_aliasing {
        let sigma: Vec<f64> = match &opts.anti_aliasing_sigma {
            None => factors.iter().map(|f| ((f - 1.0) / 2.0).max(0.0)).collect(),
            Some(s) => {
                let s: Vec<f64> = if s.len() == 1 {
                    vec![s[0]; factors.len()]
                } else {
                    s.clone()
                };
                if s.iter().any(|&x| x < 0.0) {
                    return Err(Error::value(
                        "Anti-aliasing standard deviation must be greater than or equal to zero",
                    ));
                }
                s
            }
        };
        let f = ndimage::gaussian_filter(
            &values,
            &sigma,
            ndi_mode,
            opts.cval,
            4.0,
            out_dtype == DType::Float32,
        );
        if out_dtype.is_integer() || out_dtype == DType::Bool {
            ndi_cast(&f, out_dtype).to_f64()
        } else {
            f
        }
    } else {
        values.clone()
    };
    let zoom_factors: Vec<f64> = factors.iter().map(|f| 1.0 / f).collect();
    let raw = ndimage::zoom(
        &filtered,
        &zoom_factors,
        order,
        ndi_mode,
        opts.cval,
        true,
        true,
    )?;
    let mut out = match out_dtype {
        DType::Float64 => NdArray::Float64(raw),
        DType::Float32 => NdArray::Float32(raw.mapv(|v| v as f32)),
        d => ndi_cast(&raw, d),
    };
    if opts.clip {
        clip_warp_output(&values, &mut out, &opts.mode, opts.cval);
    }
    Ok(out)
}

/// `resize` of complex data with `order=0`: no float conversion, so scipy's
/// `gaussian_filter` / `zoom` process the real and imaginary parts separately
/// (scipy's complex `zoom` does not pass `grid_mode` on, so it is off). The
/// clip is a no-op: the output values are samples of the input.
fn resize_complex_nearest(
    image: &NdArray,
    factors: &[f64],
    anti_aliasing: bool,
    opts: &ResizeOptions,
) -> Result<NdArray> {
    let ndi_mode = Mode::from_skimage(&opts.mode)?;
    let f32_ = image.dtype() == DType::Complex64;
    let parts = complex_parts(image);
    let mut out = vec![];
    for part in [parts.0, parts.1] {
        let filtered = if anti_aliasing {
            let sigma: Vec<f64> = match &opts.anti_aliasing_sigma {
                None => factors.iter().map(|f| ((f - 1.0) / 2.0).max(0.0)).collect(),
                Some(s) if s.len() == 1 => vec![s[0]; factors.len()],
                Some(s) => s.clone(),
            };
            ndimage::gaussian_filter(&part, &sigma, ndi_mode, opts.cval, 4.0, f32_)
        } else {
            part
        };
        let zoom_factors: Vec<f64> = factors.iter().map(|f| 1.0 / f).collect();
        out.push(ndimage::zoom(
            &filtered,
            &zoom_factors,
            0,
            ndi_mode,
            opts.cval,
            true,
            false,
        )?);
    }
    Ok(complex_from_parts(&out[0], &out[1], image.dtype()))
}

/// The real and imaginary parts of an array, as f64.
pub fn complex_parts(a: &NdArray) -> (ArrayD<f64>, ArrayD<f64>) {
    crate::nd_dispatch!(a, x => (x.mapv(|v| v.to_f64()), x.mapv(|v| v.imag_f64())))
}

/// A `dtype` array from real and imaginary parts (each cast as numpy does).
pub fn complex_from_parts(re: &ArrayD<f64>, im: &ArrayD<f64>, dtype: DType) -> NdArray {
    dtype_dispatch!(dtype, T => ndarray::Zip::from(re).and(im).map_collect(|&r, &i| <T as Element>::from_complex(r, i)).into_nd())
}

/// `skimage.transform._warps._clip_warp_output` (in place).
fn clip_warp_output(input: &ArrayD<f64>, output: &mut NdArray, mode: &str, cval: f64) {
    if input.is_empty() {
        return;
    }
    let has_nan = input.iter().any(|v| v.is_nan());
    let vals = input.iter().filter(|v| !has_nan || !v.is_nan());
    let (mut min_val, mut max_val) = (f64::INFINITY, f64::NEG_INFINITY);
    for &v in vals {
        min_val = min_val.min(v);
        max_val = max_val.max(v);
    }
    if mode == "constant"
        && !(min_val <= cval && cval <= max_val)
        && let Some((omin, omax)) = output.min_max()
        && omin <= cval
        && cval <= omax
    {
        min_val = min_val.min(cval);
        max_val = max_val.max(cval);
    }
    crate::nd_dispatch!(output, a => {
        a.mapv_inplace(|v| {
            let f = v.to_f64();
            if f.is_nan() { return v; }
            if f < min_val { Element::from_f64(min_val) } else if f > max_val { Element::from_f64(max_val) } else { v }
        });
    });
}

/// `skimage.measure.block_reduce(image, factors, np.mean, cval)`, i.e.
/// `skimage.transform.downscale_local_mean`. Output is float64 (float32 for
/// float32 input, as numpy's mean).
pub fn downscale_local_mean(image: &NdArray, factors: &[usize], cval: f64) -> Result<NdArray> {
    let shape = image.shape();
    if factors.len() != shape.len() {
        return Err(Error::value(
            "`block_size` must be a scalar or have the same length as `image.shape`",
        ));
    }
    if factors.iter().any(|&f| f < 1) {
        return Err(Error::value(
            "Down-sampling factors must be >= 1. Use `skimage.transform.resize` to up-sample an image.",
        ));
    }
    let dtype = image.dtype();
    let (values, imag) = complex_parts(image);
    let out_shape: Vec<usize> = shape
        .iter()
        .zip(factors)
        .map(|(&s, &f)| s.div_ceil(f))
        .collect();
    let block_n: usize = factors.iter().product();
    let row = reduce_inner_loop(&out_shape, factors);
    let mut out = ArrayD::<f64>::zeros(IxDyn(&out_shape));
    let mut out_im = ArrayD::<f64>::zeros(IxDyn(&out_shape));
    let (mut buf, mut buf_im) = (Vec::with_capacity(block_n), Vec::with_capacity(block_n));
    let rank = shape.len();
    for (oidx, o) in out.indexed_iter_mut() {
        buf.clear();
        buf_im.clear();
        let mut bidx = vec![0usize; rank];
        for _ in 0..block_n {
            let src: Vec<usize> = (0..rank).map(|d| oidx[d] * factors[d] + bidx[d]).collect();
            let inside = src.iter().zip(&shape).all(|(&i, &n)| i < n);
            buf.push(if inside { values[IxDyn(&src)] } else { cval });
            buf_im.push(if inside { imag[IxDyn(&src)] } else { 0.0 });
            for d in (0..rank).rev() {
                bidx[d] += 1;
                if bidx[d] < factors[d] {
                    break;
                }
                bidx[d] = 0;
            }
        }
        // numpy's reduction: acc = 0, then for each inner-loop row (in C
        // order) acc += pairwise_sum(row); then `mean` divides by the count
        *o = match dtype {
            // float16 is accumulated in float32 and divided there
            DType::Float32 | DType::Float16 => {
                let b: Vec<f32> = buf.iter().map(|&v| v as f32).collect();
                let mut acc = 0.0f32;
                for r in b.chunks(row) {
                    acc += ndimage::pairwise_sum_f32(r);
                }
                (acc / block_n as f32) as f64
            }
            // complex: true_divide by the count as a complex number, which
            // multiplies by its reciprocal (Smith's algorithm)
            DType::Complex64 => {
                let (b, bi): (Vec<f32>, Vec<f32>) = (
                    buf.iter().map(|&v| v as f32).collect(),
                    buf_im.iter().map(|&v| v as f32).collect(),
                );
                let (mut ar, mut ai) = (0.0f32, 0.0f32);
                for (r, ri) in b.chunks(row).zip(bi.chunks(row)) {
                    let (sr, si) = ndimage::pairwise_sum_c64(r, ri);
                    ar += sr;
                    ai += si;
                }
                let scl = 1.0f32 / block_n as f32;
                out_im[oidx.clone()] = (ai * scl) as f64;
                (ar * scl) as f64
            }
            DType::Complex128 => {
                let (mut ar, mut ai) = (0.0f64, 0.0f64);
                for (r, ri) in buf.chunks(row).zip(buf_im.chunks(row)) {
                    let (sr, si) = ndimage::pairwise_sum_c128(r, ri);
                    ar += sr;
                    ai += si;
                }
                let scl = 1.0 / block_n as f64;
                out_im[oidx.clone()] = ai * scl;
                ar * scl
            }
            _ => {
                let mut acc = 0.0f64;
                for r in buf.chunks(row) {
                    acc += ndimage::pairwise_sum(r);
                }
                acc / block_n as f64
            }
        };
    }
    Ok(match dtype {
        DType::Float32 => NdArray::Float32(out.mapv(|v| v as f32)),
        DType::Float16 => NdArray::Float16(out.mapv(|v| half::f16::from_f32(v as f32))),
        DType::Complex64 | DType::Complex128 => complex_from_parts(&out, &out_im, dtype),
        _ => NdArray::Float64(out),
    })
}

/// Length of the inner loop numpy's `mean` over `view_as_blocks` runs: its
/// iterator walks the axes in memory order (`out_0, f_0, ..., out_n, f_n`),
/// drops length-1 axes and coalesces adjacent reduction axes, so the
/// innermost run of block axes is summed as one contiguous row.
fn reduce_inner_loop(out_shape: &[usize], factors: &[usize]) -> usize {
    let mut row = 1;
    for d in (0..factors.len()).rev() {
        row *= factors[d];
        if out_shape[d] != 1 {
            break;
        }
    }
    row
}

/// `skimage.filters.gaussian` with `preserve_range=False` (img_as_float first).
pub fn gaussian(image: &NdArray, sigma: f64, mode: Mode, cval: f64, truncate: f64) -> NdArray {
    let (v, f32_) = img_as_float(image);
    let sigmas = vec![sigma; v.ndim()];
    let out = ndimage::gaussian_filter(&v, &sigmas, mode, cval, truncate, f32_);
    wrap_float(out, f32_)
}

/// `skimage.transform.pyramid_reduce`.
pub fn pyramid_reduce(
    image: &NdArray,
    downscale: f64,
    sigma: Option<f64>,
    order: usize,
    mode: &str,
    cval: f64,
) -> Result<NdArray> {
    if downscale <= 1.0 {
        return Err(Error::value("scale factor must be greater than 1"));
    }
    let (values, f32_) = img_as_float(image);
    let out_shape: Vec<usize> = values
        .shape()
        .iter()
        .map(|&d| (d as f64 / downscale).ceil() as usize)
        .collect();
    let sigma = sigma.unwrap_or(2.0 * downscale / 6.0);
    let smoothed = ndimage::gaussian_filter(
        &values,
        &vec![sigma; values.ndim()],
        Mode::parse(mode)?,
        cval,
        4.0,
        f32_,
    );
    let opts = ResizeOptions {
        order: Some(order),
        mode: mode.to_string(),
        cval,
        anti_aliasing: Some(false),
        ..Default::default()
    };
    resize(&wrap_float(smoothed, f32_), &out_shape, &opts)
}

/// `skimage.transform.pyramid_gaussian` (`max_layer=-1` for "until 1 pixel").
pub fn pyramid_gaussian(image: &NdArray, max_layer: i64, downscale: f64) -> Result<Vec<NdArray>> {
    if downscale <= 1.0 {
        return Err(Error::value("scale factor must be greater than 1"));
    }
    let (values, f32_) = img_as_float(image);
    let first = wrap_float(values, f32_);
    let mut out = vec![first.clone()];
    let mut layer = 0;
    let mut prev = first;
    let mut current_shape = prev.shape();
    while layer != max_layer {
        layer += 1;
        let layer_image = pyramid_reduce(&prev, downscale, None, 1, "reflect", 0.0)?;
        let prev_shape = current_shape;
        current_shape = layer_image.shape();
        prev = layer_image.clone();
        if current_shape == prev_shape {
            break;
        }
        out.push(layer_image);
    }
    Ok(out)
}

/// `skimage.transform.pyramid_laplacian`.
pub fn pyramid_laplacian(image: &NdArray, max_layer: i64, downscale: f64) -> Result<Vec<NdArray>> {
    if downscale <= 1.0 {
        return Err(Error::value("scale factor must be greater than 1"));
    }
    let (values, f32_) = img_as_float(image);
    let sigma = 2.0 * downscale / 6.0;
    let smooth = |v: &ArrayD<f64>| {
        ndimage::gaussian_filter(v, &vec![sigma; v.ndim()], Mode::Reflect, 0.0, 4.0, f32_)
    };
    let mut current_shape = values.shape().to_vec();
    let mut smoothed = smooth(&values);
    let mut out = vec![wrap_float(&values - &smoothed, f32_)];
    let max_layer = if max_layer == -1 {
        let m = *current_shape.iter().max().unwrap_or(&1) as f64;
        (m.ln() / downscale.ln()).ceil() as i64
    } else {
        max_layer
    };
    for _ in 0..max_layer {
        let out_shape: Vec<usize> = current_shape
            .iter()
            .map(|&d| (d as f64 / downscale).ceil() as usize)
            .collect();
        let opts = ResizeOptions {
            order: Some(1),
            anti_aliasing: Some(false),
            ..Default::default()
        };
        let resized = resize(&wrap_float(smoothed.clone(), f32_), &out_shape, &opts)?.to_f64();
        smoothed = smooth(&resized);
        current_shape = resized.shape().to_vec();
        out.push(wrap_float(&resized - &smoothed, f32_));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_nearest_halves() {
        let a = NdArray::from_f64_vec(DType::UInt8, &[4, 4], (0..16).map(|v| v as f64).collect())
            .unwrap();
        let opts = ResizeOptions {
            order: Some(0),
            preserve_range: true,
            anti_aliasing: Some(false),
            ..Default::default()
        };
        let r = resize(&a, &[2, 2], &opts).unwrap();
        assert_eq!(r.dtype(), DType::UInt8);
        assert_eq!(r.shape(), vec![2, 2]);
        // grid-mode sample positions 0.5 -> floor(0.5+0.5)=1 ; 2.5 -> 3
        assert_eq!(r.to_f64_vec(), vec![5.0, 7.0, 13.0, 15.0]);
    }

    #[test]
    fn local_mean() {
        let a = NdArray::from_f64_vec(DType::UInt8, &[2, 3], vec![1., 3., 5., 3., 5., 7.]).unwrap();
        let r = downscale_local_mean(&a, &[2, 2], 0.0).unwrap();
        assert_eq!(r.to_f64_vec(), vec![3.0, 3.0]);
    }
}
