//! A faithful port of the parts of `scipy.ndimage` used by ome-zarr:
//! `gaussian_filter`, `correlate1d`, `spline_filter`, `zoom` and
//! `map_coordinates`.
//!
//! The loops follow scipy's C implementation (`ni_interpolation.c`,
//! `ni_filters.c`, `ni_splines.c`) including floating point summation order,
//! so that integer outputs (which truncate) match bit for bit.

use crate::par::*;
use ndarray::{ArrayD, IxDyn};

use crate::error::{Error, Result};

/// Whether scipy's compiled kernels on this platform fuse multiply-adds:
/// the macOS arm64 wheels (clang, `-ffp-contract=on`) do; the Linux wheels
/// (aarch64 and x86-64, built without contraction: their `_nd_image` has no
/// fused instruction at all) and x86-64 baseline builds do not.
pub const SCIPY_FUSES: bool = cfg!(all(target_arch = "aarch64", target_os = "macos"));

/// `a * b + c` as scipy's compiled C computes it on this platform: one
/// rounding with fused multiply-adds (macOS arm64), two otherwise.
#[inline(always)]
pub fn fmadd(a: f64, b: f64, c: f64) -> f64 {
    if SCIPY_FUSES {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// One step `acc += a * b` of the correlation loop of `NI_Correlate1D`, as
/// scipy's build executes it. On macOS arm64 the loop is vectorised in
/// blocks of 8 iterations (products rounded, then added strictly in order)
/// and the remaining `n % 8` iterations run in the scalar loop, which uses
/// fused multiply-adds; elsewhere every step is a plain multiply and add.
#[inline(always)]
fn accumulate(acc: f64, a: f64, b: f64, k: usize, n: usize) -> f64 {
    if SCIPY_FUSES && k >= n - n % 8 {
        a.mul_add(b, acc)
    } else {
        acc + a * b
    }
}

/// Boundary modes (scipy names).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    /// `'reflect'`: (d c b a | a b c d | d c b a)
    Reflect,
    /// `'mirror'`: (d c b | a b c d | c b a)
    Mirror,
    /// `'nearest'`: (a a a a | a b c d | d d d d)
    Nearest,
    /// `'constant'`: no interpolation beyond the edge, `cval` outside.
    Constant,
    /// `'grid-constant'`: pad with `cval` and interpolate.
    GridConstant,
    /// `'wrap'`
    Wrap,
    /// `'grid-wrap'`
    GridWrap,
}

impl Mode {
    /// Parse a scipy mode name.
    pub fn parse(s: &str) -> Result<Mode> {
        Ok(match s {
            "reflect" | "grid-mirror" => Mode::Reflect,
            "mirror" => Mode::Mirror,
            "nearest" => Mode::Nearest,
            "constant" => Mode::Constant,
            "grid-constant" => Mode::GridConstant,
            "wrap" => Mode::Wrap,
            "grid-wrap" => Mode::GridWrap,
            other => {
                return Err(Error::value(format!(
                    "boundary mode not supported: {other}"
                )));
            }
        })
    }

    /// Translate a `numpy.pad`-style mode (as used by scikit-image) to the
    /// corresponding ndimage mode (`skimage._shared.utils._to_ndimage_mode`).
    pub fn from_skimage(s: &str) -> Result<Mode> {
        Ok(match s {
            "constant" => Mode::GridConstant,
            "edge" => Mode::Nearest,
            "symmetric" => Mode::Reflect,
            "reflect" => Mode::Mirror,
            "wrap" => Mode::GridWrap,
            other => {
                return Err(Error::value(format!(
                    "Unknown mode: '{other}', or cannot translate mode. The mode should be one of \
                     'constant', 'edge', 'symmetric', 'reflect', or 'wrap'."
                )));
            }
        })
    }

    fn spline_mode(self) -> Mode {
        match self {
            Mode::Constant | Mode::Wrap => Mode::Mirror,
            m => m,
        }
    }
}

macro_rules! complex_pairwise {
    ($name:ident, $t:ty) => {
        /// numpy's pairwise sum of a complex vector given as (real, imaginary)
        /// parts (`@TYPE@_pairwise_sum` for complex types: the floats are
        /// interleaved, so the unrolled loop sums four complex values at a time).
        pub fn $name(re: &[$t], im: &[$t]) -> ($t, $t) {
            let k = re.len();
            let n = 2 * k;
            if n < 8 {
                let (mut rr, mut ri) = (-0.0 as $t, -0.0 as $t);
                for c in 0..k {
                    rr += re[c];
                    ri += im[c];
                }
                (rr, ri)
            } else if n <= 128 {
                let mut r = [re[0], im[0], re[1], im[1], re[2], im[2], re[3], im[3]];
                let mut c = 4;
                while c < k - (k % 4) {
                    for j in 0..4 {
                        r[2 * j] += re[c + j];
                        r[2 * j + 1] += im[c + j];
                    }
                    c += 4;
                }
                let mut rr = (r[0] + r[2]) + (r[4] + r[6]);
                let mut ri = (r[1] + r[3]) + (r[5] + r[7]);
                while c < k {
                    rr += re[c];
                    ri += im[c];
                    c += 1;
                }
                (rr, ri)
            } else {
                let mut n2 = n / 2;
                n2 -= n2 % 8;
                let h = n2 / 2;
                let (a, b) = $name(&re[..h], &im[..h]);
                let (c, d) = $name(&re[h..], &im[h..]);
                (a + c, b + d)
            }
        }
    };
}

complex_pairwise!(pairwise_sum_c64, f32);
complex_pairwise!(pairwise_sum_c128, f64);

/// numpy's pairwise summation in float32 (float32 reductions).
pub fn pairwise_sum_f32(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 8 {
        // numpy starts at -0.0 to preserve the sign of zero
        let mut res = -0.0f32;
        for &v in a {
            res += v;
        }
        res
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            for (j, rj) in r.iter_mut().enumerate() {
                *rj += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum_f32(&a[..n2]) + pairwise_sum_f32(&a[n2..])
    }
}

/// numpy's pairwise summation (`pairwise_sum` in `loops_utils.h`).
pub fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        // numpy starts at -0.0 to preserve the sign of zero
        let mut res = -0.0;
        for &v in a {
            res += v;
        }
        res
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            for (j, rj) in r.iter_mut().enumerate() {
                *rj += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// `scipy.ndimage._filters._gaussian_kernel1d` for order 0:
/// `phi = exp(-0.5 / sigma**2 * x**2); phi /= phi.sum()`.
pub fn gaussian_kernel1d(sigma: f64, radius: usize) -> Vec<f64> {
    let c = -0.5 / (sigma * sigma);
    let r = radius as i64;
    let mut phi: Vec<f64> = (-r..=r).map(|x| (c * (x * x) as f64).exp()).collect();
    let s = pairwise_sum(&phi);
    for v in phi.iter_mut() {
        *v /= s;
    }
    phi
}

/// Index into a line of length `len` for position `i` (possibly outside),
/// following `NI_ExtendLine`.
fn extend_index(i: i64, len: i64, mode: Mode) -> Option<i64> {
    if i >= 0 && i < len {
        return Some(i);
    }
    let mode = if len == 1 && mode == Mode::Mirror {
        Mode::Nearest
    } else {
        mode
    };
    match mode {
        Mode::Nearest => Some(if i < 0 { 0 } else { len - 1 }),
        Mode::Mirror => {
            let p = 2 * len - 2;
            let mut m = i.rem_euclid(p);
            if m >= len {
                m = p - m;
            }
            Some(m)
        }
        Mode::Reflect => {
            let p = 2 * len;
            let mut m = i.rem_euclid(p);
            if m >= len {
                m = p - 1 - m;
            }
            Some(m)
        }
        Mode::Wrap | Mode::GridWrap => Some(i.rem_euclid(len)),
        Mode::Constant | Mode::GridConstant => None,
    }
}

fn strides_of(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1];
    }
    s
}

/// Starting offsets of every line along `axis` in a C-contiguous array.
fn line_starts(shape: &[usize], axis: usize) -> Vec<usize> {
    let strides = strides_of(shape);
    let mut other: Vec<usize> = (0..shape.len()).filter(|&d| d != axis).collect();
    other.sort();
    let counts: Vec<usize> = other.iter().map(|&d| shape[d]).collect();
    let total: usize = counts.iter().product();
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; counts.len()];
    for _ in 0..total {
        out.push(idx.iter().zip(&other).map(|(&i, &d)| i * strides[d]).sum());
        for k in (0..counts.len()).rev() {
            idx[k] += 1;
            if idx[k] < counts[k] {
                break;
            }
            idx[k] = 0;
        }
    }
    out
}

/// `scipy.ndimage.correlate1d` (origin 0) on f64 data. If `round_f32`, each
/// output is rounded to f32 (output dtype float32).
pub fn correlate1d(
    input: &ArrayD<f64>,
    weights: &[f64],
    axis: usize,
    mode: Mode,
    cval: f64,
    round_f32: bool,
) -> ArrayD<f64> {
    let shape = input.shape().to_vec();
    let input = input.as_standard_layout();
    let data = input.as_slice().unwrap();
    let mut out = vec![0.0f64; data.len()];
    let len = shape[axis];
    if len == 0 || data.is_empty() {
        return ArrayD::from_shape_vec(IxDyn(&shape), out).unwrap();
    }
    let stride = strides_of(&shape)[axis];
    let filter_size = weights.len();
    let size1 = filter_size / 2;
    let size2 = filter_size - size1 - 1;
    // symmetry test
    let mut symmetric = 0i32;
    if filter_size & 1 == 1 {
        symmetric = 1;
        for ii in 1..=filter_size / 2 {
            if (weights[ii + size1] - weights[size1 - ii]).abs() > f64::EPSILON {
                symmetric = 0;
                break;
            }
        }
        if symmetric == 0 {
            symmetric = -1;
            for ii in 1..=filter_size / 2 {
                if (weights[size1 + ii] + weights[size1 - ii]).abs() > f64::EPSILON {
                    symmetric = 0;
                    break;
                }
            }
        }
    }
    let starts = line_starts(&shape, axis);
    let lines: Vec<Vec<f64>> = starts
        .par_iter()
        .map(|&start| {
            let mut line = vec![0.0f64; len + size1 + size2];
            let mut res = vec![0.0f64; len];
            // fill extended line buffer
            for (k, slot) in line.iter_mut().enumerate() {
                let i = k as i64 - size1 as i64;
                *slot = match extend_index(i, len as i64, mode) {
                    Some(j) => data[start + j as usize * stride],
                    None => cval,
                };
            }
            let fw = |j: i64| weights[(j + size1 as i64) as usize];
            for ll in 0..len {
                let c = ll + size1; // centre position in buffer
                let il = |j: i64| line[(c as i64 + j) as usize];
                let mut v;
                if symmetric > 0 {
                    v = il(0) * fw(0);
                    let n = size1;
                    for (k, jj) in (-(size1 as i64)..0).enumerate() {
                        v = accumulate(v, il(jj) + il(-jj), fw(jj), k, n);
                    }
                } else if symmetric < 0 {
                    v = il(0) * fw(0);
                    let n = size1;
                    for (k, jj) in (-(size1 as i64)..0).enumerate() {
                        v = accumulate(v, il(jj) - il(-jj), fw(jj), k, n);
                    }
                } else {
                    v = il(size2 as i64) * fw(size2 as i64);
                    let n = size1 + size2;
                    for (k, jj) in (-(size1 as i64)..size2 as i64).enumerate() {
                        v = accumulate(v, il(jj), fw(jj), k, n);
                    }
                }
                if round_f32 {
                    v = v as f32 as f64;
                }
                res[ll] = v;
            }
            res
        })
        .collect();
    for (start, res) in starts.iter().zip(lines) {
        for (ll, v) in res.into_iter().enumerate() {
            out[start + ll * stride] = v;
        }
    }
    ArrayD::from_shape_vec(IxDyn(&shape), out).unwrap()
}

/// `scipy.ndimage.gaussian_filter` (order 0) on f64 data.
///
/// `sigmas` has one value per axis; axes with sigma <= 1e-15 are skipped.
/// `round_f32` rounds each pass to f32 (when the input dtype is float32).
pub fn gaussian_filter(
    input: &ArrayD<f64>,
    sigmas: &[f64],
    mode: Mode,
    cval: f64,
    truncate: f64,
    round_f32: bool,
) -> ArrayD<f64> {
    let mut current = input.as_standard_layout().into_owned();
    for (axis, &sigma) in sigmas.iter().enumerate() {
        if sigma <= 1e-15 {
            continue;
        }
        let lw = (truncate * sigma + 0.5) as usize;
        let mut weights = gaussian_kernel1d(sigma, lw);
        weights.reverse();
        current = correlate1d(&current, &weights, axis, mode, cval, round_f32);
    }
    current
}

/// `map_coordinate` from `ni_interpolation.c`.
fn map_coordinate(mut x: f64, len: i64, mode: Mode) -> f64 {
    let lenf = len as f64;
    if x < 0.0 {
        match mode {
            Mode::Mirror => {
                if len <= 1 {
                    x = 0.0;
                } else {
                    let sz2 = (2 * len - 2) as f64;
                    x = fmadd(sz2, ((-x / sz2) as i64) as f64, x);
                    x = if x <= 1.0 - lenf { x + sz2 } else { -x };
                }
            }
            Mode::Reflect => {
                if len <= 1 {
                    x = 0.0;
                } else {
                    let sz2 = (2 * len) as f64;
                    if x < -sz2 {
                        x = fmadd(sz2, ((-x / sz2) as i64) as f64, x);
                    }
                    x = if x < -lenf {
                        x + sz2
                    } else {
                        (if x > -1e-15 { 1e-15 } else { -x }) - 1.0
                    };
                }
            }
            Mode::Wrap => {
                if len <= 1 {
                    x = 0.0;
                } else {
                    let sz = (len - 1) as f64;
                    x = fmadd(sz, (((-x / sz) as i64) + 1) as f64, x);
                }
            }
            Mode::GridWrap => {
                if len <= 1 {
                    x = 0.0;
                } else {
                    x = fmadd(lenf, (((-1.0 - x) / lenf) as i64 + 1) as f64, x);
                }
            }
            Mode::Nearest => x = 0.0,
            Mode::Constant => x = -1.0,
            Mode::GridConstant => {}
        }
    } else if x > lenf - 1.0 {
        match mode {
            Mode::Mirror => {
                if len <= 1 {
                    x = 0.0;
                } else {
                    let sz2 = (2 * len - 2) as f64;
                    x = fmadd(-sz2, ((x / sz2) as i64) as f64, x);
                    if x >= lenf {
                        x = sz2 - x;
                    }
                }
            }
            Mode::Reflect => {
                if len <= 1 {
                    x = 0.0;
                } else {
                    let sz2 = (2 * len) as f64;
                    x = fmadd(-sz2, ((x / sz2) as i64) as f64, x);
                    if x >= lenf {
                        x = sz2 - x - 1.0;
                    }
                }
            }
            Mode::Wrap => {
                if len <= 1 {
                    x = 0.0;
                } else {
                    let sz = (len - 1) as f64;
                    x = fmadd(-sz, ((x / sz) as i64) as f64, x);
                }
            }
            Mode::GridWrap => {
                if len <= 1 {
                    x = 0.0;
                } else {
                    x = fmadd(-lenf, ((x / lenf) as i64) as f64, x);
                }
            }
            Mode::Nearest => x = lenf - 1.0,
            Mode::Constant => x = -1.0,
            Mode::GridConstant => {}
        }
    }
    x
}

/// `get_spline_interpolation_weights` from `ni_splines.c`.
fn spline_weights(x: f64, order: usize, w: &mut [f64]) {
    let mut x = x;
    x -= if order & 1 == 1 {
        x.floor()
    } else {
        (x + 0.5).floor()
    };
    let mut y = x;
    let mut z = 1.0 - x;
    match order {
        1 => {
            w[0] = 1.0 - x;
        }
        2 => {
            w[1] = fmadd(-x, x, 0.75);
            y = 0.5 - x;
            w[0] = 0.5 * y * y;
        }
        3 => {
            w[1] = fmadd(y * y * (y - 2.0), 3.0, 4.0) / 6.0;
            w[2] = fmadd(z * z * (z - 2.0), 3.0, 4.0) / 6.0;
            w[0] = z * z * z / 6.0;
        }
        4 => {
            let t = x * x;
            w[2] = fmadd(t, fmadd(t, 0.25, -0.625), 115.0 / 192.0);
            y = 1.0 + x;
            w[1] = fmadd(
                y,
                fmadd(y, y * (5.0 - y) / 6.0 - 1.25, 5.0 / 24.0),
                55.0 / 96.0,
            );
            w[3] = fmadd(
                z,
                fmadd(z, z * (5.0 - z) / 6.0 - 1.25, 5.0 / 24.0),
                55.0 / 96.0,
            );
            y = 0.5 - x;
            let t = y * y;
            w[0] = t * t / 24.0;
        }
        5 => {
            let t = y * y;
            w[2] = fmadd(t, fmadd(t, 0.25 - y / 12.0, -0.5), 0.55);
            let t = z * z;
            w[3] = fmadd(t, fmadd(t, 0.25 - z / 12.0, -0.5), 0.55);
            y += 1.0;
            w[1] = fmadd(
                y,
                fmadd(y, fmadd(y, fmadd(y, y / 24.0 - 0.375, 1.25), -1.75), 0.625),
                0.425,
            );
            z += 1.0;
            w[4] = fmadd(
                z,
                fmadd(z, fmadd(z, fmadd(z, z / 24.0 - 0.375, 1.25), -1.75), 0.625),
                0.425,
            );
            y = 1.0 - x;
            let t = y * y;
            w[0] = y * t * t / 120.0;
        }
        _ => {}
    }
    if order > 0 {
        w[order] = 1.0;
        for i in 0..order {
            w[order] -= w[i];
        }
    }
}

fn filter_poles(order: usize) -> Vec<f64> {
    match order {
        2 => vec![-0.171_572_875_253_809_9],
        3 => vec![-0.267_949_192_431_122_7],
        4 => vec![-0.361_341_225_900_220_2, -0.013_725_429_297_339_121],
        5 => vec![-0.430_575_347_099_973_8, -0.043_096_288_203_264_65],
        _ => vec![],
    }
}

fn apply_spline_filter_line(c: &mut [f64], poles: &[f64], mode: Mode) {
    let n = c.len();
    let mut gain = 1.0;
    for &z in poles {
        gain *= (1.0 - z) * (1.0 - 1.0 / z);
    }
    for v in c.iter_mut() {
        *v *= gain;
    }
    let reflect = matches!(mode, Mode::Nearest | Mode::Reflect);
    let grid_wrap = mode == Mode::GridWrap;
    for &z in poles {
        // causal init
        if grid_wrap {
            let mut z_i = z;
            for i in 1..n {
                c[0] = fmadd(z_i, c[n - i], c[0]);
                z_i *= z;
            }
            c[0] /= 1.0 - z_i;
        } else if reflect {
            let z_n = z.powf(n as f64);
            let mut z_i = z;
            let mut sum = fmadd(z_n, c[n - 1], c[0]);
            for i in 1..n {
                sum = fmadd(z_i, fmadd(z_n, c[n - 1 - i], c[i]), sum);
                z_i *= z;
            }
            c[0] += sum * z / fmadd(-z_n, z_n, 1.0);
        } else {
            let z_n_1 = z.powf((n - 1) as f64);
            let mut z_i = z;
            c[0] = fmadd(z_n_1, c[n - 1], c[0]);
            for i in 1..n - 1 {
                c[0] = fmadd(z_i, fmadd(z_n_1, c[n - 1 - i], c[i]), c[0]);
                z_i *= z;
            }
            c[0] /= fmadd(-z_n_1, z_n_1, 1.0);
        }
        for i in 1..n {
            c[i] = fmadd(z, c[i - 1], c[i]);
        }
        // anticausal init
        if grid_wrap {
            let mut z_i = z;
            for i in 0..n - 1 {
                c[n - 1] = fmadd(z_i, c[i], c[n - 1]);
                z_i *= z;
            }
            c[n - 1] *= z / (z_i - 1.0);
        } else if reflect {
            c[n - 1] *= z / (z - 1.0);
        } else {
            c[n - 1] = fmadd(z, c[n - 2], c[n - 1]) * z / fmadd(z, z, -1.0);
        }
        for i in (0..n - 1).rev() {
            c[i] = z * (c[i + 1] - c[i]);
        }
    }
}

/// `scipy.ndimage.spline_filter1d` (output float64).
pub fn spline_filter1d(input: &ArrayD<f64>, order: usize, axis: usize, mode: Mode) -> ArrayD<f64> {
    let mut out = input.as_standard_layout().into_owned();
    if order <= 1 {
        return out;
    }
    let shape = out.shape().to_vec();
    let len = shape[axis];
    if len <= 1 {
        return out;
    }
    let stride = strides_of(&shape)[axis];
    let poles = filter_poles(order);
    let data = out.as_slice_mut().unwrap();
    let mut line = vec![0.0; len];
    for start in line_starts(&shape, axis) {
        for (i, l) in line.iter_mut().enumerate() {
            *l = data[start + i * stride];
        }
        apply_spline_filter_line(&mut line, &poles, mode);
        for (i, l) in line.iter().enumerate() {
            data[start + i * stride] = *l;
        }
    }
    out
}

/// `scipy.ndimage.spline_filter` (output float64).
pub fn spline_filter(input: &ArrayD<f64>, order: usize, mode: Mode) -> ArrayD<f64> {
    let mut out = input.as_standard_layout().into_owned();
    if order <= 1 {
        return out;
    }
    for axis in 0..out.ndim() {
        out = spline_filter1d(&out, order, axis, mode);
    }
    out
}

/// Pad with `npad` on all sides (for 'nearest' and 'grid-constant' prefiltering).
fn prepad(input: &ArrayD<f64>, mode: Mode, cval: f64) -> (ArrayD<f64>, usize) {
    let npad = match mode {
        Mode::Nearest | Mode::GridConstant => 12usize,
        _ => return (input.to_owned(), 0),
    };
    let in_shape = input.shape().to_vec();
    let shape: Vec<usize> = in_shape.iter().map(|&s| s + 2 * npad).collect();
    let input = input.as_standard_layout();
    let src = input.as_slice().unwrap();
    let in_strides = strides_of(&in_shape);
    let total: usize = shape.iter().product();
    let mut data = Vec::with_capacity(total);
    let mut idx = vec![0usize; shape.len()];
    for _ in 0..total {
        let mut off = 0usize;
        let mut inside = true;
        for d in 0..shape.len() {
            let j = idx[d] as i64 - npad as i64;
            let n = in_shape[d] as i64;
            let j = if mode == Mode::Nearest {
                j.clamp(0, n - 1)
            } else if j < 0 || j >= n {
                inside = false;
                0
            } else {
                j
            };
            off += j as usize * in_strides[d];
        }
        data.push(if inside { src[off] } else { cval });
        for d in (0..shape.len()).rev() {
            idx[d] += 1;
            if idx[d] < shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    let out = ArrayD::from_shape_vec(IxDyn(&shape), data).unwrap();
    (out, npad)
}

/// Per-axis precomputation of interpolation positions for one output coordinate.
#[derive(Clone)]
struct AxisPos {
    /// `None` = point uses the constant border (`zeros`).
    start: Option<i64>,
    /// Indices of the filter footprint along this axis (len order+1).
    idx: Vec<usize>,
    /// For grid-constant: which footprint entries are outside.
    outside: Vec<bool>,
    weights: Vec<f64>,
}

fn axis_position(cc_in: f64, len: usize, order: usize, mode: Mode, npad: usize) -> AxisPos {
    let len_i = len as i64;
    let mut cc = cc_in + npad as f64;
    if mode != Mode::GridConstant && mode != Mode::Nearest {
        cc = map_coordinate(cc, len_i, mode);
    }
    let mut pos = AxisPos {
        start: None,
        idx: vec![0; order + 1],
        outside: vec![false; order + 1],
        weights: vec![1.0; order + 1],
    };
    if cc > -1.0 || mode == Mode::GridConstant || mode == Mode::Nearest {
        let start = if order & 1 == 1 {
            cc.floor() as i64 - (order / 2) as i64
        } else {
            (cc + 0.5).floor() as i64 - (order / 2) as i64
        };
        pos.start = Some(start);
        let spline_mode = mode.spline_mode();
        for h in 0..=order {
            let i = start + h as i64;
            if mode == Mode::GridConstant {
                pos.outside[h] = i < 0 || i >= len_i;
                pos.idx[h] = i.clamp(0, (len_i - 1).max(0)) as usize;
            } else if i < 0 || i >= len_i {
                pos.idx[h] = map_coordinate(i as f64, len_i, spline_mode) as i64 as usize;
            } else {
                pos.idx[h] = i as usize;
            }
        }
        if order > 0 {
            spline_weights(cc, order, &mut pos.weights);
        }
    }
    pos
}

/// Evaluate the spline at one point given per-axis positions.
fn interpolate_point(
    data: &[f64],
    strides: &[usize],
    positions: &[&AxisPos],
    order: usize,
    mode: Mode,
    cval: f64,
) -> f64 {
    if positions.iter().any(|p| p.start.is_none()) {
        return cval;
    }
    let rank = positions.len();
    let fsize = order + 1;
    let filter_size = fsize.pow(rank as u32);
    let mut ff_buf = [0usize; 32];
    let ff = &mut ff_buf[..rank];
    let mut t = 0.0;
    for _ in 0..filter_size {
        let mut is_cval = false;
        if mode == Mode::GridConstant {
            for (j, p) in positions.iter().enumerate() {
                if p.outside[ff[j]] {
                    is_cval = true;
                }
            }
        }
        let mut coeff = if is_cval {
            cval
        } else {
            let mut off = 0usize;
            for (j, p) in positions.iter().enumerate() {
                off += p.idx[ff[j]] * strides[j];
            }
            data[off]
        };
        if order > 0 {
            for (j, p) in positions.iter().enumerate() {
                coeff *= p.weights[ff[j]];
            }
        }
        t += coeff;
        for j in (0..rank).rev() {
            if ff[j] < order {
                ff[j] += 1;
                break;
            }
            ff[j] = 0;
        }
    }
    t
}

/// Output size of `scipy.ndimage.zoom` for a zoom factor: `round(n * z)`
/// using Python's round-half-even.
pub fn zoom_output_shape(shape: &[usize], zoom: &[f64]) -> Vec<usize> {
    shape
        .iter()
        .zip(zoom)
        .map(|(&n, &z)| round_half_even(n as f64 * z) as usize)
        .collect()
}

/// Python 3 `round()` (banker's rounding).
pub fn round_half_even(x: f64) -> f64 {
    x.round_ties_even()
}

/// `scipy.ndimage.zoom(input, zoom_factors, ...)` returning the raw
/// interpolated values (f64) before conversion to the output dtype. Use
/// [`crate::array::Element::from_f64_ndi`] for integer outputs.
///
/// The output shape is `round(shape * zoom)` (Python rounding). If all zoom
/// factors are 1 and `prefilter` is set, the input is returned unchanged
/// (scipy's early exit).
pub fn zoom(
    input: &ArrayD<f64>,
    zoom_factors: &[f64],
    order: usize,
    mode: Mode,
    cval: f64,
    prefilter: bool,
    grid_mode: bool,
) -> Result<ArrayD<f64>> {
    if order > 5 {
        return Err(Error::value("spline order not supported"));
    }
    let in_shape = input.shape().to_vec();
    if zoom_factors.len() != in_shape.len() {
        return Err(Error::value("zoom factors must match input rank"));
    }
    let output_shape = zoom_output_shape(&in_shape, zoom_factors);
    let output_shape = output_shape.as_slice();
    if prefilter && zoom_factors.iter().all(|&z| z == 1.0) {
        return Ok(input.as_standard_layout().into_owned());
    }
    let (filtered, npad) = if prefilter && order > 1 {
        let (padded, npad) = prepad(input, mode, cval);
        (spline_filter(&padded, order, mode), npad)
    } else {
        (input.as_standard_layout().into_owned(), 0)
    };
    let fshape = filtered.shape().to_vec();
    let zooms: Vec<f64> = in_shape
        .iter()
        .zip(output_shape)
        .map(|(&i, &o)| {
            let (num, div) = if grid_mode {
                (i as f64, o as f64)
            } else {
                (i as f64 - 1.0, o as f64 - 1.0)
            };
            if div != 0.0 { num / div } else { 1.0 }
        })
        .collect();
    // precompute per-axis positions
    let per_axis: Vec<Vec<AxisPos>> = (0..in_shape.len())
        .map(|jj| {
            (0..output_shape[jj])
                .map(|kk| {
                    let mut cc = kk as f64;
                    if grid_mode {
                        cc += 0.5;
                        cc *= zooms[jj];
                        cc -= 0.5;
                    } else {
                        cc *= zooms[jj];
                    }
                    axis_position(cc, fshape[jj], order, mode, npad)
                })
                .collect()
        })
        .collect();
    let data = filtered.as_slice().unwrap();
    let strides = strides_of(&fshape);
    let total: usize = output_shape.iter().product();
    let rank = output_shape.len();
    let mut out = vec![0.0f64; total];
    if total == 0 {
        return Ok(ArrayD::from_shape_vec(IxDyn(output_shape), out).unwrap());
    }
    let row = output_shape[rank - 1];
    // every output point is independent: compute rows in parallel
    out.par_chunks_mut(row).enumerate().for_each(|(r, chunk)| {
        let mut idx = vec![0usize; rank];
        let mut rem = r;
        for j in (0..rank - 1).rev() {
            idx[j] = rem % output_shape[j];
            rem /= output_shape[j];
        }
        let mut positions: Vec<&AxisPos> = (0..rank).map(|j| &per_axis[j][idx[j]]).collect();
        for (k, v) in chunk.iter_mut().enumerate() {
            positions[rank - 1] = &per_axis[rank - 1][k];
            *v = interpolate_point(data, &strides, &positions, order, mode, cval);
        }
    });
    Ok(ArrayD::from_shape_vec(IxDyn(output_shape), out).unwrap())
}

/// `scipy.ndimage.map_coordinates` returning f64 values.
///
/// `coordinates` is `[ndim][npoints]`.
pub fn map_coordinates(
    input: &ArrayD<f64>,
    coordinates: &[Vec<f64>],
    order: usize,
    mode: Mode,
    cval: f64,
    prefilter: bool,
) -> Result<Vec<f64>> {
    if order > 5 {
        return Err(Error::value("spline order not supported"));
    }
    if coordinates.len() != input.ndim() {
        return Err(Error::value("invalid shape for coordinate array"));
    }
    let (filtered, npad) = if prefilter && order > 1 {
        let (padded, npad) = prepad(input, mode, cval);
        (spline_filter(&padded, order, mode), npad)
    } else {
        (input.as_standard_layout().into_owned(), 0)
    };
    let fshape = filtered.shape().to_vec();
    let data = filtered.as_slice().unwrap();
    let strides = strides_of(&fshape);
    let npoints = coordinates.first().map(|c| c.len()).unwrap_or(0);
    let mut out = Vec::with_capacity(npoints);
    for p in 0..npoints {
        let pos: Vec<AxisPos> = (0..fshape.len())
            .map(|d| axis_position(coordinates[d][p], fshape[d], order, mode, npad))
            .collect();
        let refs: Vec<&AxisPos> = pos.iter().collect();
        out.push(interpolate_point(data, &strides, &refs, order, mode, cval));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_sums_to_one() {
        let k = gaussian_kernel1d(0.5, 2);
        assert_eq!(k.len(), 5);
        assert!((k.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pairwise() {
        let v: Vec<f64> = (0..300).map(|x| x as f64 * 0.1).collect();
        assert!((pairwise_sum(&v) - v.iter().sum::<f64>()).abs() < 1e-9);
    }

    #[test]
    fn zoom_identity() {
        let a = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let z = zoom(&a, &[1.0, 1.0], 1, Mode::Mirror, 0.0, true, true).unwrap();
        assert_eq!(z, a);
    }

    #[test]
    fn zoom_half_linear_grid() {
        // grid_mode zoom by 0.5: output i samples input at 2i+0.5
        let a = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0., 2., 4., 6.]).unwrap();
        let z = zoom(&a, &[0.5], 1, Mode::Mirror, 0.0, true, true).unwrap();
        assert_eq!(z.as_slice().unwrap(), &[1.0, 5.0]);
    }

    #[test]
    fn spline_interp_exact_at_nodes() {
        let a = ArrayD::from_shape_vec(IxDyn(&[6]), vec![1., 3., 2., 5., 4., 0.]).unwrap();
        let coords = vec![vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.5, -2.0]];
        let v = map_coordinates(&a, &coords, 3, Mode::Constant, f64::NAN, true).unwrap();
        for (i, x) in [1., 3., 2., 5., 4., 0.].iter().enumerate() {
            assert!((v[i] - x).abs() < 1e-9, "{} vs {}", v[i], x);
        }
        assert!(v[6].is_nan() && v[7].is_nan());
    }
}
