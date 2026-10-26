//! Functions for generating synthetic data (`ome_zarr.data`).
//!
//! The sample images are scikit-image's `coins` and `astronaut` (both public
//! domain), embedded in the binary. The coins segmentation is a port of the
//! scikit-image pipeline (Otsu threshold, closing, clear_border,
//! remove_small_objects, label) and matches it exactly.

use std::collections::HashMap;

use ndarray::{Array2, ArrayD, IxDyn, s};

use crate::array::{LazyArray, NdArray};
use crate::classes::{
    ChannelColor, LabelsArg, OmeZarrImage, OmeZarrLabels, OmeZarrMultiscale, Version,
};
use crate::error::{Error, Result};
use crate::format::{CURRENT_FORMAT, Format};
use crate::zarr::{Group, Mode};

const COINS_PNG: &[u8] = include_bytes!("../assets/coins.png");
const ASTRONAUT_PNG: &[u8] = include_bytes!("../assets/astronaut.png");

/// Dimension holding channels in [`rgb_to_5d`] output.
pub const CHANNEL_DIMENSION: usize = 1;

#[cfg(feature = "render")]
fn decode(bytes: &[u8]) -> Result<threers::PngImage> {
    threers::decode_png(bytes)
        .map_err(|e| Error::value(format!("could not decode sample image: {e}")))
}

/// A decoded sample image, RGBA8.
#[cfg(not(feature = "render"))]
struct PngImage {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

/// The sample images are decoded with threers' PNG reader.
#[cfg(not(feature = "render"))]
fn decode(_bytes: &[u8]) -> Result<PngImage> {
    Err(Error::NotImplemented(
        "the sample images need the `render` feature".into(),
    ))
}

/// `skimage.data.coins()`: 303x384 greyscale uint8.
pub fn coins_image() -> Result<Array2<u8>> {
    let img = decode(COINS_PNG)?;
    let (w, h) = (img.width as usize, img.height as usize);
    Ok(Array2::from_shape_fn((h, w), |(y, x)| {
        img.rgba[(y * w + x) * 4]
    }))
}

/// `skimage.data.astronaut()`: 512x512 RGB uint8 as (y, x, c).
pub fn astronaut_image() -> Result<ndarray::Array3<u8>> {
    let img = decode(ASTRONAUT_PNG)?;
    let (w, h) = (img.width as usize, img.height as usize);
    Ok(ndarray::Array3::from_shape_fn((h, w, 3), |(y, x, c)| {
        img.rgba[(y * w + x) * 4 + c]
    }))
}

/// `skimage.filters.threshold_otsu` for integer images.
pub fn threshold_otsu(image: &Array2<u8>) -> u8 {
    let first = image[[0, 0]];
    if image.iter().all(|&v| v == first) {
        return first;
    }
    let min = *image.iter().min().unwrap() as usize;
    let max = *image.iter().max().unwrap() as usize;
    let mut counts = vec![0i64; max - min + 1];
    for &v in image.iter() {
        counts[v as usize - min] += 1;
    }
    let centers: Vec<i64> = (min as i64..=max as i64).collect();
    let n = counts.len();
    let mut weight1 = vec![0i64; n];
    let mut cs1 = vec![0i64; n];
    let (mut a, mut b) = (0i64, 0i64);
    for i in 0..n {
        a += counts[i];
        b += counts[i] * centers[i];
        weight1[i] = a;
        cs1[i] = b;
    }
    let mean1: Vec<f64> = (0..n).map(|i| cs1[i] as f64 / weight1[i] as f64).collect();
    let mut weight2 = vec![0i64; n];
    let mut cs2r = vec![0i64; n];
    let (mut a, mut b) = (0i64, 0i64);
    for i in (0..n).rev() {
        a += counts[i];
        weight2[i] = a;
    }
    // np.cumsum((counts * centers)[::-1]) / weight2[::-1], reversed back
    for (k, i) in (0..n).rev().enumerate() {
        b += counts[i] * centers[i];
        cs2r[k] = b;
    }
    let mean2: Vec<f64> = (0..n)
        .map(|i| {
            let k = n - 1 - i;
            cs2r[k] as f64 / weight2[i] as f64
        })
        .collect();
    let mut best = 0usize;
    let mut best_v = f64::NEG_INFINITY;
    for i in 0..n - 1 {
        let d = mean1[i] - mean2[i + 1];
        let v = (weight1[i] * weight2[i + 1]) as f64 * (d * d);
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    centers[best] as u8
}

fn reflect(i: i64, n: i64) -> usize {
    // scipy 'reflect': (d c b a | a b c d | d c b a)
    let p = 2 * n;
    let mut m = i.rem_euclid(p);
    if m >= n {
        m = p - 1 - m;
    }
    m as usize
}

/// Grey (here: binary) max/min filter over offsets `lo..=hi` in both axes,
/// with scipy's 'reflect' boundary.
fn rect_filter(img: &Array2<bool>, lo: i64, hi: i64, max: bool) -> Array2<bool> {
    let (h, w) = img.dim();
    Array2::from_shape_fn((h, w), |(y, x)| {
        let mut acc = !max;
        for dy in lo..=hi {
            for dx in lo..=hi {
                let v = img[[
                    reflect(y as i64 + dy, h as i64),
                    reflect(x as i64 + dx, w as i64),
                ]];
                if max {
                    acc |= v;
                } else {
                    acc &= v;
                }
            }
        }
        acc
    })
}

/// `skimage.morphology.closing(image, footprint_rectangle((k, k)))` for binary images.
pub fn binary_closing_rect(img: &Array2<bool>, k: usize) -> Array2<bool> {
    // pad_footprint(pad_end=False) puts the extra row/column at the front; the
    // dilation then covers offsets [-(k-1)/2 .. k/2] and the erosion the mirror.
    let k = k as i64;
    let (lo, hi) = if k % 2 == 0 {
        (-(k / 2) + 1, k / 2)
    } else {
        (-(k / 2), k / 2)
    };
    let dil = rect_filter(img, lo, hi, true);
    rect_filter(&dil, -hi, -lo, false)
}

/// Connected-component labelling with labels assigned in raster order of
/// each component's first pixel (`skimage.measure.label` /
/// `scipy.ndimage.label`). `full` selects 8-connectivity, otherwise 4.
pub fn label(img: &Array2<bool>, full: bool) -> (Array2<i32>, usize) {
    let (h, w) = img.dim();
    let mut out = Array2::<i32>::zeros((h, w));
    let mut next = 0i32;
    let mut stack = vec![];
    for y in 0..h {
        for x in 0..w {
            if !img[[y, x]] || out[[y, x]] != 0 {
                continue;
            }
            next += 1;
            out[[y, x]] = next;
            stack.push((y, x));
            while let Some((cy, cx)) = stack.pop() {
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        if (dy == 0 && dx == 0) || (!full && dy != 0 && dx != 0) {
                            continue;
                        }
                        let (ny, nx) = (cy as i64 + dy, cx as i64 + dx);
                        if ny < 0 || nx < 0 || ny >= h as i64 || nx >= w as i64 {
                            continue;
                        }
                        let (ny, nx) = (ny as usize, nx as usize);
                        if img[[ny, nx]] && out[[ny, nx]] == 0 {
                            out[[ny, nx]] = next;
                            stack.push((ny, nx));
                        }
                    }
                }
            }
        }
    }
    (out, next as usize)
}

/// `skimage.segmentation.clear_border` for binary images.
pub fn clear_border(img: &Array2<bool>) -> Array2<bool> {
    let (labels, _) = label(img, true);
    let (h, w) = img.dim();
    let mut touching = std::collections::HashSet::new();
    for y in 0..h {
        for x in 0..w {
            if y == 0 || x == 0 || y == h - 1 || x == w - 1 {
                touching.insert(labels[[y, x]]);
            }
        }
    }
    let mut out = img.clone();
    for ((y, x), v) in out.indexed_iter_mut() {
        if touching.contains(&labels[[y, x]]) {
            *v = false;
        }
    }
    out
}

/// `skimage.morphology.remove_small_objects(img, max_size=max_size)` for
/// binary images (connectivity 1).
pub fn remove_small_objects(img: &Array2<bool>, max_size: usize) -> Array2<bool> {
    let (ccs, _) = label(img, false);
    let mut sizes: HashMap<i32, usize> = HashMap::new();
    for &v in ccs.iter() {
        *sizes.entry(v).or_default() += 1;
    }
    let mut out = img.clone();
    for ((y, x), v) in out.indexed_iter_mut() {
        if sizes[&ccs[[y, x]]] <= max_size {
            *v = false;
        }
    }
    out
}

/// The coins image (cropped by 50 px) and its segmentation.
pub fn coins_arrays() -> Result<(Array2<u8>, Array2<i32>)> {
    let full = coins_image()?;
    let (h, w) = full.dim();
    let image = full.slice(s![50..h - 50, 50..w - 50]).to_owned();
    let thresh = threshold_otsu(&image);
    let bw = binary_closing_rect(&image.mapv(|v| v > thresh), 4);
    let cleared = remove_small_objects(&clear_border(&bw), 20);
    let (labels, _) = label(&cleared, true);
    Ok((image, labels))
}

fn chunks_for(shape: &[usize]) -> Vec<u64> {
    shape
        .iter()
        .map(|&s| if s > 8 { (s / 8) as u64 } else { 1 })
        .collect()
}

fn axes(s: &str) -> Vec<String> {
    s.chars().map(|c| c.to_string()).collect()
}

/// Sample data from skimage: the coins image and labels.
pub fn coins() -> Result<(OmeZarrMultiscale, OmeZarrLabels)> {
    let (image, labels) = coins_arrays()?;
    let shape = image.shape().to_vec();
    let chunks = chunks_for(&shape);
    let img_data = LazyArray::from_ndarray_chunked(NdArray::UInt8(image.into_dyn()), &chunks);
    let lbl_data = LazyArray::from_ndarray_chunked(NdArray::Int32(labels.into_dyn()), &chunks);
    let img = OmeZarrImage::new(img_data, &axes("yx"), None, None, None, "coins")?;
    let lbl = OmeZarrImage::new(lbl_data, &axes("yx"), None, None, None, "coins")?;
    let img_ms = OmeZarrMultiscale::builder(img)
        .contrast_limits(vec![(0.0, 255.0)])
        .channel_colors(vec![ChannelColor::from("FF0000")])
        .build()?;
    let lbl_ms = OmeZarrLabels::from_image(lbl)?;
    Ok((img_ms, lbl_ms))
}

/// Sample data from skimage: the astronaut (tiled 2x2, 3 channels) and two
/// circle labels.
pub fn astronaut() -> Result<(OmeZarrMultiscale, OmeZarrLabels)> {
    let astro = astronaut_image()?;
    let (h, w, _) = astro.dim();
    let pixels = ArrayD::from_shape_fn(IxDyn(&[3, 2 * h, 2 * w]), |i| {
        astro[[i[1] % h, i[2] % w, i[0]]]
    });
    let (y, x) = (2 * h, 2 * w);
    let mut label = Array2::<i8>::zeros((y, x));
    make_circle(100, 100, 1, &mut label.slice_mut(s![200..300, 200..300]));
    make_circle(150, 150, 2, &mut label.slice_mut(s![250..400, 250..400]));
    let chunks = chunks_for(pixels.shape());
    let chunks_labels = chunks_for(label.shape());
    let img = OmeZarrImage::new(
        LazyArray::from_ndarray_chunked(NdArray::UInt8(pixels), &chunks),
        &axes("cyx"),
        None,
        None,
        None,
        "astronaut",
    )?;
    let lbl = OmeZarrImage::new(
        LazyArray::from_ndarray_chunked(NdArray::Int8(label.into_dyn()), &chunks_labels),
        &axes("yx"),
        None,
        None,
        None,
        "astronaut_labels",
    )?;
    let img_ms = OmeZarrMultiscale::builder(img)
        .contrast_limits(vec![(0.0, 255.0), (0.0, 255.0), (0.0, 255.0)])
        .channel_colors(vec!["FF0000".into(), "00FF00".into(), "0000FF".into()])
        .channel_names(&["Red", "Green", "Blue"])
        .build()?;
    let lbl_ms = OmeZarrLabels::from_image(lbl)?;
    Ok((img_ms, lbl_ms))
}

/// Apply a 2D circular mask to the given array.
///
/// ```
/// let mut a = ndarray::Array2::<i8>::zeros((8, 8));
/// ome_zarr::data::make_circle(8, 8, 1, &mut a.view_mut());
/// assert_eq!(a.row(0).to_vec(), vec![0; 8]);
/// assert_eq!(a.row(1).to_vec(), vec![0, 0, 1, 1, 1, 1, 1, 0]);
/// assert_eq!(a.row(4).to_vec(), vec![0, 1, 1, 1, 1, 1, 1, 1]);
/// ```
pub fn make_circle<T: Copy + num_from_i64::FromI64>(
    h: usize,
    w: usize,
    value: i64,
    target: &mut ndarray::ArrayViewMut2<T>,
) {
    let cx = (w / 2) as i64;
    let cy = (h / 2) as i64;
    let r = (w.min(h) / 2) as i64;
    for y in 0..h {
        for x in 0..w {
            let (dx, dy) = (x as i64 - cx, y as i64 - cy);
            if dx * dx + dy * dy < r * r {
                target[[y, x]] = T::from_i64(value);
            }
        }
    }
}

/// Minimal integer conversion used by [`make_circle`].
pub mod num_from_i64 {
    pub trait FromI64 {
        fn from_i64(v: i64) -> Self;
    }
    macro_rules! imp {
        ($($t:ty),*) => { $(impl FromI64 for $t { fn from_i64(v: i64) -> Self { v as $t } })* };
    }
    imp!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);
}

/// Convert an RGB (y, x, c) or greyscale (y, x) image into 5D (t, c, z, y, x).
pub fn rgb_to_5d(pixels: &NdArray) -> Result<NdArray> {
    let sh = pixels.shape();
    match sh.len() {
        2 => pixels.reshape(&[1, 1, 1, sh[0], sh[1]]),
        3 => {
            let (y, x, c) = (sh[0], sh[1], sh[2]);
            Ok(crate::nd_map!(pixels, a => {
                ArrayD::from_shape_fn(IxDyn(&[1, c, 1, y, x]), |i| a[IxDyn(&[i[3], i[4], i[1]])])
            }))
        }
        n => Err(Error::Assertion(format!("expecting 2 or 3d: ({n})"))),
    }
}

/// Which synthetic dataset to create.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleData {
    Coins,
    Astronaut,
}

/// Generate a synthetic image pyramid with labels.
pub fn create_zarr(zarr_directory: &str, method: SampleData, fmt: Format) -> Result<Group> {
    let (mut image, label) = match method {
        SampleData::Coins => coins()?,
        SampleData::Astronaut => astronaut()?,
    };
    let name = label.name().to_string();
    image.set_labels(Some(LabelsArg::Map(vec![(name, label)])));
    let version = match fmt {
        Format::V04 => Version::V04,
        Format::V05 => Version::V05,
        other => {
            return Err(Error::value(format!(
                "Unsupported OME-Zarr version: {}",
                other.version()
            )));
        }
    };
    image.to_ome_zarr(zarr_directory, None, version, true, true)?;
    Group::open_url(zarr_directory, Mode::Append, None)
}

/// [`create_zarr`] with the coins data and the current format.
pub fn create_zarr_default(zarr_directory: &str) -> Result<Group> {
    create_zarr(zarr_directory, SampleData::Coins, CURRENT_FORMAT)
}
