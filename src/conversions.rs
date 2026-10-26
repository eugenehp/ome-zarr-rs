//! Simple conversion helpers.

/// Get rgba (0-1) e.g. (1, 0.5, 0, 1) from integer.
///
/// ```
/// use ome_zarr::conversions::int_to_rgba;
/// assert_eq!(int_to_rgba(0), [0.0, 0.0, 0.0, 0.0]);
/// let c: Vec<f64> = int_to_rgba(100100).iter().map(|x| (x * 1000.0).round() / 1000.0).collect();
/// assert_eq!(c, vec![0.0, 0.004, 0.529, 0.016]);
/// ```
pub fn int_to_rgba(v: i32) -> [f64; 4] {
    int_to_rgba_255(v).map(|x| x as f64 / 255.0)
}

/// Get rgba (0-255) from integer.
///
/// ```
/// use ome_zarr::conversions::int_to_rgba_255;
/// assert_eq!(int_to_rgba_255(0), [0, 0, 0, 0]);
/// assert_eq!(int_to_rgba_255(100100), [0, 1, 135, 4]);
/// ```
pub fn int_to_rgba_255(v: i32) -> [u8; 4] {
    v.to_be_bytes()
}

/// Convert a color tuple into a (signed, big endian) integer.
///
/// ```
/// use ome_zarr::conversions::rgba_to_int;
/// assert_eq!(rgba_to_int(0, 0, 0, 0), 0);
/// assert_eq!(rgba_to_int(0, 1, 135, 4), 100100);
/// ```
pub fn rgba_to_int(r: u8, g: u8, b: u8, a: u8) -> i32 {
    i32::from_be_bytes([r, g, b, a])
}

/// Parse a 6-digit hex colour such as `"FF0000"` (optional leading `#`)
/// into rgb in 0-1.
pub fn hex_to_rgb(color: &str) -> Option<[f64; 3]> {
    let c = color.trim_start_matches('#');
    if c.len() < 6 {
        return None;
    }
    let mut out = [0.0; 3];
    for (i, o) in out.iter_mut().enumerate() {
        *o = u8::from_str_radix(&c[i * 2..i * 2 + 2], 16).ok()? as f64 / 255.0;
    }
    Some(out)
}
