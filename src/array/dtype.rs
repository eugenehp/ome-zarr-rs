//! Data types supported for OME-Zarr arrays.

use std::fmt;

use crate::error::{Error, Result};

/// The numeric data types supported by ome-zarr.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float16,
    Float32,
    Float64,
    Complex64,
    Complex128,
}

impl DType {
    /// All supported types.
    pub const ALL: [DType; 14] = [
        DType::Bool,
        DType::UInt8,
        DType::UInt16,
        DType::UInt32,
        DType::UInt64,
        DType::Int8,
        DType::Int16,
        DType::Int32,
        DType::Int64,
        DType::Float16,
        DType::Float32,
        DType::Float64,
        DType::Complex64,
        DType::Complex128,
    ];

    /// numpy-style name, e.g. `uint8`. Also the zarr v3 data type name.
    pub fn name(&self) -> &'static str {
        match self {
            DType::Bool => "bool",
            DType::UInt8 => "uint8",
            DType::UInt16 => "uint16",
            DType::UInt32 => "uint32",
            DType::UInt64 => "uint64",
            DType::Int8 => "int8",
            DType::Int16 => "int16",
            DType::Int32 => "int32",
            DType::Int64 => "int64",
            DType::Float16 => "float16",
            DType::Float32 => "float32",
            DType::Float64 => "float64",
            DType::Complex64 => "complex64",
            DType::Complex128 => "complex128",
        }
    }

    /// Size of one element in bytes (numpy `itemsize`).
    pub fn itemsize(&self) -> usize {
        match self {
            DType::Bool | DType::UInt8 | DType::Int8 => 1,
            DType::UInt16 | DType::Int16 | DType::Float16 => 2,
            DType::UInt32 | DType::Int32 | DType::Float32 => 4,
            DType::UInt64 | DType::Int64 | DType::Float64 | DType::Complex64 => 8,
            DType::Complex128 => 16,
        }
    }

    pub fn is_integer(&self) -> bool {
        !matches!(
            self,
            DType::Bool
                | DType::Float16
                | DType::Float32
                | DType::Float64
                | DType::Complex64
                | DType::Complex128
        )
    }

    pub fn is_float(&self) -> bool {
        matches!(self, DType::Float16 | DType::Float32 | DType::Float64)
    }

    pub fn is_complex(&self) -> bool {
        matches!(self, DType::Complex64 | DType::Complex128)
    }

    pub fn is_signed(&self) -> bool {
        matches!(
            self,
            DType::Int8
                | DType::Int16
                | DType::Int32
                | DType::Int64
                | DType::Float16
                | DType::Float32
                | DType::Float64
                | DType::Complex64
                | DType::Complex128
        )
    }

    /// Zarr v2 `dtype` string (little endian), e.g. `|u1`, `<u2`.
    pub fn zarr_v2(&self) -> &'static str {
        match self {
            DType::Bool => "|b1",
            DType::UInt8 => "|u1",
            DType::UInt16 => "<u2",
            DType::UInt32 => "<u4",
            DType::UInt64 => "<u8",
            DType::Int8 => "|i1",
            DType::Int16 => "<i2",
            DType::Int32 => "<i4",
            DType::Int64 => "<i8",
            DType::Float16 => "<f2",
            DType::Float32 => "<f4",
            DType::Float64 => "<f8",
            DType::Complex64 => "<c8",
            DType::Complex128 => "<c16",
        }
    }

    /// Parse a numpy/zarr v3 name (`uint8`) or a zarr v2 type string (`<u1`).
    pub fn parse(s: &str) -> Result<DType> {
        let s = s.trim();
        for d in DType::ALL {
            if d.name() == s {
                return Ok(d);
            }
        }
        let t = s.trim_start_matches(['<', '>', '|', '=']);
        let d = match t {
            "b1" | "?" => DType::Bool,
            "u1" | "B" => DType::UInt8,
            "u2" | "H" => DType::UInt16,
            "u4" | "I" => DType::UInt32,
            "u8" | "Q" => DType::UInt64,
            "i1" | "b" => DType::Int8,
            "i2" | "h" => DType::Int16,
            "i4" | "i" => DType::Int32,
            "i8" | "q" => DType::Int64,
            "f2" | "e" => DType::Float16,
            "f4" | "f" => DType::Float32,
            "c8" | "F" => DType::Complex64,
            "c16" | "D" | "complex" => DType::Complex128,
            "f8" | "d" | "float" => DType::Float64,
            "int" => DType::Int64,
            _ => return Err(Error::value(format!("unsupported data type: {s}"))),
        };
        Ok(d)
    }

    /// Largest representable value as f64 (used by viewers / contrast limits).
    pub fn max_value(&self) -> f64 {
        match self {
            DType::Bool => 1.0,
            DType::UInt8 => u8::MAX as f64,
            DType::UInt16 => u16::MAX as f64,
            DType::UInt32 => u32::MAX as f64,
            DType::UInt64 => u64::MAX as f64,
            DType::Int8 => i8::MAX as f64,
            DType::Int16 => i16::MAX as f64,
            DType::Int32 => i32::MAX as f64,
            DType::Int64 => i64::MAX as f64,
            DType::Float16 => half::f16::MAX.to_f64(),
            DType::Float32 | DType::Complex64 => f32::MAX as f64,
            DType::Float64 | DType::Complex128 => f64::MAX,
        }
    }

    /// Smallest representable value as f64.
    pub fn min_value(&self) -> f64 {
        match self {
            DType::Bool => 0.0,
            DType::UInt8 | DType::UInt16 | DType::UInt32 | DType::UInt64 => 0.0,
            DType::Int8 => i8::MIN as f64,
            DType::Int16 => i16::MIN as f64,
            DType::Int32 => i32::MIN as f64,
            DType::Int64 => i64::MIN as f64,
            DType::Float16 => half::f16::MIN.to_f64(),
            DType::Float32 | DType::Complex64 => f32::MIN as f64,
            DType::Float64 | DType::Complex128 => f64::MIN,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Scalar element trait implemented for all supported element types.
pub trait Element:
    Copy
    + Send
    + Sync
    + PartialEq
    + Default
    + fmt::Debug
    + 'static
    + zarrs::array::Element
    + zarrs::array::ElementOwned
{
    const DTYPE: DType;
    /// Lossless-ish conversion to f64 (numpy `astype(float)`).
    fn to_f64(self) -> f64;
    /// numpy `astype` from float: truncation toward zero (saturating).
    fn from_f64(v: f64) -> Self;
    /// scipy.ndimage output conversion: round half away from zero, clamp to range.
    fn from_f64_ndi(v: f64) -> Self;
    /// Cast from an i64 (used e.g. for label values).
    fn from_i64(v: i64) -> Self {
        Self::from_f64(v as f64)
    }
    /// Value as i64 if integral.
    fn to_i64(self) -> i64 {
        self.to_f64() as i64
    }
    /// The imaginary part (0 for real types).
    fn imag_f64(self) -> f64 {
        0.0
    }
    /// From a complex value (the real part for real types, as numpy's cast).
    fn from_complex(re: f64, _im: f64) -> Self {
        Self::from_f64(re)
    }
    /// numpy's ordering: lexicographic on (real, imaginary) for complex types.
    fn np_lt(self, other: Self) -> bool {
        let (a, b) = (self.to_f64(), other.to_f64());
        a < b || (a == b && self.imag_f64() < other.imag_f64())
    }
}

impl Element for half::f16 {
    const DTYPE: DType = DType::Float16;
    #[inline]
    fn to_f64(self) -> f64 {
        self.to_f64()
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        half::f16::from_f64(v)
    }
    #[inline]
    fn from_f64_ndi(v: f64) -> Self {
        half::f16::from_f64(v)
    }
}

macro_rules! impl_complex {
    ($f:ty, $d:expr) => {
        impl Element for num_complex::Complex<$f> {
            const DTYPE: DType = $d;
            /// The real part (numpy `astype(float)` discards the imaginary part).
            #[inline]
            fn to_f64(self) -> f64 {
                self.re as f64
            }
            #[inline]
            fn from_f64(v: f64) -> Self {
                num_complex::Complex::new(v as $f, 0.0)
            }
            #[inline]
            fn from_f64_ndi(v: f64) -> Self {
                num_complex::Complex::new(v as $f, 0.0)
            }
            #[inline]
            fn imag_f64(self) -> f64 {
                self.im as f64
            }
            #[inline]
            fn from_complex(re: f64, im: f64) -> Self {
                num_complex::Complex::new(re as $f, im as $f)
            }
        }
    };
}

impl_complex!(f32, DType::Complex64);
impl_complex!(f64, DType::Complex128);

macro_rules! impl_uint {
    ($t:ty, $d:expr) => {
        impl Element for $t {
            const DTYPE: DType = $d;
            #[inline]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline]
            fn from_f64(v: f64) -> Self {
                v as $t
            }
            #[inline]
            fn from_f64_ndi(v: f64) -> Self {
                let t = if v > 0.0 { v + 0.5 } else { 0.0 };
                t as $t
            }
            #[inline]
            fn from_i64(v: i64) -> Self {
                v as $t
            }
            #[inline]
            fn to_i64(self) -> i64 {
                self as i64
            }
        }
    };
}

macro_rules! impl_int {
    ($t:ty, $d:expr) => {
        impl Element for $t {
            const DTYPE: DType = $d;
            #[inline]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline]
            fn from_f64(v: f64) -> Self {
                v as $t
            }
            #[inline]
            fn from_f64_ndi(v: f64) -> Self {
                let t = if v > 0.0 { v + 0.5 } else { v - 0.5 };
                t as $t
            }
            #[inline]
            fn from_i64(v: i64) -> Self {
                v as $t
            }
            #[inline]
            fn to_i64(self) -> i64 {
                self as i64
            }
        }
    };
}

impl_uint!(u8, DType::UInt8);
impl_uint!(u16, DType::UInt16);
impl_uint!(u32, DType::UInt32);
impl_uint!(u64, DType::UInt64);
impl_int!(i8, DType::Int8);
impl_int!(i16, DType::Int16);
impl_int!(i32, DType::Int32);
impl_int!(i64, DType::Int64);

impl Element for f32 {
    const DTYPE: DType = DType::Float32;
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn from_f64_ndi(v: f64) -> Self {
        v as f32
    }
}

impl Element for f64 {
    const DTYPE: DType = DType::Float64;
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn from_f64_ndi(v: f64) -> Self {
        v
    }
}

impl Element for bool {
    const DTYPE: DType = DType::Bool;
    #[inline]
    fn to_f64(self) -> f64 {
        if self { 1.0 } else { 0.0 }
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        v != 0.0
    }
    #[inline]
    fn from_f64_ndi(v: f64) -> Self {
        v != 0.0
    }
    #[inline]
    fn from_i64(v: i64) -> Self {
        v != 0
    }
    #[inline]
    fn to_i64(self) -> i64 {
        self as i64
    }
}
