//! A dynamically-typed n-dimensional in-memory array (the `numpy.ndarray` analogue).

use std::ops::Range;

use ndarray::{ArrayD, IxDyn, SliceInfoElem};

use super::dtype::{DType, Element};
use crate::error::{Error, Result};

/// A dynamically typed, dynamically dimensioned in-memory array.
#[derive(Debug, Clone, PartialEq)]
pub enum NdArray {
    Bool(ArrayD<bool>),
    UInt8(ArrayD<u8>),
    UInt16(ArrayD<u16>),
    UInt32(ArrayD<u32>),
    UInt64(ArrayD<u64>),
    Int8(ArrayD<i8>),
    Int16(ArrayD<i16>),
    Int32(ArrayD<i32>),
    Int64(ArrayD<i64>),
    Float16(ArrayD<half::f16>),
    Float32(ArrayD<f32>),
    Float64(ArrayD<f64>),
    Complex64(ArrayD<num_complex::Complex32>),
    Complex128(ArrayD<num_complex::Complex64>),
}

/// Dispatch a generic expression over every variant of an [`NdArray`].
#[macro_export]
macro_rules! nd_dispatch {
    ($value:expr, $a:ident => $body:expr) => {
        match $value {
            $crate::array::NdArray::Bool($a) => $body,
            $crate::array::NdArray::UInt8($a) => $body,
            $crate::array::NdArray::UInt16($a) => $body,
            $crate::array::NdArray::UInt32($a) => $body,
            $crate::array::NdArray::UInt64($a) => $body,
            $crate::array::NdArray::Int8($a) => $body,
            $crate::array::NdArray::Int16($a) => $body,
            $crate::array::NdArray::Int32($a) => $body,
            $crate::array::NdArray::Int64($a) => $body,
            $crate::array::NdArray::Float16($a) => $body,
            $crate::array::NdArray::Float32($a) => $body,
            $crate::array::NdArray::Float64($a) => $body,
            $crate::array::NdArray::Complex64($a) => $body,
            $crate::array::NdArray::Complex128($a) => $body,
        }
    };
}

/// Dispatch a generic expression over every variant, re-wrapping the result in the
/// same variant.
#[macro_export]
macro_rules! nd_map {
    ($value:expr, $a:ident => $body:expr) => {
        match $value {
            $crate::array::NdArray::Bool($a) => $crate::array::NdArray::Bool($body),
            $crate::array::NdArray::UInt8($a) => $crate::array::NdArray::UInt8($body),
            $crate::array::NdArray::UInt16($a) => $crate::array::NdArray::UInt16($body),
            $crate::array::NdArray::UInt32($a) => $crate::array::NdArray::UInt32($body),
            $crate::array::NdArray::UInt64($a) => $crate::array::NdArray::UInt64($body),
            $crate::array::NdArray::Int8($a) => $crate::array::NdArray::Int8($body),
            $crate::array::NdArray::Int16($a) => $crate::array::NdArray::Int16($body),
            $crate::array::NdArray::Int32($a) => $crate::array::NdArray::Int32($body),
            $crate::array::NdArray::Int64($a) => $crate::array::NdArray::Int64($body),
            $crate::array::NdArray::Float16($a) => $crate::array::NdArray::Float16($body),
            $crate::array::NdArray::Float32($a) => $crate::array::NdArray::Float32($body),
            $crate::array::NdArray::Float64($a) => $crate::array::NdArray::Float64($body),
            $crate::array::NdArray::Complex64($a) => $crate::array::NdArray::Complex64($body),
            $crate::array::NdArray::Complex128($a) => $crate::array::NdArray::Complex128($body),
        }
    };
}

/// Run a generic block with a type alias `$t` bound to the element type for `dtype`.
#[macro_export]
macro_rules! dtype_dispatch {
    ($dtype:expr, $t:ident => $body:expr) => {
        match $dtype {
            $crate::array::DType::Bool => {
                type $t = bool;
                $body
            }
            $crate::array::DType::UInt8 => {
                type $t = u8;
                $body
            }
            $crate::array::DType::UInt16 => {
                type $t = u16;
                $body
            }
            $crate::array::DType::UInt32 => {
                type $t = u32;
                $body
            }
            $crate::array::DType::UInt64 => {
                type $t = u64;
                $body
            }
            $crate::array::DType::Int8 => {
                type $t = i8;
                $body
            }
            $crate::array::DType::Int16 => {
                type $t = i16;
                $body
            }
            $crate::array::DType::Int32 => {
                type $t = i32;
                $body
            }
            $crate::array::DType::Int64 => {
                type $t = i64;
                $body
            }
            $crate::array::DType::Float16 => {
                type $t = half::f16;
                $body
            }
            $crate::array::DType::Float32 => {
                type $t = f32;
                $body
            }
            $crate::array::DType::Complex64 => {
                type $t = num_complex::Complex32;
                $body
            }
            $crate::array::DType::Complex128 => {
                type $t = num_complex::Complex64;
                $body
            }
            $crate::array::DType::Float64 => {
                type $t = f64;
                $body
            }
        }
    };
}

/// Conversion between typed `ArrayD<T>` and [`NdArray`].
pub trait IntoNdArray {
    fn into_nd(self) -> NdArray;
}

/// Extract a typed array out of an [`NdArray`].
pub trait FromNdArray: Sized {
    fn from_nd(a: NdArray) -> Option<Self>;
    fn from_nd_ref(a: &NdArray) -> Option<&Self>;
}

macro_rules! impl_conv {
    ($t:ty, $v:ident) => {
        impl IntoNdArray for ArrayD<$t> {
            fn into_nd(self) -> NdArray {
                NdArray::$v(self)
            }
        }
        impl From<ArrayD<$t>> for NdArray {
            fn from(a: ArrayD<$t>) -> Self {
                NdArray::$v(a)
            }
        }
        impl FromNdArray for ArrayD<$t> {
            fn from_nd(a: NdArray) -> Option<Self> {
                match a {
                    NdArray::$v(x) => Some(x),
                    _ => None,
                }
            }
            fn from_nd_ref(a: &NdArray) -> Option<&Self> {
                match a {
                    NdArray::$v(x) => Some(x),
                    _ => None,
                }
            }
        }
    };
}

impl_conv!(bool, Bool);
impl_conv!(u8, UInt8);
impl_conv!(u16, UInt16);
impl_conv!(u32, UInt32);
impl_conv!(u64, UInt64);
impl_conv!(i8, Int8);
impl_conv!(i16, Int16);
impl_conv!(i32, Int32);
impl_conv!(i64, Int64);
impl_conv!(f32, Float32);
impl_conv!(f64, Float64);
impl_conv!(half::f16, Float16);
impl_conv!(num_complex::Complex32, Complex64);
impl_conv!(num_complex::Complex64, Complex128);

/// Convert a typed array of `S` into one of `T` via f64 (numpy astype semantics).
pub fn cast_array<S: Element, T: Element>(a: &ArrayD<S>) -> ArrayD<T> {
    a.mapv(|v| T::from_f64(v.to_f64()))
}

impl NdArray {
    /// Allocate a zero-filled array.
    pub fn zeros(dtype: DType, shape: &[usize]) -> NdArray {
        dtype_dispatch!(dtype, T => ArrayD::<T>::default(IxDyn(shape)).into_nd())
    }

    /// Array filled with a value (cast to dtype).
    pub fn full(dtype: DType, shape: &[usize], value: f64) -> NdArray {
        dtype_dispatch!(dtype, T => ArrayD::<T>::from_elem(IxDyn(shape), T::from_f64(value)).into_nd())
    }

    /// Build an array from a flat vector of f64 values cast to `dtype`.
    pub fn from_f64_vec(dtype: DType, shape: &[usize], data: Vec<f64>) -> Result<NdArray> {
        let a =
            ArrayD::from_shape_vec(IxDyn(shape), data).map_err(|e| Error::value(e.to_string()))?;
        Ok(NdArray::Float64(a).astype(dtype))
    }

    pub fn dtype(&self) -> DType {
        match self {
            NdArray::Bool(_) => DType::Bool,
            NdArray::UInt8(_) => DType::UInt8,
            NdArray::UInt16(_) => DType::UInt16,
            NdArray::UInt32(_) => DType::UInt32,
            NdArray::UInt64(_) => DType::UInt64,
            NdArray::Int8(_) => DType::Int8,
            NdArray::Int16(_) => DType::Int16,
            NdArray::Int32(_) => DType::Int32,
            NdArray::Int64(_) => DType::Int64,
            NdArray::Float16(_) => DType::Float16,
            NdArray::Float32(_) => DType::Float32,
            NdArray::Float64(_) => DType::Float64,
            NdArray::Complex64(_) => DType::Complex64,
            NdArray::Complex128(_) => DType::Complex128,
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        crate::nd_dispatch!(self, a => a.shape().to_vec())
    }

    pub fn shape_u64(&self) -> Vec<u64> {
        self.shape().iter().map(|&s| s as u64).collect()
    }

    pub fn ndim(&self) -> usize {
        crate::nd_dispatch!(self, a => a.ndim())
    }

    pub fn len(&self) -> usize {
        crate::nd_dispatch!(self, a => a.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// numpy `astype`.
    pub fn astype(&self, dtype: DType) -> NdArray {
        if self.dtype() == dtype {
            return self.clone();
        }
        crate::nd_dispatch!(self, a => {
            dtype_dispatch!(dtype, T => cast_array::<_, T>(a).into_nd())
        })
    }

    /// Convert to an f64 array (numpy `astype(float)`).
    pub fn to_f64(&self) -> ArrayD<f64> {
        match self {
            NdArray::Float64(a) => a.clone(),
            other => crate::nd_dispatch!(other, a => a.mapv(|v| v.to_f64())),
        }
    }

    /// Extract a typed view, failing if the dtype does not match.
    pub fn as_typed<T: Element>(&self) -> Result<&ArrayD<T>>
    where
        ArrayD<T>: FromNdArray,
    {
        <ArrayD<T> as FromNdArray>::from_nd_ref(self).ok_or_else(|| {
            Error::type_err(format!("expected {} array, got {}", T::DTYPE, self.dtype()))
        })
    }

    /// Consume into a typed array, failing if the dtype does not match.
    pub fn into_typed<T: Element>(self) -> Result<ArrayD<T>>
    where
        ArrayD<T>: FromNdArray,
    {
        let d = self.dtype();
        <ArrayD<T> as FromNdArray>::from_nd(self)
            .ok_or_else(|| Error::type_err(format!("expected {} array, got {}", T::DTYPE, d)))
    }

    /// Copy out a rectangular region.
    pub fn slice_region(&self, region: &[Range<usize>]) -> NdArray {
        let info: Vec<SliceInfoElem> = region
            .iter()
            .map(|r| SliceInfoElem::Slice {
                start: r.start as isize,
                end: Some(r.end as isize),
                step: 1,
            })
            .collect();
        crate::nd_map!(self, a => a.slice(info.as_slice()).to_owned())
    }

    /// Write `src` into a rectangular region of `self` (dtype cast if necessary).
    pub fn assign_region(&mut self, region: &[Range<usize>], src: &NdArray) -> Result<()> {
        let info: Vec<SliceInfoElem> = region
            .iter()
            .map(|r| SliceInfoElem::Slice {
                start: r.start as isize,
                end: Some(r.end as isize),
                step: 1,
            })
            .collect();
        let dtype = self.dtype();
        let src = if src.dtype() == dtype {
            src.clone()
        } else {
            src.astype(dtype)
        };
        macro_rules! assign {
            ($($v:ident),*) => {
                match (self, src) {
                    $( (NdArray::$v(dst), NdArray::$v(s)) => {
                        let mut view = dst.slice_mut(info.as_slice());
                        if view.shape() != s.shape() {
                            return Err(Error::value(format!(
                                "could not broadcast input array from shape {:?} into shape {:?}",
                                s.shape(), view.shape()
                            )));
                        }
                        view.assign(&s);
                        Ok(())
                    } )*
                    _ => unreachable!(),
                }
            };
        }
        assign!(
            Bool, UInt8, UInt16, UInt32, UInt64, Int8, Int16, Int32, Int64, Float16, Float32,
            Float64, Complex64, Complex128
        )
    }

    /// Element at a (multi-)index, as f64.
    pub fn get_f64(&self, index: &[usize]) -> Option<f64> {
        crate::nd_dispatch!(self, a => a.get(IxDyn(index)).map(|v| v.to_f64()))
    }

    /// Minimum and maximum values (as f64). NaNs propagate like numpy.
    /// numpy's `min()` / `max()` of a complex array (lexicographic on
    /// (real, imaginary)) as ((re, im), (re, im)); real arrays have im = 0.
    pub fn min_max_complex(&self) -> Option<((f64, f64), (f64, f64))> {
        if self.is_empty() {
            return None;
        }
        crate::nd_dispatch!(self, a => {
            let mut it = a.iter();
            let first = *it.next().unwrap();
            let (mut lo, mut hi) = (first, first);
            for &v in it {
                if v.np_lt(lo) { lo = v; }
                if hi.np_lt(v) { hi = v; }
            }
            Some(((lo.to_f64(), lo.imag_f64()), (hi.to_f64(), hi.imag_f64())))
        })
    }

    pub fn min_max(&self) -> Option<(f64, f64)> {
        if self.is_empty() {
            return None;
        }
        crate::nd_dispatch!(self, a => {
            let mut it = a.iter();
            let first = it.next().unwrap().to_f64();
            let (mut lo, mut hi) = (first, first);
            for v in it {
                let v = v.to_f64();
                if v.is_nan() || lo.is_nan() {
                    lo = f64::NAN;
                    hi = f64::NAN;
                    continue;
                }
                if v < lo { lo = v; }
                if v > hi { hi = v; }
            }
            Some((lo, hi))
        })
    }

    /// Sorted unique values (numpy `unique`), as f64.
    pub fn unique_f64(&self) -> Vec<f64> {
        let mut v: Vec<f64> =
            crate::nd_dispatch!(self, a => a.iter().map(|x| x.to_f64()).collect());
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v.dedup();
        v
    }

    /// Flat iterator over values as f64 in C order.
    pub fn to_f64_vec(&self) -> Vec<f64> {
        crate::nd_dispatch!(self, a => a.iter().map(|x| x.to_f64()).collect())
    }

    /// Ensure standard (C-contiguous) layout.
    pub fn as_standard_layout(&self) -> NdArray {
        crate::nd_map!(self, a => a.as_standard_layout().into_owned())
    }

    /// Reshape (C order).
    pub fn reshape(&self, shape: &[usize]) -> Result<NdArray> {
        let total: usize = shape.iter().product();
        if total != self.len() {
            return Err(Error::value(format!(
                "cannot reshape array of size {} into shape {:?}",
                self.len(),
                shape
            )));
        }
        Ok(crate::nd_map!(self, a => {
            let v: Vec<_> = a.iter().cloned().collect();
            ArrayD::from_shape_vec(IxDyn(shape), v).unwrap()
        }))
    }

    /// Whether every element equals the other array's element (dtype-agnostic).
    pub fn values_equal(&self, other: &NdArray) -> bool {
        self.shape() == other.shape() && self.to_f64_vec() == other.to_f64_vec()
    }
}
