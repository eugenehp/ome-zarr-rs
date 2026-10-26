//! A small lazy, chunked n-dimensional array: the `dask.array` analogue.
//!
//! A [`LazyArray`] knows its shape, dtype and chunking, and computes any
//! rectangular region on demand. Arrays are built from in-memory data, zarr
//! arrays, zeros, or from other lazy arrays by rechunking, slicing, casting,
//! concatenating and block-wise mapping (`map_blocks`).

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::ops::Range;
use std::sync::{Arc, Mutex};

use crate::par::*;

use super::chunks::{Chunks, auto_chunks, block_offsets, blockdims_from_blockshape, chunksize};
use super::dtype::DType;
use super::nd::NdArray;
use crate::error::{Error, Result};

/// A region of an array: one half-open range per dimension.
pub type Region = Vec<Range<u64>>;

/// Something that can produce regions of an n-dimensional array.
pub trait ChunkedSource: crate::par::MaybeSendSync {
    fn shape(&self) -> &[u64];
    fn dtype(&self) -> DType;
    fn chunks(&self) -> &Chunks;
    /// Compute the data in `region` (which lies inside the array).
    fn read(&self, region: &[Range<u64>]) -> Result<NdArray>;
    /// Short description for debugging.
    fn describe(&self) -> String;
}

/// The (real, imaginary) minimum and maximum of a complex array.
pub type ComplexMinMax = ((f64, f64), (f64, f64));

/// A lazily evaluated, chunked n-dimensional array (like `dask.array.Array`).
#[derive(Clone)]
pub struct LazyArray(Arc<dyn ChunkedSource>);

impl LazyArray {
    /// dask's `repr()`: `dask.array<from-zarr, shape=(2, 3), dtype=uint8,
    /// chunksize=(1, 3), chunktype=numpy.ndarray>`, named by the outermost operation.
    pub fn dask_repr(&self) -> String {
        let d = self.0.describe();
        let name = d.split('(').next().unwrap_or("array");
        let shape: Vec<usize> = self.shape().iter().map(|&n| n as usize).collect();
        let chunks: Vec<usize> = self.chunksize().iter().map(|&n| n as usize).collect();
        format!(
            "dask.array<{name}, shape={}, dtype={}, chunksize={}, chunktype=numpy.ndarray>",
            crate::pyjson::shape_repr(&shape),
            self.dtype(),
            crate::pyjson::shape_repr(&chunks)
        )
    }
}

impl fmt::Debug for LazyArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LazyArray<{}, shape={:?}, dtype={}, chunksize={:?}>",
            self.0.describe(),
            self.shape(),
            self.dtype(),
            self.chunksize()
        )
    }
}

fn full_region(shape: &[u64]) -> Region {
    shape.iter().map(|&s| 0..s).collect()
}

fn region_shape(region: &[Range<u64>]) -> Vec<usize> {
    region.iter().map(|r| (r.end - r.start) as usize).collect()
}

fn check_region(shape: &[u64], region: &[Range<u64>]) -> Result<()> {
    if region.len() != shape.len() {
        return Err(Error::value(format!(
            "region has {} dimensions but array has {}",
            region.len(),
            shape.len()
        )));
    }
    for (r, &s) in region.iter().zip(shape) {
        if r.start > r.end || r.end > s {
            return Err(Error::value(format!(
                "region {region:?} out of bounds for shape {shape:?}"
            )));
        }
    }
    Ok(())
}

/// Iterate over all multi-indices of a grid with `counts` entries per dimension.
pub(crate) fn grid_indices(counts: &[usize]) -> Vec<Vec<usize>> {
    let total: usize = counts.iter().product();
    let mut out = Vec::with_capacity(total);
    if counts.contains(&0) {
        return out;
    }
    let mut idx = vec![0usize; counts.len()];
    for _ in 0..total {
        out.push(idx.clone());
        for d in (0..counts.len()).rev() {
            idx[d] += 1;
            if idx[d] < counts[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    if counts.is_empty() {
        out.truncate(1);
    }
    out
}

impl LazyArray {
    /// Wrap a custom source.
    pub fn from_source(src: Arc<dyn ChunkedSource>) -> LazyArray {
        LazyArray(src)
    }

    /// `da.from_array(x)` with automatic chunking.
    pub fn from_ndarray(data: NdArray) -> LazyArray {
        let shape = data.shape_u64();
        let chunks = auto_chunks(&shape, data.dtype().itemsize());
        LazyArray(Arc::new(MemSource {
            shape,
            chunks,
            data: Arc::new(data.as_standard_layout()),
        }))
    }

    /// `da.from_array(x, chunks=blockshape)`.
    pub fn from_ndarray_chunked(data: NdArray, blockshape: &[u64]) -> LazyArray {
        let shape = data.shape_u64();
        let chunks = blockdims_from_blockshape(&shape, blockshape);
        LazyArray(Arc::new(MemSource {
            shape,
            chunks,
            data: Arc::new(data.as_standard_layout()),
        }))
    }

    /// `da.zeros(shape, dtype)` with the given block shape (or auto-chunked).
    pub fn zeros(shape: &[u64], dtype: DType, blockshape: Option<&[u64]>) -> LazyArray {
        let chunks = match blockshape {
            Some(b) => blockdims_from_blockshape(shape, b),
            None => auto_chunks(shape, dtype.itemsize()),
        };
        LazyArray(Arc::new(ZerosSource {
            shape: shape.to_vec(),
            chunks,
            dtype,
        }))
    }

    pub fn shape(&self) -> &[u64] {
        self.0.shape()
    }

    pub fn ndim(&self) -> usize {
        self.0.shape().len()
    }

    pub fn dtype(&self) -> DType {
        self.0.dtype()
    }

    pub fn chunks(&self) -> &Chunks {
        self.0.chunks()
    }

    /// Largest chunk along each dimension.
    pub fn chunksize(&self) -> Vec<u64> {
        chunksize(self.0.chunks())
    }

    /// Number of elements.
    pub fn size(&self) -> u64 {
        self.shape().iter().product()
    }

    /// Size in bytes.
    pub fn nbytes(&self) -> u64 {
        self.size() * self.dtype().itemsize() as u64
    }

    /// Compute a region.
    pub fn read(&self, region: &[Range<u64>]) -> Result<NdArray> {
        check_region(self.shape(), region)?;
        if region.iter().any(|r| r.start == r.end) {
            return Ok(NdArray::zeros(self.dtype(), &region_shape(region)));
        }
        self.0.read(region)
    }

    /// Compute the whole array.
    pub fn compute(&self) -> Result<NdArray> {
        self.read(&full_region(self.shape()))
    }

    /// Change chunking (metadata only; data is unchanged).
    pub fn rechunk(&self, blockshape: &[u64]) -> LazyArray {
        let bs: Vec<u64> = blockshape
            .iter()
            .zip(self.shape())
            .map(|(&c, &s)| if c == 0 { s.max(1) } else { c })
            .collect();
        self.rechunk_to(blockdims_from_blockshape(self.shape(), &bs))
    }

    /// Change chunking to explicit per-dimension chunks.
    pub fn rechunk_to(&self, chunks: Chunks) -> LazyArray {
        if &chunks == self.chunks() {
            return self.clone();
        }
        LazyArray(Arc::new(RechunkSource {
            inner: self.clone(),
            chunks,
        }))
    }

    /// Cast to another dtype (numpy `astype`).
    pub fn astype(&self, dtype: DType) -> LazyArray {
        if dtype == self.dtype() {
            return self.clone();
        }
        LazyArray(Arc::new(AstypeSource {
            inner: self.clone(),
            dtype,
        }))
    }

    /// Lazily slice a rectangular region (`x[a:b, c:d]`).
    pub fn slice(&self, region: &[Range<u64>]) -> Result<LazyArray> {
        check_region(self.shape(), region)?;
        if region
            .iter()
            .zip(self.shape())
            .all(|(r, &s)| r.start == 0 && r.end == s)
        {
            return Ok(self.clone());
        }
        let shape: Vec<u64> = region.iter().map(|r| r.end - r.start).collect();
        let chunks: Chunks = region
            .iter()
            .zip(self.chunks())
            .map(|(r, dim_chunks)| {
                let offs = block_offsets(dim_chunks);
                let mut v = vec![];
                for b in 0..dim_chunks.len() {
                    let s = offs[b].max(r.start);
                    let e = offs[b + 1].min(r.end);
                    if e > s {
                        v.push(e - s);
                    }
                }
                if v.is_empty() {
                    v.push(0);
                }
                v
            })
            .collect();
        Ok(LazyArray(Arc::new(SliceSource {
            inner: self.clone(),
            offset: region.iter().map(|r| r.start).collect(),
            shape,
            chunks,
        })))
    }

    /// `da.concatenate(arrays, axis)`.
    pub fn concatenate(arrays: &[LazyArray], axis: usize) -> Result<LazyArray> {
        if arrays.is_empty() {
            return Err(Error::value("Need array(s) to concatenate"));
        }
        let first = &arrays[0];
        let ndim = first.ndim();
        if axis >= ndim {
            return Err(Error::value(format!(
                "axis {axis} out of bounds for {ndim}D arrays"
            )));
        }
        for a in arrays {
            if a.ndim() != ndim {
                return Err(Error::value(
                    "all arrays must have the same number of dimensions",
                ));
            }
            for d in 0..ndim {
                if d != axis && a.shape()[d] != first.shape()[d] {
                    return Err(Error::value(format!(
                        "all the input array dimensions except for the concatenation axis must match exactly: {:?} vs {:?}",
                        a.shape(),
                        first.shape()
                    )));
                }
            }
        }
        if arrays.len() == 1 {
            return Ok(first.clone());
        }
        // dtype promotion: use the "largest" dtype (simple numpy-like result_type)
        let dtype = arrays
            .iter()
            .map(|a| a.dtype())
            .fold(first.dtype(), promote_types);
        let parts: Vec<LazyArray> = arrays.iter().map(|a| a.astype(dtype)).collect();
        let mut shape = first.shape().to_vec();
        shape[axis] = parts.iter().map(|a| a.shape()[axis]).sum();
        let mut chunks = first.chunks().clone();
        chunks[axis] = parts
            .iter()
            .flat_map(|a| a.chunks()[axis].clone())
            .collect();
        let mut offsets = vec![0u64];
        for p in &parts {
            offsets.push(offsets.last().unwrap() + p.shape()[axis]);
        }
        Ok(LazyArray(Arc::new(ConcatSource {
            parts,
            axis,
            offsets,
            shape,
            chunks,
            dtype,
        })))
    }

    /// `da.map_blocks(func, self, dtype=dtype, chunks=out_blockshape)`.
    ///
    /// Each input block (as given by the current chunking) is passed to `func`;
    /// the result array has a regular grid of `out_blockshape` blocks, one per
    /// input block (like dask with a tuple of ints for `chunks`). Edge blocks
    /// may be returned smaller than `out_blockshape` as long as every element
    /// subsequently read exists.
    pub fn map_blocks(
        &self,
        func: BlockFn,
        out_blockshape: &[u64],
        dtype: DType,
    ) -> Result<LazyArray> {
        if out_blockshape.len() != self.ndim() {
            return Err(Error::value(
                "out_blockshape must match array dimensionality",
            ));
        }
        let nblocks: Vec<usize> = self.chunks().iter().map(|c| c.len()).collect();
        let shape: Vec<u64> = nblocks
            .iter()
            .zip(out_blockshape)
            .map(|(&n, &b)| n as u64 * b)
            .collect();
        let chunks: Chunks = nblocks
            .iter()
            .zip(out_blockshape)
            .map(|(&n, &b)| vec![b; n])
            .collect();
        Ok(LazyArray(Arc::new(BlockwiseSource {
            input: self.clone(),
            in_offsets: self.chunks().iter().map(|c| block_offsets(c)).collect(),
            out_block: out_blockshape.to_vec(),
            shape,
            chunks,
            dtype,
            func,
            cache: Mutex::new(BlockCache::default()),
        })))
    }

    /// Iterate over the regions of all blocks.
    pub fn block_regions(&self) -> Vec<Region> {
        let offs: Vec<Vec<u64>> = self.chunks().iter().map(|c| block_offsets(c)).collect();
        let counts: Vec<usize> = self.chunks().iter().map(|c| c.len()).collect();
        grid_indices(&counts)
            .into_iter()
            .map(|idx| {
                idx.iter()
                    .enumerate()
                    .map(|(d, &b)| offs[d][b]..offs[d][b + 1])
                    .collect()
            })
            .collect()
    }

    /// Minimum and maximum (computed block by block, in parallel).
    pub fn min_max(&self) -> Result<Option<(f64, f64)>> {
        let parts: Vec<Option<(f64, f64)>> = self
            .block_regions()
            .into_par_iter()
            .map(|r| self.read(&r).map(|a| a.min_max()))
            .collect::<Result<_>>()?;
        let mut out: Option<(f64, f64)> = None;
        for (lo, hi) in parts.into_iter().flatten() {
            out = Some(match out {
                None => (lo, hi),
                Some((a, b)) => {
                    if lo.is_nan() || a.is_nan() {
                        (f64::NAN, f64::NAN)
                    } else {
                        (a.min(lo), b.max(hi))
                    }
                }
            });
        }
        Ok(out)
    }

    /// numpy's lexicographic `min()` / `max()` of a complex array.
    pub fn min_max_complex(&self) -> Result<Option<ComplexMinMax>> {
        let parts: Vec<Option<ComplexMinMax>> = self
            .block_regions()
            .into_par_iter()
            .map(|r| self.read(&r).map(|a| a.min_max_complex()))
            .collect::<Result<_>>()?;
        let lt = |a: (f64, f64), b: (f64, f64)| a.0 < b.0 || (a.0 == b.0 && a.1 < b.1);
        let mut out: Option<((f64, f64), (f64, f64))> = None;
        for (lo, hi) in parts.into_iter().flatten() {
            out = Some(match out {
                None => (lo, hi),
                Some((a, b)) => (
                    if lt(lo, a) { lo } else { a },
                    if lt(b, hi) { hi } else { b },
                ),
            });
        }
        Ok(out)
    }

    /// Sorted unique values (`da.unique(x).compute()`).
    pub fn unique(&self) -> Result<Vec<f64>> {
        let parts: Vec<Vec<f64>> = self
            .block_regions()
            .into_par_iter()
            .map(|r| self.read(&r).map(|a| a.unique_f64()))
            .collect::<Result<_>>()?;
        let mut v: Vec<f64> = parts.into_iter().flatten().collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v.dedup();
        Ok(v)
    }
}

/// Block function used by [`LazyArray::map_blocks`].
#[cfg(not(target_arch = "wasm32"))]
pub type BlockFn = Arc<dyn Fn(&NdArray) -> Result<NdArray> + Send + Sync>;
/// Block function used by [`LazyArray::map_blocks`].
#[cfg(target_arch = "wasm32")]
pub type BlockFn = Arc<dyn Fn(&NdArray) -> Result<NdArray>>;

/// numpy-like type promotion (sufficient for concatenating same-kind data).
pub fn promote_types(a: DType, b: DType) -> DType {
    if a == b {
        return a;
    }
    use DType::*;
    if a.is_float() || b.is_float() {
        if a == Float64
            || b == Float64
            || a.itemsize() >= 4 && !a.is_float()
            || b.itemsize() >= 4 && !b.is_float()
        {
            return Float64;
        }
        return Float32;
    }
    if a == Bool {
        return b;
    }
    if b == Bool {
        return a;
    }
    let size = a.itemsize().max(b.itemsize());
    let signed = a.is_signed() || b.is_signed();
    let mixed = a.is_signed() != b.is_signed();
    let size = if mixed {
        let unsigned_size = if a.is_signed() {
            b.itemsize()
        } else {
            a.itemsize()
        };
        let signed_size = if a.is_signed() {
            a.itemsize()
        } else {
            b.itemsize()
        };
        if unsigned_size >= signed_size {
            (unsigned_size * 2).min(8)
        } else {
            signed_size
        }
    } else {
        size
    };
    match (signed, size) {
        (false, 1) => UInt8,
        (false, 2) => UInt16,
        (false, 4) => UInt32,
        (false, _) => UInt64,
        (true, 1) => Int8,
        (true, 2) => Int16,
        (true, 4) => Int32,
        (true, _) => Int64,
    }
}

struct MemSource {
    shape: Vec<u64>,
    chunks: Chunks,
    data: Arc<NdArray>,
}

impl ChunkedSource for MemSource {
    fn shape(&self) -> &[u64] {
        &self.shape
    }
    fn dtype(&self) -> DType {
        self.data.dtype()
    }
    fn chunks(&self) -> &Chunks {
        &self.chunks
    }
    fn read(&self, region: &[Range<u64>]) -> Result<NdArray> {
        let r: Vec<Range<usize>> = region
            .iter()
            .map(|r| r.start as usize..r.end as usize)
            .collect();
        Ok(self.data.slice_region(&r))
    }
    fn describe(&self) -> String {
        "array".into()
    }
}

struct ZerosSource {
    shape: Vec<u64>,
    chunks: Chunks,
    dtype: DType,
}

impl ChunkedSource for ZerosSource {
    fn shape(&self) -> &[u64] {
        &self.shape
    }
    fn dtype(&self) -> DType {
        self.dtype
    }
    fn chunks(&self) -> &Chunks {
        &self.chunks
    }
    fn read(&self, region: &[Range<u64>]) -> Result<NdArray> {
        Ok(NdArray::zeros(self.dtype, &region_shape(region)))
    }
    fn describe(&self) -> String {
        "zeros".into()
    }
}

struct RechunkSource {
    inner: LazyArray,
    chunks: Chunks,
}

impl ChunkedSource for RechunkSource {
    fn shape(&self) -> &[u64] {
        self.inner.shape()
    }
    fn dtype(&self) -> DType {
        self.inner.dtype()
    }
    fn chunks(&self) -> &Chunks {
        &self.chunks
    }
    fn read(&self, region: &[Range<u64>]) -> Result<NdArray> {
        self.inner.read(region)
    }
    fn describe(&self) -> String {
        format!("rechunk({})", self.inner.0.describe())
    }
}

struct AstypeSource {
    inner: LazyArray,
    dtype: DType,
}

impl ChunkedSource for AstypeSource {
    fn shape(&self) -> &[u64] {
        self.inner.shape()
    }
    fn dtype(&self) -> DType {
        self.dtype
    }
    fn chunks(&self) -> &Chunks {
        self.inner.chunks()
    }
    fn read(&self, region: &[Range<u64>]) -> Result<NdArray> {
        Ok(self.inner.read(region)?.astype(self.dtype))
    }
    fn describe(&self) -> String {
        format!("astype({})", self.inner.0.describe())
    }
}

struct SliceSource {
    inner: LazyArray,
    offset: Vec<u64>,
    shape: Vec<u64>,
    chunks: Chunks,
}

impl ChunkedSource for SliceSource {
    fn shape(&self) -> &[u64] {
        &self.shape
    }
    fn dtype(&self) -> DType {
        self.inner.dtype()
    }
    fn chunks(&self) -> &Chunks {
        &self.chunks
    }
    fn read(&self, region: &[Range<u64>]) -> Result<NdArray> {
        let r: Region = region
            .iter()
            .zip(&self.offset)
            .map(|(r, &o)| r.start + o..r.end + o)
            .collect();
        self.inner.read(&r)
    }
    fn describe(&self) -> String {
        format!("getitem({})", self.inner.0.describe())
    }
}

struct ConcatSource {
    parts: Vec<LazyArray>,
    axis: usize,
    offsets: Vec<u64>,
    shape: Vec<u64>,
    chunks: Chunks,
    dtype: DType,
}

impl ChunkedSource for ConcatSource {
    fn shape(&self) -> &[u64] {
        &self.shape
    }
    fn dtype(&self) -> DType {
        self.dtype
    }
    fn chunks(&self) -> &Chunks {
        &self.chunks
    }
    fn read(&self, region: &[Range<u64>]) -> Result<NdArray> {
        let shape = region_shape(region);
        let mut out = NdArray::zeros(self.dtype, &shape);
        let ax = self.axis;
        let r = &region[ax];
        for (i, part) in self.parts.iter().enumerate() {
            let (ps, pe) = (self.offsets[i], self.offsets[i + 1]);
            let s = ps.max(r.start);
            let e = pe.min(r.end);
            if e <= s {
                continue;
            }
            let mut sub = region.to_vec();
            sub[ax] = s - ps..e - ps;
            let data = part.read(&sub)?;
            let mut dst: Vec<Range<usize>> = shape.iter().map(|&n| 0..n).collect();
            dst[ax] = (s - r.start) as usize..(e - r.start) as usize;
            out.assign_region(&dst, &data)?;
        }
        Ok(out)
    }
    fn describe(&self) -> String {
        "concatenate".into()
    }
}

/// Bounded cache of computed blocks (keyed by block index).
#[derive(Default)]
struct BlockCache {
    map: HashMap<Vec<usize>, Arc<NdArray>>,
    order: VecDeque<Vec<usize>>,
    bytes: usize,
}

const BLOCK_CACHE_BYTES: usize = 512 * 1024 * 1024;

impl BlockCache {
    fn get(&self, k: &[usize]) -> Option<Arc<NdArray>> {
        self.map.get(k).cloned()
    }
    fn insert(&mut self, k: Vec<usize>, v: Arc<NdArray>) {
        let size = v.len() * v.dtype().itemsize();
        if self.map.contains_key(&k) {
            return;
        }
        while self.bytes + size > BLOCK_CACHE_BYTES && !self.order.is_empty() {
            if let Some(old) = self.order.pop_front()
                && let Some(a) = self.map.remove(&old)
            {
                self.bytes -= a.len() * a.dtype().itemsize();
            }
        }
        self.bytes += size;
        self.order.push_back(k.clone());
        self.map.insert(k, v);
    }
}

struct BlockwiseSource {
    input: LazyArray,
    in_offsets: Vec<Vec<u64>>,
    out_block: Vec<u64>,
    shape: Vec<u64>,
    chunks: Chunks,
    dtype: DType,
    func: BlockFn,
    cache: Mutex<BlockCache>,
}

impl BlockwiseSource {
    fn block(&self, idx: &[usize]) -> Result<Arc<NdArray>> {
        if let Some(b) = self.cache.lock().unwrap().get(idx) {
            return Ok(b);
        }
        let region: Region = idx
            .iter()
            .enumerate()
            .map(|(d, &b)| self.in_offsets[d][b]..self.in_offsets[d][b + 1])
            .collect();
        let input = self.input.read(&region)?;
        let out = (self.func)(&input)?;
        let out = if out.dtype() != self.dtype {
            out.astype(self.dtype)
        } else {
            out
        };
        let out = Arc::new(out);
        self.cache.lock().unwrap().insert(idx.to_vec(), out.clone());
        Ok(out)
    }
}

impl ChunkedSource for BlockwiseSource {
    fn shape(&self) -> &[u64] {
        &self.shape
    }
    fn dtype(&self) -> DType {
        self.dtype
    }
    fn chunks(&self) -> &Chunks {
        &self.chunks
    }
    fn read(&self, region: &[Range<u64>]) -> Result<NdArray> {
        let shape = region_shape(region);
        let mut out = NdArray::zeros(self.dtype, &shape);
        let ranges: Vec<Range<usize>> = region
            .iter()
            .zip(&self.out_block)
            .map(|(r, &b)| (r.start / b) as usize..((r.end - 1) / b) as usize + 1)
            .collect();
        let counts: Vec<usize> = ranges.iter().map(|r| r.end - r.start).collect();
        for rel in grid_indices(&counts) {
            let idx: Vec<usize> = rel.iter().zip(&ranges).map(|(&i, r)| i + r.start).collect();
            let block = self.block(&idx)?;
            let bshape = block.shape();
            let mut src = Vec::with_capacity(idx.len());
            let mut dst = Vec::with_capacity(idx.len());
            for d in 0..idx.len() {
                let bstart = idx[d] as u64 * self.out_block[d];
                let s = bstart.max(region[d].start);
                let e = (bstart + self.out_block[d]).min(region[d].end);
                let ls = (s - bstart) as usize;
                let le = (e - bstart) as usize;
                if le > bshape[d] {
                    return Err(Error::value(format!(
                        "block {idx:?} has shape {bshape:?}, smaller than required by declared chunks {:?}",
                        self.out_block
                    )));
                }
                src.push(ls..le);
                dst.push((s - region[d].start) as usize..(e - region[d].start) as usize);
            }
            out.assign_region(&dst, &block.slice_region(&src))?;
        }
        Ok(out)
    }
    fn describe(&self) -> String {
        format!("map_blocks({})", self.input.0.describe())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    fn arange(shape: &[usize]) -> NdArray {
        let n: usize = shape.iter().product();
        NdArray::Int32(ArrayD::from_shape_vec(IxDyn(shape), (0..n as i32).collect()).unwrap())
    }

    #[test]
    fn slice_concat() {
        let a = LazyArray::from_ndarray_chunked(arange(&[4, 6]), &[2, 4]);
        let b = LazyArray::zeros(&[4, 2], DType::Int32, Some(&[2, 2]));
        let c = LazyArray::concatenate(&[a.clone(), b], 1).unwrap();
        assert_eq!(c.shape(), &[4, 8]);
        assert_eq!(c.chunks()[1], vec![4, 2, 2]);
        let v = c.read(&[1..2, 4..8]).unwrap().to_f64_vec();
        assert_eq!(v, vec![10.0, 11.0, 0.0, 0.0]);
        let s = a.slice(&[1..3, 3..6]).unwrap();
        assert_eq!(s.chunks(), &vec![vec![1, 1], vec![1, 2]]);
        assert_eq!(
            s.compute().unwrap().to_f64_vec(),
            vec![9.0, 10.0, 11.0, 15.0, 16.0, 17.0]
        );
    }

    #[test]
    fn map_blocks_identity_crop() {
        let a = LazyArray::from_ndarray_chunked(arange(&[5, 5]), &[2, 2]);
        let f: BlockFn = Arc::new(|b: &NdArray| Ok(b.clone()));
        let m = a.map_blocks(f, &[2, 2], DType::Int32).unwrap();
        assert_eq!(m.shape(), &[6, 6]);
        let m = m.slice(&[0..5, 0..5]).unwrap();
        assert!(m.compute().unwrap().values_equal(&arange(&[5, 5])));
    }

    #[test]
    fn grid() {
        assert_eq!(grid_indices(&[2, 2]).len(), 4);
        assert_eq!(grid_indices(&[]).len(), 1);
    }
}
