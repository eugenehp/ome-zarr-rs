//! Chunk-size helpers mirroring `dask.array` chunk normalisation.

/// Per-dimension chunk sizes (dask `chunks`), e.g. `[[64, 64, 32], [100]]`.
pub type Chunks = Vec<Vec<u64>>;

/// Default dask `array.chunk-size` (128 MiB).
pub const DEFAULT_CHUNK_BYTES: f64 = 134_217_728.0;

/// Expand a regular block shape into dask-style per-dimension chunk tuples
/// (`blockdims_from_blockshape`).
pub fn blockdims_from_blockshape(shape: &[u64], blockshape: &[u64]) -> Chunks {
    shape
        .iter()
        .zip(blockshape)
        .map(|(&s, &c)| {
            if s == 0 {
                return vec![0];
            }
            let c = c.max(1).min(s.max(1));
            let mut v = vec![c; (s / c) as usize];
            if s % c != 0 {
                v.push(s % c);
            }
            v
        })
        .collect()
}

/// Largest chunk along each dimension (dask `chunksize`).
pub fn chunksize(chunks: &Chunks) -> Vec<u64> {
    chunks
        .iter()
        .map(|c| c.iter().copied().max().unwrap_or(0))
        .collect()
}

/// Number of blocks along each dimension.
pub fn numblocks(chunks: &Chunks) -> Vec<usize> {
    chunks.iter().map(|c| c.len()).collect()
}

fn round_to(c: f64, s: u64) -> u64 {
    if c <= s as f64 {
        (c as u64).max(1)
    } else {
        ((c / s as f64).floor() as u64) * s
    }
}

/// dask `auto_chunks` (without previous chunks) for every dimension.
pub fn auto_blockshape(shape: &[u64], itemsize: usize, limit_bytes: f64) -> Vec<u64> {
    let mut chunks: Vec<Option<u64>> = vec![None; shape.len()];
    loop {
        let autos: Vec<usize> = (0..shape.len()).filter(|&i| chunks[i].is_none()).collect();
        if autos.is_empty() {
            break;
        }
        let largest_block: f64 = chunks.iter().flatten().map(|&c| c as f64).product();
        let limit = limit_bytes.max(1.0);
        let size = (limit / itemsize as f64 / largest_block).powf(1.0 / autos.len() as f64);
        let small: Vec<usize> = autos
            .iter()
            .copied()
            .filter(|&i| (shape[i] as f64) < size)
            .collect();
        if !small.is_empty() {
            for i in small {
                chunks[i] = Some(shape[i]);
            }
            continue;
        }
        for i in autos {
            chunks[i] = Some(round_to(size, shape[i]));
        }
        break;
    }
    chunks.into_iter().map(|c| c.unwrap_or(1)).collect()
}

/// dask `normalize_chunks("auto", shape, dtype)`.
pub fn auto_chunks(shape: &[u64], itemsize: usize) -> Chunks {
    let bs = auto_blockshape(shape, itemsize, DEFAULT_CHUNK_BYTES);
    blockdims_from_blockshape(shape, &bs)
}

/// Offsets of each block start along a dimension.
pub fn block_offsets(dim_chunks: &[u64]) -> Vec<u64> {
    let mut out = Vec::with_capacity(dim_chunks.len() + 1);
    let mut acc = 0;
    out.push(0);
    for &c in dim_chunks {
        acc += c;
        out.push(acc);
    }
    out
}

/// Expand `chunks` to match `shape` (writer `_retuple`).
///
/// E.g. if chunks is (64, 64) and shape is (3, 4, 5, 1028, 1028)
/// return (3, 4, 5, 64, 64).
pub fn retuple(chunks: &[u64], shape: &[u64]) -> Vec<u64> {
    if chunks.len() >= shape.len() {
        return chunks[chunks.len() - shape.len()..].to_vec();
    }
    let dims_to_add = shape.len() - chunks.len();
    let mut out = shape[..dims_to_add].to_vec();
    out.extend_from_slice(chunks);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_small() {
        assert_eq!(
            auto_blockshape(&[256, 256], 1, DEFAULT_CHUNK_BYTES),
            vec![256, 256]
        );
    }

    #[test]
    fn test_auto_big() {
        // 20000x20000 uint8 = 400MB -> sqrt(128MiB) = 11585.2 per dim
        let b = auto_blockshape(&[20000, 20000], 1, DEFAULT_CHUNK_BYTES);
        assert_eq!(b, vec![11585, 11585]);
    }

    #[test]
    fn test_blockdims() {
        assert_eq!(
            blockdims_from_blockshape(&[10, 4], &[3, 4]),
            vec![vec![3, 3, 3, 1], vec![4]]
        );
    }

    #[test]
    fn test_retuple() {
        assert_eq!(
            retuple(&[64, 64], &[3, 4, 5, 1028, 1028]),
            vec![3, 4, 5, 64, 64]
        );
        assert_eq!(retuple(&[1, 2, 3], &[5, 5, 5]), vec![1, 2, 3]);
    }
}
