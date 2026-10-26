//! The small dense linear algebra numpy does for coordinate transforms:
//! `matrix @ matrix`, `np.dot` and `np.linalg.inv`, reproducing numpy's
//! results.
//!
//! numpy computes these with its BLAS / LAPACK. On macOS that is Accelerate:
//! `np.linalg.inv` is Accelerate's `dgesv` (called here directly, so results
//! are identical), and tiny matrix products are a forward fused multiply-add
//! chain. Elsewhere a partial-pivoting LU is used.

/// The BLAS behind numpy on this platform, for the summation order of small
/// products (measured against numpy: Accelerate on macOS arm64, OpenBLAS's
/// kernels on Linux aarch64, OpenBLAS's generic kernels on x86-64).
#[derive(Clone, Copy, PartialEq)]
enum Blas {
    Accelerate,
    OpenBlasArm,
    OpenBlasX86,
}

const BLAS: Blas = if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
    Blas::Accelerate
} else if cfg!(target_arch = "aarch64") {
    Blas::OpenBlasArm
} else {
    Blas::OpenBlasX86
};

/// `s = a0*b0; s = fma(ak, bk, s)`.
fn fused_chain(a: &[f64], b: &[f64]) -> f64 {
    let mut s = a[0] * b[0];
    for k in 1..a.len() {
        s = a[k].mul_add(b[k], s);
    }
    s
}

/// `s = a0*b0; s = s + ak*bk`.
fn plain_chain(a: &[f64], b: &[f64]) -> f64 {
    let mut s = a[0] * b[0];
    for k in 1..a.len() {
        s += a[k] * b[k];
    }
    s
}

/// Two fused accumulators over the even / odd terms, then added.
fn fused_two_acc(a: &[f64], b: &[f64]) -> f64 {
    let (mut e, mut o) = (a[0] * b[0], a[1] * b[1]);
    for k in 2..a.len() {
        if k % 2 == 0 {
            e = a[k].mul_add(b[k], e);
        } else {
            o = a[k].mul_add(b[k], o);
        }
    }
    e + o
}

/// Element (row `i` of `nrows`) of a matrix product (`dgemm`).
fn gemm_dot(a: &[f64], b: &[f64], i: usize, nrows: usize) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    match BLAS {
        Blas::OpenBlasArm => fused_chain(a, b),
        Blas::OpenBlasX86 => plain_chain(a, b),
        // 4 terms: rows in pairs use two accumulators, a last odd row the chain
        Blas::Accelerate if a.len() == 4 && i < nrows - nrows % 2 => fused_two_acc(a, b),
        Blas::Accelerate => fused_chain(a, b),
    }
}

/// Element of a matrix-vector product (`dgemv`).
fn gemv_dot(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    match BLAS {
        Blas::OpenBlasArm => fused_chain(a, b),
        Blas::OpenBlasX86 | Blas::Accelerate => plain_chain(a, b),
    }
}

/// `np.dot(a, b)` of two vectors, as a one-row matrix product.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    gemm_dot(a, b, 0, 1)
}

/// `a @ b` for row-major matrices (`dgemm`).
pub fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let k = b.len();
    let m = b.first().map(|r| r.len()).unwrap_or(0);
    let cols: Vec<Vec<f64>> = (0..m).map(|j| (0..k).map(|l| b[l][j]).collect()).collect();
    let n = a.len();
    a.iter()
        .enumerate()
        .map(|(i, row)| cols.iter().map(|col| gemm_dot(row, col, i, n)).collect())
        .collect()
}

/// `a @ b.T` with `b` given by its rows (transformnd's `coords @ lm.T`).
/// With a transposed operand Accelerate computes one row (and the trailing
/// row of an odd count) as a matrix-vector product.
pub fn matmul_bt(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    a.iter()
        .enumerate()
        .map(|(i, row)| {
            b.iter()
                .map(|col| {
                    if row.is_empty() {
                        0.0
                    } else {
                        match BLAS {
                            Blas::OpenBlasArm => fused_chain(row, col),
                            Blas::OpenBlasX86 => plain_chain(row, col),
                            Blas::Accelerate if i >= n - n % 2 => plain_chain(row, col),
                            Blas::Accelerate => fused_chain(row, col),
                        }
                    }
                })
                .collect()
        })
        .collect()
}

/// `np.dot(m, v)` (`dgemv`).
pub fn matvec(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter().map(|row| gemv_dot(row, v)).collect()
}

/// `np.linalg.inv`; `None` if the matrix is singular (numpy's `LinAlgError`).
pub fn inv(m: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = m.len();
    if n == 0 || m.iter().any(|r| r.len() != n) {
        return None;
    }
    #[cfg(target_os = "macos")]
    {
        accelerate_inv(m)
    }
    #[cfg(not(target_os = "macos"))]
    {
        openblas_inv(m)
    }
}

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    #[link_name = "dgesv$NEWLAPACK"]
    fn dgesv_newlapack(
        n: *const i32,
        nrhs: *const i32,
        a: *mut f64,
        lda: *const i32,
        ipiv: *mut i32,
        b: *mut f64,
        ldb: *const i32,
        info: *mut i32,
    );
}

/// numpy's `inv`: `dgesv(A, I)` on a column-major copy of `A`.
#[cfg(target_os = "macos")]
fn accelerate_inv(m: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = m.len();
    let mut a: Vec<f64> = (0..n * n).map(|k| m[k % n][k / n]).collect();
    let mut b: Vec<f64> = (0..n * n)
        .map(|k| if k % n == k / n { 1.0 } else { 0.0 })
        .collect();
    let nn = n as i32;
    let mut ipiv = vec![0i32; n];
    let mut info = 0i32;
    // SAFETY: buffers are n*n (A, B) and n (IPIV), as dgesv requires.
    unsafe {
        dgesv_newlapack(
            &nn,
            &nn,
            a.as_mut_ptr(),
            &nn,
            ipiv.as_mut_ptr(),
            b.as_mut_ptr(),
            &nn,
            &mut info,
        );
    }
    if info != 0 {
        return None;
    }
    Some(
        (0..n)
            .map(|i| (0..n).map(|j| b[j * n + i]).collect())
            .collect(),
    )
}

/// numpy's `inv` on OpenBLAS: `dgesv(A, I)`, i.e. OpenBLAS's left-looking
/// `getf2` (dot products for the upper part, a matrix-vector update for the
/// rest, reciprocal pivot scaling) and `getrs` (row swaps, then unit-lower
/// and upper triangular solves multiplying by the inverted diagonal), with
/// the multiply-add fusion and summation of its kernels as measured against
/// numpy: Linux aarch64 fuses (except the upper solve's odd last row), x86-64's
/// generic kernels do not and accumulate the matrix-vector update first.
/// Matrices up to 4x4 are exact; larger ones use blocked kernels.
pub fn openblas_inv(m: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = m.len();
    let arm = BLAS == Blas::OpenBlasArm;
    let fused = arm;
    let gemv_sum = !arm;
    // aarch64's upper solve handles rows in pairs: the odd last row of an
    // odd-sized matrix updates the others without fusion
    let upper_fused = |k: usize| arm && !(n % 2 == 1 && k == n - 1);
    let madd = |x: f64, y: f64, acc: f64, f: bool| if f { x.mul_add(y, acc) } else { acc + x * y };
    let mut a: Vec<Vec<f64>> = m.to_vec();
    let mut ipiv = vec![0usize; n];
    for j in 0..n {
        let mut b: Vec<f64> = (0..n).map(|i| a[i][j]).collect();
        for i in 0..j {
            b.swap(i, ipiv[i]);
        }
        for i in 1..j {
            let mut s = a[i][0] * b[0];
            for k in 1..i {
                s = madd(a[i][k], b[k], s, fused);
            }
            b[i] -= s;
        }
        if j > 0 {
            if gemv_sum {
                for i in j..n {
                    let mut s = a[i][0] * b[0];
                    for k in 1..j {
                        s = madd(a[i][k], b[k], s, fused);
                    }
                    b[i] -= s;
                }
            } else {
                for k in 0..j {
                    let t = -b[k];
                    for i in j..n {
                        b[i] = madd(t, a[i][k], b[i], fused);
                    }
                }
            }
        }
        // the first maximum
        let mut jp = j;
        for i in j + 1..n {
            if b[i].abs() > b[jp].abs() {
                jp = i;
            }
        }
        ipiv[j] = jp;
        for (i, v) in b.iter().enumerate() {
            a[i][j] = *v;
        }
        let t = b[jp];
        if t == 0.0 {
            return None;
        }
        if t.abs() >= f64::MIN_POSITIVE {
            let r = 1.0 / t;
            if jp != j {
                for c in 0..=j {
                    let tmp = a[j][c];
                    a[j][c] = a[jp][c];
                    a[jp][c] = tmp;
                }
            }
            for row in a.iter_mut().skip(j + 1) {
                row[j] *= r;
            }
        }
    }
    let mut x: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|c| if i == c { 1.0 } else { 0.0 }).collect())
        .collect();
    for i in 0..n {
        x.swap(i, ipiv[i]);
    }
    for c in 0..n {
        let mut v: Vec<f64> = (0..n).map(|i| x[i][c]).collect();
        if arm {
            for k in 0..n {
                for i in k + 1..n {
                    v[i] = madd(-v[k], a[i][k], v[i], fused);
                }
            }
            for k in (0..n).rev() {
                v[k] *= 1.0 / a[k][k];
                for i in 0..k {
                    v[i] = madd(-v[k], a[i][k], v[i], upper_fused(k));
                }
            }
        } else {
            // x86-64's generic kernels: the lower solve sums each row's
            // terms; the upper solve works in 2-row blocks (aligned from the
            // top), subtracting a solved block from the rows above as one sum
            for i in 1..n {
                let mut s = a[i][0] * v[0];
                for k in 1..i {
                    s += a[i][k] * v[k];
                }
                v[i] -= s;
            }
            let mut hi = n;
            while hi > 0 {
                let lo = if hi.is_multiple_of(2) { hi - 2 } else { hi - 1 };
                for k in (lo..hi).rev() {
                    v[k] *= 1.0 / a[k][k];
                    for i in lo..k {
                        v[i] -= v[k] * a[i][k];
                    }
                }
                for i in 0..lo {
                    let mut s = a[i][lo] * v[lo];
                    for k in lo + 1..hi {
                        s += a[i][k] * v[k];
                    }
                    v[i] -= s;
                }
                hi = lo;
            }
        }
        for i in 0..n {
            x[i][c] = v[i];
        }
    }
    Some(x)
}

/// `np.allclose(a, b)` with the default tolerances.
pub fn allclose(a: &[Vec<f64>], b: &[Vec<f64>]) -> bool {
    a.iter().zip(b).all(|(x, y)| {
        x.iter()
            .zip(y)
            .all(|(p, q)| (p - q).abs() <= 1e-8 + 1e-5 * q.abs())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_and_products() {
        let m = vec![
            vec![2.0, 1.0, 3.0],
            vec![0.5, -1.0, 2.0],
            vec![0.0, 0.0, 1.0],
        ];
        let i = inv(&m).unwrap();
        let p = matmul(&m, &i);
        for (r, row) in p.iter().enumerate() {
            for (c, v) in row.iter().enumerate() {
                assert!((v - if r == c { 1.0 } else { 0.0 }).abs() < 1e-12);
            }
        }
        assert!(inv(&[vec![1.0, 2.0], vec![2.0, 4.0]]).is_none());
        assert_eq!(openblas_inv(&m).unwrap().len(), 3);
    }
}
