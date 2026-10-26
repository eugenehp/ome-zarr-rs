//! Functional coordinate transforms and a graph of coordinate spaces: a port
//! of the parts of [transformnd](https://github.com/clbarnes/transformnd)
//! used by ome-zarr scenes.
//!
//! Coordinates are `N x D` row-major point lists (`Vec<Vec<f64>>`).

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::hash::Hash;

use crate::array::LazyArray;
use crate::error::{Error, Result};
use crate::ndimage::{self, Mode};

/// N points of dimensionality D.
pub type Coords = Vec<Vec<f64>>;

/// Source and target dimensionality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NDims {
    pub source: usize,
    pub target: usize,
}

impl NDims {
    pub fn new(source: usize, target: usize) -> NDims {
        NDims { source, target }
    }
    pub fn invert(&self) -> NDims {
        NDims {
            source: self.target,
            target: self.source,
        }
    }
}

impl fmt::Display for NDims {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.source == self.target {
            write!(f, "{}D", self.source)
        } else {
            write!(f, "{}->{}D", self.source, self.target)
        }
    }
}

/// Augmented affine matrix `(Do + 1) x (Di + 1)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Affine {
    pub matrix: Vec<Vec<f64>>,
}

impl Affine {
    pub fn new(matrix: Vec<Vec<f64>>) -> Result<Affine> {
        if matrix.is_empty() || matrix.iter().any(|r| r.len() != matrix[0].len()) {
            return Err(Error::value("Affine matrix must be 2D"));
        }
        let last = matrix.last().unwrap();
        let n = last.len();
        let expected: Vec<f64> = (0..n).map(|i| if i == n - 1 { 1.0 } else { 0.0 }).collect();
        if !crate::linalg::allclose(std::slice::from_ref(last), std::slice::from_ref(&expected)) {
            return Err(Error::value(format!(
                "Transformation matrix is not affine (expected bottom row {}, got {}).",
                crate::pyjson::np_array_str(&expected),
                crate::pyjson::np_array_str(last)
            )));
        }
        Ok(Affine { matrix })
    }

    /// Build from a `(Do, Di)` linear map and optional translation.
    pub fn from_linear_map(linear: &[Vec<f64>], translation: Option<&[f64]>) -> Result<Affine> {
        let rows = linear.len();
        let cols = linear.first().map(|r| r.len()).unwrap_or(0);
        if let Some(t) = translation
            && t.len() != rows
        {
            return Err(Error::value(
                "Translation array must be the same length as linear map columns",
            ));
        }
        let mut m = vec![vec![0.0; cols + 1]; rows + 1];
        for i in 0..rows {
            m[i][..cols].copy_from_slice(&linear[i]);
            if let Some(t) = translation {
                m[i][cols] = t[i];
            }
        }
        m[rows][cols] = 1.0;
        Affine::new(m)
    }

    pub fn identity(ndim: usize) -> Affine {
        let m = (0..=ndim)
            .map(|i| (0..=ndim).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        Affine { matrix: m }
    }

    pub fn translation(t: &[f64]) -> Affine {
        let mut a = Affine::identity(t.len());
        for (i, v) in t.iter().enumerate() {
            a.matrix[i][t.len()] = *v;
        }
        a
    }

    pub fn scaling(s: &[f64]) -> Affine {
        let mut a = Affine::identity(s.len());
        for (i, v) in s.iter().enumerate() {
            a.matrix[i][i] = *v;
        }
        a
    }

    pub fn ndims(&self) -> NDims {
        NDims::new(self.matrix[0].len() - 1, self.matrix.len() - 1)
    }

    /// `self @ rhs`: rhs is applied first.
    pub fn compose(&self, rhs: &Affine) -> Result<Affine> {
        if self.ndims().source != rhs.ndims().target {
            return Err(Error::value(
                "Cannot compose affine transformations of different dimensionality",
            ));
        }
        Ok(Affine {
            matrix: crate::linalg::matmul(&self.matrix, &rhs.matrix),
        })
    }

    /// `np.linalg.inv` of the augmented matrix; `None` if singular.
    pub fn invert(&self) -> Option<Affine> {
        crate::linalg::inv(&self.matrix).map(|m| Affine { matrix: m })
    }

    pub fn is_identity(&self) -> bool {
        let n = self.matrix.len();
        self.matrix[0].len() == n
            && self.matrix.iter().enumerate().all(|(i, r)| {
                r.iter()
                    .enumerate()
                    .all(|(j, v)| *v == if i == j { 1.0 } else { 0.0 })
            })
    }

    /// The linear part, or `None` if it is square and `allclose` to the
    /// identity (transformnd then skips the multiplication).
    fn linear_map(&self) -> Option<Vec<Vec<f64>>> {
        let nd = self.ndims();
        let lm: Vec<Vec<f64>> = self.matrix[..nd.target]
            .iter()
            .map(|r| r[..nd.source].to_vec())
            .collect();
        if nd.source == nd.target {
            let eye: Vec<Vec<f64>> = (0..nd.source)
                .map(|i| {
                    (0..nd.source)
                        .map(|j| if i == j { 1.0 } else { 0.0 })
                        .collect()
                })
                .collect();
            if crate::linalg::allclose(&lm, &eye) {
                return None;
            }
        }
        Some(lm)
    }

    /// The translation, or `None` if `allclose` to zero.
    fn translation_part(&self) -> Option<Vec<f64>> {
        let nd = self.ndims();
        let t: Vec<f64> = self.matrix[..nd.target]
            .iter()
            .map(|r| r[nd.source])
            .collect();
        let zeros = vec![vec![0.0; t.len()]];
        if crate::linalg::allclose(&zeros, std::slice::from_ref(&t)) {
            None
        } else {
            Some(t)
        }
    }

    /// transformnd's `Affine.apply`: `coords @ lm.T`, then `+= t`.
    pub fn apply(&self, coords: &Coords) -> Coords {
        let lm = self.linear_map();
        let t = self.translation_part();
        let mut out: Coords = match &lm {
            Some(lm) => {
                // coords @ lm.T: one product (numpy's BLAS summation order
                // can depend on the row)
                crate::linalg::matmul_bt(coords, lm)
            }
            None => coords.clone(),
        };
        if let Some(t) = t {
            for p in out.iter_mut() {
                for (v, d) in p.iter_mut().zip(&t) {
                    *v += d;
                }
            }
        }
        out
    }
}

/// One component of a [`Transform::ByDimension`].
#[derive(Debug, Clone, PartialEq)]
pub struct SubTransform {
    pub transform: Transform,
    pub input_axes: Vec<usize>,
    pub output_axes: Vec<usize>,
}

impl SubTransform {
    pub fn new(
        transform: Transform,
        input_axes: Vec<usize>,
        output_axes: Option<Vec<usize>>,
    ) -> Result<SubTransform> {
        let output_axes = output_axes.unwrap_or_else(|| input_axes.clone());
        let nd = transform.ndims();
        if nd.source != input_axes.len() {
            return Err(Error::value(format!(
                "Subtransform input dimensionality ({}) must match length of input_axes",
                nd.source
            )));
        }
        if nd.target != output_axes.len() {
            return Err(Error::value(format!(
                "Subtransform output dimensionality ({}) must match length of output_axes",
                nd.target
            )));
        }
        Ok(SubTransform {
            transform,
            input_axes,
            output_axes,
        })
    }
}

/// A vector field used by [`Transform::Coordinates`] and [`Transform::Displacements`].
#[derive(Debug, Clone)]
pub struct VectorField {
    /// One `Di`-dimensional array of values per output dimension.
    pub components: Vec<ndarray::ArrayD<f64>>,
    pub index_transform: Option<Box<Transform>>,
    pub interpolation_order: usize,
    pub ndims: NDims,
    /// Round interpolated vectors to float32 (the field's dtype).
    pub float32: bool,
}

impl PartialEq for VectorField {
    fn eq(&self, other: &Self) -> bool {
        self.components == other.components && self.index_transform == other.index_transform
    }
}

impl VectorField {
    /// Build from an array with `Di + 1` dimensions whose `vector_axis` holds
    /// the vectors.
    pub fn new(
        field: &LazyArray,
        index_transform: Option<Transform>,
        interpolation_order: usize,
        vector_axis: usize,
    ) -> Result<VectorField> {
        let computed = field.compute()?;
        let float32 = computed.dtype() == crate::array::DType::Float32;
        let data = computed.to_f64();
        let sh = data.shape().to_vec();
        if vector_axis >= sh.len() {
            return Err(Error::value("vector_axis out of range"));
        }
        let in_ndim = sh.len() - 1;
        let tgt = sh[vector_axis];
        let source = match &index_transform {
            None => in_ndim,
            Some(t) => {
                if t.ndims().target != in_ndim {
                    return Err(Error::value(
                        "If index_transform is given, its output dimensionality must match the vector field's shape",
                    ));
                }
                t.ndims().source
            }
        };
        let components = (0..tgt)
            .map(|i| data.index_axis(ndarray::Axis(vector_axis), i).to_owned())
            .collect();
        Ok(VectorField {
            components,
            index_transform: index_transform.map(Box::new),
            interpolation_order,
            ndims: NDims::new(source, tgt),
            float32,
        })
    }

    fn vectors(&self, coords: &Coords) -> Result<Coords> {
        let idx = match &self.index_transform {
            Some(t) => t.apply(coords)?,
            None => coords.clone(),
        };
        let nd = self.components.first().map(|c| c.ndim()).unwrap_or(0);
        let cols: Vec<Vec<f64>> = (0..nd)
            .map(|d| idx.iter().map(|p| p[d]).collect())
            .collect();
        let mut out = vec![vec![0.0; self.components.len()]; coords.len()];
        for (k, comp) in self.components.iter().enumerate() {
            // ome-zarr vector fields are dask arrays, which transformnd samples
            // with dask_image's map_coordinates: no spline prefilter, output in
            // the field's dtype
            let v = ndimage::map_coordinates(
                comp,
                &cols,
                self.interpolation_order,
                Mode::Constant,
                f64::NAN,
                false,
            )?;
            for (i, x) in v.into_iter().enumerate() {
                out[i][k] = if self.float32 { x as f32 as f64 } else { x };
            }
        }
        Ok(out)
    }
}

/// A functional coordinate transform.
#[derive(Debug, Clone, PartialEq)]
pub enum Transform {
    Identity(usize),
    Translate(Vec<f64>),
    Scale(Vec<f64>),
    Affine(Affine),
    /// Output column `i` is input column `permutation[i]`.
    MapAxis(Vec<usize>),
    ProjectAxis {
        dropped: Vec<usize>,
        created: Vec<usize>,
        ndims: NDims,
        idxs: Vec<Option<usize>>,
    },
    ByDimension(Vec<SubTransform>),
    Sequence(Sequence),
    Coordinates(Box<VectorField>),
    Displacements(Box<VectorField>),
    Bijection(Box<Transform>, Box<Transform>),
}

/// Chain of transforms applied one after another.
#[derive(Debug, Clone, PartialEq)]
pub struct Sequence {
    pub transforms: Vec<Transform>,
    pub ndims: NDims,
}

impl Sequence {
    /// Combine transforms (non-empty; dimensionalities must chain).
    pub fn new(transforms: Vec<Transform>) -> Result<Sequence> {
        if transforms.is_empty() {
            return Err(Error::value(
                "Empty transform sequence; use TransformSequence.empty(ndim)",
            ));
        }
        for (idx, w) in transforms.windows(2).enumerate() {
            if w[0].ndims().target != w[1].ndims().source {
                return Err(Error::value(format!(
                    "Incompatible dimensionality: transform {idx}'s target is {}D and the next source is {}D",
                    w[0].ndims().target,
                    w[1].ndims().source
                )));
            }
        }
        let ndims = NDims::new(
            transforms[0].ndims().source,
            transforms.last().unwrap().ndims().target,
        );
        Ok(Sequence { transforms, ndims })
    }

    pub fn empty(ndim: usize) -> Sequence {
        Sequence {
            transforms: vec![],
            ndims: NDims::new(ndim, ndim),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }

    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    /// Flatten nested sequences (optionally replacing bijections by their forward transform).
    pub fn flatten(&self, drop_inverse: bool) -> Sequence {
        let mut out = vec![];
        for t in &self.transforms {
            let t = match t {
                Transform::Bijection(f, _) if drop_inverse => (**f).clone(),
                other => other.clone(),
            };
            match t {
                Transform::Sequence(s) => out.extend(s.flatten(drop_inverse).transforms),
                other => out.push(other),
            }
        }
        Sequence {
            transforms: out,
            ndims: self.ndims,
        }
    }

    /// Compose consecutive affines and drop identities.
    pub fn simplify(&self, drop_inverse: bool) -> Sequence {
        if self.is_empty() {
            return self.clone();
        }
        let mut out = vec![];
        let mut affine: Option<Affine> = None;
        for t in self.flatten(drop_inverse).transforms {
            if t.is_identity() {
                continue;
            }
            match t.to_affine() {
                None => {
                    if let Some(a) = affine.take()
                        && !a.is_identity()
                    {
                        out.push(Transform::Affine(a));
                    }
                    out.push(t);
                }
                Some(na) => {
                    affine = Some(match affine {
                        None => na,
                        Some(a) => na.compose(&a).unwrap_or(na),
                    });
                }
            }
        }
        if let Some(a) = affine
            && !a.is_identity()
        {
            out.push(Transform::Affine(a));
        }
        if out.is_empty() {
            return Sequence::empty(self.ndims.source);
        }
        Sequence::new(out).unwrap_or_else(|_| self.clone())
    }

    pub fn apply(&self, coords: &Coords) -> Result<Coords> {
        let mut c = coords.clone();
        for t in &self.transforms {
            c = t.apply(&c)?;
        }
        Ok(c)
    }

    pub fn invert(&self) -> Option<Sequence> {
        if self.is_empty() {
            return Some(Sequence::empty(self.ndims.source));
        }
        let ts: Option<Vec<Transform>> = self.transforms.iter().rev().map(|t| t.invert()).collect();
        Sequence::new(ts?).ok()
    }

    pub fn to_affine(&self) -> Option<Affine> {
        if self.is_empty() {
            return Some(Affine::identity(self.ndims.source));
        }
        let s = self.simplify(true);
        if s.is_empty() {
            return Some(Affine::identity(self.ndims.source));
        }
        if s.len() != 1 {
            return None;
        }
        s.transforms[0].to_affine()
    }
}

impl Transform {
    /// `ProjectAxis` constructor (at least one of the dimensionalities must be given).
    pub fn project_axis(
        dropped: Option<Vec<usize>>,
        created: Option<Vec<usize>>,
        source_ndim: Option<usize>,
        target_ndim: Option<usize>,
    ) -> Result<Transform> {
        let mut dropped = dropped.unwrap_or_default();
        let mut created = created.unwrap_or_default();
        dropped.sort();
        dropped.dedup();
        created.sort();
        created.dedup();
        let (src, tgt) = match (source_ndim, target_ndim) {
            (Some(s), t) => {
                let nd = (s + created.len())
                    .checked_sub(dropped.len())
                    .ok_or_else(|| Error::value("invalid ProjectAxis"))?;
                if let Some(t) = t
                    && t != nd
                {
                    return Err(Error::value("Operations do not match expected target ndim"));
                }
                (s, nd)
            }
            (None, Some(t)) => {
                let nd = (t + dropped.len())
                    .checked_sub(created.len())
                    .ok_or_else(|| Error::value("invalid ProjectAxis"))?;
                (nd, t)
            }
            (None, None) => {
                return Err(Error::value(
                    "At least one of source_ndim or target_ndim must be given",
                ));
            }
        };
        let mut idxs: Vec<Option<usize>> = (0..src).map(Some).collect();
        for &d in dropped.iter().rev() {
            if d >= idxs.len() {
                return Err(Error::value("dropped axis out of range"));
            }
            idxs.remove(d);
        }
        for &c in &created {
            if c > idxs.len() {
                return Err(Error::value("created axis out of range"));
            }
            idxs.insert(c, None);
        }
        Ok(Transform::ProjectAxis {
            dropped,
            created,
            ndims: NDims::new(src, tgt),
            idxs,
        })
    }

    /// `ByDimension` constructor (input and output axes must cover 0..N).
    pub fn by_dimension(subs: Vec<SubTransform>) -> Result<Transform> {
        let mut si: Vec<usize> = subs.iter().flat_map(|s| s.input_axes.clone()).collect();
        si.sort();
        if si != (0..si.len()).collect::<Vec<_>>() {
            return Err(Error::value("N-length input axes must go from 0 to N-1"));
        }
        let mut so: Vec<usize> = subs.iter().flat_map(|s| s.output_axes.clone()).collect();
        so.sort();
        if so != (0..so.len()).collect::<Vec<_>>() {
            return Err(Error::value("N-length output axes must go from 0 to N-1"));
        }
        Ok(Transform::ByDimension(subs))
    }

    /// `MapAxis` constructor (must be a permutation).
    pub fn map_axis(permutation: Vec<usize>) -> Result<Transform> {
        let mut s = permutation.clone();
        s.sort();
        if s.iter().enumerate().any(|(a, b)| a != *b) {
            return Err(Error::value(
                "N-D permutation must contain all dimensions [0, N) exactly once",
            ));
        }
        Ok(Transform::MapAxis(permutation))
    }

    /// Displacements field (source and target dimensionality must match).
    pub fn displacements(field: VectorField) -> Result<Transform> {
        if field.ndims.source != field.ndims.target {
            return Err(Error::value("Displacements cannot change dimensionality"));
        }
        Ok(Transform::Displacements(Box::new(field)))
    }

    /// Source and target dimensionality.
    pub fn ndims(&self) -> NDims {
        match self {
            Transform::Identity(n) => NDims::new(*n, *n),
            Transform::Translate(t) => NDims::new(t.len(), t.len()),
            Transform::Scale(s) => NDims::new(s.len(), s.len()),
            Transform::Affine(a) => a.ndims(),
            Transform::MapAxis(p) => NDims::new(p.len(), p.len()),
            Transform::ProjectAxis { ndims, .. } => *ndims,
            Transform::ByDimension(subs) => {
                let i: usize = subs.iter().map(|s| s.input_axes.len()).sum();
                let o: usize = subs.iter().map(|s| s.output_axes.len()).sum();
                NDims::new(i, o)
            }
            Transform::Sequence(s) => s.ndims,
            Transform::Coordinates(v) | Transform::Displacements(v) => v.ndims,
            Transform::Bijection(f, _) => f.ndims(),
        }
    }

    /// Whether this is a no-op.
    pub fn is_identity(&self) -> bool {
        match self {
            Transform::Identity(_) => true,
            Transform::Translate(t) => t.iter().all(|v| *v == 0.0),
            Transform::Scale(s) => s.iter().all(|v| *v == 1.0),
            Transform::Affine(a) => a.is_identity(),
            Transform::MapAxis(p) => p.iter().enumerate().all(|(a, b)| a == *b),
            Transform::ProjectAxis {
                dropped, created, ..
            } => dropped.is_empty() && created.is_empty(),
            Transform::ByDimension(subs) => subs
                .iter()
                .all(|s| s.input_axes == s.output_axes && s.transform.is_identity()),
            Transform::Sequence(s) => s.transforms.iter().all(|t| t.is_identity()),
            Transform::Bijection(f, i) => f.is_identity() && i.is_identity(),
            _ => false,
        }
    }

    fn validate_coords(&self, coords: &Coords) -> Result<()> {
        let d = self.ndims().source;
        if let Some(p) = coords.iter().find(|p| p.len() != d) {
            return Err(Error::value(format!(
                "Coords must have dimensionality {d}, got {}",
                p.len()
            )));
        }
        Ok(())
    }

    /// Apply to N x D coordinates.
    pub fn apply(&self, coords: &Coords) -> Result<Coords> {
        if !matches!(self, Transform::Sequence(_) | Transform::Bijection(..)) {
            self.validate_coords(coords)?;
        }
        Ok(match self {
            Transform::Identity(_) => coords.clone(),
            Transform::Translate(t) => coords
                .iter()
                .map(|p| p.iter().zip(t).map(|(a, b)| a + b).collect())
                .collect(),
            Transform::Scale(s) => coords
                .iter()
                .map(|p| p.iter().zip(s).map(|(a, b)| a * b).collect())
                .collect(),
            Transform::Affine(a) => a.apply(coords),
            Transform::MapAxis(perm) => coords
                .iter()
                .map(|p| perm.iter().map(|&i| p[i]).collect())
                .collect(),
            Transform::ProjectAxis { idxs, .. } => coords
                .iter()
                .map(|p| {
                    idxs.iter()
                        .map(|i| i.map(|i| p[i]).unwrap_or(0.0))
                        .collect()
                })
                .collect(),
            Transform::ByDimension(subs) => {
                let nd = self.ndims();
                let mut out = vec![vec![0.0; nd.target]; coords.len()];
                for s in subs {
                    let sub: Coords = coords
                        .iter()
                        .map(|p| s.input_axes.iter().map(|&i| p[i]).collect())
                        .collect();
                    let t = s.transform.apply(&sub)?;
                    for (row, tr) in out.iter_mut().zip(t) {
                        for (k, &o) in s.output_axes.iter().enumerate() {
                            row[o] = tr[k];
                        }
                    }
                }
                out
            }
            Transform::Sequence(s) => s.apply(coords)?,
            Transform::Coordinates(v) => v.vectors(coords)?,
            Transform::Displacements(v) => {
                let vecs = v.vectors(coords)?;
                coords
                    .iter()
                    .zip(vecs)
                    .map(|(p, d)| p.iter().zip(d).map(|(a, b)| a + b).collect())
                    .collect()
            }
            Transform::Bijection(f, _) => f.apply(coords)?,
        })
    }

    /// The inverse, if possible.
    pub fn invert(&self) -> Option<Transform> {
        match self {
            Transform::Identity(n) => Some(Transform::Identity(*n)),
            Transform::Translate(t) => Some(Transform::Translate(t.iter().map(|v| -v).collect())),
            Transform::Scale(s) => Some(Transform::Scale(s.iter().map(|v| 1.0 / v).collect())),
            Transform::Affine(a) => a.invert().map(Transform::Affine),
            Transform::MapAxis(p) => {
                let mut inv: Vec<usize> = (0..p.len()).collect();
                inv.sort_by_key(|&i| p[i]);
                Some(Transform::MapAxis(inv))
            }
            Transform::ProjectAxis {
                dropped,
                created,
                ndims,
                ..
            } => Transform::project_axis(
                Some(created.clone()),
                Some(dropped.clone()),
                Some(ndims.target),
                Some(ndims.source),
            )
            .ok(),
            Transform::ByDimension(subs) => {
                let inv: Option<Vec<SubTransform>> = subs
                    .iter()
                    .rev()
                    .map(|s| {
                        Some(SubTransform {
                            transform: s.transform.invert()?,
                            input_axes: s.output_axes.clone(),
                            output_axes: s.input_axes.clone(),
                        })
                    })
                    .collect();
                Some(Transform::ByDimension(inv?))
            }
            Transform::Sequence(s) => s.invert().map(Transform::Sequence),
            Transform::Bijection(f, i) => Some(Transform::Bijection(i.clone(), f.clone())),
            Transform::Coordinates(_) | Transform::Displacements(_) => None,
        }
    }

    /// Convert to an affine if possible.
    pub fn to_affine(&self) -> Option<Affine> {
        match self {
            Transform::Identity(n) => Some(Affine::identity(*n)),
            Transform::Translate(t) => Some(Affine::translation(t)),
            Transform::Scale(s) => Some(Affine::scaling(s)),
            Transform::Affine(a) => Some(a.clone()),
            Transform::MapAxis(p) => {
                let n = p.len();
                let m: Vec<Vec<f64>> = p
                    .iter()
                    .map(|&j| (0..n).map(|k| if k == j { 1.0 } else { 0.0 }).collect())
                    .collect();
                Affine::from_linear_map(&m, None).ok()
            }
            Transform::ProjectAxis { ndims, .. } => {
                let eye: Coords = (0..ndims.source)
                    .map(|i| {
                        (0..ndims.source)
                            .map(|j| if i == j { 1.0 } else { 0.0 })
                            .collect()
                    })
                    .collect();
                let out = self.apply(&eye).ok()?;
                // out is (Di x Do); the linear map is its transpose (Do x Di)
                let lm: Vec<Vec<f64>> = (0..ndims.target)
                    .map(|o| (0..ndims.source).map(|i| out[i][o]).collect())
                    .collect();
                Affine::from_linear_map(&lm, None).ok()
            }
            Transform::Sequence(s) => s.to_affine(),
            Transform::Bijection(f, i) => {
                let fa = f.to_affine()?;
                let ia = i.to_affine()?.invert()?;
                if fa == ia { Some(fa) } else { None }
            }
            _ => None,
        }
    }
}

/// A transform between two named spaces.
#[derive(Debug, Clone, PartialEq)]
pub struct Spaced<S> {
    pub transform: Transform,
    pub source: S,
    pub target: S,
}

impl<S: Clone> Spaced<S> {
    pub fn new(transform: Transform, source: S, target: S) -> Spaced<S> {
        Spaced {
            transform,
            source,
            target,
        }
    }

    pub fn invert(&self) -> Option<Spaced<S>> {
        Some(Spaced {
            transform: self.transform.invert()?,
            source: self.target.clone(),
            target: self.source.clone(),
        })
    }
}

/// Transform between any number of coordinate spaces, finding the shortest
/// path of transforms between two spaces.
#[derive(Debug, Clone, Default)]
pub struct TransformGraph<S: Eq + Hash + Clone> {
    nodes: HashMap<S, usize>,
    node_order: Vec<S>,
    edges: Vec<(S, S, Transform)>,
}

impl<S: Eq + Hash + Clone + crate::pyjson::PyDisplay> TransformGraph<S> {
    pub fn new() -> Self {
        TransformGraph {
            nodes: HashMap::new(),
            node_order: vec![],
            edges: vec![],
        }
    }

    /// Dimensionality of a space, if known.
    pub fn ndim(&self, space: &S) -> Option<usize> {
        self.nodes.get(space).copied()
    }

    /// All spaces, in insertion order.
    pub fn nodes(&self) -> &[S] {
        &self.node_order
    }

    /// Number of spaces.
    pub fn node_count(&self) -> usize {
        self.node_order.len()
    }

    /// All edges.
    pub fn edges(&self) -> &[(S, S, Transform)] {
        &self.edges
    }

    fn add_space(&mut self, space: &S, ndim: usize) -> Result<()> {
        match self.nodes.get(space) {
            None => {
                self.nodes.insert(space.clone(), ndim);
                self.node_order.push(space.clone());
                Ok(())
            }
            Some(&n) if n != ndim => Err(Error::value(format!(
                "Space {} is {n}D, got {ndim}D",
                space.py_str()
            ))),
            _ => Ok(()),
        }
    }

    /// Add a transform as an edge (a bijection adds its forward transform).
    pub fn add_transform(&mut self, spaced: Spaced<S>) -> Result<()> {
        let t = match spaced.transform {
            Transform::Bijection(f, _) => *f,
            t => t,
        };
        let nd = t.ndims();
        self.add_space(&spaced.source, nd.source)?;
        self.add_space(&spaced.target, nd.target)?;
        self.edges.push((spaced.source, spaced.target, t));
        Ok(())
    }

    fn shortest_path(&self, src: &S, tgt: &S) -> Option<Vec<usize>> {
        // BFS over edges; returns edge indices
        let mut prev: HashMap<S, usize> = HashMap::new();
        let mut visited: HashSet<S> = HashSet::new();
        let mut q = VecDeque::new();
        visited.insert(src.clone());
        q.push_back(src.clone());
        while let Some(n) = q.pop_front() {
            if &n == tgt {
                break;
            }
            for (i, (s, t, _)) in self.edges.iter().enumerate() {
                if s == &n && !visited.contains(t) {
                    visited.insert(t.clone());
                    prev.insert(t.clone(), i);
                    q.push_back(t.clone());
                }
            }
        }
        if !visited.contains(tgt) {
            return None;
        }
        let mut path = vec![];
        let mut cur = tgt.clone();
        while &cur != src {
            let e = prev[&cur];
            path.push(e);
            cur = self.edges[e].0.clone();
        }
        path.reverse();
        Some(path)
    }

    /// The shortest transform sequence between two spaces. Unless `full`,
    /// consecutive affines are composed and bijections' inverses dropped.
    pub fn get_sequence(&self, source: &S, target: &S, full: bool) -> Result<Sequence> {
        let src_ndim = self
            .ndim(source)
            .ok_or_else(|| Error::value(format!("Unknown source space {}", source.py_str())))?;
        if source == target {
            return Ok(Sequence::empty(src_ndim));
        }
        if self.ndim(target).is_none() {
            // (sic) transformnd names the source space here
            return Err(Error::value(format!(
                "Unknown target space {}",
                source.py_str()
            )));
        }
        let path = self.shortest_path(source, target).ok_or_else(|| {
            Error::NoPath(format!(
                "No path between {} and {}.",
                source.py_str(),
                target.py_str()
            ))
        })?;
        let seq = Sequence::new(path.iter().map(|&e| self.edges[e].2.clone()).collect())?;
        Ok(if full { seq } else { seq.simplify(true) })
    }

    /// Transform coordinates from one space to another.
    pub fn transform(&self, source: &S, target: &S, coords: &Coords) -> Result<Coords> {
        self.get_sequence(source, target, false)?.apply(coords)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_compose_and_invert() {
        let a = Transform::Scale(vec![2.0, 3.0]);
        let b = Transform::Translate(vec![1.0, 1.0]);
        let s = Sequence::new(vec![a, b]).unwrap();
        let p = s.apply(&vec![vec![1.0, 1.0]]).unwrap();
        assert_eq!(p, vec![vec![3.0, 4.0]]);
        let aff = s.to_affine().unwrap();
        assert_eq!(
            aff.matrix,
            vec![
                vec![2.0, 0.0, 1.0],
                vec![0.0, 3.0, 1.0],
                vec![0.0, 0.0, 1.0]
            ]
        );
        let inv = s.invert().unwrap();
        assert_eq!(inv.apply(&p).unwrap(), vec![vec![1.0, 1.0]]);
    }

    #[test]
    fn project_axis() {
        let t = Transform::project_axis(None, Some(vec![0]), Some(2), None).unwrap();
        assert_eq!(t.ndims(), NDims::new(2, 3));
        assert_eq!(
            t.apply(&vec![vec![5.0, 6.0]]).unwrap(),
            vec![vec![0.0, 5.0, 6.0]]
        );
        let a = t.to_affine().unwrap();
        assert_eq!(a.ndims(), NDims::new(2, 3));
    }

    #[test]
    fn graph_paths() {
        let mut g = TransformGraph::new();
        let t = Spaced::new(Transform::Scale(vec![2.0]), "a", "b");
        g.add_transform(t.clone()).unwrap();
        g.add_transform(t.invert().unwrap()).unwrap();
        g.add_transform(Spaced::new(Transform::Translate(vec![1.0]), "b", "c"))
            .unwrap();
        assert_eq!(g.node_count(), 3);
        assert_eq!(
            g.transform(&"a", &"c", &vec![vec![1.0]]).unwrap(),
            vec![vec![3.0]]
        );
        assert!(g.get_sequence(&"c", &"a", false).is_err());
        assert!(g.get_sequence(&"a", &"a", false).unwrap().is_empty());
    }
}
