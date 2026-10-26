//! OME-NGFF 0.6 coordinate systems and coordinate transformations
//! (`ome_zarr_models.v06.coordinate_transforms`).

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::zarr::JsonMap;

fn is_none_bool(b: &Option<bool>) -> bool {
    b.is_none()
}

/// An element of a coordinate system's axes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Axis {
    /// Axis name.
    pub name: Option<String>,
    /// Axis type, e.g. "space", "time", "channel", "coordinate", "displacement".
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub type_: Option<String>,
    /// Whether coordinates on this axis are discrete.
    #[serde(default, skip_serializing_if = "is_none_bool")]
    pub discrete: Option<bool>,
    /// Axis units.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<Value>,
    /// Longer name for axis.
    #[serde(rename = "longName", default, skip_serializing_if = "Option::is_none")]
    pub long_name: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

impl Axis {
    pub fn new(name: &str, type_: Option<&str>) -> Axis {
        Axis {
            name: Some(name.to_string()),
            type_: type_.map(String::from),
            ..Default::default()
        }
    }

    pub fn name_str(&self) -> &str {
        self.name.as_deref().unwrap_or("")
    }

    /// The unit as a string, if it is one.
    pub fn unit_str(&self) -> Option<String> {
        match &self.unit {
            Some(Value::String(s)) => Some(s.clone()),
            Some(Value::Null) | None => None,
            Some(other) => Some(other.to_string()),
        }
    }
}

/// A named coordinate system.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoordinateSystem {
    pub name: String,
    pub axes: Vec<Axis>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

impl CoordinateSystem {
    pub fn new(name: &str, axes: Vec<Axis>) -> Result<CoordinateSystem> {
        let cs = CoordinateSystem {
            name: name.to_string(),
            axes,
            extra: JsonMap::new(),
        };
        cs.validate()?;
        Ok(cs)
    }

    pub fn from_json(v: &Value) -> Result<CoordinateSystem> {
        crate::models::pydantic::validate_model("CoordinateSystem", v)?;
        let cs: CoordinateSystem = serde_json::from_value(v.clone())
            .map_err(|e| Error::value(format!("invalid coordinate system: {e}")))?;
        cs.validate()?;
        Ok(cs)
    }

    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(Error::value("Coordinate system name must not be empty"));
        }
        if self.axes.is_empty() {
            return Err(Error::value("Coordinate system must have at least 1 axis"));
        }
        let names: Vec<&Option<String>> = self.axes.iter().map(|a| &a.name).collect();
        unique_items(&names)?;
        Ok(())
    }

    pub fn ndim(&self) -> usize {
        self.axes.len()
    }

    pub fn axis_names(&self) -> Vec<String> {
        self.axes.iter().map(|a| a.name_str().to_string()).collect()
    }
}

/// Error if items are not unique (`ome_zarr_models.common.validation.unique_items_validator`).
pub fn unique_items<T: PartialEq + std::fmt::Debug>(items: &[T]) -> Result<()> {
    for (i, a) in items.iter().enumerate() {
        if items[i + 1..].contains(a) {
            return Err(Error::value(format!("Non-unique values in {items:?}.")));
        }
    }
    Ok(())
}

/// Identifies a coordinate system by name and/or zarr path.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct CoordinateSystemIdentifier {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

impl CoordinateSystemIdentifier {
    pub fn name(name: &str) -> Self {
        CoordinateSystemIdentifier {
            name: Some(name.into()),
            path: None,
        }
    }
    pub fn path(path: &str) -> Self {
        CoordinateSystemIdentifier {
            name: None,
            path: Some(path.into()),
        }
    }
    pub fn name_path(name: &str, path: &str) -> Self {
        CoordinateSystemIdentifier {
            name: Some(name.into()),
            path: Some(path.into()),
        }
    }
    pub fn validate(&self) -> Result<()> {
        if self.name.is_none() && self.path.is_none() {
            return Err(Error::value(
                "At least one of 'name' or 'path' must be set.",
            ));
        }
        Ok(())
    }
}

/// Raised when a transform cannot be converted to an affine.
pub const NO_AFFINE: &str = "NoAffineError";

macro_rules! transform_struct {
    ($(#[$m:meta])* $name:ident { $($(#[$fm:meta])* pub $f:ident : $t:ty),* $(,)? }) => {
        $(#[$m])*
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
        pub struct $name {
            #[serde(default, skip_serializing_if = "Option::is_none")]
            pub input: Option<CoordinateSystemIdentifier>,
            #[serde(default, skip_serializing_if = "Option::is_none")]
            pub output: Option<CoordinateSystemIdentifier>,
            #[serde(default, skip_serializing_if = "Option::is_none")]
            pub name: Option<String>,
            $($(#[$fm])* pub $f: $t,)*
        }
    };
}

transform_struct!(
    /// Identity transformation.
    Identity {}
);
transform_struct!(
    /// Axis mapping transform.
    MapAxis {
        #[serde(rename = "mapAxis")]
        pub map_axis: Vec<i64>,
    }
);
transform_struct!(
    /// Projects coordinates between different dimensionalities.
    ProjectAxis {
        #[serde(rename = "createdOutputs", default, skip_serializing_if = "Option::is_none")]
        pub created_outputs: Option<Vec<i64>>,
        #[serde(rename = "droppedInputs", default, skip_serializing_if = "Option::is_none")]
        pub dropped_inputs: Option<Vec<i64>>,
    }
);
transform_struct!(
    /// Translation transformation.
    Translation { pub translation: Vec<f64> }
);
transform_struct!(
    /// Scale transformation.
    Scale { pub scale: Vec<f64> }
);
transform_struct!(
    /// Affine transform (augmented matrix rows, `ndim x (ndim + 1)`).
    Affine {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub affine: Option<Vec<Vec<f64>>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub path: Option<String>,
    }
);
transform_struct!(
    /// Rotation transform.
    Rotation {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub rotation: Option<Vec<Vec<f64>>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub path: Option<String>,
    }
);
transform_struct!(
    /// Sequence transformation.
    Sequence { pub transformations: Vec<Transform> }
);
transform_struct!(
    /// Displacement field transform.
    Displacements {
        pub path: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub interpolation: Option<String>,
    }
);
transform_struct!(
    /// Coordinate field transform.
    Coordinates {
        pub path: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub interpolation: Option<String>,
    }
);
transform_struct!(
    /// A transform with an explicit inverse.
    Bijection {
        pub forward: Box<Transform>,
        pub inverse: Box<Transform>,
    }
);
transform_struct!(
    /// A transform that operates on a subset of dimensions.
    ByDimension { pub transformations: Vec<ByDimensionTransform> }
);

/// A transformation item within a byDimension transformation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ByDimensionTransform {
    pub transformation: Transform,
    #[serde(rename = "inputAxes")]
    pub input_axes: Vec<i64>,
    #[serde(rename = "outputAxes")]
    pub output_axes: Vec<i64>,
}

impl Default for Transform {
    fn default() -> Self {
        Transform::Identity(Identity::default())
    }
}

/// Any coordinate transformation (discriminated by `type`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Transform {
    #[serde(rename = "identity")]
    Identity(Identity),
    #[serde(rename = "mapAxis")]
    MapAxis(MapAxis),
    #[serde(rename = "projectAxis")]
    ProjectAxis(ProjectAxis),
    #[serde(rename = "translation")]
    Translation(Translation),
    #[serde(rename = "scale")]
    Scale(Scale),
    #[serde(rename = "affine")]
    Affine(Affine),
    #[serde(rename = "rotation")]
    Rotation(Rotation),
    #[serde(rename = "sequence")]
    Sequence(Sequence),
    #[serde(rename = "displacements")]
    Displacements(Displacements),
    #[serde(rename = "coordinates")]
    Coordinates(Coordinates),
    #[serde(rename = "bijection")]
    Bijection(Bijection),
    #[serde(rename = "byDimension")]
    ByDimension(ByDimension),
}

macro_rules! each_variant {
    ($s:expr, $t:ident => $body:expr) => {
        match $s {
            Transform::Identity($t) => $body,
            Transform::MapAxis($t) => $body,
            Transform::ProjectAxis($t) => $body,
            Transform::Translation($t) => $body,
            Transform::Scale($t) => $body,
            Transform::Affine($t) => $body,
            Transform::Rotation($t) => $body,
            Transform::Sequence($t) => $body,
            Transform::Displacements($t) => $body,
            Transform::Coordinates($t) => $body,
            Transform::Bijection($t) => $body,
            Transform::ByDimension($t) => $body,
        }
    };
}

fn inverse_name(name: &Option<String>) -> Option<String> {
    name.as_ref().map(|n| format!("{n}_inverse"))
}

/// `np.linalg.inv` of a square matrix (see [`crate::linalg::inv`]).
pub fn invert_matrix(m: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    crate::linalg::inv(m)
}

fn matvec(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    crate::linalg::matvec(m, v)
}

impl Affine {
    /// Build from a linear matrix and a translation vector.
    pub fn from_matrix_vector(matrix: &[Vec<f64>], vector: &[f64]) -> Affine {
        Affine {
            affine: Some(
                matrix
                    .iter()
                    .zip(vector)
                    .map(|(r, &t)| r.iter().copied().chain([t]).collect())
                    .collect(),
            ),
            ..Default::default()
        }
    }

    pub fn affine_matrix(&self) -> Result<&Vec<Vec<f64>>> {
        match (&self.affine, &self.path) {
            (Some(a), _) => Ok(a),
            (None, Some(_)) => Err(Error::NotImplemented(
                "Loading affine matrix from a Zarr array not yet implemented".into(),
            )),
            _ => Err(Error::value("Both self.affine and self.path are None")),
        }
    }

    /// The linear part (each row without its last element).
    pub fn matrix(&self) -> Result<Vec<Vec<f64>>> {
        Ok(self
            .affine_matrix()?
            .iter()
            .map(|r| r[..r.len() - 1].to_vec())
            .collect())
    }

    /// The translation part.
    pub fn translation(&self) -> Result<Vec<f64>> {
        Ok(self
            .affine_matrix()?
            .iter()
            .map(|r| r[r.len() - 1])
            .collect())
    }
}

impl Rotation {
    pub fn rotation_matrix(&self) -> Result<&Vec<Vec<f64>>> {
        match (&self.rotation, &self.path) {
            (Some(a), _) => Ok(a),
            (None, Some(_)) => Err(Error::NotImplemented(
                "Loading rotation matrix from a Zarr array not yet implemented".into(),
            )),
            _ => Err(Error::value("Both self.rotation and self.path are None")),
        }
    }
}

impl Transform {
    /// Parse and validate from JSON.
    pub fn from_json(v: &Value) -> Result<Transform> {
        crate::models::pydantic::validate_model("Transform", v)?;
        let t: Transform = serde_json::from_value(v.clone())
            .map_err(|e| Error::value(format!("invalid transform: {e}")))?;
        t.validate()?;
        Ok(t)
    }

    /// Serialize (only set fields, like `model_dump(exclude_none=True)`).
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).expect("transform serialization")
    }

    /// The `type` string.
    pub fn type_name(&self) -> &'static str {
        match self {
            Transform::Identity(_) => "identity",
            Transform::MapAxis(_) => "mapAxis",
            Transform::ProjectAxis(_) => "projectAxis",
            Transform::Translation(_) => "translation",
            Transform::Scale(_) => "scale",
            Transform::Affine(_) => "affine",
            Transform::Rotation(_) => "rotation",
            Transform::Sequence(_) => "sequence",
            Transform::Displacements(_) => "displacements",
            Transform::Coordinates(_) => "coordinates",
            Transform::Bijection(_) => "bijection",
            Transform::ByDimension(_) => "byDimension",
        }
    }

    pub fn input(&self) -> Option<&CoordinateSystemIdentifier> {
        each_variant!(self, t => t.input.as_ref())
    }

    pub fn output(&self) -> Option<&CoordinateSystemIdentifier> {
        each_variant!(self, t => t.output.as_ref())
    }

    pub fn name(&self) -> Option<&str> {
        each_variant!(self, t => t.name.as_deref())
    }

    /// Copy with new input/output (`model_copy(update=...)`).
    pub fn with_io(
        &self,
        input: Option<CoordinateSystemIdentifier>,
        output: Option<CoordinateSystemIdentifier>,
    ) -> Transform {
        let mut t = self.clone();
        each_variant!(&mut t, x => {
            x.input = input;
            x.output = output;
        });
        t
    }

    /// Copy with a new name.
    pub fn with_name(&self, name: Option<String>) -> Transform {
        let mut t = self.clone();
        each_variant!(&mut t, x => {
            x.name = name;
        });
        t
    }

    /// Validate the model (pydantic validators).
    pub fn validate(&self) -> Result<()> {
        if let Some(i) = self.input() {
            i.validate()?;
        }
        if let Some(o) = self.output() {
            o.validate()?;
        }
        if self.input().is_none() != self.output().is_none() {
            return Err(Error::value(format!(
                "Either both input and output must be defined or both must be omitted. Got input={:?} and output={:?}.",
                self.input(),
                self.output()
            )));
        }
        match self {
            Transform::MapAxis(t) => {
                let n = t.map_axis.len() as i64;
                let s: HashSet<i64> = t.map_axis.iter().copied().collect();
                let want: HashSet<i64> = (0..n).collect();
                if s != want {
                    return Err(Error::value(format!(
                        "Not all axes present from 0 to {}: {:?}",
                        n - 1,
                        t.map_axis
                    )));
                }
            }
            Transform::Affine(t) => {
                if t.affine.is_none() && t.path.is_none() {
                    return Err(Error::value("One of 'affine' or 'path' must be given"));
                }
                if let Some(a) = &t.affine {
                    let lens: Vec<usize> = a.iter().map(|r| r.len()).collect();
                    if lens.iter().any(|&l| l != lens[0]) {
                        return Err(Error::value(format!(
                            "Row lengths in affine matrix ({lens:?}) are not all equal."
                        )));
                    }
                }
            }
            Transform::Rotation(t) => {
                if t.rotation.is_none() && t.path.is_none() {
                    return Err(Error::value("One of 'rotation' or 'path' must be given"));
                }
                if let Some(m) = &t.rotation {
                    let n = m.len();
                    for i in 0..n {
                        for j in 0..n {
                            let d: f64 = m[i].iter().zip(&m[j]).map(|(a, b)| a * b).sum();
                            let want = if i == j { 1.0 } else { 0.0 };
                            if (d - want).abs() > 1e-8 + 1e-5 * want.abs() {
                                return Err(Error::value(format!(
                                    "Provided matrix is not a pure rotation matrix: {m:?}"
                                )));
                            }
                        }
                    }
                }
            }
            Transform::ProjectAxis(t) => {
                for v in [&t.created_outputs, &t.dropped_inputs]
                    .into_iter()
                    .flatten()
                {
                    if v.is_empty() || v.len() > 3 {
                        return Err(Error::value(
                            "createdOutputs/droppedInputs must have 1 to 3 items",
                        ));
                    }
                    unique_items(v)?;
                }
                if t.created_outputs.is_none() && t.dropped_inputs.is_none() {
                    return Err(Error::value(
                        "At least one of 'createdOutputs' or 'droppedInputs' must be set.",
                    ));
                }
            }
            Transform::Sequence(t) => {
                for x in &t.transformations {
                    x.validate()?;
                }
            }
            Transform::Bijection(t) => {
                t.forward.validate()?;
                t.inverse.validate()?;
            }
            Transform::ByDimension(t) => {
                for x in &t.transformations {
                    x.transformation.validate()?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Dimensionality, if the transform has one.
    pub fn ndim(&self) -> Option<usize> {
        match self {
            Transform::MapAxis(t) => Some(t.map_axis.len()),
            Transform::Translation(t) => Some(t.translation.len()),
            Transform::Scale(t) => Some(t.scale.len()),
            Transform::Affine(t) => t.affine.as_ref().map(|a| a.len()),
            Transform::Rotation(t) => t.rotation.as_ref().map(|a| a.len()),
            Transform::Sequence(t) => {
                let first = t.transformations.first()?.ndim()?;
                for x in &t.transformations[1..] {
                    if x.ndim()? != first {
                        return None;
                    }
                }
                Some(first)
            }
            _ => None,
        }
    }

    /// Whether [`Transform::get_inverse`] is available.
    pub fn has_inverse(&self) -> Result<bool> {
        Ok(match self {
            Transform::Identity(_)
            | Transform::MapAxis(_)
            | Transform::Translation(_)
            | Transform::Scale(_) => true,
            Transform::Bijection(_) => true,
            Transform::Affine(_)
            | Transform::Rotation(_)
            | Transform::Displacements(_)
            | Transform::Coordinates(_) => false,
            Transform::Sequence(t) => {
                for x in &t.transformations {
                    if !x.has_inverse()? {
                        return Ok(false);
                    }
                }
                true
            }
            Transform::ByDimension(t) => {
                for x in &t.transformations {
                    if !x.transformation.has_inverse()? {
                        return Ok(false);
                    }
                }
                true
            }
            Transform::ProjectAxis(_) => {
                return Err(Error::NotImplemented("ProjectAxis.has_inverse".into()));
            }
        })
    }

    /// The inverse transform.
    pub fn get_inverse(&self) -> Result<Transform> {
        let (inp, out) = (self.output().cloned(), self.input().cloned());
        let name = inverse_name(&each_variant!(self, t => t.name.clone()));
        Ok(match self {
            Transform::Identity(_) => Transform::Identity(Identity {
                input: inp,
                output: out,
                name,
            }),
            Transform::MapAxis(t) => {
                let n = t.map_axis.len() as i64;
                let inv = (0..n)
                    .map(|i| t.map_axis.iter().position(|&x| x == i).unwrap() as i64)
                    .collect();
                Transform::MapAxis(MapAxis {
                    input: inp,
                    output: out,
                    name,
                    map_axis: inv,
                })
            }
            Transform::Translation(t) => Transform::Translation(Translation {
                input: inp,
                output: out,
                name,
                translation: t.translation.iter().map(|v| -v).collect(),
            }),
            Transform::Scale(t) => Transform::Scale(Scale {
                input: inp,
                output: out,
                name,
                scale: t.scale.iter().map(|v| 1.0 / v).collect(),
            }),
            Transform::Affine(t) => {
                let m = t.matrix()?;
                let tr = t.translation()?;
                let mi = invert_matrix(&m).ok_or_else(|| Error::value("Singular matrix"))?;
                let ti: Vec<f64> = matvec(&mi, &tr).iter().map(|v| -v).collect();
                let mut a = Affine::from_matrix_vector(&mi, &ti);
                a.input = inp;
                a.output = out;
                a.name = name;
                Transform::Affine(a)
            }
            Transform::Rotation(t) => {
                let mi = invert_matrix(t.rotation_matrix()?)
                    .ok_or_else(|| Error::value("Singular matrix"))?;
                Transform::Rotation(Rotation {
                    input: inp,
                    output: out,
                    name: None,
                    rotation: Some(mi),
                    path: None,
                })
            }
            Transform::Sequence(t) => Transform::Sequence(Sequence {
                input: inp,
                output: out,
                name,
                transformations: t
                    .transformations
                    .iter()
                    .rev()
                    .map(|x| x.get_inverse())
                    .collect::<Result<_>>()?,
            }),
            Transform::Bijection(t) => Transform::Bijection(Bijection {
                input: inp,
                output: out,
                name,
                forward: t.inverse.clone(),
                inverse: t.forward.clone(),
            }),
            Transform::ByDimension(t) => {
                if !self.has_inverse()? {
                    return Err(Error::value("Not all transformations have an inverse"));
                }
                Transform::ByDimension(ByDimension {
                    input: None,
                    output: None,
                    name: None,
                    transformations: t
                        .transformations
                        .iter()
                        .map(|x| {
                            Ok(ByDimensionTransform {
                                transformation: x.transformation.get_inverse()?,
                                input_axes: x.output_axes.clone(),
                                output_axes: x.input_axes.clone(),
                            })
                        })
                        .collect::<Result<_>>()?,
                })
            }
            Transform::Displacements(_) | Transform::Coordinates(_) | Transform::ProjectAxis(_) => {
                return Err(Error::NotImplemented(format!(
                    "inverse of {}",
                    self.type_name()
                )));
            }
        })
    }

    /// Apply the transform to a single point.
    pub fn transform_point(&self, point: &[f64]) -> Result<Vec<f64>> {
        Ok(match self {
            Transform::Identity(_) => point.to_vec(),
            Transform::MapAxis(t) => t.map_axis.iter().map(|&i| point[i as usize]).collect(),
            Transform::Translation(t) => {
                if point.len() != t.translation.len() {
                    return Err(Error::value(format!(
                        "Length of point ({}) does not match length of translation vector ({})",
                        point.len(),
                        t.translation.len()
                    )));
                }
                point
                    .iter()
                    .zip(&t.translation)
                    .map(|(p, t)| p + t)
                    .collect()
            }
            Transform::Scale(t) => {
                if point.len() != t.scale.len() {
                    return Err(Error::value(
                        "Length of point does not match length of scale vector",
                    ));
                }
                point.iter().zip(&t.scale).map(|(p, s)| p * s).collect()
            }
            Transform::Affine(t) => {
                let a = t.affine_matrix()?;
                if point.len() != a.len() {
                    return Err(Error::value(format!(
                        "Dimensionality of point ({}) does not match dimensionality of transform ({})",
                        point.len(),
                        a.len()
                    )));
                }
                let m = t.matrix()?;
                matvec(&m, point)
                    .iter()
                    .zip(t.translation()?)
                    .map(|(a, b)| a + b)
                    .collect()
            }
            Transform::Rotation(t) => matvec(t.rotation_matrix()?, point),
            Transform::Sequence(t) => {
                let mut p = point.to_vec();
                for x in &t.transformations {
                    p = x.transform_point(&p)?;
                }
                p
            }
            Transform::Bijection(t) => t.forward.transform_point(point)?,
            Transform::ByDimension(t) => {
                let mut out = point.to_vec();
                for x in &t.transformations {
                    let cin: Vec<f64> = x.input_axes.iter().map(|&i| point[i as usize]).collect();
                    let cout = x.transformation.transform_point(&cin)?;
                    for (c, &i) in cout.iter().zip(&x.output_axes) {
                        out[i as usize] = *c;
                    }
                }
                out
            }
            Transform::Displacements(_) | Transform::Coordinates(_) | Transform::ProjectAxis(_) => {
                return Err(Error::NotImplemented(format!(
                    "Transforming using a {} transform not yet implemented",
                    self.type_name()
                )));
            }
        })
    }

    /// Convert to an equivalent affine transform.
    pub fn as_affine(&self) -> Result<Affine> {
        let no = || Error::value(NO_AFFINE);
        let ident = |n: usize| -> Vec<Vec<f64>> {
            (0..n)
                .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
                .collect()
        };
        let mut a = match self {
            Transform::MapAxis(t) => {
                let n = t.map_axis.len();
                let mut m = vec![vec![0.0; n]; n];
                for (row, &col) in t.map_axis.iter().enumerate() {
                    m[row][col as usize] = 1.0;
                }
                Affine::from_matrix_vector(&m, &vec![0.0; n])
            }
            Transform::Translation(t) => {
                Affine::from_matrix_vector(&ident(t.translation.len()), &t.translation)
            }
            Transform::Scale(t) => {
                let mut m = ident(t.scale.len());
                for (i, s) in t.scale.iter().enumerate() {
                    m[i][i] = *s;
                }
                Affine::from_matrix_vector(&m, &vec![0.0; t.scale.len()])
            }
            Transform::Affine(t) => t.clone(),
            Transform::Rotation(t) => {
                let r = t.rotation_matrix()?;
                Affine::from_matrix_vector(r, &vec![0.0; r.len()])
            }
            Transform::Sequence(t) => {
                let n = self.ndim().ok_or_else(no)?;
                let mut m = ident(n);
                let mut tr = vec![0.0; n];
                for x in &t.transformations {
                    let na = x.as_affine()?;
                    let nm = na.matrix()?;
                    m = crate::linalg::matmul(&nm, &m);
                    tr = matvec(&nm, &tr)
                        .iter()
                        .zip(na.translation()?)
                        .map(|(a, b)| a + b)
                        .collect();
                }
                return Ok(Affine::from_matrix_vector(&m, &tr));
            }
            Transform::Bijection(t) => t.forward.as_affine()?,
            _ => return Err(no()),
        };
        a.input = self.input().cloned();
        a.output = self.output().cloned();
        a.name = self.name().map(String::from);
        Ok(a)
    }
}

/// Python `repr()` of a float as pydantic models print it.
fn py_float(x: f64) -> String {
    if x.is_nan() {
        "nan".into()
    } else if x.is_infinite() {
        if x > 0.0 { "inf".into() } else { "-inf".into() }
    } else {
        crate::pyjson::float_repr(x)
    }
}

/// Python `repr()` of a tuple.
fn py_tuple(items: Vec<String>) -> String {
    if items.len() == 1 {
        format!("({},)", items[0])
    } else {
        format!("({})", items.join(", "))
    }
}

fn opt_repr<T>(v: &Option<T>, f: impl Fn(&T) -> String) -> String {
    v.as_ref().map(f).unwrap_or_else(|| "None".into())
}

fn floats_repr(v: &[f64]) -> String {
    py_tuple(v.iter().map(|&x| py_float(x)).collect())
}

fn ints_repr(v: &[i64]) -> String {
    py_tuple(v.iter().map(|x| x.to_string()).collect())
}

fn matrix_repr(m: &[Vec<f64>]) -> String {
    py_tuple(m.iter().map(|r| floats_repr(r)).collect())
}

fn str_opt(v: &Option<String>) -> String {
    opt_repr(v, |s| crate::pyjson::repr_str(s))
}

impl CoordinateSystemIdentifier {
    /// pydantic `repr()`: `CoordinateSystemIdentifier(name='a', path=None)`.
    pub fn py_repr(&self) -> String {
        format!(
            "CoordinateSystemIdentifier(name={}, path={})",
            str_opt(&self.name),
            str_opt(&self.path)
        )
    }
}

impl ByDimensionTransform {
    /// pydantic `repr()`.
    pub fn py_repr(&self) -> String {
        format!(
            "ByDimensionTransform(transformation={}, inputAxes={}, outputAxes={})",
            self.transformation.py_repr(),
            ints_repr(&self.input_axes),
            ints_repr(&self.output_axes)
        )
    }
}

impl Transform {
    /// The ome-zarr-models class name (`Scale`, `MapAxis`, ...).
    pub fn class_name(&self) -> &'static str {
        match self {
            Transform::Identity(_) => "Identity",
            Transform::MapAxis(_) => "MapAxis",
            Transform::ProjectAxis(_) => "ProjectAxis",
            Transform::Translation(_) => "Translation",
            Transform::Scale(_) => "Scale",
            Transform::Affine(_) => "Affine",
            Transform::Rotation(_) => "Rotation",
            Transform::Sequence(_) => "Sequence",
            Transform::Displacements(_) => "Displacements",
            Transform::Coordinates(_) => "Coordinates",
            Transform::Bijection(_) => "Bijection",
            Transform::ByDimension(_) => "ByDimension",
        }
    }

    /// The model's fields in declaration order, each as its Python `repr()`.
    fn py_fields(&self) -> Vec<(&'static str, String)> {
        let (input, output, name) =
            each_variant!(self, t => (t.input.clone(), t.output.clone(), t.name.clone()));
        let mut f = vec![
            ("type", crate::pyjson::repr_str(self.type_name())),
            ("input", opt_repr(&input, |c| c.py_repr())),
            ("output", opt_repr(&output, |c| c.py_repr())),
            ("name", str_opt(&name)),
        ];
        match self {
            Transform::Identity(_) => {}
            Transform::MapAxis(t) => f.push(("mapAxis", ints_repr(&t.map_axis))),
            Transform::ProjectAxis(t) => {
                f.push((
                    "createdOutputs",
                    opt_repr(&t.created_outputs, |v| ints_repr(v)),
                ));
                f.push((
                    "droppedInputs",
                    opt_repr(&t.dropped_inputs, |v| ints_repr(v)),
                ));
            }
            Transform::Translation(t) => f.push(("translation", floats_repr(&t.translation))),
            Transform::Scale(t) => f.push(("scale", floats_repr(&t.scale))),
            Transform::Affine(t) => {
                f.push(("affine", opt_repr(&t.affine, |m| matrix_repr(m))));
                f.push(("path", str_opt(&t.path)));
            }
            Transform::Rotation(t) => {
                f.push(("rotation", opt_repr(&t.rotation, |m| matrix_repr(m))));
                f.push(("path", str_opt(&t.path)));
            }
            Transform::Sequence(t) => f.push((
                "transformations",
                py_tuple(t.transformations.iter().map(|x| x.py_repr()).collect()),
            )),
            Transform::Displacements(t) => {
                f.push(("path", crate::pyjson::repr_str(&t.path)));
                f.push(("interpolation", str_opt(&t.interpolation)));
            }
            Transform::Coordinates(t) => {
                f.push(("path", crate::pyjson::repr_str(&t.path)));
                f.push(("interpolation", str_opt(&t.interpolation)));
            }
            Transform::Bijection(t) => {
                f.push(("forward", t.forward.py_repr()));
                f.push(("inverse", t.inverse.py_repr()));
            }
            Transform::ByDimension(t) => f.push((
                "transformations",
                py_tuple(t.transformations.iter().map(|x| x.py_repr()).collect()),
            )),
        }
        f
    }

    /// pydantic `str()`: `type='scale' input=None output=None name=None scale=(1.0, 2.0)`.
    pub fn py_str(&self) -> String {
        self.py_fields()
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// pydantic `repr()`: `Scale(type='scale', input=None, ...)`.
    pub fn py_repr(&self) -> String {
        let fields: Vec<String> = self
            .py_fields()
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        format!("{}({})", self.class_name(), fields.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn roundtrip_json() {
        let v = json!({"type": "sequence", "input": {"path": "s0"}, "output": {"name": "physical"},
            "transformations": [{"type": "scale", "scale": [1.0, 2.0]}, {"type": "translation", "translation": [0.0, 0.5]}]});
        let t = Transform::from_json(&v).unwrap();
        assert_eq!(t.to_json(), v);
        assert_eq!(t.ndim(), Some(2));
        let p = t.transform_point(&[1.0, 1.0]).unwrap();
        assert_eq!(p, vec![1.0, 2.5]);
        let inv = t.get_inverse().unwrap();
        assert_eq!(inv.transform_point(&p).unwrap(), vec![1.0, 1.0]);
        assert_eq!(
            inv.input(),
            Some(&CoordinateSystemIdentifier::name("physical"))
        );
    }

    #[test]
    fn validation() {
        assert!(Transform::from_json(&json!({"type": "mapAxis", "mapAxis": [0, 2]})).is_err());
        assert!(Transform::from_json(&json!({"type": "affine"})).is_err());
        assert!(
            Transform::from_json(
                &json!({"type": "rotation", "rotation": [[1.0, 1.0], [0.0, 1.0]]})
            )
            .is_err()
        );
        assert!(
            Transform::from_json(&json!({"type": "scale", "scale": [1.0], "input": {"name": "a"}}))
                .is_err()
        );
        assert!(Transform::from_json(&json!({"type": "projectAxis"})).is_err());
        assert!(Transform::from_json(&json!({"type": "nope"})).is_err());
    }

    #[test]
    fn map_axis_inverse() {
        let t = Transform::from_json(&json!({"type": "mapAxis", "mapAxis": [1, 2, 0]})).unwrap();
        let p = t.transform_point(&[11.0, 12.0, 13.0]).unwrap();
        assert_eq!(p, vec![12.0, 13.0, 11.0]);
        assert_eq!(
            t.get_inverse().unwrap().transform_point(&p).unwrap(),
            vec![11.0, 12.0, 13.0]
        );
    }

    #[test]
    fn affine() {
        let t =
            Transform::from_json(&json!({"type": "translation", "translation": [1.0, -2.0, 3.4]}))
                .unwrap();
        let a = t.as_affine().unwrap();
        assert_eq!(a.affine.unwrap()[1], vec![0.0, 1.0, 0.0, -2.0]);
    }
}
