//! OME-NGFF multiscales metadata, version 0.6 (with conversion to and from
//! 0.4 / 0.5). Port of `ome_zarr_models.v06.multiscales` and the
//! `to_version` machinery of the v04 / v05 models.

use log::warn;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::transforms::{
    Axis, CoordinateSystem, CoordinateSystemIdentifier, Scale, Sequence, Transform, Translation,
};
use crate::error::{Error, Result};
use crate::zarr::JsonMap;

/// An element of `Multiscale.datasets`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Dataset {
    pub path: String,
    #[serde(rename = "coordinateTransformations")]
    pub coordinate_transformations: Vec<Transform>,
}

impl Dataset {
    /// Construct a dataset from a scale and optional translation
    /// (`Dataset.build`).
    pub fn build(
        path: &str,
        scale: &[f64],
        translation: Option<&[f64]>,
        output_cs: &str,
    ) -> Dataset {
        let input = Some(CoordinateSystemIdentifier::path(path));
        let output = Some(CoordinateSystemIdentifier::name(output_cs));
        let t = match translation {
            None => Transform::Scale(Scale {
                input,
                output,
                name: None,
                scale: scale.to_vec(),
            }),
            Some(tr) => Transform::Sequence(Sequence {
                input,
                output,
                name: None,
                transformations: vec![
                    Transform::Scale(Scale {
                        scale: scale.to_vec(),
                        ..Default::default()
                    }),
                    Transform::Translation(Translation {
                        translation: tr.to_vec(),
                        ..Default::default()
                    }),
                ],
            }),
        };
        Dataset {
            path: path.to_string(),
            coordinate_transformations: vec![t],
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.coordinate_transformations.len() != 1 {
            return Err(Error::value(
                "datasets must have exactly one coordinate transformation",
            ));
        }
        for t in &self.coordinate_transformations {
            if !matches!(
                t,
                Transform::Scale(_) | Transform::Identity(_) | Transform::Sequence(_)
            ) {
                return Err(Error::value(format!(
                    "dataset transformation must be scale, identity or sequence, got {}",
                    t.type_name()
                )));
            }
            t.validate()?;
            let input = t
                .input()
                .ok_or_else(|| Error::value("Transformations in datasets must have an input."))?;
            let ipath = input.path.as_ref().ok_or_else(|| {
                Error::value("Input for dataset transforms must have a path field.")
            })?;
            if ipath != &self.path {
                return Err(Error::value(format!(
                    "Input for a dataset transform must be the dataset array path: '{}'. Got '{ipath}' instead.",
                    self.path
                )));
            }
            let output = t
                .output()
                .ok_or_else(|| Error::value("Transformations in datasets must have an output."))?;
            if output.name.is_none() {
                return Err(Error::value(
                    "Output name for dataset transforms must be set.",
                ));
            }
        }
        if let Some(Transform::Sequence(s)) = self.coordinate_transformations.first()
            && s.transformations.len() == 2
            && let (Transform::Scale(sc), Transform::Translation(tr)) =
                (&s.transformations[0], &s.transformations[1])
            && sc.scale.len() != tr.translation.len()
        {
            return Err(Error::value(format!(
                "The length of the scale and translation vectors must be the same. Got {} and {}.",
                sc.scale.len(),
                tr.translation.len()
            )));
        }
        Ok(())
    }

    /// The scale vector of this dataset's transformation.
    pub fn scale(&self, ndim: usize) -> Option<Vec<f64>> {
        match self.coordinate_transformations.first()? {
            Transform::Scale(s) => Some(s.scale.clone()),
            Transform::Identity(_) => Some(vec![1.0; ndim]),
            Transform::Sequence(s) => match s.transformations.first()? {
                Transform::Scale(sc) => Some(sc.scale.clone()),
                _ => None,
            },
            _ => None,
        }
    }

    /// The translation vector, if any.
    pub fn translation(&self) -> Option<Vec<f64>> {
        match self.coordinate_transformations.first()? {
            Transform::Sequence(s) => s.transformations.iter().find_map(|t| match t {
                Transform::Translation(tr) => Some(tr.translation.clone()),
                _ => None,
            }),
            _ => None,
        }
    }

    /// The v0.4/0.5 form: `[{"type": "scale", ...}, {"type": "translation", ...}]`.
    pub fn to_v05_json(&self) -> Result<Value> {
        let mut scale = None;
        let mut translation = None;
        for t in &self.coordinate_transformations {
            match t {
                Transform::Scale(s) => scale = Some(s.scale.clone()),
                Transform::Sequence(seq) => {
                    for sub in &seq.transformations {
                        match sub {
                            Transform::Scale(s) => scale = Some(s.scale.clone()),
                            Transform::Translation(tr) => {
                                translation = Some(tr.translation.clone())
                            }
                            other => {
                                return Err(Error::value(format!(
                                    "Unsupported transform type: {}",
                                    other.type_name()
                                )));
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        let scale = scale.ok_or_else(|| Error::value("No scale transform found"))?;
        let mut cts = vec![json!({"type": "scale", "scale": scale})];
        if let Some(tr) = translation {
            cts.push(json!({"type": "translation", "translation": tr}));
        }
        Ok(json!({"path": self.path, "coordinateTransformations": cts}))
    }
}

/// An element of multiscales metadata (OME-NGFF 0.6).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Multiscale {
    #[serde(rename = "coordinateSystems")]
    pub coordinate_systems: Vec<CoordinateSystem>,
    pub datasets: Vec<Dataset>,
    #[serde(
        rename = "coordinateTransformations",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub coordinate_transformations: Option<Vec<Transform>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<Value>,
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub type_: Option<Value>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

fn check_length<T>(items: &[T], valid: &[usize], name: &str) -> Result<()> {
    if !valid.contains(&items.len()) {
        return Err(Error::value(format!(
            "Length of {name} ({}) not valid. Allowed values are {valid:?}.",
            items.len()
        )));
    }
    Ok(())
}

/// Validate axis counts and types (`_ensure_axis_types`), shared by 0.4-0.6.
pub fn validate_axis_types(types: &[Option<String>]) -> Result<()> {
    let count = |t: &str| types.iter().filter(|x| x.as_deref() == Some(t)).count();
    let n_space = count("space");
    check_length(&vec![(); n_space], &[2, 3], "space axes")?;
    check_length(&vec![(); count("time")], &[0, 1], "time axes")?;
    check_length(&vec![(); count("channel")], &[0, 1], "channel axes")?;
    let custom = types
        .iter()
        .filter(|t| !matches!(t.as_deref(), Some("space") | Some("time") | Some("channel")))
        .count();
    check_length(&vec![(); custom], &[0, 1], "custom axes")?;
    if !types[types.len() - n_space..]
        .iter()
        .all(|t| t.as_deref() == Some("space"))
    {
        return Err(Error::value(format!(
            "All space axes must be at the end of the axes list. Got axes with order: {types:?}."
        )));
    }
    if count("time") == 1 && types[0].as_deref() != Some("time") {
        return Err(Error::value(
            "Time axis must be at the beginning of axis list.",
        ));
    }
    Ok(())
}

impl Multiscale {
    /// Parse and validate from JSON.
    pub fn from_json(v: &Value) -> Result<Multiscale> {
        crate::models::pydantic::validate_model("Multiscale", v)?;
        let m: Multiscale = serde_json::from_value(v.clone())
            .map_err(|e| Error::value(format!("invalid multiscale: {e}")))?;
        m.validate()?;
        Ok(m)
    }

    /// Serialize (`model_dump` + `_recursive_pop_nones`).
    pub fn to_json(&self) -> Value {
        crate::utils::recursive_pop_nones(
            &serde_json::to_value(self).expect("serialize multiscale"),
        )
    }

    /// Validate (pydantic validators of `Multiscale` v0.6).
    pub fn validate(&self) -> Result<()> {
        if self.coordinate_systems.is_empty() {
            return Err(Error::value("coordinateSystems must have at least 1 item"));
        }
        if self.datasets.is_empty() {
            return Err(Error::value("datasets must have at least 1 item"));
        }
        for cs in &self.coordinate_systems {
            cs.validate()?;
        }
        for d in &self.datasets {
            d.validate()?;
        }
        let names: Vec<&String> = self.coordinate_systems.iter().map(|c| &c.name).collect();
        if super::transforms::unique_items(&names).is_err() {
            return Err(Error::value(format!(
                "Duplicate coordinate system names: {names:?}"
            )));
        }
        // ordered scales
        let mut scales: Vec<Vec<f64>> = vec![];
        for d in &self.datasets {
            match &d.coordinate_transformations[0] {
                Transform::Scale(s) => scales.push(s.scale.clone()),
                Transform::Identity(_) => {}
                Transform::Sequence(seq) => match seq.transformations.first() {
                    Some(Transform::Scale(s)) => scales.push(s.scale.clone()),
                    _ => {
                        return Err(Error::value(
                            "Sequence transformation inside a multiscales must start with a scale transform",
                        ));
                    }
                },
                _ => {
                    return Err(Error::value(
                        "Transform inside a multiscales must be a scale or sequence",
                    ));
                }
            }
        }
        for i in 0..scales.len().saturating_sub(1) {
            let (s1, s2) = (&scales[i], &scales[i + 1]);
            if !s1.iter().zip(s2).all(|(a, b)| a <= b) {
                return Err(Error::value(format!(
                    "Dataset {i} has a lower resolution (scales = {s1:?}) than dataset {} (scales = {s2:?}).",
                    i + 1
                )));
            }
        }
        // same output for all datasets
        let outputs: Vec<Option<&CoordinateSystemIdentifier>> = self
            .datasets
            .iter()
            .map(|d| d.coordinate_transformations[0].output())
            .collect();
        if outputs.iter().any(|o| *o != outputs[0]) {
            return Err(Error::value(format!(
                "All coordinate transform outputs inside 'datasets' must be identical. Got {outputs:?}"
            )));
        }
        let ndim = self.intrinsic_coordinate_system()?.ndim();
        for (i, d) in self.datasets.iter().enumerate() {
            for t in &d.coordinate_transformations {
                if let Some(n) = t.ndim()
                    && n != ndim
                {
                    return Err(Error::value(format!(
                        "The length of axes does not match the dimensionality of the {} transform in \
                         datasets[{i}].coordinateTransformations. Got {ndim} axes, but the {} transform has dimensionality {n}",
                        t.type_name(),
                        t.type_name()
                    )));
                }
            }
        }
        if let Some(cts) = &self.coordinate_transformations {
            for t in cts {
                t.validate()?;
                if let Some(n) = t.ndim()
                    && n != ndim
                {
                    return Err(Error::value(format!(
                        "The length of axes does not match the dimensionality of the {} transform in \
                         coordinateTransformations. Got {ndim} axes, but the {} transform has dimensionality {n}",
                        t.type_name(),
                        t.type_name()
                    )));
                }
            }
            let cs_names: Vec<&String> = self.coordinate_systems.iter().map(|c| &c.name).collect();
            for t in cts {
                let input = t.input().ok_or_else(|| {
                    Error::value("Transformations in coordinateTransformations must have an input.")
                })?;
                let output = t.output().ok_or_else(|| {
                    Error::value(
                        "Transformations in coordinateTransformations must have an output.",
                    )
                })?;
                let iname = input.name.as_ref().ok_or_else(|| {
                    Error::value("Input for coordinate transformations must provide a coordinate system name")
                })?;
                let oname = output.name.as_ref().ok_or_else(|| {
                    Error::value("Output for coordinate transformations must provide a coordinate system name")
                })?;
                if !cs_names.contains(&iname) {
                    return Err(Error::value(format!(
                        "Invalid input in coordinate transformation '{:?}': {iname}. Must be one of {cs_names:?}.",
                        t.name()
                    )));
                }
                if output.path.as_deref().unwrap_or("").is_empty() && !cs_names.contains(&oname) {
                    return Err(Error::value(format!(
                        "Invalid output in coordinate transformation '{:?}': {oname}. Must be one of {cs_names:?} or the path field must be set.",
                        t.name()
                    )));
                }
                if let Some(p) = &output.path
                    && p.contains("..")
                {
                    return Err(Error::value(format!(
                        "Output paths in coordinate transformations must be downpointing. Got '{p}'."
                    )));
                }
            }
        }
        for cs in &self.coordinate_systems {
            check_length(&cs.axes, &[2, 3, 4, 5], "axes")?;
            let types: Vec<Option<String>> = cs.axes.iter().map(|a| a.type_.clone()).collect();
            validate_axis_types(&types)?;
        }
        Ok(())
    }

    /// The physical coordinate system (output of all dataset transforms).
    pub fn intrinsic_coordinate_system(&self) -> Result<&CoordinateSystem> {
        let output = self
            .datasets
            .first()
            .and_then(|d| d.coordinate_transformations.first())
            .and_then(|t| t.output())
            .ok_or_else(|| Error::value("Output coordinate system in datasets[0] not found."))?;
        self.coordinate_systems
            .iter()
            .find(|cs| Some(&cs.name) == output.name.as_ref())
            .ok_or_else(|| {
                Error::value(format!(
                    "No coordinate systems have the name '{:?}'",
                    output.name
                ))
            })
    }

    /// Dimensionality.
    pub fn ndim(&self) -> Result<usize> {
        Ok(self.intrinsic_coordinate_system()?.ndim())
    }

    /// Name as a string if set.
    pub fn name_str(&self) -> Option<String> {
        match &self.name {
            Some(Value::String(s)) => Some(s.clone()),
            Some(Value::Null) | None => None,
            Some(v) => Some(v.to_string()),
        }
    }

    /// Convert to the v0.5 JSON form (`to_version("0.5").model_dump(by_alias=True)`
    /// followed by `_recursive_pop_nones`).
    pub fn to_v05_json(&self) -> Result<Value> {
        let axes: Vec<Value> = self
            .intrinsic_coordinate_system()?
            .axes
            .iter()
            .map(|a| json!({"name": a.name, "type": a.type_, "unit": a.unit}))
            .collect();
        let datasets: Vec<Value> = self
            .datasets
            .iter()
            .map(|d| d.to_v05_json())
            .collect::<Result<_>>()?;
        if self.coordinate_transformations.is_some() {
            warn!(
                "Coordinate transformations defined in multiscales > coordinateTransformations can currently \
                 not be converted to v0.5, as they are not supported in this version."
            );
        }
        let v = json!({
            "axes": axes,
            "datasets": datasets,
            "metadata": self.metadata,
            "name": self.name,
            "type": self.type_,
        });
        Ok(crate::utils::recursive_pop_nones(&v))
    }

    /// Convert to the v0.4 JSON form (same structure as 0.5).
    pub fn to_v04_json(&self) -> Result<Value> {
        self.to_v05_json()
    }

    /// Convert v0.4 or v0.5 multiscales JSON to v0.6, as
    /// `Multiscale.model_validate(json).to_version("0.6")`.
    pub fn from_v05_json(v: &Value) -> Result<Multiscale> {
        crate::models::pydantic::validate_model("MultiscaleV05", v)?;
        Self::from_v05_json_opt(v, "physical", "output")
    }

    /// See [`Multiscale::from_v05_json`]; with explicit coordinate system names.
    pub fn from_v05_json_opt(
        v: &Value,
        default_cs_name: &str,
        output_cs_name: &str,
    ) -> Result<Multiscale> {
        let obj = v
            .as_object()
            .ok_or_else(|| Error::value("multiscales metadata must be an object"))?;
        let axes_json = obj
            .get("axes")
            .ok_or_else(|| Error::value("Field required: axes"))?;
        let axes_arr = axes_json
            .as_array()
            .ok_or_else(|| Error::value("axes must be a list"))?;
        let axes: Vec<Axis> = axes_arr
            .iter()
            .map(|a| match a {
                Value::Object(o) => Ok(Axis {
                    name: o.get("name").and_then(|n| n.as_str()).map(String::from),
                    type_: o.get("type").and_then(|n| n.as_str()).map(String::from),
                    unit: o.get("unit").cloned().filter(|u| !u.is_null()),
                    ..Default::default()
                }),
                _ => Err(Error::value(format!("invalid axis: {a}"))),
            })
            .collect::<Result<_>>()?;
        check_length(&axes, &[2, 3, 4, 5], "axes")?;
        let types: Vec<Option<String>> = axes.iter().map(|a| a.type_.clone()).collect();
        validate_axis_types(&types)?;
        let names: Vec<&Option<String>> = axes.iter().map(|a| &a.name).collect();
        if super::transforms::unique_items(&names).is_err() {
            return Err(Error::value(format!(
                "Axis names must be unique. Got {names:?}"
            )));
        }
        let ndim = axes.len();
        let ds_json = obj
            .get("datasets")
            .and_then(|d| d.as_array())
            .ok_or_else(|| Error::value("Field required: datasets"))?;
        if ds_json.is_empty() {
            return Err(Error::value("datasets must have at least 1 item"));
        }
        let mut datasets = vec![];
        let mut vector_scales: Vec<Vec<f64>> = vec![];
        for d in ds_json {
            let path = d
                .get("path")
                .and_then(|p| p.as_str())
                .ok_or_else(|| Error::value(format!("dataset has no path: {d}")))?;
            let cts = d
                .get("coordinateTransformations")
                .and_then(|c| c.as_array())
                .ok_or_else(|| {
                    Error::value(format!("dataset has no coordinateTransformations: {d}"))
                })?;
            let (scale, translation) = parse_v05_transforms(cts)?;
            if scale.len() != ndim {
                return Err(Error::value(format!(
                    "The length of axes does not match the dimensionality of the scale transform in datasets. \
                     Got {ndim} axes, but the scale transform has dimensionality {}",
                    scale.len()
                )));
            }
            if let Some(t) = &translation
                && t.len() != scale.len()
            {
                return Err(Error::value(format!(
                    "The transforms have inconsistent dimensionality. Got transforms with dimensionality = ({}, {}).",
                    scale.len(),
                    t.len()
                )));
            }
            vector_scales.push(scale.clone());
            datasets.push(Dataset::build(
                path,
                &scale,
                translation.as_deref(),
                default_cs_name,
            ));
        }
        for i in 0..vector_scales.len().saturating_sub(1) {
            let (s1, s2) = (&vector_scales[i], &vector_scales[i + 1]);
            if !s1.iter().zip(s2).all(|(a, b)| a <= b) {
                return Err(Error::value(format!(
                    "Dataset {i} has a lower resolution (scales = {s1:?}) than dataset {} (scales = {s2:?}).",
                    i + 1
                )));
            }
        }
        let mut cs = vec![CoordinateSystem {
            name: default_cs_name.into(),
            axes: axes.clone(),
            extra: JsonMap::new(),
        }];
        let mut top = None;
        if let Some(ct) = obj
            .get("coordinateTransformations")
            .filter(|c| !c.is_null())
        {
            let cts = ct
                .as_array()
                .ok_or_else(|| Error::value("coordinateTransformations must be a list"))?;
            let (scale, translation) = parse_v05_transforms(cts)?;
            let t = match translation {
                None => Transform::Scale(Scale {
                    scale,
                    ..Default::default()
                }),
                Some(tr) => Transform::Sequence(Sequence {
                    transformations: vec![
                        Transform::Scale(Scale {
                            scale,
                            ..Default::default()
                        }),
                        Transform::Translation(Translation {
                            translation: tr,
                            ..Default::default()
                        }),
                    ],
                    ..Default::default()
                }),
            };
            top = Some(vec![t.with_io(
                Some(CoordinateSystemIdentifier::name(default_cs_name)),
                Some(CoordinateSystemIdentifier::name(output_cs_name)),
            )]);
            cs.push(CoordinateSystem {
                name: output_cs_name.into(),
                axes: axes
                    .iter()
                    .map(|a| Axis {
                        name: a.name.clone(),
                        type_: a.type_.clone(),
                        ..Default::default()
                    })
                    .collect(),
                extra: JsonMap::new(),
            });
        }
        let m = Multiscale {
            coordinate_systems: cs,
            datasets,
            coordinate_transformations: top,
            metadata: obj.get("metadata").cloned().filter(|v| !v.is_null()),
            name: obj.get("name").cloned().filter(|v| !v.is_null()),
            type_: obj.get("type").cloned().filter(|v| !v.is_null()),
            extra: JsonMap::new(),
        };
        m.validate()?;
        Ok(m)
    }

    /// [`Multiscale::from_v05_json`] for v0.4 metadata (validated with the v0.4 model).
    pub fn from_v04_json(v: &Value) -> Result<Multiscale> {
        crate::models::pydantic::validate_model("MultiscaleV04", v)?;
        Self::from_v05_json_opt(v, "physical", "output")
    }

    /// Get a coordinate system by name.
    pub fn coordinate_system(&self, name: &str) -> Option<&CoordinateSystem> {
        self.coordinate_systems.iter().find(|c| c.name == name)
    }
}

/// Parse `[scale]` or `[scale, translation]` v0.4/0.5 transforms (swapping a
/// reversed `[translation, scale]` pair as ome-zarr-models does).
fn parse_v05_transforms(cts: &[Value]) -> Result<(Vec<f64>, Option<Vec<f64>>)> {
    check_length(cts, &[1, 2], "transforms")?;
    let types: Vec<&str> = cts
        .iter()
        .map(|t| t.get("type").and_then(|x| x.as_str()).unwrap_or(""))
        .collect();
    let vec_of = |t: &Value, k: &str| -> Result<Vec<f64>> {
        let a = t.get(k).and_then(|x| x.as_array()).ok_or_else(|| {
            if t.get("path").is_some() {
                Error::NotImplemented(format!(
                    "Conversion of path-based {k} transforms is not currently supported."
                ))
            } else {
                Error::value(format!("Missing {k} in {t}"))
            }
        })?;
        a.iter()
            .map(|x| {
                x.as_f64()
                    .ok_or_else(|| Error::value(format!("'{k}' values must all be numbers")))
            })
            .collect()
    };
    match types.as_slice() {
        ["scale"] => Ok((vec_of(&cts[0], "scale")?, None)),
        ["scale", "translation"] => Ok((
            vec_of(&cts[0], "scale")?,
            Some(vec_of(&cts[1], "translation")?),
        )),
        // ome-zarr-models has a swap for `[translation, scale]`, but it checks
        // the parsed base `Transform` class, so it never applies: the order
        // is rejected.
        other => Err(Error::value(format!(
            "Expected a scale or (scale, translation) transform. Got {other:?} instead."
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v05() -> Value {
        json!({
            "axes": [{"name": "z", "type": "space"}, {"name": "y", "type": "space", "unit": "micrometer"}, {"name": "x", "type": "space"}],
            "datasets": [
                {"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 1]}]},
                {"path": "s1", "coordinateTransformations": [{"type": "scale", "scale": [1, 2, 2]}, {"type": "translation", "translation": [0, 0.5, 0.5]}]}
            ],
            "name": "img"
        })
    }

    #[test]
    fn v05_roundtrip() {
        let m = Multiscale::from_v05_json(&v05()).unwrap();
        assert_eq!(m.ndim().unwrap(), 3);
        assert_eq!(m.intrinsic_coordinate_system().unwrap().name, "physical");
        let back = m.to_v05_json().unwrap();
        assert_eq!(
            back["datasets"][1]["coordinateTransformations"][1]["translation"],
            json!([0.0, 0.5, 0.5])
        );
        assert_eq!(back["axes"][1]["unit"], "micrometer");
        assert!(back["axes"][0].get("unit").is_none());
        let m2 = Multiscale::from_json(&m.to_json()).unwrap();
        assert_eq!(m, m2);
    }

    #[test]
    fn bad_order() {
        let mut v = v05();
        v["datasets"][1]["coordinateTransformations"][0]["scale"] = json!([1, 0.5, 0.5]);
        assert!(Multiscale::from_v05_json(&v).is_err());
    }

    #[test]
    fn bad_axes() {
        let mut v = v05();
        v["axes"][0]["type"] = json!("time");
        v["axes"] = json!([{"name": "y", "type": "space"}, {"name": "t", "type": "time"}, {"name": "x", "type": "space"}]);
        assert!(Multiscale::from_v05_json(&v).is_err());
    }
}
