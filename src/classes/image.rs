//! Class-based interface: [`OmeZarrImage`] (single scale),
//! [`OmeZarrMultiscale`] (image pyramid with labels and omero metadata) and
//! [`OmeZarrLabels`] (label pyramid).

use std::collections::BTreeMap;

use log::warn;
use serde_json::{Value, json};

use crate::array::LazyArray;
use crate::error::{Error, Result};
use crate::format::Format;
use crate::models::image_label::{Color, Property, Source};
use crate::models::omero::{Channel, Window};
use crate::models::transforms::{
    Axis, CoordinateSystem, CoordinateSystemIdentifier, Scale, Sequence, Transform, Translation,
};
use crate::models::{Dataset, ImageLabel, Multiscale, Omero};
use crate::scale::{Methods, ScaleFactors, build_pyramid};
use crate::utils::recursive_pop_nones;
use crate::writer::{
    DelayedWrite, GroupLike, StorageOptionsArg, check_group_fmt, write_pyramid_to_zarr,
};
use crate::zarr::{Group, JsonMap, Mode, NodeKind};

/// Axis types that are discrete.
pub const DISCRETE_DIMS: [&str; 3] = ["coordinate", "displacement", "channel"];

/// Default colours for channels.
pub const DEFAULT_COLORS: [&str; 10] = [
    "00FFFF", // cyan
    "FF00FF", // magenta
    "FFFF00", // yellow
    "FF0000", // red
    "00FF00", // green
    "0000FF", // blue
    "FFFFFF", // white
    "FFA500", // orange
    "800080", // purple
    "008000", // dark green
];

/// An ordered map from axis name to physical scale.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ScaleMap(pub Vec<(String, f64)>);

impl ScaleMap {
    pub fn from_pairs(v: Vec<(String, f64)>) -> ScaleMap {
        ScaleMap(v)
    }
    /// All axes with scale 1.0.
    pub fn ones(dims: &[String]) -> ScaleMap {
        ScaleMap(dims.iter().map(|d| (d.clone(), 1.0)).collect())
    }
    pub fn get(&self, k: &str) -> Option<f64> {
        self.0.iter().find(|(d, _)| d == k).map(|(_, v)| *v)
    }
    pub fn keys(&self) -> Vec<String> {
        self.0.iter().map(|(k, _)| k.clone()).collect()
    }
    pub fn values(&self) -> Vec<f64> {
        self.0.iter().map(|(_, v)| *v).collect()
    }
    pub fn contains(&self, k: &str) -> bool {
        self.0.iter().any(|(d, _)| d == k)
    }
}

impl<const N: usize> From<[(&str, f64); N]> for ScaleMap {
    fn from(v: [(&str, f64); N]) -> Self {
        ScaleMap(v.iter().map(|(k, v)| (k.to_string(), *v)).collect())
    }
}

/// OME-Zarr versions supported by the class-based writer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Version {
    V04,
    V05,
    #[default]
    V06,
}

impl Version {
    pub fn as_str(&self) -> &'static str {
        match self {
            Version::V04 => "0.4",
            Version::V05 => "0.5",
            Version::V06 => "0.6",
        }
    }
    pub fn parse(s: &str) -> Result<Version> {
        match s {
            "0.4" => Ok(Version::V04),
            "0.5" => Ok(Version::V05),
            "0.6" => Ok(Version::V06),
            other => Err(Error::value(format!(
                "Unsupported OME-Zarr version: {other}"
            ))),
        }
    }
    /// The version matching a [`Format`] (0.4 or 0.5).
    pub fn from_format(f: Format) -> Version {
        match f {
            Format::V05 => Version::V05,
            _ => Version::V04,
        }
    }
    fn format(&self) -> Format {
        match self {
            Version::V04 => Format::V04,
            _ => Format::V05,
        }
    }
}

/// Single-scale image representation with metadata.
#[derive(Debug, Clone)]
pub struct OmeZarrImage {
    pub data: LazyArray,
    pub axes: Vec<String>,
    pub scale: ScaleMap,
    pub axes_units: Option<BTreeMap<String, String>>,
    pub axes_types: BTreeMap<String, Option<String>>,
    pub name: String,
}

impl OmeZarrImage {
    /// Validate and build an image (`OMEZarrImage.__post_init__`).
    pub fn new(
        data: LazyArray,
        axes: &[String],
        scale: Option<ScaleMap>,
        axes_units: Option<BTreeMap<String, String>>,
        axes_types: Option<BTreeMap<String, Option<String>>>,
        name: &str,
    ) -> Result<OmeZarrImage> {
        if axes.len() != data.ndim() {
            return Err(Error::value(format!(
                "Number of dimensions in data ({}) does not match number of dims ({})",
                data.ndim(),
                axes.len()
            )));
        }
        let scale = scale.unwrap_or_else(|| ScaleMap::ones(axes));
        let invalid: Vec<&String> = scale
            .0
            .iter()
            .map(|(k, _)| k)
            .filter(|k| !axes.contains(k))
            .collect();
        if !invalid.is_empty() {
            return Err(Error::value(format!(
                "Scale contains invalid ax(i)(e)s: {}. Valid axes are: {}",
                crate::pyjson::set_repr(&invalid.iter().map(|s| s.to_string()).collect::<Vec<_>>()),
                crate::pyjson::set_repr(axes)
            )));
        }
        let missing: Vec<&String> = axes.iter().filter(|a| !scale.contains(a)).collect();
        if !missing.is_empty() {
            warn!(
                "Scale value not provided for ax(i)(e)s '{missing:?}'. Using default scale of 1.0."
            );
        }
        let scale = ScaleMap(
            axes.iter()
                .map(|d| (d.clone(), scale.get(d).unwrap_or(1.0)))
                .collect(),
        );
        let axes_types = match axes_types {
            Some(t) if !t.is_empty() => t,
            _ => axes
                .iter()
                .map(|d| {
                    let t = match d.as_str() {
                        "t" => Some("time"),
                        "c" => Some("channel"),
                        "z" | "y" | "x" => Some("space"),
                        _ => None,
                    };
                    (d.clone(), t.map(String::from))
                })
                .collect(),
        };
        Ok(OmeZarrImage {
            data,
            axes: axes.to_vec(),
            scale,
            axes_units,
            axes_types,
            name: name.to_string(),
        })
    }

    /// Convenience constructor from an axes string like `"czyx"`.
    pub fn from_axes(data: LazyArray, axes: &str) -> Result<OmeZarrImage> {
        let axes: Vec<String> = axes.chars().map(|c| c.to_string()).collect();
        OmeZarrImage::new(data, &axes, None, None, None, "image")
    }

    /// Builder-style: set the scale.
    pub fn with_scale(mut self, scale: ScaleMap) -> Result<OmeZarrImage> {
        let s = OmeZarrImage::new(
            self.data.clone(),
            &self.axes,
            Some(scale),
            self.axes_units.clone(),
            Some(self.axes_types.clone()),
            &self.name,
        )?;
        self.scale = s.scale;
        Ok(self)
    }

    /// Builder-style: set the name.
    pub fn with_name(mut self, name: &str) -> OmeZarrImage {
        self.name = name.to_string();
        self
    }

    /// Builder-style: set axis units.
    pub fn with_units(mut self, units: &[(&str, &str)]) -> OmeZarrImage {
        self.axes_units = Some(
            units
                .iter()
                .map(|(a, u)| (a.to_string(), u.to_string()))
                .collect(),
        );
        self
    }

    /// Builder-style: set axis types.
    pub fn with_types(mut self, types: &[(&str, &str)]) -> OmeZarrImage {
        self.axes_types = types
            .iter()
            .map(|(a, t)| (a.to_string(), Some(t.to_string())))
            .collect();
        self
    }
}

/// Shared state of multiscale image and label pyramids.
#[derive(Debug, Clone)]
pub struct MultiscaleBase {
    pub images: Vec<OmeZarrImage>,
    /// The OME-NGFF 0.6 multiscales metadata.
    pub metadata: Multiscale,
    pub name: String,
}

impl MultiscaleBase {
    /// Build the pyramid and metadata.
    pub fn new(
        image: OmeZarrImage,
        scale_factors: Option<ScaleFactors>,
        coordinate_transformations: Option<Vec<Transform>>,
        coordinate_systems: Option<Vec<CoordinateSystem>>,
        method: Option<Methods>,
        default_cs_name: &str,
    ) -> Result<MultiscaleBase> {
        let scale_factors = scale_factors.unwrap_or_default();
        let method = method.unwrap_or(Methods::Resize);
        let pyramid = build_pyramid(&image.data, &scale_factors, &image.axes, method)?;
        let full = image.data.shape().to_vec();
        let mut scales: Vec<Vec<(String, f64)>> = vec![];
        for level in &pyramid {
            let s: Vec<(String, f64)> = image
                .axes
                .iter()
                .zip(full.iter().zip(level.shape()))
                .map(|(d, (&f, &l))| {
                    let r = f as f64 / l as f64;
                    (d.clone(), image.scale.get(d).map(|s| r * s).unwrap_or(1.0))
                })
                .collect();
            scales.push(s);
        }
        let translations: Vec<Vec<f64>> = scales
            .iter()
            .map(|s| {
                s.iter()
                    .zip(&scales[0])
                    .map(|((_, v), (_, v0))| v / 2.0 - v0 / 2.0)
                    .collect()
            })
            .collect();
        let mut images = vec![];
        let mut datasets = vec![];
        for (idx, (level, level_scale)) in pyramid.into_iter().zip(&scales).enumerate() {
            images.push(OmeZarrImage {
                data: level,
                axes: image.axes.clone(),
                scale: ScaleMap(level_scale.clone()),
                axes_units: image.axes_units.clone(),
                axes_types: image.axes_types.clone(),
                name: image.name.clone(),
            });
            let path = format!("s{idx}");
            let t = Transform::Sequence(Sequence {
                input: Some(CoordinateSystemIdentifier::path(&path)),
                output: Some(CoordinateSystemIdentifier::name(default_cs_name)),
                name: None,
                transformations: vec![
                    Transform::Scale(Scale {
                        scale: level_scale.iter().map(|(_, v)| *v).collect(),
                        ..Default::default()
                    }),
                    Transform::Translation(Translation {
                        translation: translations[idx].clone(),
                        ..Default::default()
                    }),
                ],
            });
            datasets.push(Dataset {
                path,
                coordinate_transformations: vec![t],
            });
        }
        let units = image.axes_units.clone().unwrap_or_default();
        let axes: Vec<Axis> = image
            .axes
            .iter()
            .map(|d| {
                let t = image.axes_types.get(d).cloned().flatten();
                let discrete = t.as_deref().is_some_and(|t| DISCRETE_DIMS.contains(&t));
                Axis {
                    name: Some(d.clone()),
                    type_: t,
                    unit: units.get(d).map(|u| json!(u)),
                    discrete: Some(discrete),
                    ..Default::default()
                }
            })
            .collect();
        let mut cs = vec![CoordinateSystem::new(default_cs_name, axes)?];
        cs.extend(coordinate_systems.unwrap_or_default());
        if let Some(tfs) = &coordinate_transformations {
            let cs_names: Vec<&String> = cs.iter().map(|c| &c.name).collect();
            for (idx, tf) in tfs.iter().enumerate() {
                let input = tf.input().ok_or_else(|| {
                    Error::value(format!("Coordinate transformation at index {idx} is missing input coordinate system: {}", tf.py_str()))
                })?;
                let output = tf.output().ok_or_else(|| {
                    Error::value(format!("Coordinate transformation at index {idx} is missing output coordinate system: {}", tf.py_str()))
                })?;
                let dn = Some(default_cs_name.to_string());
                if input.name != dn && output.name != dn {
                    return Err(Error::value(format!(
                        "Coordinate transformation at index {idx} must have either input or output coordinate system name \
                         matching the default coordinate system name '{default_cs_name}': {}",
                        tf.py_str()
                    )));
                }
                let in_ok = input.name.as_ref().is_some_and(|n| cs_names.contains(&n));
                let out_ok = output.name.as_ref().is_some_and(|n| cs_names.contains(&n));
                if !in_ok || !out_ok {
                    return Err(Error::value(format!(
                        "Coordinate transformation {} has input coordinate system name '{}' or output coordinate \
                         system name '{}' which does not match any of the defined coordinate systems: {}",
                        tf.py_str(),
                        input.name.as_deref().unwrap_or("None"),
                        output.name.as_deref().unwrap_or("None"),
                        crate::pyjson::str_list_repr(
                            &cs_names.iter().map(|s| s.to_string()).collect::<Vec<_>>()
                        )
                    )));
                }
            }
        }
        let metadata = Multiscale {
            coordinate_systems: cs,
            datasets,
            coordinate_transformations,
            metadata: None,
            name: Some(json!(image.name)),
            type_: None,
            extra: JsonMap::new(),
        };
        metadata.validate()?;
        Ok(MultiscaleBase {
            images,
            metadata,
            name: image.name.clone(),
        })
    }

    /// Get a coordinate system defined on this multiscale by its name.
    pub fn get_local_coordinate_system(&self, name: &str) -> Option<&CoordinateSystem> {
        self.metadata.coordinate_system(name)
    }

    /// Write the image pyramid and the multiscales metadata.
    fn write_base(
        &self,
        group: GroupLike,
        storage_options: Option<&StorageOptionsArg>,
        version: Version,
        compute: bool,
        overwrite: bool,
    ) -> Result<(Group, Vec<DelayedWrite>)> {
        let store_exists = match &group {
            GroupLike::Group(_) => true,
            GroupLike::Path(p) => std::path::Path::new(p).exists(),
        };
        if store_exists && !overwrite {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "store exists but overwrite=False",
            )));
        }
        if overwrite
            && let GroupLike::Path(p) = &group
            && std::path::Path::new(p).exists()
        {
            std::fs::remove_dir_all(p)?;
        }
        let fmt = version.format();
        let (group, fmt) = check_group_fmt(group, Some(fmt), "a")?;
        let pyramid: Vec<LazyArray> = self.images.iter().map(|i| i.data.clone()).collect();
        let cs = self.metadata.intrinsic_coordinate_system()?;
        let axes: Vec<String> = cs.axis_names();
        let delayed = write_pyramid_to_zarr(
            &pyramid,
            &group,
            fmt,
            &self.images[0].scale,
            &axes,
            None,
            None,
            storage_options,
            Some(&self.name),
            compute,
            None,
        )?;
        // normalised dataset paths s0, s1, ...
        let mut md = self.metadata.clone();
        for (idx, ds) in md.datasets.iter_mut().enumerate() {
            let path = format!("s{idx}");
            let t = &ds.coordinate_transformations[0];
            let input = t.input().ok_or_else(|| {
                Error::value(format!(
                    "Transform input cannot be None in dataset {idx} transform {}",
                    t.py_str()
                ))
            })?;
            let mut input = input.clone();
            input.path = Some(path.clone());
            ds.coordinate_transformations = vec![t.with_io(Some(input), t.output().cloned())];
            ds.path = path;
        }
        match version {
            Version::V04 => {
                let mut d = md.to_v04_json()?;
                d.as_object_mut()
                    .unwrap()
                    .insert("version".into(), json!("0.4"));
                group.set_attr("multiscales", json!([d]))?;
            }
            Version::V05 => {
                group.set_attr(
                    "ome",
                    json!({"version": "0.5", "multiscales": [md.to_v05_json()?]}),
                )?;
            }
            Version::V06 => {
                group.set_attr(
                    "ome",
                    json!({"version": "0.6", "multiscales": [md.to_json()]}),
                )?;
            }
        }
        Ok((group, delayed))
    }
}

/// Labels passed to an [`OmeZarrMultiscale`].
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum LabelsArg {
    One(OmeZarrLabels),
    List(Vec<OmeZarrLabels>),
    Map(Vec<(String, OmeZarrLabels)>),
}

fn parse_labels(labels: Option<LabelsArg>) -> Option<Vec<(String, OmeZarrLabels)>> {
    match labels? {
        LabelsArg::One(l) => Some(vec![(l.base.name.clone(), l)]),
        LabelsArg::List(v) => Some(v.into_iter().map(|l| (l.base.name.clone(), l)).collect()),
        LabelsArg::Map(m) => Some(m),
    }
}

/// Channel colour given as hex or RGB(A).
#[derive(Debug, Clone, PartialEq)]
pub enum ChannelColor {
    Hex(String),
    Rgb(Vec<u8>),
}

impl From<&str> for ChannelColor {
    fn from(s: &str) -> Self {
        ChannelColor::Hex(s.to_string())
    }
}

/// Container for multiscale image pyramid with OME-Zarr metadata.
#[derive(Debug, Clone)]
pub struct OmeZarrMultiscale {
    pub base: MultiscaleBase,
    labels: Option<Vec<(String, OmeZarrLabels)>>,
    pub omero: Option<Omero>,
}

/// Builder for [`OmeZarrMultiscale`] (the keyword arguments of the Python class).
#[derive(Debug, Clone)]
pub struct MultiscaleBuilder {
    image: OmeZarrImage,
    scale_factors: Option<ScaleFactors>,
    method: Option<Methods>,
    coordinate_transformations: Option<Vec<Transform>>,
    coordinate_systems: Option<Vec<CoordinateSystem>>,
    default_cs_name: String,
    labels: Option<LabelsArg>,
    channel_names: Option<Vec<String>>,
    channel_colors: Option<Vec<ChannelColor>>,
    contrast_limits: Option<Vec<(f64, f64)>>,
}

impl MultiscaleBuilder {
    pub fn scale_factors(mut self, s: Option<ScaleFactors>) -> Self {
        self.scale_factors = s;
        self
    }
    /// No downsampling (a single level).
    pub fn no_downsampling(mut self) -> Self {
        self.scale_factors = Some(ScaleFactors::Ints(vec![]));
        self
    }
    pub fn method(mut self, m: Option<Methods>) -> Self {
        self.method = m;
        self
    }
    pub fn coordinate_transformations(mut self, t: Vec<Transform>) -> Self {
        self.coordinate_transformations = Some(t);
        self
    }
    pub fn coordinate_systems(mut self, cs: Vec<CoordinateSystem>) -> Self {
        self.coordinate_systems = Some(cs);
        self
    }
    pub fn default_coordinate_system_name(mut self, n: &str) -> Self {
        self.default_cs_name = n.into();
        self
    }
    pub fn labels(mut self, l: LabelsArg) -> Self {
        self.labels = Some(l);
        self
    }
    pub fn channel_names(mut self, n: &[&str]) -> Self {
        self.channel_names = Some(n.iter().map(|s| s.to_string()).collect());
        self
    }
    pub fn channel_colors(mut self, c: Vec<ChannelColor>) -> Self {
        self.channel_colors = Some(c);
        self
    }
    pub fn contrast_limits(mut self, c: Vec<(f64, f64)>) -> Self {
        self.contrast_limits = Some(c);
        self
    }
    pub fn build(self) -> Result<OmeZarrMultiscale> {
        let base = MultiscaleBase::new(
            self.image,
            self.scale_factors,
            self.coordinate_transformations,
            self.coordinate_systems,
            self.method,
            &self.default_cs_name,
        )?;
        let mut ms = OmeZarrMultiscale {
            base,
            labels: parse_labels(self.labels),
            omero: None,
        };
        ms.parse_omero_metadata(
            self.channel_names,
            self.channel_colors,
            self.contrast_limits,
        )?;
        Ok(ms)
    }
}

impl OmeZarrMultiscale {
    /// Start building a multiscale from a single-scale image.
    pub fn builder(image: OmeZarrImage) -> MultiscaleBuilder {
        MultiscaleBuilder {
            image,
            scale_factors: None,
            method: Some(Methods::Resize),
            coordinate_transformations: None,
            coordinate_systems: None,
            default_cs_name: "physical".into(),
            labels: None,
            channel_names: None,
            channel_colors: None,
            contrast_limits: None,
        }
    }

    /// `OMEZarrMultiscale(image)` with default arguments.
    pub fn new(image: OmeZarrImage) -> Result<OmeZarrMultiscale> {
        Self::builder(image).build()
    }

    /// List of images at each pyramid level.
    pub fn images(&self) -> &[OmeZarrImage] {
        &self.base.images
    }

    pub fn metadata(&self) -> &Multiscale {
        &self.base.metadata
    }

    pub fn name(&self) -> &str {
        &self.base.name
    }

    /// Labels keyed by name.
    pub fn labels(&self) -> Option<&Vec<(String, OmeZarrLabels)>> {
        self.labels.as_ref()
    }

    /// Get a label by name.
    pub fn label(&self, name: &str) -> Option<&OmeZarrLabels> {
        self.labels
            .as_ref()?
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, l)| l)
    }

    /// Set labels.
    pub fn set_labels(&mut self, labels: Option<LabelsArg>) {
        self.labels = parse_labels(labels);
    }

    fn parse_omero_metadata(
        &mut self,
        channel_names: Option<Vec<String>>,
        channel_colors: Option<Vec<ChannelColor>>,
        contrast_limits: Option<Vec<(f64, f64)>>,
    ) -> Result<()> {
        let img = &self.base.images[0];
        let n_channels = match img.axes.iter().position(|a| a == "c") {
            None => 1,
            Some(i) => img.data.shape()[i] as usize,
        };
        // messages show the parameter as Python's repr
        let check = |len: Option<usize>, what: String| -> Result<()> {
            if let Some(l) = len
                && l != n_channels
            {
                return Err(Error::type_err(format!(
                    "Length of {what} ({l}) does not match number of channels ({n_channels})"
                )));
            }
            Ok(())
        };
        let num = |v: f64| {
            if v.fract() == 0.0 && v.abs() < 1e16 {
                format!("{}", v as i64)
            } else {
                crate::pyjson::float_repr(v)
            }
        };
        if let Some(n) = &channel_names {
            check(Some(n.len()), crate::pyjson::str_list_repr(n))?;
        }
        if let Some(c) = &channel_colors {
            let items: Vec<String> = c
                .iter()
                .map(|c| match c {
                    ChannelColor::Hex(h) => crate::pyjson::repr_str(h),
                    ChannelColor::Rgb(v) => format!(
                        "[{}]",
                        v.iter()
                            .map(|x| x.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                })
                .collect();
            check(Some(c.len()), format!("[{}]", items.join(", ")))?;
        }
        if let Some(c) = &contrast_limits {
            let items: Vec<String> = c
                .iter()
                .map(|(a, b)| format!("({}, {})", num(*a), num(*b)))
                .collect();
            check(Some(c.len()), format!("[{}]", items.join(", ")))?;
        }
        let dtype_max = (img.data.dtype().itemsize() * 255) as f64;
        let mut channels = vec![];
        for i in 0..n_channels {
            let name = channel_names
                .as_ref()
                .map(|n| n[i].clone())
                .unwrap_or_else(|| format!("Channel {i}"));
            let color = match channel_colors.as_ref().map(|c| &c[i]) {
                Some(ChannelColor::Hex(h)) => h.clone(),
                Some(ChannelColor::Rgb(v)) => format!("{:02x}{:02x}{:02x}", v[0], v[1], v[2]),
                None => DEFAULT_COLORS[i % DEFAULT_COLORS.len()].to_string(),
            };
            let color = color.trim_start_matches('#').to_string();
            let (start, end) = contrast_limits
                .as_ref()
                .map(|c| c[i])
                .unwrap_or((0.0, dtype_max));
            let mut extra = JsonMap::new();
            extra.insert("label".into(), json!(name));
            extra.insert("active".into(), json!(true));
            channels.push(Channel {
                color,
                window: Window {
                    max: dtype_max,
                    min: 0.0,
                    start,
                    end,
                    extra: JsonMap::new(),
                },
                extra,
            });
        }
        let invalid = channels
            .iter()
            .find_map(|c| crate::models::omero::validate_rgb_hex(&c.color).err());
        match invalid {
            Some(e) => warn!("Failed to validate Omero metadata: {e}"),
            None => {
                self.omero = Some(Omero {
                    channels,
                    extra: JsonMap::new(),
                })
            }
        }
        Ok(())
    }

    /// Write the multiscale image pyramid, metadata and labels.
    pub fn to_ome_zarr(
        &self,
        group: impl Into<GroupLike>,
        storage_options: Option<&StorageOptionsArg>,
        version: Version,
        compute: bool,
        overwrite: bool,
    ) -> Result<Vec<DelayedWrite>> {
        let (group, mut delayed) =
            self.base
                .write_base(group.into(), storage_options, version, compute, overwrite)?;
        if let Some(omero) = &self.omero {
            let d = omero.to_json();
            match version {
                Version::V04 => group.set_attr("omero", d)?,
                _ => {
                    let attrs = group.attrs()?;
                    let mut ome = attrs
                        .get("ome")
                        .and_then(|o| o.as_object())
                        .cloned()
                        .ok_or_else(|| Error::value("OME-Zarr attributes not found in group"))?;
                    ome.insert("omero".into(), d);
                    group.set_attr("ome", Value::Object(ome))?;
                }
            }
        }
        if let Some(labels) = &self.labels {
            let label_group = group.require_group("labels")?;
            let mut list = vec![];
            for (label_name, ms_labels) in labels {
                let mut ms_labels = ms_labels.clone();
                ms_labels.base.name = label_name.clone();
                list.push(json!(label_name));
                if label_group.contains(label_name) && !overwrite {
                    warn!(
                        "Label group {label_name} already exists in store. Skipping writing this label since overwrite=False."
                    );
                    continue;
                }
                let sub = label_group.require_group(label_name)?;
                delayed.extend(ms_labels.to_ome_zarr(
                    GroupLike::Group(sub),
                    storage_options,
                    version,
                    compute,
                    true,
                )?);
            }
            match version {
                Version::V04 => label_group.set_attr("labels", Value::Array(list))?,
                _ => label_group
                    .set_attr("ome", json!({"version": version.as_str(), "labels": list}))?,
            }
        }
        Ok(delayed)
    }

    /// Load a multiscale image pyramid (with omero metadata and labels).
    pub fn from_ome_zarr(group: impl Into<GroupLike>) -> Result<OmeZarrMultiscale> {
        let (group, base, version, _) = read_base(group.into())?;
        let mut ms = OmeZarrMultiscale {
            base,
            labels: None,
            omero: None,
        };
        ms.read_additional_metadata(&group, &version)?;
        Ok(ms)
    }

    fn read_additional_metadata(&mut self, group: &Group, version: &str) -> Result<()> {
        let attrs = group.attrs()?;
        let legacy = matches!(version, "0.1" | "0.2" | "0.3" | "0.4");
        let modern = version == "0.5" || version.starts_with("0.6");
        let omero_dict = if legacy {
            attrs.get("omero").cloned()
        } else if modern {
            attrs.get("ome").and_then(|o| o.get("omero")).cloned()
        } else {
            None
        };
        if let Some(o) = omero_dict.filter(|o| !o.is_null()) {
            match Omero::from_json(&o) {
                Ok(o) => self.omero = Some(o),
                Err(e) => warn!("Invalid Omero metadata: {e}"),
            }
        }
        let mut list: Vec<String> = vec![];
        if group.child_kind("labels") == Some(NodeKind::Group) {
            let lattrs = group.group("labels")?.attrs()?;
            let v = if legacy {
                lattrs.get("labels").cloned()
            } else if modern {
                lattrs.get("ome").and_then(|o| o.get("labels")).cloned()
            } else {
                None
            };
            list = v
                .and_then(|v| v.as_array().cloned())
                .unwrap_or_default()
                .iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect();
        }
        if !list.is_empty() {
            let mut loaded = vec![];
            for name in list {
                let sub = match group.group(&format!("labels/{name}")) {
                    Ok(g) => g,
                    Err(_) => {
                        warn!("Label {name} is not a zarr.Group, skipping");
                        continue;
                    }
                };
                loaded.push((name, OmeZarrLabels::from_ome_zarr(GroupLike::Group(sub))?));
            }
            self.labels = Some(loaded);
        }
        Ok(())
    }

    /// Find a coordinate system belonging either to this image, or a label
    /// below it. Must be named.
    pub fn get_descendant_coordinate_system(
        &self,
        csid: &CoordinateSystemIdentifier,
    ) -> Option<&CoordinateSystem> {
        let name = csid.name.as_ref()?;
        if let Some(p) = &csid.path
            && p != "."
        {
            return self.label(p)?.base.get_local_coordinate_system(name);
        }
        self.base.get_local_coordinate_system(name)
    }
}

/// Container for label images with OME-Zarr metadata.
#[derive(Debug, Clone)]
pub struct OmeZarrLabels {
    pub base: MultiscaleBase,
    pub image_label: Option<ImageLabel>,
}

impl OmeZarrLabels {
    /// Build a label pyramid (default method nearest); with
    /// `auto_parse_labels`, image-label metadata is generated from the values.
    pub fn new(
        image: OmeZarrImage,
        scale_factors: Option<ScaleFactors>,
        method: Option<Methods>,
        auto_parse_labels: bool,
    ) -> Result<OmeZarrLabels> {
        let base = MultiscaleBase::new(image, scale_factors, None, None, method, "physical")?;
        let mut l = OmeZarrLabels {
            base,
            image_label: None,
        };
        if auto_parse_labels {
            l.parse_image_label_metadata()?;
        }
        Ok(l)
    }

    /// `OMEZarrLabels(image)` with defaults (nearest, auto-parsed labels).
    pub fn from_image(image: OmeZarrImage) -> Result<OmeZarrLabels> {
        OmeZarrLabels::new(image, None, Some(Methods::Nearest), true)
    }

    pub fn images(&self) -> &[OmeZarrImage] {
        &self.base.images
    }

    pub fn metadata(&self) -> &Multiscale {
        &self.base.metadata
    }

    pub fn name(&self) -> &str {
        &self.base.name
    }

    fn parse_image_label_metadata(&mut self) -> Result<()> {
        let values = self.base.images[0].data.unique()?;
        let mut colors = vec![];
        let mut props = vec![];
        let bad: Vec<(usize, f64)> = values
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, v)| v.fract() != 0.0 || !v.is_finite())
            .collect();
        if !bad.is_empty() {
            return Err(label_value_errors(&bad));
        }
        for v in values {
            let lv = v as i64;
            colors.push(Color {
                label_value: lv,
                // [np.random.randint(0, 255) for _ in range(3)] + [255]
                rgba: Some([
                    crate::utils::np_random_randint(0, 255) as u8,
                    crate::utils::np_random_randint(0, 255) as u8,
                    crate::utils::np_random_randint(0, 255) as u8,
                    255,
                ]),
                extra: JsonMap::new(),
            });
            props.push(Property {
                label_value: lv,
                extra: JsonMap::new(),
            });
        }
        let l = ImageLabel {
            colors: Some(colors),
            properties: Some(props),
            source: Some(Source {
                image: Some("../..".into()),
                extra: JsonMap::new(),
            }),
            version: None,
            extra: JsonMap::new(),
        };
        l.validate()?;
        self.image_label = Some(l);
        Ok(())
    }

    /// Write the label pyramid and metadata.
    pub fn to_ome_zarr(
        &self,
        group: impl Into<GroupLike>,
        storage_options: Option<&StorageOptionsArg>,
        version: Version,
        compute: bool,
        overwrite: bool,
    ) -> Result<Vec<DelayedWrite>> {
        let (group, delayed) =
            self.base
                .write_base(group.into(), storage_options, version, compute, overwrite)?;
        if let Some(il) = &self.image_label {
            let mut meta = il.to_json();
            match version {
                Version::V04 => {
                    meta["version"] = json!("0.4");
                    group.set_attr("image-label", meta)?;
                }
                _ => {
                    if version == Version::V05 {
                        meta["version"] = json!("0.5");
                    }
                    let attrs = group.attrs()?;
                    let mut ome = attrs
                        .get("ome")
                        .and_then(|o| o.as_object())
                        .cloned()
                        .unwrap_or_default();
                    ome.insert("image-label".into(), meta);
                    group.set_attr("ome", Value::Object(ome))?;
                }
            }
        }
        Ok(delayed)
    }

    /// Load a label pyramid.
    pub fn from_ome_zarr(group: impl Into<GroupLike>) -> Result<OmeZarrLabels> {
        let (group, base, version, _) = read_base(group.into())?;
        let mut l = OmeZarrLabels {
            base,
            image_label: None,
        };
        let attrs = group.attrs()?;
        let legacy = matches!(version.as_str(), "0.1" | "0.2" | "0.3" | "0.4");
        let modern = version == "0.5" || version.starts_with("0.6");
        let d = if legacy {
            attrs.get("image-label").cloned()
        } else if modern {
            attrs.get("ome").and_then(|o| o.get("image-label")).cloned()
        } else {
            None
        };
        if let Some(d) = d.filter(|d| !d.is_null()) {
            match ImageLabel::from_json(&d) {
                Ok(il) => l.image_label = Some(il),
                Err(e) => warn!("Invalid image-label metadata: {e}"),
            }
        }
        Ok(l)
    }

    /// A coordinate system on this label. As in ome-zarr-py, whose condition
    /// (`path is not None or path != "."`) is always true, this returns `None`.
    pub fn get_descendant_coordinate_system(
        &self,
        csid: &CoordinateSystemIdentifier,
    ) -> Option<&CoordinateSystem> {
        let _name = csid.name.as_ref()?;
        None
    }

    /// Set image-label metadata (validated).
    pub fn set_image_label(&mut self, value: Option<ImageLabel>) -> Result<()> {
        if let Some(v) = &value {
            v.validate()?;
        }
        self.image_label = value;
        Ok(())
    }
}

/// Either kind of loaded multiscale.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum AnyMultiscale {
    Image(OmeZarrMultiscale),
    Labels(OmeZarrLabels),
}

/// Load a multiscale, returning labels if the group has image-label metadata
/// (`OMEZarrMultiscaleBase.from_ome_zarr`).
pub fn read_multiscale(group: impl Into<GroupLike>) -> Result<AnyMultiscale> {
    let g = group.into();
    let (group, _, _, is_label) = read_base(g.clone())?;
    if is_label {
        Ok(AnyMultiscale::Labels(OmeZarrLabels::from_ome_zarr(
            GroupLike::Group(group),
        )?))
    } else {
        Ok(AnyMultiscale::Image(OmeZarrMultiscale::from_ome_zarr(
            GroupLike::Group(group),
        )?))
    }
}

/// Safely extract the OME-Zarr version from group attributes (`_get_version`).
pub fn get_version(group: &Group) -> Result<String> {
    let attrs = group.attrs()?;
    if let Some(v) = attrs
        .get("ome")
        .and_then(|o| o.get("version"))
        .and_then(|v| v.as_str())
    {
        return Ok(v.to_string());
    }
    if let Some(v) = attrs
        .get("multiscales")
        .and_then(|m| m.as_array())
        .and_then(|m| m.first())
        .and_then(|m| m.get("version"))
        .and_then(|v| v.as_str())
    {
        return Ok(v.to_string());
    }
    Err(Error::value(
        "Could not find 'version' in group attributes. Expected location: group.attrs['ome']['version'] (v0.5+) \
         or group.attrs['multiscales'][0]['version'] (v0.4 or lower)",
    ))
}

const NOT_MULTISCALES: &str = "Multiscales metadata not found in group attributes. Opening groups other than \
    multiscales (i.e., HCS, Plates, Wells) is currently not supported.";

/// Read metadata from legacy OME-Zarr versions (0.1, 0.2, 0.3).
pub fn read_legacy_metadata(attrs: &JsonMap, version: &str) -> Result<Multiscale> {
    let ms = attrs
        .get("multiscales")
        .and_then(|m| m.as_array())
        .and_then(|m| m.first())
        .ok_or_else(|| Error::value(NOT_MULTISCALES))?;
    let axes_order: Vec<String> = if version == "0.3" {
        ms.get("axes")
            .and_then(|a| a.as_array())
            .ok_or_else(|| Error::value("Metadata version 0.3 requires 'axes' field in metadata"))?
            .iter()
            .map(|a| {
                a.as_str()
                    .map(String::from)
                    .ok_or_else(|| Error::value(format!("invalid axis {a}")))
            })
            .collect::<Result<_>>()?
    } else {
        "tczyx".chars().map(|c| c.to_string()).collect()
    };
    let axes: Vec<Axis> = axes_order
        .iter()
        .map(|a| {
            let t = match a.as_str() {
                "t" => "time",
                "c" => "channel",
                "z" | "y" | "x" => "space",
                other => return Err(Error::key(format!("'{other}'"))),
            };
            Ok(Axis::new(a, Some(t)))
        })
        .collect::<Result<_>>()?;
    let spatial = |a: &Axis| matches!(a.name_str(), "z" | "y" | "x");
    let mut datasets = vec![];
    for (idx, ds) in ms
        .get("datasets")
        .and_then(|d| d.as_array())
        .cloned()
        .unwrap_or_default()
        .iter()
        .enumerate()
    {
        let scale: Vec<f64> = axes
            .iter()
            .map(|a| {
                if spatial(a) {
                    2f64.powi(idx as i32)
                } else {
                    1.0
                }
            })
            .collect();
        let path = ds
            .get("path")
            .and_then(|p| p.as_str())
            .map(String::from)
            .unwrap_or_else(|| format!("s{idx}"));
        let t = if idx == 0 {
            Transform::Scale(Scale {
                input: Some(CoordinateSystemIdentifier::path(&path)),
                output: Some(CoordinateSystemIdentifier::name("physical")),
                name: None,
                scale,
            })
        } else {
            let translate: Vec<f64> = axes
                .iter()
                .map(|a| {
                    if spatial(a) {
                        2f64.powi(idx as i32 - 1) - 0.5
                    } else {
                        0.0
                    }
                })
                .collect();
            Transform::Sequence(Sequence {
                input: Some(CoordinateSystemIdentifier::path(&path)),
                output: Some(CoordinateSystemIdentifier::name("physical")),
                name: None,
                transformations: vec![
                    Transform::Scale(Scale {
                        scale,
                        ..Default::default()
                    }),
                    Transform::Translation(Translation {
                        translation: translate,
                        ..Default::default()
                    }),
                ],
            })
        };
        datasets.push(Dataset {
            path,
            coordinate_transformations: vec![t],
        });
    }
    let m = Multiscale {
        coordinate_systems: vec![CoordinateSystem {
            name: "physical".into(),
            axes,
            extra: JsonMap::new(),
        }],
        datasets,
        coordinate_transformations: None,
        metadata: ms.get("metadata").cloned().filter(|v| !v.is_null()),
        name: Some(ms.get("name").cloned().unwrap_or(json!("image"))),
        type_: ms.get("type").cloned().filter(|v| !v.is_null()),
        extra: JsonMap::new(),
    };
    m.validate()?;
    Ok(m)
}

/// Read the multiscales metadata of any supported version and build the
/// images. Returns (group, base, version, is_label).
fn read_base(group: GroupLike) -> Result<(Group, MultiscaleBase, String, bool)> {
    let group = match group {
        GroupLike::Group(g) => g,
        GroupLike::Path(p) => Group::open_url(&p, Mode::Read, None)?,
    };
    let version = get_version(&group)?;
    let attrs = group.attrs()?;
    let (metadata, is_label) = if matches!(version.as_str(), "0.1" | "0.2" | "0.3") {
        (
            read_legacy_metadata(&attrs, &version)?,
            attrs.contains_key("image-label"),
        )
    } else if version.starts_with("0.4") {
        let ms = attrs
            .get("multiscales")
            .and_then(|m| m.as_array())
            .and_then(|m| m.first())
            .filter(|m| !m.is_null())
            .ok_or_else(|| Error::value(NOT_MULTISCALES))?;
        (
            Multiscale::from_v04_json(ms)?,
            attrs.contains_key("image-label"),
        )
    } else if version.starts_with("0.5") {
        let ome = attrs
            .get("ome")
            .and_then(|o| o.as_object())
            .cloned()
            .unwrap_or_default();
        let ms = ome
            .get("multiscales")
            .and_then(|m| m.as_array())
            .and_then(|m| m.first())
            .ok_or_else(|| Error::value(NOT_MULTISCALES))?;
        (
            Multiscale::from_v05_json(ms)?,
            ome.contains_key("image-label"),
        )
    } else if version.contains("0.6") {
        let ome = attrs
            .get("ome")
            .and_then(|o| o.as_object())
            .cloned()
            .unwrap_or_default();
        let ms = ome
            .get("multiscales")
            .and_then(|m| m.as_array())
            .and_then(|m| m.first())
            .ok_or_else(|| Error::value(NOT_MULTISCALES))?;
        (Multiscale::from_json(ms)?, ome.contains_key("image-label"))
    } else {
        return Err(Error::value(format!(
            "Unsupported OME-Zarr version: {version}"
        )));
    };
    let cs = metadata.intrinsic_coordinate_system()?.clone();
    let name = metadata.name_str().unwrap_or_else(|| "image".into());
    let mut images = vec![];
    for ds in &metadata.datasets {
        let data = group.array(&ds.path)?.to_lazy();
        let scale = ds.scale(cs.ndim()).unwrap_or_else(|| vec![1.0; cs.ndim()]);
        let units: BTreeMap<String, String> = cs
            .axes
            .iter()
            .filter_map(|a| a.unit_str().map(|u| (a.name_str().to_string(), u)))
            .collect();
        let types: BTreeMap<String, Option<String>> = cs
            .axes
            .iter()
            .filter(|a| a.type_.is_some())
            .map(|a| (a.name_str().to_string(), a.type_.clone()))
            .collect();
        let axes = cs.axis_names();
        let sm = ScaleMap(axes.iter().cloned().zip(scale).collect());
        images.push(OmeZarrImage::new(
            data,
            &axes,
            Some(sm),
            if units.is_empty() { None } else { Some(units) },
            Some(types),
            &name,
        )?);
    }
    Ok((
        group,
        MultiscaleBase {
            images,
            metadata,
            name,
        },
        version,
        is_label,
    ))
}

/// Remove nones (re-exported helper for metadata dicts).
pub fn pop_nones(v: &Value) -> Value {
    recursive_pop_nones(v)
}

/// pydantic's `ValidationError` for `LabelBase` with non-integer label values
/// (each is reported under both `colors` and `properties`).
fn label_value_errors(bad: &[(usize, f64)]) -> Error {
    let entry = |field: &str, i: usize, v: f64| {
        let (kind, msg) = if v.is_finite() {
            (
                "int_from_float",
                "Input should be a valid integer, got a number with a fractional part",
            )
        } else {
            ("finite_number", "Input should be a finite number")
        };
        let iv = crate::pyjson::repr(
            &serde_json::Number::from_f64(v)
                .map(Value::Number)
                .unwrap_or(Value::Null),
        );
        let iv = if v.is_finite() {
            iv
        } else if v.is_nan() {
            "nan".into()
        } else if v > 0.0 {
            "inf".into()
        } else {
            "-inf".into()
        };
        format!(
            "{field}.{i}.label-value\n  {msg} [type={kind}, input_value={iv}, input_type=float]\n    \
             For further information visit https://errors.pydantic.dev/2.12/v/{kind}"
        )
    };
    let mut lines: Vec<String> = bad.iter().map(|&(i, v)| entry("colors", i, v)).collect();
    lines.extend(bad.iter().map(|&(i, v)| entry("properties", i, v)));
    let n = lines.len();
    Error::Validation(format!(
        "{n} validation error{} for LabelBase\n{}",
        if n == 1 { "" } else { "s" },
        lines.join("\n")
    ))
}
