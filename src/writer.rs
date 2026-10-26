//! Image writer utility (`ome_zarr.writer`).

use std::collections::BTreeMap;

use log::{debug, info, warn};
use serde_json::{Value, json};

use crate::array::{LazyArray, retuple};
use crate::axes::{Axes, AxesInput};
use crate::classes::image::{OmeZarrImage, OmeZarrLabels, OmeZarrMultiscale, ScaleMap, Version};
use crate::error::{Error, Result};
use crate::format::{CURRENT_FORMAT, Format};
use crate::models::{ImageLabel, Omero};
use crate::pyjson::py_str;
use crate::scale::{Methods, ScaleFactors, Scaler};
use crate::zarr::{ArraySpec, Compressor, Group, JsonMap, Mode, ZArray, store_lazy};

/// Spatial dimension names.
pub const SPATIAL_DIMS: [&str; 3] = ["x", "y", "z"];

/// A group, or a path / URL at which to open or create one.
#[derive(Debug, Clone)]
pub enum GroupLike {
    Group(Group),
    Path(String),
}

impl From<Group> for GroupLike {
    fn from(g: Group) -> Self {
        GroupLike::Group(g)
    }
}

impl From<&Group> for GroupLike {
    fn from(g: &Group) -> Self {
        GroupLike::Group(g.clone())
    }
}

impl From<&str> for GroupLike {
    fn from(s: &str) -> Self {
        GroupLike::Path(s.to_string())
    }
}

impl From<String> for GroupLike {
    fn from(s: String) -> Self {
        GroupLike::Path(s)
    }
}

impl From<&std::path::Path> for GroupLike {
    fn from(p: &std::path::Path) -> Self {
        GroupLike::Path(p.to_string_lossy().to_string())
    }
}

impl From<std::path::PathBuf> for GroupLike {
    fn from(p: std::path::PathBuf) -> Self {
        GroupLike::Path(p.to_string_lossy().to_string())
    }
}

impl From<&std::path::PathBuf> for GroupLike {
    fn from(p: &std::path::PathBuf) -> Self {
        GroupLike::Path(p.to_string_lossy().to_string())
    }
}

/// Options passed on to the storage backend when creating arrays.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StorageOptions {
    /// Zarr chunk shape (trailing dims; leading dims are filled from the shape).
    pub chunks: Option<Vec<u64>>,
    /// `chunks="auto"`: not allowed by ome-zarr-py, which warns and uses the
    /// chunk size of the (dask) array; `chunks` is ignored when set.
    pub auto_chunks: bool,
    /// Zarr v3 shard shape (trailing dims; leading dims are filled from the shape).
    pub shards: Option<Vec<u64>>,
    /// Compressor (default: zstd level 0, as zarr-python).
    pub compressor: Option<Compressor>,
    /// Fill value.
    pub fill_value: Option<f64>,
}

impl StorageOptions {
    pub fn chunks(chunks: &[u64]) -> StorageOptions {
        StorageOptions {
            chunks: Some(chunks.to_vec()),
            ..Default::default()
        }
    }
}

/// Storage options for all levels, or one per level.
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::large_enum_variant)] // mirrors the Python argument; built once per call
pub enum StorageOptionsArg {
    All(StorageOptions),
    PerLevel(Vec<StorageOptions>),
}

impl From<StorageOptions> for StorageOptionsArg {
    fn from(s: StorageOptions) -> Self {
        StorageOptionsArg::All(s)
    }
}

impl From<Vec<StorageOptions>> for StorageOptionsArg {
    fn from(s: Vec<StorageOptions>) -> Self {
        StorageOptionsArg::PerLevel(s)
    }
}

fn resolve_storage_options(
    opts: Option<&StorageOptionsArg>,
    level: usize,
) -> Result<StorageOptions> {
    Ok(match opts {
        None => StorageOptions::default(),
        Some(StorageOptionsArg::All(o)) => o.clone(),
        Some(StorageOptionsArg::PerLevel(v)) => v
            .get(level)
            .cloned()
            .ok_or_else(|| Error::value(format!("no storage_options for level {level}")))?,
    })
}

/// A write that has been prepared but not executed (`dask.delayed`).
#[derive(Debug, Clone)]
pub struct DelayedWrite {
    pub array: ZArray,
    pub data: LazyArray,
}

impl DelayedWrite {
    /// Compute the data and write it.
    pub fn compute(&self) -> Result<()> {
        store_lazy(&self.data, &self.array)
    }
}

/// Compute a list of delayed writes (`dask.compute(*delayed)`).
pub fn compute_all(jobs: &[DelayedWrite]) -> Result<()> {
    for j in jobs {
        j.compute()?;
    }
    Ok(())
}

/// Returns list of axes valid for fmt.version or raise exception if invalid.
pub fn get_valid_axes(
    ndim: Option<usize>,
    axes: Option<&AxesInput>,
    axes_units: Option<&JsonMap>,
    fmt: Format,
) -> Result<Option<Value>> {
    if matches!(fmt, Format::V01 | Format::V02) {
        if axes.is_some() {
            info!("axes ignored for version 0.1 or 0.2");
        }
        return Ok(None);
    }
    let axes = match axes {
        Some(a) => a.clone(),
        None => match ndim {
            Some(2) => {
                info!("Auto using axes ['y', 'x'] for 2D data");
                AxesInput::from_chars("yx")
            }
            Some(5) => {
                info!("Auto using axes ['t', 'c', 'z', 'y', 'x'] for 5D data");
                AxesInput::from_chars("tczyx")
            }
            _ => {
                return Err(Error::value(
                    "axes must be provided. Can't be guessed for 3D or 4D data",
                ));
            }
        },
    };
    if let Some(n) = ndim
        && axes.len() != n
    {
        return Err(Error::value(format!(
            "axes length ({}) must match number of dimensions ({n})",
            axes.len()
        )));
    }
    let obj = Axes::new(Some(&axes), axes_units, fmt)?;
    Ok(Some(obj.to_list(fmt)?))
}

/// Extract dimension names from axes (`_extract_dims_from_axes`).
pub fn extract_dims_from_axes(axes: Option<&Value>) -> Result<Vec<String>> {
    let Some(axes) = axes else {
        return Ok("tczyx".chars().map(|c| c.to_string()).collect());
    };
    AxesInput::from_json(axes)?.names()
}

/// Validate the images of a well (`_validate_well_images`).
pub fn validate_well_images(images: &[Value]) -> Result<Vec<Value>> {
    let valid_keys = ["acquisition", "path"];
    let mut out = vec![];
    for image in images {
        match image {
            Value::String(s) => out.push(json!({"path": s})),
            Value::Object(o) => {
                if o.keys().any(|k| !valid_keys.contains(&k.as_str())) {
                    debug!("{image} contains unspecified keys");
                }
                let path = o.get("path").ok_or_else(|| {
                    Error::value(format!("{} must contain a path key", py_str(image)))
                })?;
                if !path.is_string() {
                    return Err(Error::type_err(format!(
                        "{} path must be of string type",
                        py_str(image)
                    )));
                }
                if let Some(a) = o.get("acquisition")
                    && !(a.is_i64() || a.is_u64() || a.is_boolean())
                {
                    return Err(Error::type_err(format!(
                        "{} acquisition must be of int type",
                        py_str(image)
                    )));
                }
                out.push(image.clone());
            }
            other => {
                return Err(Error::type_err(format!(
                    "Unrecognized type for {}",
                    py_str(other)
                )));
            }
        }
    }
    Ok(out)
}

/// Validate plate acquisitions (`_validate_plate_acquisitions`).
pub fn validate_plate_acquisitions(acquisitions: &[Value]) -> Result<Vec<Value>> {
    let valid = [
        "id",
        "name",
        "maximumfieldcount",
        "description",
        "starttime",
        "endtime",
    ];
    for a in acquisitions {
        let o = a
            .as_object()
            .ok_or_else(|| Error::type_err(format!("{} must be a dictionary", py_str(a))))?;
        if o.keys().any(|k| !valid.contains(&k.as_str())) {
            debug!("{a} contains unspecified keys");
        }
        let id = o
            .get("id")
            .ok_or_else(|| Error::value(format!("{} must contain an id key", py_str(a))))?;
        if !(id.is_i64() || id.is_u64() || id.is_boolean()) {
            return Err(Error::type_err(format!(
                "{} id must be of int type",
                py_str(a)
            )));
        }
    }
    Ok(acquisitions.to_vec())
}

/// Validate plate rows or columns (`_validate_plate_rows_columns`).
pub fn validate_plate_rows_columns(items: &[String]) -> Result<Vec<Value>> {
    let mut seen = std::collections::HashSet::new();
    if !items.iter().all(|i| seen.insert(i)) {
        return Err(Error::value(format!(
            "{} must contain unique elements",
            crate::pyjson::str_list_repr(items)
        )));
    }
    items
        .iter()
        .map(|e| {
            if e.is_empty() || !e.chars().all(|c| c.is_alphanumeric()) {
                Err(Error::value(format!(
                    "{e} must contain alphanumeric characters"
                )))
            } else {
                Ok(json!({"name": e}))
            }
        })
        .collect()
}

/// Validate datasets (`_validate_datasets`).
pub fn validate_datasets(datasets: &[Value], dims: i64, fmt: Format) -> Result<Vec<Value>> {
    if datasets.is_empty() {
        return Err(Error::value("Empty datasets list"));
    }
    let mut transformations = vec![];
    for d in datasets {
        let o = d
            .as_object()
            .ok_or_else(|| Error::type_err(format!("Unrecognized type for {}", py_str(d))))?;
        if !o.get("path").is_some_and(crate::format::is_truthy) {
            return Err(Error::value("no 'path' in dataset"));
        }
        if let Some(t) = o.get("coordinateTransformations").filter(|t| !t.is_null()) {
            let list = t.as_array().ok_or_else(|| {
                Error::type_err(format!("coordinateTransformations must be a list: {t}"))
            })?;
            transformations.push(list.clone());
        }
    }
    fmt.validate_coordinate_transformations(dims, datasets.len(), Some(&transformations))?;
    Ok(datasets.to_vec())
}

/// Validate plate wells (`_validate_plate_wells`).
pub fn validate_plate_wells(
    wells: &[Value],
    rows: &[String],
    columns: &[String],
    fmt: Format,
) -> Result<Vec<Value>> {
    if wells.is_empty() {
        return Err(Error::value("Empty wells list"));
    }
    let mut out = vec![];
    for w in wells {
        match w {
            Value::String(s) => {
                let d = fmt.generate_well_dict(s, rows, columns)?;
                fmt.validate_well_dict(&d, rows, columns)?;
                out.push(Value::Object(d));
            }
            Value::Object(o) => {
                fmt.validate_well_dict(o, rows, columns)?;
                out.push(w.clone());
            }
            other => {
                return Err(Error::type_err(format!(
                    "Unrecognized type for {}",
                    py_str(other)
                )));
            }
        }
    }
    Ok(out)
}

/// The `Blosc(cname="zstd", clevel=5, shuffle=SHUFFLE)` compressor.
pub fn blosc_compressor() -> Compressor {
    Compressor::blosc_zstd()
}

/// Create the group if a path (according to `fmt`), or check `fmt` is
/// compatible with the group.
pub fn check_group_fmt(
    group: impl Into<GroupLike>,
    fmt: Option<Format>,
    mode: &str,
) -> Result<(Group, Format)> {
    let group = match group.into() {
        GroupLike::Group(g) => g,
        GroupLike::Path(p) => {
            Group::open_url(&p, Mode::parse(mode)?, fmt.map(|f| f.zarr_format()))?
        }
    };
    let fmt = check_format(&group, fmt)?;
    Ok((group, fmt))
}

/// Check if the format is valid for the given group.
pub fn check_format(group: &Group, fmt: Option<Format>) -> Result<Format> {
    let zf = group.zarr_format();
    match fmt {
        Some(f) => {
            if f.zarr_format() != zf {
                return Err(Error::value(format!(
                    "Group is zarr_format: {zf} but OME-Zarr {} is {}",
                    f.version(),
                    f.zarr_format()
                )));
            }
            Ok(f)
        }
        None if zf == 2 => Ok(Format::V04),
        None => Ok(CURRENT_FORMAT),
    }
}

/// Options for [`write_multiscale`] and [`write_multiscale_labels`].
#[derive(Debug, Clone, Default)]
pub struct MultiscaleOptions {
    pub fmt: Option<Format>,
    pub axes: Option<AxesInput>,
    /// Deprecated: per-level transformations (take precedence over `scale`).
    pub coordinate_transformations: Option<Vec<Vec<Value>>>,
    pub storage_options: Option<StorageOptionsArg>,
    pub name: Option<String>,
    /// Compute immediately (default true when built with [`MultiscaleOptions::new`]).
    pub compute: bool,
    pub scale: Option<ScaleMap>,
    pub axes_units: Option<JsonMap>,
    /// The `metadata` keyword (e.g. `{"omero": {...}}`).
    pub metadata: Option<JsonMap>,
}

impl MultiscaleOptions {
    pub fn new() -> MultiscaleOptions {
        MultiscaleOptions {
            compute: true,
            ..Default::default()
        }
    }
}

fn deprecation_ct() {
    warn!(
        "The 'coordinate_transformations' argument is deprecated and will be removed in a future version. \
         Please use the `scale` argument to specify the physical pixel size for each dimension instead. \
         When `coordinate_transformations` is provided, it takes precedence over `scale`, so `scale` is not applied."
    );
}

/// Write a pyramid with multiscale metadata to disk.
pub fn write_multiscale(
    pyramid: &[LazyArray],
    group: impl Into<GroupLike>,
    opts: &MultiscaleOptions,
) -> Result<Vec<DelayedWrite>> {
    let (group, fmt) = check_group_fmt(group, opts.fmt, "a")?;
    let dims = pyramid
        .first()
        .ok_or_else(|| Error::value("pyramid must not be empty"))?
        .ndim();
    let axes = get_valid_axes(
        Some(dims),
        opts.axes.as_ref(),
        opts.axes_units.as_ref(),
        fmt,
    )?;
    let scale = match &opts.scale {
        Some(s) => s.clone(),
        None => ScaleMap::ones(&extract_dims_from_axes(axes.as_ref())?),
    };
    if opts.coordinate_transformations.is_some() {
        deprecation_ct();
    }
    let axes_names = scale.keys();
    write_pyramid_to_zarr(
        pyramid,
        &group,
        fmt,
        &scale,
        &axes_names,
        opts.axes_units.as_ref(),
        opts.coordinate_transformations.clone(),
        opts.storage_options.as_ref(),
        opts.name.as_deref(),
        opts.compute,
        opts.metadata.as_ref(),
    )
}

/// Validate omero metadata passed through `metadata={"omero": ...}`.
fn validate_omero_dict(omero: &Value) -> Result<()> {
    if omero.is_null() {
        return Err(Error::key(
            "If `'omero'` is present, value cannot be `None`.",
        ));
    }
    let channels = omero["channels"]
        .as_array()
        .ok_or_else(|| Error::key("channels"))?;
    for c in channels {
        if let Some(color) = c.get("color")
            && !(color.as_str().is_some_and(|s| s.chars().count() == 6))
        {
            return Err(Error::type_err("`'color'` must be a hex code string."));
        }
        if let Some(window) = c.get("window") {
            let w = window
                .as_object()
                .ok_or_else(|| Error::type_err("`'window'` must be a dict."))?;
            for p in ["min", "max", "start", "end"] {
                let v = w
                    .get(p)
                    .ok_or_else(|| Error::key(format!("`'{p}'` not found in `'window'`.")))?;
                if !v.is_number() {
                    return Err(Error::type_err(format!("`'{p}'` must be an int or float.")));
                }
            }
        }
    }
    Ok(())
}

/// Write the multiscales metadata in the group.
pub fn write_multiscales_metadata(
    group: impl Into<GroupLike>,
    datasets: &[Value],
    fmt: Option<Format>,
    axes: Option<&AxesInput>,
    axes_units: Option<&JsonMap>,
    name: Option<&str>,
    metadata: Option<&JsonMap>,
) -> Result<()> {
    let (group, fmt) = check_group_fmt(group, fmt, "a")?;
    let mut ndim: i64 = -1;
    let mut axes_v = None;
    if let Some(a) = axes {
        if matches!(fmt, Format::V01 | Format::V02) {
            info!("axes ignored for version 0.1 or 0.2");
        } else {
            axes_v = get_valid_axes(None, Some(a), axes_units, fmt)?;
            if let Some(Value::Array(v)) = &axes_v {
                ndim = v.len() as i64;
            }
        }
    }
    let mut metadata = metadata.cloned().unwrap_or_default();
    if let Some(omero) = metadata.shift_remove("omero") {
        validate_omero_dict(&omero)?;
        let mut m = JsonMap::new();
        m.insert("omero".into(), omero);
        add_metadata(&group, &m, Some(fmt))?;
    }
    let mut ms = JsonMap::new();
    ms.insert(
        "datasets".into(),
        Value::Array(validate_datasets(datasets, ndim, fmt)?),
    );
    ms.insert(
        "name".into(),
        json!(name.map(String::from).unwrap_or_else(|| group.name())),
    );
    if !metadata.is_empty() {
        ms.insert("metadata".into(), Value::Object(metadata));
    }
    if let Some(a) = axes_v {
        ms.insert("axes".into(), a);
    }
    if matches!(fmt, Format::V01 | Format::V02 | Format::V03 | Format::V04) {
        ms.insert("version".into(), json!(fmt.version()));
    } else {
        let mut m = JsonMap::new();
        m.insert("version".into(), json!(fmt.version()));
        add_metadata(&group, &m, Some(fmt))?;
    }
    let mut m = JsonMap::new();
    m.insert("multiscales".into(), json!([Value::Object(ms)]));
    add_metadata(&group, &m, Some(fmt))
}

/// Write the plate metadata in the group.
#[allow(clippy::too_many_arguments)]
pub fn write_plate_metadata(
    group: impl Into<GroupLike>,
    rows: &[String],
    columns: &[String],
    wells: &[Value],
    fmt: Option<Format>,
    acquisitions: Option<&[Value]>,
    field_count: Option<i64>,
    name: Option<&str>,
) -> Result<()> {
    let (group, fmt) = check_group_fmt(group, fmt, "a")?;
    let mut plate = JsonMap::new();
    plate.insert(
        "columns".into(),
        Value::Array(validate_plate_rows_columns(columns)?),
    );
    plate.insert(
        "rows".into(),
        Value::Array(validate_plate_rows_columns(rows)?),
    );
    plate.insert(
        "wells".into(),
        Value::Array(validate_plate_wells(wells, rows, columns, fmt)?),
    );
    if let Some(n) = name {
        plate.insert("name".into(), json!(n));
    }
    if let Some(fc) = field_count {
        plate.insert("field_count".into(), json!(fc));
    }
    if let Some(a) = acquisitions {
        plate.insert(
            "acquisitions".into(),
            Value::Array(validate_plate_acquisitions(a)?),
        );
    }
    plate.insert("version".into(), json!(fmt.version()));
    if matches!(fmt, Format::V01 | Format::V02 | Format::V03 | Format::V04) {
        group.set_attr("plate", Value::Object(plate))
    } else {
        group.set_attr(
            "ome",
            json!({"version": fmt.version(), "plate": Value::Object(plate)}),
        )
    }
}

/// Convenience wrapper taking `&str` slices.
pub fn write_plate_metadata_simple(
    group: impl Into<GroupLike>,
    rows: &[&str],
    columns: &[&str],
    wells: &[&str],
    fmt: Option<Format>,
) -> Result<()> {
    let rows: Vec<String> = rows.iter().map(|s| s.to_string()).collect();
    let cols: Vec<String> = columns.iter().map(|s| s.to_string()).collect();
    let wells: Vec<Value> = wells.iter().map(|s| json!(s)).collect();
    write_plate_metadata(group, &rows, &cols, &wells, fmt, None, None, None)
}

/// Write the well metadata in the group.
pub fn write_well_metadata(
    group: impl Into<GroupLike>,
    images: &[Value],
    fmt: Option<Format>,
) -> Result<()> {
    let (group, fmt) = check_group_fmt(group, fmt, "a")?;
    let mut well = JsonMap::new();
    well.insert("images".into(), Value::Array(validate_well_images(images)?));
    if matches!(fmt, Format::V01 | Format::V02 | Format::V03 | Format::V04) {
        well.insert("version".into(), json!(fmt.version()));
        group.set_attr("well", Value::Object(well))
    } else {
        group.set_attr(
            "ome",
            json!({"version": fmt.version(), "well": Value::Object(well)}),
        )
    }
}

/// Options for [`write_image`] and [`write_labels`].
#[derive(Debug, Clone)]
pub struct ImageOptions {
    pub scale_factors: Option<ScaleFactors>,
    pub name: Option<String>,
    pub method: Option<Methods>,
    /// Deprecated: use `scale_factors` and `method`.
    pub scaler: Option<Scaler>,
    pub fmt: Option<Format>,
    pub axes: Option<AxesInput>,
    /// Deprecated and ignored (a warning is logged).
    pub coordinate_transformations: Option<Vec<Vec<Value>>>,
    pub storage_options: Option<StorageOptionsArg>,
    pub compute: bool,
    pub scale: Option<ScaleMap>,
    pub axes_units: Option<JsonMap>,
    /// `omero` metadata for [`write_image`].
    pub omero: Option<Omero>,
    /// `image-label` metadata for [`write_labels`].
    pub image_label: Option<ImageLabel>,
    /// Label metadata (`write_label_metadata` keywords) for [`write_labels`].
    pub label_metadata: Option<JsonMap>,
}

impl Default for ImageOptions {
    fn default() -> Self {
        ImageOptions {
            scale_factors: Some(ScaleFactors::default()),
            name: None,
            method: None,
            scaler: None,
            fmt: None,
            axes: None,
            coordinate_transformations: None,
            storage_options: None,
            compute: true,
            scale: None,
            axes_units: None,
            omero: None,
            image_label: None,
            label_metadata: None,
        }
    }
}

fn scaler_scale_factors(scaler: &Scaler, dims: &[String]) -> ScaleFactors {
    ScaleFactors::PerDim(
        (1..=scaler.max_layer)
            .map(|i| {
                dims.iter()
                    .map(|d| {
                        (
                            d.clone(),
                            if d == "y" || d == "x" {
                                2f64.powi(i as i32)
                            } else {
                                1.0
                            },
                        )
                    })
                    .collect::<BTreeMap<_, _>>()
            })
            .collect(),
    )
}

fn units_to_strings(units: Option<&JsonMap>) -> Option<BTreeMap<String, String>> {
    units.map(|u| {
        u.iter()
            .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
            .collect()
    })
}

/// Write an image to the zarr store according to the OME-Zarr specification,
/// supporting multiscale pyramids.
pub fn write_image(
    image: &LazyArray,
    group: impl Into<GroupLike>,
    opts: &ImageOptions,
) -> Result<Vec<DelayedWrite>> {
    let mut method = opts.method.unwrap_or(Methods::Resize);
    let (group, fmt) = check_group_fmt(group, opts.fmt, "a")?;
    if matches!(fmt, Format::V01 | Format::V02 | Format::V03) {
        return Err(Error::Deprecation(format!(
            "Writing ome-zarr v{} is deprecated and has been removed in version 0.15.0.",
            fmt.version()
        )));
    }
    let axes = get_valid_axes(
        Some(image.ndim()),
        opts.axes.as_ref(),
        opts.axes_units.as_ref(),
        fmt,
    )?;
    let dims = extract_dims_from_axes(axes.as_ref())?;
    let scale = opts.scale.clone().unwrap_or_else(|| ScaleMap::ones(&dims));
    if opts.coordinate_transformations.is_some() {
        warn!(
            "The 'coordinate_transformations' argument is deprecated and will be removed in a future version. \
             Please use the `scale` argument to specify the physical pixel size for each dimension instead. "
        );
    }
    let mut scale_factors = opts.scale_factors.clone();
    if let Some(scaler) = &opts.scaler {
        warn!(
            "The 'scaler' argument is deprecated and will be removed in a future version. Please use the 'scale_factors' argument instead."
        );
        scale_factors = Some(scaler_scale_factors(scaler, &dims));
        method = match scaler.method.as_str() {
            "local_mean" => Methods::LocalMean,
            "nearest" => Methods::Nearest,
            "resize_image" => Methods::Resize,
            "laplacian" => {
                warn!("Laplacian downsampling is not supported anymore.Falling back to `resize`");
                Methods::Resize
            }
            "zoom" => Methods::Zoom,
            _ => Methods::Resize,
        };
    }
    let single = OmeZarrImage::new(
        image.clone(),
        &dims,
        Some(scale),
        units_to_strings(opts.axes_units.as_ref()),
        None,
        opts.name.as_deref().unwrap_or("image"),
    )?;
    let mut multiscale = OmeZarrMultiscale::builder(single)
        .scale_factors(scale_factors)
        .method(Some(method))
        .build()?;
    multiscale.omero = opts.omero.clone();
    multiscale.to_ome_zarr(
        GroupLike::Group(group),
        opts.storage_options.as_ref(),
        Version::from_format(fmt),
        opts.compute,
        true,
    )
}

/// The shared implementation of pyramid writing (`_write_pyramid_to_zarr`).
#[allow(clippy::too_many_arguments)]
pub fn write_pyramid_to_zarr(
    pyramid: &[LazyArray],
    group: &Group,
    fmt: Format,
    scale: &ScaleMap,
    axes: &[String],
    axes_units: Option<&JsonMap>,
    coordinate_transformations: Option<Vec<Vec<Value>>>,
    storage_options: Option<&StorageOptionsArg>,
    name: Option<&str>,
    compute: bool,
    metadata: Option<&JsonMap>,
) -> Result<Vec<DelayedWrite>> {
    let fmt = check_format(group, Some(fmt))?;
    let scale = ScaleMap::from_pairs(
        axes.iter()
            .map(|d| (d.clone(), scale.get(d).unwrap_or(1.0)))
            .collect(),
    );
    let zarr_format = fmt.zarr_format();
    let level0 = resolve_storage_options(storage_options, 0)?;
    let global_compressor = level0.compressor.clone();
    let mut shapes = vec![];
    let mut datasets = vec![];
    let mut delayed = vec![];
    for (idx, level) in pyramid.iter().enumerate() {
        let options = resolve_storage_options(storage_options, idx)?;
        let compressor = global_compressor
            .clone()
            .or(options.compressor.clone())
            .unwrap_or_default();
        let shape = level.shape().to_vec();
        if options.auto_chunks {
            let cs: Vec<usize> = level.chunksize().iter().map(|&c| c as usize).collect();
            warn!(
                "Setting `chunks` to `auto` is not allowed. Defaulting to the chunksize of dask array: {}.",
                crate::pyjson::shape_repr(&cs)
            );
        }
        let user_chunks = if options.auto_chunks {
            None
        } else {
            options.chunks.clone()
        };
        let (chunks, shards, level_image) = match (&user_chunks, &options.shards) {
            (Some(c), None) => {
                let c = retuple(c, &shape);
                let li = level.rechunk(&c);
                (c, None, li)
            }
            (c, Some(s)) => {
                let c = retuple(c.as_deref().unwrap_or(&level.chunksize()), &shape);
                let s = retuple(s, &shape);
                let li = level.rechunk(&s);
                (c, Some(s), li)
            }
            (None, None) => (level.chunksize(), None, level.clone()),
        };
        if shards.is_some() && zarr_format == 2 {
            return Err(Error::value("Sharding is only supported for zarr_format 3"));
        }
        shapes.push(shape.clone());
        debug!(
            "write dask.array to_zarr shape: {:?}, dtype: {}",
            shape,
            level_image.dtype()
        );
        let mut spec = ArraySpec::new(&shape, level_image.dtype(), &chunks, zarr_format);
        spec.shards = shards;
        spec.compressor = compressor;
        spec.fill_value = options.fill_value.unwrap_or(0.0);
        spec.separator = "/".into();
        if zarr_format != 2 {
            spec.dimension_names = Some(axes.to_vec());
        }
        let array = group.create_array(&format!("s{idx}"), &spec, false)?;
        delayed.push(DelayedWrite {
            array,
            data: level_image,
        });
        datasets.push(json!({"path": format!("s{idx}")}));
    }
    if compute {
        compute_all(&delayed)?;
        delayed.clear();
    }
    let mut cts = coordinate_transformations;
    if cts.is_none() {
        cts = fmt.generate_coordinate_transformations(&shapes);
        if let Some(cts) = cts.as_mut() {
            for transform in cts.iter_mut() {
                let s: Vec<f64> = transform[0]["scale"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .zip(axes)
                    .map(|(v, d)| v.as_f64().unwrap() * scale.get(d).unwrap_or(1.0))
                    .collect();
                transform[0]["scale"] = json!(s);
            }
        }
    }
    fmt.validate_coordinate_transformations(
        pyramid[0].ndim() as i64,
        datasets.len(),
        cts.as_deref(),
    )?;
    if let Some(cts) = cts {
        for (d, t) in datasets.iter_mut().zip(cts) {
            d["coordinateTransformations"] = Value::Array(t);
        }
    }
    write_multiscales_metadata(
        group,
        &datasets,
        Some(fmt),
        Some(&AxesInput::Names(axes.to_vec())),
        axes_units,
        name,
        metadata,
    )?;
    Ok(delayed)
}

/// Write image-label metadata to the group. The label data must have been
/// written to a sub-group with the same name.
pub fn write_label_metadata(
    group: impl Into<GroupLike>,
    name: &str,
    colors: Option<&[Value]>,
    properties: Option<&[Value]>,
    fmt: Option<Format>,
    metadata: Option<&JsonMap>,
) -> Result<()> {
    let (group, fmt) = check_group_fmt(group, fmt, "a")?;
    let label_group = group.group(name)?;
    let mut image_label = metadata.cloned().unwrap_or_default();
    if let Some(c) = colors {
        image_label.insert("colors".into(), Value::Array(c.to_vec()));
    }
    if let Some(p) = properties {
        image_label.insert("properties".into(), Value::Array(p.to_vec()));
    }
    image_label.insert("version".into(), json!(fmt.version()));
    let mut label_list = get_metadata(&group)?
        .get("labels")
        .and_then(|l| l.as_array())
        .cloned()
        .unwrap_or_default();
    label_list.push(json!(name));
    let mut m = JsonMap::new();
    m.insert("labels".into(), Value::Array(label_list));
    add_metadata(&group, &m, Some(fmt))?;
    let mut m = JsonMap::new();
    m.insert("image-label".into(), Value::Object(image_label));
    add_metadata(&label_group, &m, Some(fmt))
}

/// The OME metadata of a group (`attrs["ome"]` for zarr v3).
pub fn get_metadata(group: &Group) -> Result<JsonMap> {
    let attrs = group.attrs()?;
    if group.zarr_format() == 3 {
        Ok(attrs
            .get("ome")
            .and_then(|o| o.as_object())
            .cloned()
            .unwrap_or_default())
    } else {
        Ok(attrs)
    }
}

/// Open a group by path and return its OME metadata.
pub fn get_metadata_path(path: &str) -> Result<JsonMap> {
    get_metadata(&Group::open_url(path, Mode::Read, None)?)
}

/// Add metadata to the group, merging dicts (`attrs["ome"]` for v0.5+).
pub fn add_metadata(
    group: impl Into<GroupLike>,
    metadata: &JsonMap,
    fmt: Option<Format>,
) -> Result<()> {
    let (group, fmt) = check_group_fmt(group, fmt, "a")?;
    let legacy = matches!(fmt, Format::V01 | Format::V02 | Format::V03 | Format::V04);
    let mut all = group.attrs()?;
    let mut attrs = if legacy {
        all.clone()
    } else {
        all.get("ome")
            .and_then(|o| o.as_object())
            .cloned()
            .unwrap_or_default()
    };
    for (k, v) in metadata {
        match (v, attrs.get_mut(k)) {
            (Value::Object(new), Some(Value::Object(old))) => {
                for (k2, v2) in new {
                    old.insert(k2.clone(), v2.clone());
                }
            }
            _ => {
                attrs.insert(k.clone(), v.clone());
            }
        }
    }
    if legacy {
        for (k, v) in attrs {
            all.insert(k, v);
        }
    } else {
        all.insert("ome".into(), Value::Object(attrs));
    }
    group.set_attrs(&all)
}

/// Options for [`write_multiscale_labels`].
#[derive(Debug, Clone, Default)]
pub struct LabelsOptions {
    pub multiscale: MultiscaleOptions,
    /// `write_label_metadata` keywords (colors, properties, ...).
    pub label_metadata: Option<JsonMap>,
}

/// Write pyramidal image labels to disk, in the sub-group `labels/{name}`.
pub fn write_multiscale_labels(
    pyramid: &[LazyArray],
    group: impl Into<GroupLike>,
    name: &str,
    opts: &LabelsOptions,
) -> Result<Vec<DelayedWrite>> {
    let mo = &opts.multiscale;
    let (group, fmt) = check_group_fmt(group, mo.fmt, "a")?;
    let sub_group = group.require_group(&format!("labels/{name}"))?;
    let scale = match &mo.scale {
        Some(s) => s.clone(),
        None => {
            let a = get_valid_axes(
                Some(pyramid[0].ndim()),
                mo.axes.as_ref(),
                mo.axes_units.as_ref(),
                fmt,
            )?;
            ScaleMap::ones(&extract_dims_from_axes(a.as_ref())?)
        }
    };
    if mo.coordinate_transformations.is_some() {
        deprecation_ct();
    }
    let delayed = write_pyramid_to_zarr(
        pyramid,
        &sub_group,
        fmt,
        &scale,
        &scale.keys(),
        mo.axes_units.as_ref(),
        mo.coordinate_transformations.clone(),
        mo.storage_options.as_ref(),
        Some(name),
        mo.compute,
        mo.metadata.as_ref(),
    )?;
    let (colors, properties, rest) = split_label_metadata(opts.label_metadata.as_ref());
    write_label_metadata(
        group.group("labels")?,
        name,
        colors.as_deref(),
        properties.as_deref(),
        Some(fmt),
        Some(&rest),
    )?;
    Ok(delayed)
}

fn split_label_metadata(m: Option<&JsonMap>) -> (Option<Vec<Value>>, Option<Vec<Value>>, JsonMap) {
    let mut rest = m.cloned().unwrap_or_default();
    let colors = rest
        .shift_remove("colors")
        .and_then(|c| c.as_array().cloned());
    let properties = rest
        .shift_remove("properties")
        .and_then(|c| c.as_array().cloned());
    (colors, properties, rest)
}

/// Write image label data to disk, including multiscale and image-label
/// metadata, in the sub-group `labels/{name}`.
pub fn write_labels(
    labels: &LazyArray,
    group: impl Into<GroupLike>,
    name: &str,
    opts: &ImageOptions,
) -> Result<Vec<DelayedWrite>> {
    let (group, fmt) = check_group_fmt(group, opts.fmt, "a")?;
    let sub_group = group.require_group(&format!("labels/{name}"))?;
    if matches!(fmt, Format::V01 | Format::V02 | Format::V03) {
        return Err(Error::Deprecation(format!(
            "Writing ome-zarr v{} is deprecated and has been removed in version 0.15.0.",
            fmt.version()
        )));
    }
    let axes = get_valid_axes(
        Some(labels.ndim()),
        opts.axes.as_ref(),
        opts.axes_units.as_ref(),
        fmt,
    )?;
    let dims = extract_dims_from_axes(axes.as_ref())?;
    let scale = opts.scale.clone().unwrap_or_else(|| ScaleMap::ones(&dims));
    let method = opts.method.unwrap_or(Methods::Nearest);
    let mut scale_factors = opts.scale_factors.clone();
    if let Some(scaler) = &opts.scaler {
        warn!(
            "The 'scaler' argument is deprecated and will be removed in version 0.13.0. Please use the 'scale_factors' argument instead."
        );
        scale_factors = Some(scaler_scale_factors(scaler, &dims));
    }
    if opts.coordinate_transformations.is_some() {
        warn!(
            "The 'coordinate_transformations' argument is deprecated and will be removed in a future version."
        );
    }
    let single = OmeZarrImage::new(
        labels.clone(),
        &dims,
        Some(scale),
        units_to_strings(opts.axes_units.as_ref()),
        None,
        name,
    )?;
    let mut ms = OmeZarrLabels::new(single, scale_factors, Some(method), true)?;
    ms.image_label = opts.image_label.clone();
    let delayed = ms.to_ome_zarr(
        GroupLike::Group(sub_group),
        opts.storage_options.as_ref(),
        Version::from_format(fmt),
        opts.compute,
        true,
    )?;
    let (colors, properties, rest) = split_label_metadata(opts.label_metadata.as_ref());
    write_label_metadata(
        group.group("labels")?,
        name,
        colors.as_deref(),
        properties.as_deref(),
        Some(fmt),
        Some(&rest),
    )?;
    Ok(delayed)
}
