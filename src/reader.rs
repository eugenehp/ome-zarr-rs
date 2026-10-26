//! Reading logic for ome-zarr: [`Reader`] walks an OME-Zarr hierarchy and
//! returns [`Node`]s, each tagged with the [`Spec`]s it implements.

use std::cell::RefCell;
use std::rc::Rc;

use log::{debug, error, info, warn};
use serde_json::{Value, json};

use crate::array::LazyArray;
use crate::axes::{Axes, AxesInput};
use crate::error::{Error, Result};
use crate::format::{format_from_version, is_truthy};
use crate::io::ZarrLocation;
use crate::zarr::JsonMap;

/// Locations already visited while walking a hierarchy.
pub type Seen = Rc<RefCell<Vec<ZarrLocation>>>;

/// The kinds of specification a node can implement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecKind {
    Labels,
    Label,
    Multiscales,
    Omero,
    Plate,
    Well,
}

impl std::fmt::Display for SpecKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            SpecKind::Labels => "Labels",
            SpecKind::Label => "Label",
            SpecKind::Multiscales => "Multiscales",
            SpecKind::Omero => "OMERO",
            SpecKind::Plate => "Plate",
            SpecKind::Well => "Well",
        })
    }
}

/// Information recorded by the Multiscales spec.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiscalesSpec {
    /// Paths of the resolution levels.
    pub datasets: Vec<String>,
}

/// Information recorded by the OMERO spec.
#[derive(Debug, Clone, PartialEq)]
pub struct OmeroSpec {
    pub image_data: JsonMap,
}

/// Information recorded by the Well spec.
#[derive(Debug, Clone, PartialEq)]
pub struct WellSpec {
    pub well_data: JsonMap,
    pub ds_paths: Vec<String>,
    pub numpy_type: crate::array::DType,
    pub img_shape: Vec<u64>,
    pub img_metadata: JsonMap,
    pub img_pyramid_shapes: Vec<Vec<u64>>,
}

/// Information recorded by the Plate spec.
#[derive(Debug, Clone, PartialEq)]
pub struct PlateSpec {
    pub plate_data: JsonMap,
    pub row_names: Vec<String>,
    pub col_names: Vec<String>,
    pub well_paths: Vec<String>,
    pub row_count: usize,
    pub column_count: usize,
    pub first_field_path: String,
    pub img_paths: Vec<String>,
    pub numpy_type: crate::array::DType,
    pub axes: Value,
}

/// A specification implemented by a group (multiple may apply).
#[derive(Debug, Clone, PartialEq)]
pub enum Spec {
    Labels,
    Label,
    Multiscales(MultiscalesSpec),
    Omero(OmeroSpec),
    Plate(PlateSpec),
    Well(WellSpec),
}

impl Spec {
    pub fn kind(&self) -> SpecKind {
        match self {
            Spec::Labels => SpecKind::Labels,
            Spec::Label => SpecKind::Label,
            Spec::Multiscales(_) => SpecKind::Multiscales,
            Spec::Omero(_) => SpecKind::Omero,
            Spec::Plate(_) => SpecKind::Plate,
            Spec::Well(_) => SpecKind::Well,
        }
    }

    /// Does the location match this kind of spec?
    pub fn matches(kind: SpecKind, zarr: &ZarrLocation) -> bool {
        let attrs = zarr.root_attrs();
        match kind {
            SpecKind::Labels => attrs.contains_key("labels"),
            SpecKind::Label => attrs.contains_key("image-label"),
            SpecKind::Multiscales => !zarr.zgroup.is_empty() && attrs.contains_key("multiscales"),
            SpecKind::Omero => attrs.contains_key("omero"),
            SpecKind::Plate => attrs.contains_key("plate"),
            SpecKind::Well => attrs.contains_key("well"),
        }
    }
}

/// Container for a representation of the binary data somewhere in the data
/// hierarchy.
#[derive(Debug, Clone)]
pub struct Node {
    pub zarr: ZarrLocation,
    pub seen: Seen,
    visible: bool,
    /// Metadata collected by the specs (napari-style layer metadata).
    pub metadata: JsonMap,
    /// The arrays (resolution levels) of this node.
    pub data: Vec<LazyArray>,
    pub specs: Vec<Spec>,
    pub pre_nodes: Vec<Node>,
    pub post_nodes: Vec<Node>,
}

fn lookup(zarr: &ZarrLocation, key: &str) -> Value {
    zarr.root_attrs().get(key).cloned().unwrap_or(Value::Null)
}

/// Python-style tuple formatting, e.g. `('c', 'y', 'x')` / `(3, 512)`.
pub fn py_tuple<T: std::fmt::Display>(items: &[T], quote: bool) -> String {
    let parts: Vec<String> = items
        .iter()
        .map(|i| {
            if quote {
                format!("'{i}'")
            } else {
                i.to_string()
            }
        })
        .collect();
    if parts.len() == 1 {
        format!("({},)", parts[0])
    } else {
        format!("({})", parts.join(", "))
    }
}

impl Node {
    /// Create a node; specs matching the location are instantiated (which may
    /// add child nodes).
    pub fn new(zarr: ZarrLocation, seen: Seen, visibility: bool) -> Result<Node> {
        let mut node = Node {
            zarr: zarr.clone(),
            seen,
            visible: visibility,
            metadata: JsonMap::new(),
            data: vec![],
            specs: vec![],
            pre_nodes: vec![],
            post_nodes: vec![],
        };
        if Spec::matches(SpecKind::Labels, &zarr) {
            node.init_labels()?;
        }
        if Spec::matches(SpecKind::Label, &zarr) {
            node.init_label()?;
        }
        if Spec::matches(SpecKind::Multiscales, &zarr) {
            node.init_multiscales()?;
        }
        if Spec::matches(SpecKind::Omero, &zarr) {
            node.init_omero();
        }
        if Spec::matches(SpecKind::Plate, &zarr) {
            node.init_plate()?;
        }
        if Spec::matches(SpecKind::Well, &zarr) {
            node.init_well()?;
        }
        Ok(node)
    }

    /// A node with a fresh "seen" list (Python: `Node(zarr, list())`).
    pub fn root(zarr: ZarrLocation) -> Result<Node> {
        Node::new(zarr, Rc::new(RefCell::new(vec![])), true)
    }

    /// The first spec of the given kind.
    pub fn first(&self, kind: SpecKind) -> Option<&Spec> {
        self.specs.iter().find(|s| s.kind() == kind)
    }

    /// Alias for [`Node::first`].
    pub fn load(&self, kind: SpecKind) -> Option<&Spec> {
        self.first(kind)
    }

    /// True if this node should be displayed by default.
    pub fn visible(&self) -> bool {
        self.visible
    }

    /// Set the visibility, propagating to all subnodes; returns the old value.
    pub fn set_visible(&mut self, visibility: bool) -> bool {
        let old = self.visible;
        if old != visibility {
            self.visible = visibility;
            for n in self.pre_nodes.iter_mut().chain(self.post_nodes.iter_mut()) {
                n.set_visible(visibility);
            }
        }
        old
    }

    /// Create a child node if this location has not yet been seen.
    pub fn add(
        &mut self,
        zarr: ZarrLocation,
        prepend: bool,
        visibility: Option<bool>,
        plate_labels: bool,
    ) -> Result<bool> {
        if self.seen.borrow().contains(&zarr) && !plate_labels {
            debug!("already seen  {zarr}; stopping recursion");
            return Ok(false);
        }
        let visibility = visibility.unwrap_or(self.visible);
        self.seen.borrow_mut().push(zarr.clone());
        let node = Node::new(zarr, self.seen.clone(), visibility)?;
        if prepend {
            self.pre_nodes.push(node);
        } else {
            self.post_nodes.push(node);
        }
        Ok(true)
    }

    /// Merge this node's root attributes into `metadata`.
    pub fn write_metadata(&self, metadata: &mut JsonMap) {
        for _ in &self.specs {
            for (k, v) in self.zarr.root_attrs() {
                metadata.insert(k, v);
            }
        }
    }

    fn log_spec(&self, name: &str) {
        debug!("treating {} as {name}", self.zarr);
        for (k, v) in self.zarr.root_attrs() {
            info!("root_attr: {k}");
            debug!("{}", crate::pyjson::py_str(&v));
        }
    }

    /// Labels: the well-known "labels" group, listing label sub-groups.
    fn init_labels(&mut self) -> Result<()> {
        self.log_spec("Labels");
        self.specs.push(Spec::Labels);
        let names = lookup(&self.zarr, "labels");
        for name in names.as_array().cloned().unwrap_or_default() {
            let Some(name) = name.as_str() else { continue };
            let child = self.zarr.create(name)?;
            if child.exists() {
                self.add(child, false, None, false)?;
            }
        }
        Ok(())
    }

    /// Label: a labeled image (image-label metadata).
    fn init_label(&mut self) -> Result<()> {
        self.log_spec("Label");
        self.specs.push(Spec::Label);
        let image_label = lookup(&self.zarr, "image-label");
        let image = image_label
            .get("source")
            .and_then(|s| s.get("image"))
            .and_then(|i| i.as_str());
        let mut parent_found = false;
        if let Some(image) = image.filter(|s| !s.is_empty()) {
            let parent = self.zarr.create(image)?;
            if parent.exists() {
                debug!("delegating to parent image: {parent}");
                self.add(parent, true, Some(false), false)?;
                parent_found = true;
            }
        }
        if !parent_found {
            warn!("no parent found for Label: {image:?}");
        }
        let mut colors = JsonMap::new();
        for color in image_label
            .get("colors")
            .and_then(|c| c.as_array())
            .cloned()
            .unwrap_or_default()
        {
            let lv = color.get("label-value");
            let rgba = color.get("rgba").and_then(|r| r.as_array()).map(|r| {
                Value::Array(
                    r.iter()
                        .map(|x| json!(x.as_f64().unwrap_or(0.0) / 255.0))
                        .collect(),
                )
            });
            match lv {
                Some(Value::Bool(b)) => {
                    colors.insert(
                        if *b { "True" } else { "False" }.into(),
                        rgba.unwrap_or(Value::Null),
                    );
                }
                Some(Value::Number(n)) if n.is_i64() || n.is_u64() => {
                    colors.insert(n.to_string(), rgba.unwrap_or(Value::Null));
                }
                _ => error!("invalid color - {color}"),
            }
        }
        let mut properties = JsonMap::new();
        for props in image_label
            .get("properties")
            .and_then(|c| c.as_array())
            .cloned()
            .unwrap_or_default()
        {
            let Some(obj) = props.as_object() else {
                continue;
            };
            let lv = obj
                .get("label-value")
                .map(|v| v.to_string())
                .unwrap_or_default();
            let mut p = obj.clone();
            p.shift_remove("label-value");
            properties.insert(lv, Value::Object(p));
        }
        let name = self.zarr.basename();
        self.metadata.insert("visible".into(), json!(self.visible));
        self.metadata.insert("name".into(), json!(name));
        self.metadata.insert("color".into(), Value::Object(colors));
        let image_meta = match lookup(&self.zarr, "image") {
            Value::Null => json!({}),
            v => v,
        };
        self.metadata.insert(
            "metadata".into(),
            json!({"image": image_meta, "path": name}),
        );
        if !properties.is_empty() {
            self.metadata
                .insert("properties".into(), Value::Object(properties));
        }
        Ok(())
    }

    /// Multiscales: a pyramid of resolution levels.
    fn init_multiscales(&mut self) -> Result<()> {
        self.log_spec("Multiscales");
        let multiscales = lookup(&self.zarr, "multiscales");
        let ms0 = multiscales
            .as_array()
            .and_then(|a| a.first())
            .cloned()
            .ok_or_else(|| Error::value("multiscales metadata is empty"))?;
        let version = ms0.get("version").and_then(|v| v.as_str()).unwrap_or("0.1");
        let datasets = ms0
            .get("datasets")
            .and_then(|d| d.as_array())
            .cloned()
            .ok_or_else(|| Error::key("datasets"))?;
        let axes = ms0.get("axes").filter(|a| !a.is_null());
        let fmt = format_from_version(version)?;
        let axes_input = axes.map(AxesInput::from_json).transpose()?;
        let axes_obj = Axes::new(axes_input.as_ref(), None, fmt)?;
        self.metadata.insert(
            "axes".into(),
            axes_obj.to_list(crate::format::CURRENT_FORMAT)?,
        );
        self.metadata.insert(
            "name".into(),
            ms0.get("name").cloned().unwrap_or(Value::Null),
        );
        let paths: Vec<String> = datasets
            .iter()
            .map(|d| {
                d.get("path")
                    .and_then(|p| p.as_str())
                    .map(String::from)
                    .ok_or_else(|| Error::key("path"))
            })
            .collect::<Result<_>>()?;
        let transformations: Vec<Value> = datasets
            .iter()
            .map(|d| {
                d.get("coordinateTransformations")
                    .cloned()
                    .unwrap_or(Value::Null)
            })
            .collect();
        if transformations.iter().any(|t| !t.is_null()) {
            self.metadata.insert(
                "coordinateTransformations".into(),
                Value::Array(transformations),
            );
        }
        info!(
            "datasets {}",
            crate::pyjson::repr(&Value::Array(datasets.clone()))
        );
        let axes_names: Option<Vec<String>> = axes
            .and_then(|a| AxesInput::from_json(a).ok())
            .and_then(|a| a.names().ok());
        let loaded: Vec<LazyArray> = {
            use crate::par::*;
            paths
                .par_iter()
                .map(|p| self.zarr.load(p))
                .collect::<Result<_>>()?
        };
        for (resolution, data) in paths.iter().zip(loaded) {
            let chunk_sizes: Vec<String> = data
                .chunks()
                .iter()
                .map(|c| {
                    let (first, last) = (c[0], *c.last().unwrap());
                    if last != first {
                        format!("{first} (+ {last})")
                    } else {
                        first.to_string()
                    }
                })
                .collect();
            info!("resolution: {resolution}");
            let names = match &axes_names {
                Some(n) => py_tuple(n, true),
                None => "None".into(),
            };
            info!(" - shape {names} = {}", py_tuple(data.shape(), false));
            let quoted: Vec<String> = chunk_sizes.iter().map(|c| format!("'{c}'")).collect();
            info!(" - chunks =  [{}]", quoted.join(", "));
            info!(" - dtype = {}", data.dtype());
            self.data.push(data);
        }
        self.specs
            .push(Spec::Multiscales(MultiscalesSpec { datasets: paths }));
        let child = self.zarr.create("labels")?;
        if child.exists() {
            self.add(child, false, Some(false), false)?;
        }
        Ok(())
    }

    /// OMERO: rendering settings.
    fn init_omero(&mut self) {
        self.log_spec("OMERO");
        let image_data = lookup(&self.zarr, "omero")
            .as_object()
            .cloned()
            .unwrap_or_default();
        self.specs.push(Spec::Omero(OmeroSpec {
            image_data: image_data.clone(),
        }));
        let mut model = "unknown".to_string();
        if let Some(rdefs) = image_data.get("rdefs").filter(|r| is_truthy(r)) {
            model = rdefs
                .get("model")
                .and_then(|m| m.as_str())
                .unwrap_or("unset")
                .to_string();
        }
        let Some(channels) = image_data.get("channels").filter(|c| !c.is_null()) else {
            return;
        };
        let Some(channels) = channels.as_array() else {
            warn!("error counting channels: {channels}");
            return;
        };
        let n = channels.len();
        let mut colormaps = vec![];
        let mut contrast_limits: Option<Vec<Value>> = Some(vec![Value::Null; n]);
        let mut names: Vec<Value> = (0..n).map(|i| json!(format!("channel_{i}"))).collect();
        let mut visibles: Vec<Value> = vec![json!(true); n];
        for (idx, ch) in channels.iter().enumerate() {
            if let Some(color) = ch.get("color").filter(|c| !c.is_null()) {
                let Some(rgb) = color.as_str().and_then(crate::conversions::hex_to_rgb) else {
                    error!("Failed to parse metadata");
                    return;
                };
                // (Python's greyscale colormap is the ints [1, 1, 1])
                let rgb = if model == "greyscale" {
                    json!([1, 1, 1])
                } else {
                    json!(rgb)
                };
                colormaps.push(json!([[0, 0, 0], rgb]));
            }
            if let Some(label) = ch.get("label").filter(|c| !c.is_null()) {
                names[idx] = label.clone();
            }
            if let Some(active) = ch.get("active").filter(|c| !c.is_null()) {
                visibles[idx] = json!(is_truthy(active) && self.visible);
            }
            if let Some(window) = ch.get("window").filter(|c| !c.is_null()) {
                let start = window.get("start").filter(|v| !v.is_null());
                let end = window.get("end").filter(|v| !v.is_null());
                match (start, end) {
                    (Some(s), Some(e)) => {
                        if let Some(cl) = contrast_limits.as_mut() {
                            cl[idx] = json!([s, e]);
                        }
                    }
                    _ => contrast_limits = None,
                }
            }
        }
        self.metadata
            .insert("channel_names".into(), Value::Array(names));
        self.metadata
            .insert("visible".into(), Value::Array(visibles));
        self.metadata.insert(
            "contrast_limits".into(),
            contrast_limits.map(Value::Array).unwrap_or(Value::Null),
        );
        self.metadata
            .insert("colormap".into(), Value::Array(colormaps));
    }

    /// Well: a grid of fields of view.
    fn init_well(&mut self) -> Result<()> {
        self.log_spec("Well");
        let well_data = lookup(&self.zarr, "well")
            .as_object()
            .cloned()
            .unwrap_or_default();
        info!(
            "well_data: {}",
            crate::pyjson::repr(&Value::Object(well_data.clone()))
        );
        let image_paths: Vec<String> = well_data
            .get("images")
            .and_then(|i| i.as_array())
            .ok_or_else(|| Error::type_err("well has no images"))?
            .iter()
            .map(|i| {
                i.get("path")
                    .and_then(|p| p.as_str())
                    .map(String::from)
                    .ok_or_else(|| Error::key("path"))
            })
            .collect::<Result<_>>()?;
        let field_count = image_paths.len();
        if field_count == 0 {
            return Err(Error::value("well has no images"));
        }
        let column_count = (field_count as f64).sqrt().ceil() as usize;
        let row_count = field_count.div_ceil(column_count);
        let image_zarr = self.zarr.create(&image_paths[0])?;
        let image_node = Node::new(image_zarr.clone(), self.seen.clone(), self.visible)?;
        let ds_paths: Vec<String> = image_zarr.root_attrs()["multiscales"][0]["datasets"]
            .as_array()
            .ok_or_else(|| Error::key("multiscales"))?
            .iter()
            .filter_map(|d| d["path"].as_str().map(String::from))
            .collect();
        let naxes = image_node.metadata["axes"]
            .as_array()
            .map(|a| a.len())
            .unwrap_or(0);
        let (x_index, y_index) = (naxes - 1, naxes - 2);
        let numpy_type = image_node.data[0].dtype();
        let img_shape = image_node.data[0].shape().to_vec();
        let img_metadata = image_node.metadata.clone();
        let pyramid_shapes: Vec<Vec<u64>> =
            image_node.data.iter().map(|d| d.shape().to_vec()).collect();
        let mut pyramid = vec![];
        for (level, shape) in pyramid_shapes.iter().enumerate() {
            let mut lazy_rows = vec![];
            for row in 0..row_count {
                let mut lazy_row = vec![];
                for col in 0..column_count {
                    debug!("creating lazy_reader. row: {row} col: {col} level: {level}");
                    let field_index = column_count * row + col;
                    let mut data = None;
                    if field_index < image_paths.len() {
                        let path = format!("{}/{}", image_paths[field_index], ds_paths[level]);
                        match self.zarr.load(&path) {
                            Ok(d) => data = Some(d),
                            Err(_) => error!("Failed to load {path}"),
                        }
                    }
                    lazy_row
                        .push(data.unwrap_or_else(|| LazyArray::zeros(shape, numpy_type, None)));
                }
                lazy_rows.push(LazyArray::concatenate(&lazy_row, x_index)?);
            }
            pyramid.push(LazyArray::concatenate(&lazy_rows, y_index)?);
        }
        self.data = pyramid;
        self.metadata = img_metadata.clone();
        self.specs.push(Spec::Well(WellSpec {
            well_data,
            ds_paths,
            numpy_type,
            img_shape,
            img_metadata,
            img_pyramid_shapes: pyramid_shapes,
        }));
        Ok(())
    }

    /// Plate: a grid of wells, stitched lazily.
    fn init_plate(&mut self) -> Result<()> {
        self.log_spec("Plate");
        debug!("Plate created with ZarrLocation fmt: {}", self.zarr.fmt());
        let plate_data = lookup(&self.zarr, "plate")
            .as_object()
            .cloned()
            .unwrap_or_default();
        info!(
            "plate_data: {}",
            crate::pyjson::repr(&Value::Object(plate_data.clone()))
        );
        let names_of = |key: &str| -> Result<Vec<String>> {
            Ok(plate_data
                .get(key)
                .and_then(|r| r.as_array())
                .ok_or_else(|| Error::type_err(format!("plate has no {key}")))?
                .iter()
                .filter_map(|r| r.get("name").and_then(|n| n.as_str()).map(String::from))
                .collect())
        };
        let row_names = names_of("rows")?;
        let col_names = names_of("columns")?;
        let mut well_paths: Vec<String> = plate_data
            .get("wells")
            .and_then(|w| w.as_array())
            .ok_or_else(|| Error::type_err("plate has no wells"))?
            .iter()
            .filter_map(|w| w.get("path").and_then(|p| p.as_str()).map(String::from))
            .collect();
        well_paths.sort();
        let row_count = row_names.len();
        let column_count = col_names.len();
        let well_zarr = self.zarr.create(&well_paths[0])?;
        let well_node = Node::new(well_zarr, self.seen.clone(), self.visible)?;
        let Some(Spec::Well(well_spec)) = well_node.first(SpecKind::Well).cloned() else {
            return Err(Error::value("Could not find first well"));
        };
        let first_field_path = well_spec.well_data["images"][0]["path"]
            .as_str()
            .ok_or_else(|| Error::key("path"))?
            .to_string();
        let img0 = self
            .zarr
            .create(&format!("{}/{first_field_path}", well_paths[0]))?;
        let img_paths: Vec<String> = img0.root_attrs()["multiscales"][0]["datasets"]
            .as_array()
            .ok_or_else(|| Error::key("multiscales"))?
            .iter()
            .filter_map(|d| d["path"].as_str().map(String::from))
            .collect();
        let numpy_type = well_spec.numpy_type;
        debug!(
            "img_pyramid_shapes: [{}]",
            well_spec
                .img_pyramid_shapes
                .iter()
                .map(|s| py_tuple(s, false))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let axes = well_spec
            .img_metadata
            .get("axes")
            .cloned()
            .unwrap_or(Value::Null);
        let naxes = axes.as_array().map(|a| a.len()).unwrap_or(0);
        let mut pyramid = vec![];
        for (level, tile_shape) in well_spec.img_pyramid_shapes.iter().enumerate() {
            debug!(
                "get_stitched_grid() level: {level}, tile_shape: {}",
                py_tuple(tile_shape, false)
            );
            let mut lazy_rows = vec![];
            for row in 0..row_count {
                let mut lazy_row = vec![];
                for col in 0..column_count {
                    let well_path = format!("{}/{}", row_names[row], col_names[col]);
                    let tile = if !well_paths.contains(&well_path) {
                        debug!("empty well: {well_path}");
                        LazyArray::zeros(tile_shape, numpy_type, None)
                    } else {
                        let path = format!("{well_path}/{first_field_path}/{}", img_paths[level]);
                        debug!(
                            "creating tile... {path} with shape: {}",
                            py_tuple(tile_shape, false)
                        );
                        match self.zarr.load(&path) {
                            Ok(d) => d,
                            Err(e) => {
                                error!("Failed to load {path}: {e}");
                                LazyArray::zeros(tile_shape, numpy_type, None)
                            }
                        }
                    };
                    lazy_row.push(tile);
                }
                lazy_rows.push(LazyArray::concatenate(&lazy_row, naxes - 1)?);
            }
            pyramid.push(LazyArray::concatenate(&lazy_rows, naxes - 2)?);
        }
        self.data = pyramid;
        self.metadata = well_spec.img_metadata.clone();
        self.metadata.insert(
            "metadata".into(),
            json!({"plate": Value::Object(plate_data.clone())}),
        );
        self.specs.push(Spec::Plate(PlateSpec {
            plate_data,
            row_names,
            col_names,
            well_paths,
            row_count,
            column_count,
            first_field_path,
            img_paths,
            numpy_type,
            axes,
        }));
        Ok(())
    }
}

impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let suffix = if self.visible { "" } else { " (hidden)" };
        write!(f, "{}{suffix}", self.zarr)
    }
}

/// Parses the given Zarr instance into a collection of Nodes properly ordered
/// depending on context.
///
/// Depending on the starting point, metadata may be followed up or down the
/// Zarr hierarchy.
#[derive(Debug, Clone)]
pub struct Reader {
    pub zarr: ZarrLocation,
    pub seen: Seen,
}

impl Reader {
    pub fn new(zarr: ZarrLocation) -> Result<Reader> {
        if !zarr.exists() {
            return Err(Error::Assertion(format!("{zarr} does not exist")));
        }
        let seen = Rc::new(RefCell::new(vec![zarr.clone()]));
        Ok(Reader { zarr, seen })
    }

    /// Walk the hierarchy and return all nodes (`Reader.__call__`).
    pub fn nodes(&self) -> Result<Vec<Node>> {
        let mut out = vec![];
        self.visit(&mut |n| {
            out.push(n.clone());
            Ok(())
        })?;
        Ok(out)
    }

    /// Call `f` with each node in the order [`Reader::nodes`] returns them, as
    /// each is reached (like iterating the Python generator).
    pub fn visit(&self, f: &mut dyn FnMut(&Node) -> Result<()>) -> Result<()> {
        let mut node = Node::new(self.zarr.clone(), self.seen.clone(), true)?;
        if !node.specs.is_empty() {
            debug!("treating {} as ome-zarr", self.zarr);
            Self::descend(&node, f)
        } else if !self.zarr.zarray.is_empty() {
            debug!("treating {} as raw zarr", self.zarr);
            node.data.push(self.zarr.load("")?);
            f(&node)
        } else {
            debug!("ignoring {}", self.zarr);
            Ok(())
        }
    }

    fn descend(node: &Node, f: &mut dyn FnMut(&Node) -> Result<()>) -> Result<()> {
        for pre in &node.pre_nodes {
            Self::descend(pre, f)?;
        }
        debug!("returning {node}");
        f(node)?;
        for post in &node.post_nodes {
            Self::descend(post, f)?;
        }
        Ok(())
    }
}
