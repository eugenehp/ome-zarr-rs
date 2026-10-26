//! Hierarchy of format OME-Zarr implementations.

use std::fmt;

use log::debug;
use serde_json::{Value, json};

use crate::error::{Error, Result};
use crate::pyjson::{args_msg, repr, repr_str};
use crate::store::Store;
use crate::zarr::JsonMap;

/// An OME-NGFF format version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Format {
    /// Initial format. (2020)
    V01,
    /// Changelog: move to nested storage (April 2021)
    V02,
    /// Changelog: variable number of dimensions (up to 5),
    /// introduce axes field in multiscales (June 2021)
    V03,
    /// Changelog: axes is list of dicts,
    /// introduce coordinate_transformations in multiscales (Nov 2021)
    V04,
    /// Changelog: added FormatV05 (May 2025)
    V05,
}

/// The most recent format.
pub const CURRENT_FORMAT: Format = Format::V05;

impl Default for Format {
    fn default() -> Self {
        CURRENT_FORMAT
    }
}

impl fmt::Display for Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Format::V01 => "FormatV01",
            Format::V02 => "FormatV02",
            Format::V03 => "FormatV03",
            Format::V04 => "FormatV04",
            Format::V05 => "FormatV05",
        };
        f.write_str(name)
    }
}

/// Return an instance of each format implementation, newest to oldest.
pub fn format_implementations() -> [Format; 5] {
    [
        Format::V05,
        Format::V04,
        Format::V03,
        Format::V02,
        Format::V01,
    ]
}

/// Look up a format by its version string, e.g. `"0.4"`.
pub fn format_from_version(version: &str) -> Result<Format> {
    for fmt in format_implementations() {
        if fmt.version() == version {
            return Ok(fmt);
        }
    }
    Err(Error::value(format!("Version {version} not recognized")))
}

/// Give each format implementation a chance to take ownership of the
/// given metadata. If none matches, the default value will be returned.
pub fn detect_format(metadata: &JsonMap, default: Format) -> Format {
    if !metadata.is_empty() {
        for fmt in format_implementations() {
            if fmt.matches(metadata) {
                return fmt;
            }
        }
    }
    default
}

/// Required keys and JSON types for a plate's well entry.
#[derive(Debug, Clone, Copy)]
enum KeyType {
    Str,
    Int,
}

impl Format {
    /// The version string, e.g. `"0.4"`.
    pub fn version(&self) -> &'static str {
        match self {
            Format::V01 => "0.1",
            Format::V02 => "0.2",
            Format::V03 => "0.3",
            Format::V04 => "0.4",
            Format::V05 => "0.5",
        }
    }

    /// The zarr format (2 or 3) used by this OME-Zarr version.
    pub fn zarr_format(&self) -> u8 {
        match self {
            Format::V05 => 3,
            _ => 2,
        }
    }

    /// The chunk key encoding, as JSON.
    pub fn chunk_key_encoding(&self) -> Value {
        match self {
            Format::V01 => json!({"name": "v2", "separator": "."}),
            Format::V05 => json!({"name": "default", "separator": "/"}),
            _ => json!({"name": "v2", "separator": "/"}),
        }
    }

    /// Chunk key separator.
    pub fn dimension_separator(&self) -> &'static str {
        match self {
            Format::V01 => ".",
            _ => "/",
        }
    }

    /// Checks the metadata dict for a version.
    ///
    /// Returns the version of the first object found in the metadata,
    /// checking for 'multiscales', 'plate', 'well' etc
    pub fn get_metadata_version(metadata: &JsonMap) -> Option<String> {
        if let Some(ms) = metadata.get("multiscales").and_then(|m| m.as_array())
            && let Some(first) = ms.first()
        {
            return first
                .get("version")
                .and_then(|v| v.as_str())
                .map(String::from);
        }
        for name in ["plate", "well", "image-label"] {
            if let Some(obj) = metadata.get(name)
                && is_truthy(obj)
            {
                return obj
                    .get("version")
                    .and_then(|v| v.as_str())
                    .map(String::from);
            }
        }
        None
    }

    /// Whether this format owns the given metadata.
    pub fn matches(&self, metadata: &JsonMap) -> bool {
        let version = Format::get_metadata_version(metadata);
        debug!(
            "{} matches {}?",
            self.version(),
            version.as_deref().unwrap_or("None")
        );
        version.as_deref() == Some(self.version())
    }

    /// Create a store for the given path / URL.
    pub fn init_store(&self, path: &str, mode: &str) -> Result<Store> {
        let read_only = mode == "r";
        let store = Store::from_url(path, read_only)?;
        debug!("Created nested FsspecStore({path}, {mode})");
        Ok(store)
    }

    fn required_plate_well_keys(&self) -> &'static [(&'static str, KeyType)] {
        match self {
            Format::V01 | Format::V02 | Format::V03 => &[("path", KeyType::Str)],
            _ => &[
                ("path", KeyType::Str),
                ("rowIndex", KeyType::Int),
                ("columnIndex", KeyType::Int),
            ],
        }
    }

    /// Generate the well entry for a plate.
    pub fn generate_well_dict(
        &self,
        well: &str,
        rows: &[String],
        columns: &[String],
    ) -> Result<JsonMap> {
        let mut out = JsonMap::new();
        match self {
            Format::V01 | Format::V02 | Format::V03 => {
                out.insert("path".into(), json!(well));
            }
            _ => {
                let (row, column) = unpack2(well)?;
                let row_index = rows.iter().position(|r| r == row).ok_or_else(|| {
                    Error::value(args_msg(
                        "%s is not defined in the list of rows",
                        &[repr_str(row)],
                    ))
                })?;
                let column_index = columns.iter().position(|c| c == column).ok_or_else(|| {
                    Error::value(args_msg(
                        "%s is not defined in the list of columns",
                        &[repr_str(column)],
                    ))
                })?;
                out.insert("path".into(), json!(well));
                out.insert("rowIndex".into(), json!(row_index));
                out.insert("columnIndex".into(), json!(column_index));
            }
        }
        Ok(out)
    }

    /// Validate a well entry of a plate.
    pub fn validate_well_dict(
        &self,
        well: &JsonMap,
        rows: &[String],
        columns: &[String],
    ) -> Result<()> {
        let required = self.required_plate_well_keys();
        let wrepr = repr(&Value::Object(well.clone()));
        if well.keys().any(|k| !required.iter().any(|(r, _)| r == k)) {
            debug!("{wrepr} contains unspecified keys");
        }
        for (k, t) in required {
            let class = match t {
                KeyType::Str => "<class 'str'>",
                KeyType::Int => "<class 'int'>",
            };
            let v = well.get(*k).ok_or_else(|| {
                Error::type_err(args_msg(
                    "%s must contain a %s key of type %s",
                    &[wrepr.clone(), repr_str(k), class.into()],
                ))
            })?;
            let ok = match t {
                KeyType::Str => v.is_string(),
                // bool is an int in Python
                KeyType::Int => v.is_i64() || v.is_u64() || v.is_boolean(),
            };
            if !ok {
                return Err(Error::type_err(args_msg(
                    "%s path must be of %s type",
                    &[wrepr.clone(), class.into()],
                )));
            }
        }
        if matches!(self, Format::V04 | Format::V05) {
            let path = well["path"].as_str().unwrap_or("");
            let parts: Vec<&str> = path.split('/').collect();
            if parts.len() != 2 {
                return Err(Error::value(args_msg(
                    "%s path must exactly be composed of 2 groups",
                    &[wrepr],
                )));
            }
            let (row, column) = (parts[0], parts[1]);
            let ri = rows.iter().position(|r| r == row).ok_or_else(|| {
                Error::value(args_msg(
                    "%s is not defined in the plate rows",
                    &[repr_str(row)],
                ))
            })?;
            if py_int(&well["rowIndex"]) != Some(ri as i64) {
                return Err(Error::value(args_msg(
                    "Mismatching row index for %s",
                    &[wrepr],
                )));
            }
            let ci = columns.iter().position(|c| c == column).ok_or_else(|| {
                Error::value(args_msg(
                    "%s is not defined in the plate columns",
                    &[repr_str(column)],
                ))
            })?;
            if py_int(&well["columnIndex"]) != Some(ci as i64) {
                return Err(Error::value(args_msg(
                    "Mismatching column index for %s",
                    &[wrepr],
                )));
            }
        }
        Ok(())
    }

    /// Calculate the minimal 'scale' (and 'translation') transforms for each level
    /// of a pyramid, based on the level shapes. `None` for versions < 0.4.
    pub fn generate_coordinate_transformations(
        &self,
        shapes: &[Vec<u64>],
    ) -> Option<Vec<Vec<Value>>> {
        if !matches!(self, Format::V04 | Format::V05) {
            return None;
        }
        let data_shape = &shapes[0];
        let scale0 = vec![1.0f64; data_shape.len()];
        let mut out = vec![];
        for shape in shapes {
            assert_eq!(shape.len(), data_shape.len());
            let scale: Vec<f64> = data_shape
                .iter()
                .zip(shape)
                .map(|(&full, &level)| full as f64 / level as f64)
                .collect();
            let trans: Vec<f64> = scale
                .iter()
                .zip(&scale0)
                .map(|(s, s0)| s / 2.0 - s0 / 2.0)
                .collect();
            out.push(vec![
                json!({"type": "scale", "scale": scale}),
                json!({"type": "translation", "translation": trans}),
            ]);
        }
        Some(out)
    }

    /// Validates that each level's transformation list contains a single
    /// 'scale' transform (first) with `ndim` values and at most one translation.
    pub fn validate_coordinate_transformations(
        &self,
        ndim: i64,
        nlevels: usize,
        coordinate_transformations: Option<&[Vec<Value>]>,
    ) -> Result<()> {
        if !matches!(self, Format::V04 | Format::V05) {
            return Ok(());
        }
        let cts = coordinate_transformations
            .ok_or_else(|| Error::value("coordinate_transformations must be provided"))?;
        if cts.len() != nlevels {
            return Err(Error::value(format!(
                "coordinate_transformations count: {} must match datasets {nlevels}",
                cts.len()
            )));
        }
        let is_number = |v: &Value| v.is_number() || v.is_boolean();
        for transformations in cts {
            let trepr = repr(&Value::Array(transformations.clone()));
            let types: Vec<Option<&Value>> = transformations
                .iter()
                .map(|t| t.get("type").filter(|v| !v.is_null()))
                .collect();
            if types.iter().any(|t| t.is_none()) {
                return Err(Error::value(format!("Missing type in: {trepr}")));
            }
            let is = |t: &Option<&Value>, name: &str| t.and_then(|v| v.as_str()) == Some(name);
            if types.iter().filter(|t| is(t, "scale")).count() != 1 {
                return Err(Error::value(
                    "Must supply 1 'scale' item in coordinate_transformations",
                ));
            }
            if !is(&types[0], "scale") {
                return Err(Error::value(
                    "First coordinate_transformations must be 'scale'",
                ));
            }
            let first = &transformations[0];
            let scale = first.get("scale").ok_or_else(|| {
                Error::value(format!("Missing scale argument in: {}", repr(first)))
            })?;
            if py_len(scale)? as i64 != ndim {
                return Err(Error::value(format!(
                    "'scale' list {} must match number of image dimensions: {ndim}",
                    repr(scale)
                )));
            }
            if py_iter(scale).iter().any(|v| !is_number(v)) {
                return Err(Error::type_err(format!(
                    "'scale' values must all be numbers: {}",
                    repr(scale)
                )));
            }
            let n_trans = types.iter().filter(|t| is(t, "translation")).count();
            if n_trans > 1 {
                return Err(Error::value(
                    "Must supply 0 or 1 'translation' item incoordinate_transformations",
                ));
            } else if n_trans == 1 {
                let idx = types.iter().position(|t| is(t, "translation")).unwrap();
                let translation = transformations[idx].get("translation").ok_or_else(|| {
                    Error::value(format!("Missing scale argument in: {}", repr(first)))
                })?;
                if py_len(translation)? as i64 != ndim {
                    return Err(Error::value(format!(
                        "'translation' list {} must match image dimensions count: {ndim}",
                        repr(translation)
                    )));
                }
                if py_iter(translation).iter().any(|v| !is_number(v)) {
                    return Err(Error::type_err(format!(
                        "'translation' values must all be numbers: {}",
                        repr(translation)
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Python `a, b = s.split("/")`.
fn unpack2(well: &str) -> Result<(&str, &str)> {
    let parts: Vec<&str> = well.split('/').collect();
    match parts.len() {
        2 => Ok((parts[0], parts[1])),
        n if n > 2 => Err(Error::value("too many values to unpack (expected 2)")),
        n => Err(Error::value(format!(
            "not enough values to unpack (expected 2, got {n})"
        ))),
    }
}

/// Python `int` value of a JSON value (bools are ints).
fn py_int(v: &Value) -> Option<i64> {
    match v {
        Value::Bool(b) => Some(*b as i64),
        other => other.as_i64(),
    }
}

/// Python type name of a JSON value.
pub fn py_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) if n.is_f64() => "float",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// Python `len()` of a JSON value.
pub fn py_len(v: &Value) -> Result<usize> {
    match v {
        Value::String(s) => Ok(s.chars().count()),
        Value::Array(a) => Ok(a.len()),
        Value::Object(o) => Ok(o.len()),
        other => Err(Error::type_err(format!(
            "object of type '{}' has no len()",
            py_type_name(other)
        ))),
    }
}

/// Python iteration over a JSON value (dict keys, string characters).
pub fn py_iter(v: &Value) -> Vec<Value> {
    match v {
        Value::String(s) => s.chars().map(|c| json!(c.to_string())).collect(),
        Value::Array(a) => a.clone(),
        Value::Object(o) => o.keys().map(|k| json!(k)).collect(),
        _ => vec![],
    }
}

/// Python truthiness of a JSON value.
pub fn is_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj(v: Value) -> JsonMap {
        v.as_object().unwrap().clone()
    }

    #[test]
    fn versions() {
        assert_eq!(format_from_version("0.4").unwrap(), Format::V04);
        assert!(format_from_version("9.9").is_err());
        assert_eq!(Format::V05.zarr_format(), 3);
        assert_eq!(Format::V04.zarr_format(), 2);
    }

    #[test]
    fn detect() {
        let md = obj(json!({"multiscales": [{"version": "0.3"}]}));
        assert_eq!(detect_format(&md, Format::V05), Format::V03);
        let md = obj(json!({"plate": {"version": "0.4"}}));
        assert_eq!(detect_format(&md, Format::V05), Format::V04);
        assert_eq!(detect_format(&JsonMap::new(), Format::V02), Format::V02);
    }

    #[test]
    fn well_dict() {
        let rows = vec!["A".to_string(), "B".to_string()];
        let cols = vec!["1".to_string(), "2".to_string()];
        let w = Format::V04.generate_well_dict("B/2", &rows, &cols).unwrap();
        assert_eq!(w["rowIndex"], 1);
        Format::V04.validate_well_dict(&w, &rows, &cols).unwrap();
        assert!(Format::V04.generate_well_dict("C/1", &rows, &cols).is_err());
        let bad = obj(json!({"path": "A/1", "rowIndex": 1, "columnIndex": 0}));
        assert!(Format::V04.validate_well_dict(&bad, &rows, &cols).is_err());
    }

    #[test]
    fn coordinate_transformations() {
        let ct = Format::V04
            .generate_coordinate_transformations(&[vec![1, 64, 64], vec![1, 32, 32]])
            .unwrap();
        assert_eq!(ct[1][0]["scale"], json!([1.0, 2.0, 2.0]));
        assert_eq!(ct[1][1]["translation"], json!([0.0, 0.5, 0.5]));
        Format::V04
            .validate_coordinate_transformations(3, 2, Some(&ct))
            .unwrap();
        assert!(
            Format::V04
                .validate_coordinate_transformations(2, 2, Some(&ct))
                .is_err()
        );
        assert!(
            Format::V04
                .validate_coordinate_transformations(3, 2, None)
                .is_err()
        );
        let bad = vec![vec![json!({"type": "scale", "scale": ["a", 1, 1]})]];
        assert!(
            Format::V04
                .validate_coordinate_transformations(3, 1, Some(&bad))
                .unwrap_err()
                .is_type()
        );
    }
}
