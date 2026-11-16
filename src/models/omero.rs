//! `omero` rendering metadata (`ome_zarr_models.common.omero`).

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::pyjson::JsonMap;

/// A channel window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Window {
    pub max: f64,
    pub min: f64,
    pub start: f64,
    pub end: f64,
    #[serde(flatten)]
    pub extra: JsonMap,
}

/// A single omero channel. Extra fields (e.g. `label`, `active`) are kept.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Channel {
    /// RGB hex colour, e.g. `"FF0000"`.
    pub color: String,
    pub window: Window,
    #[serde(flatten)]
    pub extra: JsonMap,
}

impl Channel {
    pub fn label(&self) -> Option<&str> {
        self.extra.get("label").and_then(|v| v.as_str())
    }
    pub fn active(&self) -> Option<bool> {
        self.extra.get("active").and_then(|v| v.as_bool())
    }
}

/// omero model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Omero {
    pub channels: Vec<Channel>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

/// `RGBHexConstraint`: exactly six hexadecimal digits.
pub fn validate_rgb_hex(s: &str) -> Result<()> {
    if s.len() == 6 && s.chars().all(|c| c.is_ascii_hexdigit()) {
        Ok(())
    } else {
        Err(Error::value(format!(
            "String should match pattern '[0-9a-fA-F]{{6}}': {s:?}"
        )))
    }
}

impl Omero {
    /// `Omero.model_validate(...)`.
    pub fn from_json(v: &Value) -> Result<Omero> {
        let o: Omero = super::parse("Omero", "omero", v)?;
        for c in &o.channels {
            validate_rgb_hex(&c.color)?;
        }
        Ok(o)
    }

    /// `model_dump(by_alias=True)` + `_recursive_pop_nones`.
    pub fn to_json(&self) -> Value {
        super::dump_without_nones(self)
    }
}
