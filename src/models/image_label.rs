//! `image-label` metadata (`ome_zarr_models.common.image_label_types`).

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::zarr::JsonMap;

/// A colour used for a specific label value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Color {
    #[serde(rename = "label-value")]
    pub label_value: i64,
    /// RGBA colour (0-255 each).
    pub rgba: Option<[u8; 4]>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

/// Source data for the labels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Source {
    #[serde(default = "default_source_image")]
    pub image: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

fn default_source_image() -> Option<String> {
    Some("../../".into())
}

/// Additional properties of one label value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Property {
    #[serde(rename = "label-value")]
    pub label_value: i64,
    #[serde(flatten)]
    pub extra: JsonMap,
}

/// image-label metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ImageLabel {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub colors: Option<Vec<Color>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub properties: Option<Vec<Property>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<Source>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

impl ImageLabel {
    /// `LabelBase.model_validate(...)`.
    pub fn from_json(v: &Value) -> Result<ImageLabel> {
        crate::models::pydantic::validate_model("LabelBase", v)?;
        if !v.is_object() {
            return Err(Error::type_err(format!(
                "Expected an image-label dict, received {v}"
            )));
        }
        let l: ImageLabel = serde_json::from_value(v.clone())
            .map_err(|e| Error::value(format!("invalid image-label: {e}")))?;
        l.validate()?;
        Ok(l)
    }

    pub fn validate(&self) -> Result<()> {
        if let Some(colors) = &self.colors {
            let mut seen = HashSet::new();
            let mut dupes = vec![];
            for c in colors {
                if !seen.insert(c.label_value) {
                    dupes.push(c.label_value);
                }
            }
            if !dupes.is_empty() {
                return Err(Error::value(format!(
                    "Duplicated label-value: {dupes:?}.label-values must be unique across elements of `colors`."
                )));
            }
        }
        if let (Some(colors), Some(props)) = (&self.colors, &self.properties) {
            let a: HashSet<i64> = colors.iter().map(|c| c.label_value).collect();
            let b: HashSet<i64> = props.iter().map(|p| p.label_value).collect();
            if a != b {
                return Err(Error::value(format!(
                    "Inconsistent `label_value` attributes in `colors` and `properties`.The `properties` attributes \
                     have `label_values` {b:?}, The `colors` attributes have `label_values` {a:?}, "
                )));
            }
        }
        Ok(())
    }

    /// `model_dump(exclude_none=True, by_alias=True)`.
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).expect("serialize image-label")
    }
}
