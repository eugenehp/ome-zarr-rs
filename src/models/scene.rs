//! Scene metadata (`ome_zarr_models.v06.scene`).

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::transforms::{CoordinateSystem, Transform};
use crate::error::{Error, Result};

/// Metadata for OME-Zarr scene groups.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SceneAttrs {
    #[serde(rename = "coordinateTransformations", default)]
    pub coordinate_transformations: Vec<Transform>,
    #[serde(
        rename = "coordinateSystems",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub coordinate_systems: Option<Vec<CoordinateSystem>>,
}

impl SceneAttrs {
    pub fn validate(&self) -> Result<()> {
        for t in &self.coordinate_transformations {
            t.validate()?;
        }
        for cs in self.coordinate_systems.iter().flatten() {
            cs.validate()?;
        }
        Ok(())
    }

    pub fn from_json(v: &Value) -> Result<SceneAttrs> {
        crate::models::pydantic::validate_model("SceneAttrs", v)?;
        let s: SceneAttrs = serde_json::from_value(v.clone())
            .map_err(|e| Error::value(format!("invalid scene metadata: {e}")))?;
        s.validate()?;
        Ok(s)
    }

    /// `model_dump(exclude_none=True)`.
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).expect("serialize scene")
    }
}

/// The `ome` attributes of a scene group: `{"version": ..., "scene": {...}}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaseSceneAttrs {
    pub scene: SceneAttrs,
    pub version: String,
}

impl BaseSceneAttrs {
    pub fn from_json(v: &Value) -> Result<BaseSceneAttrs> {
        crate::models::pydantic::validate_model("BaseSceneAttrs", v)?;
        let s: BaseSceneAttrs = serde_json::from_value(v.clone())
            .map_err(|e| Error::value(format!("invalid scene attributes: {e}")))?;
        s.scene.validate()?;
        Ok(s)
    }
}
