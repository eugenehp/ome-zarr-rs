//! OME-NGFF metadata models: a port of the parts of
//! [ome-zarr-models](https://github.com/ome-zarr-models/ome-zarr-models-py) used by
//! ome-zarr.

pub mod image_label;
pub mod multiscales;
pub mod omero;
pub mod pydantic;
pub mod scene;
pub mod transforms;

pub use image_label::ImageLabel;
pub use multiscales::{Dataset, Multiscale};
pub use omero::Omero;
pub use scene::{BaseSceneAttrs, SceneAttrs};
pub use transforms::{
    Axis, ByDimensionTransform, CoordinateSystem, CoordinateSystemIdentifier, Transform,
};

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::error::{Error, Result};

/// `Model.model_validate(v)`'s first half: pydantic's checks for `model`
/// (with its errors), then `v` as a `T`; `what` names it in a parse error.
pub(crate) fn parse<T: DeserializeOwned>(model: &str, what: &str, v: &Value) -> Result<T> {
    pydantic::validate_model(model, v)?;
    from_value(what, v)
}

/// `v` as a `T`; `what` names it in the error.
pub(crate) fn from_value<T: DeserializeOwned>(what: &str, v: &Value) -> Result<T> {
    serde_json::from_value(v.clone()).map_err(|e| Error::value(format!("invalid {what}: {e}")))
}

/// `model_dump(exclude_none=True)`: the fields set.
pub(crate) fn dump<T: Serialize>(model: &T) -> Value {
    serde_json::to_value(model).expect("a model serializes to JSON")
}

/// `model_dump()` + `_recursive_pop_nones`: nulls dropped at every depth.
pub(crate) fn dump_without_nones<T: Serialize>(model: &T) -> Value {
    crate::pyjson::recursive_pop_nones(&dump(model))
}
