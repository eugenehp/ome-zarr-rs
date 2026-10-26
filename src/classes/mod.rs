//! The class-based interface: images, multiscales, labels and scenes.

pub mod image;
pub mod scene;
pub mod tnd_utils;

pub use image::{
    AnyMultiscale, ChannelColor, LabelsArg, OmeZarrImage, OmeZarrLabels, OmeZarrMultiscale,
    ScaleMap, Version, read_multiscale,
};
pub use scene::{OmeZarrScene, SceneImages};
