//! Scenes: several images related by coordinate transformations
//! (`ome_zarr.classes.scene`).

use serde_json::{Value, json};

use super::image::{OmeZarrMultiscale, Version};
use super::tnd_utils::{Displacements, SpaceRef, ozmp_tf_to_tnd_spaced};
use crate::error::{Error, Result};
use crate::models::transforms::{CoordinateSystem, Transform};
use crate::models::{BaseSceneAttrs, SceneAttrs};
use crate::transformnd::TransformGraph;
use crate::writer::{DelayedWrite, GroupLike};
use crate::zarr::{Group, Mode, NodeKind};

/// Images for a scene: a list (keyed by `metadata.name`) or explicit paths.
#[derive(Debug, Clone)]
pub enum SceneImages {
    List(Vec<OmeZarrMultiscale>),
    Map(Vec<(String, OmeZarrMultiscale)>),
}

/// A collection of images and the coordinate transformations between them.
#[derive(Debug, Clone)]
pub struct OmeZarrScene {
    /// Images keyed by zarr group path.
    pub images: Vec<(String, OmeZarrMultiscale)>,
    /// Displacement / coordinate field images keyed by path.
    pub coordinate_displacements: Displacements,
    metadata: SceneAttrs,
    graph: TransformGraph<SpaceRef>,
}

impl OmeZarrScene {
    pub fn new(
        images: SceneImages,
        coordinate_transformations: Vec<Transform>,
        coordinate_systems: Vec<CoordinateSystem>,
        coordinate_displacements: Option<Displacements>,
    ) -> Result<OmeZarrScene> {
        let images = match images {
            SceneImages::List(v) => v
                .into_iter()
                .map(|i| (i.metadata().name_str().unwrap_or_default(), i))
                .collect(),
            SceneImages::Map(m) => m,
        };
        for t in &coordinate_transformations {
            t.validate()?;
        }
        let metadata = SceneAttrs {
            coordinate_transformations,
            coordinate_systems: Some(coordinate_systems),
        };
        metadata.validate()?;
        let mut scene = OmeZarrScene {
            images,
            coordinate_displacements: coordinate_displacements.unwrap_or_default(),
            metadata,
            graph: TransformGraph::new(),
        };
        scene.graph = scene.setup_graph()?;
        Ok(scene)
    }

    /// Build a scene from JSON transform / coordinate system dicts.
    pub fn from_json_parts(
        images: SceneImages,
        transforms: &[Value],
        coordinate_systems: &[Value],
    ) -> Result<OmeZarrScene> {
        let t = transforms
            .iter()
            .map(Transform::from_json)
            .collect::<Result<_>>()?;
        let cs = coordinate_systems
            .iter()
            .map(CoordinateSystem::from_json)
            .collect::<Result<_>>()?;
        OmeZarrScene::new(images, t, cs, None)
    }

    pub fn coordinate_systems(&self) -> &[CoordinateSystem] {
        self.metadata.coordinate_systems.as_deref().unwrap_or(&[])
    }

    pub fn set_coordinate_systems(&mut self, value: Vec<CoordinateSystem>) {
        self.metadata.coordinate_systems = Some(value);
    }

    pub fn coordinate_transformations(&self) -> &[Transform] {
        &self.metadata.coordinate_transformations
    }

    pub fn set_coordinate_transformations(&mut self, value: Vec<Transform>) {
        self.metadata.coordinate_transformations = value;
    }

    /// The scene metadata.
    pub fn metadata(&self) -> &SceneAttrs {
        &self.metadata
    }

    /// Replace the metadata and rebuild the graph.
    pub fn set_metadata(&mut self, value: SceneAttrs) -> Result<()> {
        self.metadata = value;
        self.graph = self.setup_graph()?;
        Ok(())
    }

    /// The transform graph between all coordinate systems.
    pub fn graph(&self) -> &TransformGraph<SpaceRef> {
        &self.graph
    }

    /// Image by path.
    pub fn image(&self, path: &str) -> Option<&OmeZarrMultiscale> {
        self.images.iter().find(|(p, _)| p == path).map(|(_, i)| i)
    }

    /// Retrieve coordinate systems matching `path` (if given) and `name` (if
    /// given), keyed by (path, name).
    pub fn get_coordinate_system(
        &self,
        path: Option<&str>,
        name: Option<&str>,
    ) -> Vec<(SpaceRef, CoordinateSystem)> {
        let mut out: Vec<(SpaceRef, CoordinateSystem)> = vec![];
        let mut put = |k: SpaceRef, v: CoordinateSystem| {
            if let Some(e) = out.iter_mut().find(|(kk, _)| kk == &k) {
                e.1 = v;
            } else {
                out.push((k, v));
            }
        };
        for cs in self.coordinate_systems() {
            if name.is_none_or(|n| cs.name == n) && path.is_none_or(|p| p.is_empty()) {
                put((String::new(), cs.name.clone()), cs.clone());
            }
        }
        for (img_path, img) in &self.images {
            for cs in &img.metadata().coordinate_systems {
                if name.is_none_or(|n| cs.name == n) && path.is_none_or(|p| p == img_path) {
                    put((img_path.clone(), cs.name.clone()), cs.clone());
                }
            }
        }
        out
    }

    fn lookup_cs(&self, path: Option<&str>, name: Option<&str>) -> Result<CoordinateSystem> {
        let key = (
            path.unwrap_or("").to_string(),
            name.unwrap_or("").to_string(),
        );
        self.get_coordinate_system(path, name)
            .into_iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| v)
            .ok_or_else(|| Error::key(format!("{key:?}")))
    }

    fn setup_graph(&self) -> Result<TransformGraph<SpaceRef>> {
        let mut g = TransformGraph::new();
        for tf in self.coordinate_transformations() {
            let (Some(input), Some(output)) = (tf.input(), tf.output()) else {
                return Err(Error::value(format!(
                    "Coordinate transformation {} is missing input or output information.",
                    tf.py_str()
                )));
            };
            let source_cs = self.lookup_cs(input.path.as_deref(), input.name.as_deref())?;
            let target_cs = self.lookup_cs(output.path.as_deref(), output.name.as_deref())?;
            let t = ozmp_tf_to_tnd_spaced(
                tf,
                "",
                Some(&source_cs),
                Some(&target_cs),
                &self.coordinate_displacements,
            )?;
            let inv = t.invert();
            g.add_transform(t)?;
            if let Some(inv) = inv {
                g.add_transform(inv)?;
            }
            let subgroups: Vec<&String> =
                [&input.path, &output.path].into_iter().flatten().collect();
            for sub in subgroups {
                let Some(img) = self.image(sub) else { continue };
                for img_tf in img.metadata().coordinate_transformations.iter().flatten() {
                    let src = img_tf
                        .input()
                        .and_then(|i| img.get_descendant_coordinate_system(i));
                    let tgt = img_tf
                        .output()
                        .and_then(|o| img.get_descendant_coordinate_system(o));
                    let t = ozmp_tf_to_tnd_spaced(
                        img_tf,
                        sub,
                        src,
                        tgt,
                        &self.coordinate_displacements,
                    )?;
                    let inv = t.invert();
                    g.add_transform(t)?;
                    if let Some(inv) = inv {
                        g.add_transform(inv)?;
                    }
                }
            }
        }
        Ok(g)
    }

    /// Write the scene. With `overwrite`, the store is cleared and every image
    /// written; otherwise only images not yet in the store are written. The
    /// scene metadata is always updated.
    pub fn to_ome_zarr(
        &self,
        store: &str,
        overwrite: bool,
        compute: bool,
    ) -> Result<Vec<DelayedWrite>> {
        if overwrite && std::path::Path::new(store).exists() {
            std::fs::remove_dir_all(store)?;
        }
        let mode = if overwrite { Mode::Write } else { Mode::Append };
        let root = Group::open_url(store, mode, None)?;
        let mut delayed = vec![];
        for (img_path, img) in &self.images {
            if !overwrite && root.contains(img_path) {
                continue;
            }
            let sub = root.create_group(img_path, overwrite)?;
            delayed.extend(img.to_ome_zarr(
                GroupLike::Group(sub),
                None,
                Version::V06,
                compute,
                true,
            )?);
        }
        for (disp_path, disp) in &self.coordinate_displacements {
            let p = format!("coordinateTransformations/{disp_path}");
            if !overwrite && root.contains(disp_path) {
                continue;
            }
            let sub = root.create_group(&p, overwrite)?;
            delayed.extend(disp.to_ome_zarr(
                GroupLike::Group(sub),
                None,
                Version::V06,
                compute,
                true,
            )?);
        }
        root.set_attr(
            "ome",
            json!({"scene": self.metadata.to_json(), "version": "0.6"}),
        )?;
        Ok(delayed)
    }

    /// Load an existing scene.
    pub fn from_ome_zarr(store: impl Into<GroupLike>) -> Result<OmeZarrScene> {
        let root = match store.into() {
            GroupLike::Group(g) => g,
            GroupLike::Path(p) => Group::open_url(&p, Mode::Read, None)?,
        };
        let displacements = if root.child_kind("coordinateTransformations") == Some(NodeKind::Group)
        {
            let ct = root.group("coordinateTransformations")?;
            let mut d = vec![];
            for k in ct.group_keys()? {
                d.push((
                    k.clone(),
                    OmeZarrMultiscale::from_ome_zarr(GroupLike::Group(ct.group(&k)?))?,
                ));
            }
            Some(d)
        } else {
            None
        };
        let ome = root.attrs()?.get("ome").cloned().unwrap_or(json!({}));
        let attrs = BaseSceneAttrs::from_json(&ome)?;
        let mut images: Vec<(String, OmeZarrMultiscale)> = vec![];
        for tf in &attrs.scene.coordinate_transformations {
            for id in [tf.input(), tf.output()].into_iter().flatten() {
                let Some(path) = &id.path else { continue };
                if root.contains(path) {
                    let img =
                        OmeZarrMultiscale::from_ome_zarr(GroupLike::Group(root.group(path)?))?;
                    if let Some(e) = images.iter_mut().find(|(p, _)| p == path) {
                        e.1 = img;
                    } else {
                        images.push((path.clone(), img));
                    }
                } else {
                    return Err(Error::value(format!(
                        "Image specified in metadata at '{path}' not found in zarr group."
                    )));
                }
            }
        }
        OmeZarrScene::new(
            SceneImages::Map(images),
            attrs.scene.coordinate_transformations,
            attrs.scene.coordinate_systems.unwrap_or_default(),
            displacements,
        )
    }
}
