//! Utilities for creating a [`TransformGraph`](crate::transformnd::TransformGraph)
//! from OME-Zarr metadata (`ome_zarr.classes.tnd_utils`).

use super::image::OmeZarrMultiscale;
use crate::error::{Error, Result};
use crate::models::transforms::{CoordinateSystem, Transform as Ozm};
use crate::transformnd::{Affine, Sequence, Spaced, SubTransform, Transform, VectorField};

/// Which dimension of a vector field array contains the vectors.
pub const VECTOR_FIELD_AXIS: usize = 0;

/// A space in a scene graph: (zarr path relative to the scene root, coordinate system name).
pub type SpaceRef = (String, String);

/// Displacement / coordinate field images, keyed by path.
pub type Displacements = Vec<(String, OmeZarrMultiscale)>;

fn posix_join(a: &str, b: &str) -> String {
    if b.starts_with('/') || a.is_empty() {
        b.to_string()
    } else if a.ends_with('/') {
        format!("{a}{b}")
    } else {
        format!("{a}/{b}")
    }
}

fn basename(p: &str) -> &str {
    p.rsplit('/').next().unwrap_or(p)
}

/// Convert an OME-Zarr coordinate transformation with coordinate system
/// information to a spaced functional transform.
pub fn ozmp_tf_to_tnd_spaced(
    transform: &Ozm,
    zarr_context: &str,
    source_cs: Option<&CoordinateSystem>,
    target_cs: Option<&CoordinateSystem>,
    coordinate_displacements: &Displacements,
) -> Result<Spaced<SpaceRef>> {
    let (Some(input), Some(output)) = (transform.input(), transform.output()) else {
        return Err(Error::unsupported("Missing coordinate system", transform));
    };
    let t = ozmp_tf_to_tnd(
        transform,
        zarr_context,
        source_cs.map(|c| c.ndim()),
        target_cs.map(|c| c.ndim()),
        coordinate_displacements,
    )?;
    let mut input_path = input.path.clone().unwrap_or_default();
    let mut output_path = output.path.clone().unwrap_or_default();
    if !zarr_context.is_empty() {
        input_path = if input_path.is_empty() {
            zarr_context.to_string()
        } else {
            posix_join(zarr_context, &input_path)
        };
        output_path = if output_path.is_empty() {
            zarr_context.to_string()
        } else {
            posix_join(zarr_context, &output_path)
        };
    }
    Ok(Spaced::new(
        t,
        (input_path, input.name.clone().unwrap_or_default()),
        (output_path, output.name.clone().unwrap_or_default()),
    ))
}

/// Set up the vector field for displacements / coordinates transforms.
pub fn setup_vectorfield_args(
    transform: &Ozm,
    zarr_context: &str,
    coordinate_displacements: &Displacements,
) -> Result<VectorField> {
    let path = match transform {
        Ozm::Displacements(d) => d.path.clone(),
        Ozm::Coordinates(c) => c.path.clone(),
        _ => {
            return Err(Error::unsupported(
                "Not a vector field transform",
                transform,
            ));
        }
    };
    let mut p = path;
    if !zarr_context.is_empty() && !p.is_empty() {
        p = posix_join(zarr_context, &p);
    }
    let vfield = coordinate_displacements
        .iter()
        .find(|(k, _)| k == basename(&p))
        .map(|(_, v)| v)
        .ok_or_else(|| Error::unsupported("Missing vector field", transform))?;
    let img = &vfield.images()[0];
    // Metadata is defined from displacements to input space, but transformnd
    // expects input space to displacements, so this is inverted.
    let scale: Vec<f64> = img.scale.values()[1..].to_vec();
    let index_transform = Transform::Scale(scale).invert().unwrap();
    VectorField::new(&img.data, Some(index_transform), 3, VECTOR_FIELD_AXIS)
}

/// Convert an OME-Zarr coordinate transformation to a functional transform.
pub fn ozmp_tf_to_tnd(
    transform: &Ozm,
    zarr_context: &str,
    source_ndim: Option<usize>,
    target_ndim: Option<usize>,
    coordinate_displacements: &Displacements,
) -> Result<Transform> {
    let ndim_or = |what: &Ozm| -> Result<usize> {
        source_ndim
            .or(target_ndim)
            .ok_or_else(|| Error::unsupported("Could not infer dimensionality", what))
    };
    Ok(match transform {
        Ozm::Affine(a) => {
            let aff = match a.affine_matrix() {
                Ok(m) => m.clone(),
                Err(Error::NotImplemented(_)) => {
                    return Err(Error::unsupported(
                        "Path-form transformations are not implemented",
                        transform,
                    ));
                }
                Err(e) => return Err(e),
            };
            let lin: Vec<Vec<f64>> = aff.iter().map(|r| r[..r.len() - 1].to_vec()).collect();
            let tr: Vec<f64> = aff.iter().map(|r| r[r.len() - 1]).collect();
            Transform::Affine(Affine::from_linear_map(&lin, Some(&tr))?)
        }
        Ozm::Coordinates(_) => Transform::Coordinates(Box::new(setup_vectorfield_args(
            transform,
            zarr_context,
            coordinate_displacements,
        )?)),
        Ozm::Displacements(_) => Transform::displacements(setup_vectorfield_args(
            transform,
            zarr_context,
            coordinate_displacements,
        )?)?,
        Ozm::Identity(_) => Transform::Identity(ndim_or(transform)?),
        Ozm::MapAxis(m) => Transform::map_axis(m.map_axis.iter().map(|&i| i as usize).collect())?,
        Ozm::ProjectAxis(p) => {
            if source_ndim.is_none() && target_ndim.is_none() {
                return Err(Error::unsupported(
                    "Could not infer dimensionality",
                    transform,
                ));
            }
            Transform::project_axis(
                p.dropped_inputs
                    .as_ref()
                    .map(|v| v.iter().map(|&i| i as usize).collect()),
                p.created_outputs
                    .as_ref()
                    .map(|v| v.iter().map(|&i| i as usize).collect()),
                source_ndim,
                target_ndim,
            )?
        }
        Ozm::Scale(s) => Transform::Scale(s.scale.clone()),
        Ozm::Translation(t) => Transform::Translate(t.translation.clone()),
        Ozm::Rotation(r) => {
            let rot = match r.rotation_matrix() {
                Ok(m) => m.clone(),
                Err(Error::NotImplemented(_)) => {
                    return Err(Error::unsupported(
                        "Path-form transformations are not yet implemented",
                        transform,
                    ));
                }
                Err(e) => return Err(e),
            };
            Transform::Affine(Affine::from_linear_map(&rot, None)?)
        }
        Ozm::ByDimension(b) => {
            let subs = b
                .transformations
                .iter()
                .map(|t| {
                    SubTransform::new(
                        ozmp_tf_to_tnd(
                            &t.transformation,
                            zarr_context,
                            Some(t.input_axes.len()),
                            Some(t.output_axes.len()),
                            coordinate_displacements,
                        )?,
                        t.input_axes.iter().map(|&i| i as usize).collect(),
                        Some(t.output_axes.iter().map(|&i| i as usize).collect()),
                    )
                })
                .collect::<Result<_>>()?;
            Transform::by_dimension(subs)?
        }
        Ozm::Sequence(s) => {
            let ts = &s.transformations;
            if ts.is_empty() {
                return Ok(Transform::Sequence(Sequence::empty(ndim_or(transform)?)));
            }
            if ts.len() == 1 {
                return Ok(Transform::Sequence(Sequence::new(vec![ozmp_tf_to_tnd(
                    &ts[0],
                    zarr_context,
                    source_ndim,
                    target_ndim,
                    coordinate_displacements,
                )?])?));
            }
            let mut inner = vec![ozmp_tf_to_tnd(
                &ts[0],
                zarr_context,
                source_ndim,
                None,
                coordinate_displacements,
            )?];
            for t in &ts[1..ts.len() - 1] {
                let src = inner.last().unwrap().ndims().target;
                inner.push(ozmp_tf_to_tnd(
                    t,
                    zarr_context,
                    Some(src),
                    None,
                    coordinate_displacements,
                )?);
            }
            let src = inner.last().unwrap().ndims().target;
            inner.push(ozmp_tf_to_tnd(
                ts.last().unwrap(),
                zarr_context,
                Some(src),
                target_ndim,
                coordinate_displacements,
            )?);
            Transform::Sequence(Sequence::new(inner)?)
        }
        Ozm::Bijection(_) => {
            return Err(Error::unsupported("Unsupported transform type", transform));
        }
    })
}
