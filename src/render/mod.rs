//! GPU rendering of OME-Zarr images with [threers](https://github.com/eugenehp/threers).
//!
//! An image is shown as a 2D plane (one `z` and `t`) through a pyramid level:
//! each channel is uploaded as a [`threers::volume::GpuVolume`] and drawn with
//! its `omero` colour and window, composited additively; label images are
//! drawn on top with threers' label colouring. Rendering works headless
//! ([`render_to_png`]) and in a window ([`viewer::run`]).

#[cfg(not(target_arch = "wasm32"))]
pub mod viewer;

use std::sync::Arc;

use threers::math::{Quaternion, Vector3};
#[cfg(not(target_arch = "wasm32"))]
use threers::renderer::readback;
use threers::renderer::{PixelRect, fullscreen};
use threers::volume::{
    Data3DTexture, GpuVolume, SliceBinding, SliceParams, SliceRenderer, SliceTarget, VolumeSample,
    plane_to_voxel,
};
use threers::wgpu;

use crate::array::LazyArray;
use crate::classes::image::DEFAULT_COLORS;
use crate::conversions::hex_to_rgb;
use crate::error::{Error, Result};
use crate::io::{StoreLike, ZarrLocation};
use crate::reader::{Node, Reader, SpecKind};

/// Colour function for one channel: window `params[0].xy`, colour `params[1].rgb`.
pub const CHANNEL_WGSL: &str = r#"
fn slice_color(value : vec4<f32>, params : array<vec4<f32>, 4>) -> vec4<f32> {
    let w = max(params[0].y - params[0].x, 1e-20);
    let g = clamp((value.r - params[0].x) / w, 0.0, 1.0);
    return vec4<f32>(params[1].rgb * g, 1.0);
}
"#;

/// Rendering settings for one channel.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelSettings {
    pub label: String,
    /// Linear-ish display colour in 0-1 (display referred).
    pub color: [f32; 3],
    pub start: f32,
    pub end: f32,
    pub active: bool,
}

/// An image and its labels, ready to extract planes from.
#[derive(Debug, Clone)]
pub struct RenderSource {
    /// Axis names of the image, e.g. `["c", "z", "y", "x"]`.
    pub axes: Vec<String>,
    /// Resolution levels, highest first.
    pub levels: Vec<LazyArray>,
    pub channels: Vec<ChannelSettings>,
    /// Label images: (name, axes, levels).
    pub labels: Vec<(String, Vec<String>, Vec<LazyArray>)>,
    pub name: String,
}

fn axis_names(node: &Node) -> Vec<String> {
    match node.metadata.get("axes").and_then(|a| a.as_array()) {
        Some(a) => a
            .iter()
            .map(|x| {
                x.get("name")
                    .and_then(|n| n.as_str())
                    .or(x.as_str())
                    .unwrap_or("")
                    .to_string()
            })
            .collect(),
        None => "tczyx".chars().map(|c| c.to_string()).collect(),
    }
}

impl RenderSource {
    /// Open an OME-Zarr image (or plate / well, which are stitched) for rendering.
    pub fn open(path: impl Into<StoreLike>) -> Result<RenderSource> {
        let loc = ZarrLocation::open(path)?;
        if !loc.exists() {
            return Err(Error::Assertion(format!("not a zarr: {}", loc.path())));
        }
        let nodes = Reader::new(loc)?.nodes()?;
        let image = nodes
            .iter()
            .find(|n| !n.data.is_empty() && n.first(SpecKind::Label).is_none())
            .or_else(|| nodes.iter().find(|n| !n.data.is_empty()))
            .ok_or_else(|| Error::value("no image data found"))?;
        let axes = axis_names(image);
        let levels = image.data.clone();
        let labels = nodes
            .iter()
            .filter(|n| n.first(SpecKind::Label).is_some() && !n.data.is_empty())
            .map(|n| (n.zarr.basename(), axis_names(n), n.data.clone()))
            .collect();
        let mut src = RenderSource {
            axes,
            levels,
            channels: vec![],
            labels,
            name: image
                .metadata
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("image")
                .to_string(),
        };
        src.channels = src.channel_settings(image)?;
        Ok(src)
    }

    fn axis(&self, name: &str) -> Option<usize> {
        self.axes.iter().position(|a| a == name)
    }

    /// Size of an axis at level 0 (1 if absent).
    pub fn size(&self, name: &str) -> u64 {
        self.axis(name)
            .map(|i| self.levels[0].shape()[i])
            .unwrap_or(1)
    }

    /// (height, width) of a level.
    pub fn plane_shape(&self, level: usize) -> (u64, u64) {
        let s = self.levels[level].shape();
        let y = self.axis("y").map(|i| s[i]).unwrap_or(s[s.len() - 2]);
        let x = self.axis("x").map(|i| s[i]).unwrap_or(s[s.len() - 1]);
        (y, x)
    }

    fn channel_settings(&self, node: &Node) -> Result<Vec<ChannelSettings>> {
        let n = self.size("c") as usize;
        let omero = node
            .zarr
            .root_attrs()
            .get("omero")
            .and_then(|o| o.get("channels"))
            .and_then(|c| c.as_array())
            .cloned();
        let mut out = vec![];
        for i in 0..n {
            let ch = omero.as_ref().and_then(|o| o.get(i));
            let color = ch
                .and_then(|c| c.get("color"))
                .and_then(|c| c.as_str())
                .and_then(hex_to_rgb)
                .unwrap_or_else(|| {
                    if n == 1 {
                        [1.0, 1.0, 1.0]
                    } else {
                        hex_to_rgb(DEFAULT_COLORS[i % DEFAULT_COLORS.len()]).unwrap()
                    }
                });
            let window = ch.and_then(|c| c.get("window"));
            let start = window.and_then(|w| w.get("start")).and_then(|v| v.as_f64());
            let end = window.and_then(|w| w.get("end")).and_then(|v| v.as_f64());
            let (start, end) = match (start, end) {
                (Some(s), Some(e)) => (s as f32, e as f32),
                _ => {
                    // auto contrast from the smallest level
                    let last = self.levels.last().unwrap();
                    let (lo, hi) = last.min_max()?.unwrap_or((0.0, 1.0));
                    (lo as f32, hi as f32)
                }
            };
            out.push(ChannelSettings {
                label: ch
                    .and_then(|c| c.get("label"))
                    .and_then(|l| l.as_str())
                    .map(String::from)
                    .unwrap_or(format!("Channel {i}")),
                color: [color[0] as f32, color[1] as f32, color[2] as f32],
                start,
                end,
                active: ch
                    .and_then(|c| c.get("active"))
                    .and_then(|a| a.as_bool())
                    .unwrap_or(true),
            });
        }
        Ok(out)
    }

    fn region(
        axes: &[String],
        shape: &[u64],
        t: u64,
        c: u64,
        z_frac: f64,
        t_frac: f64,
    ) -> Vec<std::ops::Range<u64>> {
        axes.iter()
            .zip(shape)
            .map(|(a, &n)| match a.as_str() {
                "t" => {
                    let i = if t_frac >= 0.0 {
                        ((t_frac * n as f64) as u64).min(n - 1)
                    } else {
                        t.min(n - 1)
                    };
                    i..i + 1
                }
                "c" => c.min(n - 1)..c.min(n - 1) + 1,
                "z" => {
                    let i = ((z_frac * n as f64) as u64).min(n.saturating_sub(1));
                    i..i + 1
                }
                _ => 0..n,
            })
            .collect()
    }

    /// Extract the planes of one level: one f32 plane per channel and one
    /// u64 plane per label image (resampled by nearest neighbour onto the
    /// level's y/x grid if the label pyramid differs).
    pub fn plane(&self, level: usize, z: u64, t: u64) -> Result<Plane> {
        let shape = self.levels[level].shape().to_vec();
        let (h, w) = self.plane_shape(level);
        let z0 = self.size("z").max(1);
        let z_frac = (z as f64 + 0.5) / z0 as f64;
        let t_frac = (t as f64 + 0.5) / self.size("t").max(1) as f64;
        let n = self.size("c");
        let mut channels = vec![];
        for c in 0..n {
            let r = Self::region(&self.axes, &shape, t, c, z_frac, t_frac);
            let data = self.levels[level].read(&r)?;
            channels.push(data.to_f64_vec().into_iter().map(|v| v as f32).collect());
        }
        let mut labels = vec![];
        for (name, axes, levels) in &self.labels {
            // the label level whose plane is closest to (h, w)
            let best = levels
                .iter()
                .enumerate()
                .min_by_key(|(_, l)| {
                    let s = l.shape();
                    let (ly, lx) = (s[s.len() - 2], s[s.len() - 1]);
                    ly.abs_diff(h) + lx.abs_diff(w)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);
            let l = &levels[best];
            let ls = l.shape().to_vec();
            let r = Self::region(axes, &ls, t, 0, z_frac, t_frac);
            let vals = l.read(&r)?.to_f64_vec();
            let (ly, lx) = (ls[ls.len() - 2] as usize, ls[ls.len() - 1] as usize);
            let mut out = vec![0u64; (h * w) as usize];
            for y in 0..h as usize {
                let sy = (y * ly / h as usize).min(ly - 1);
                for x in 0..w as usize {
                    let sx = (x * lx / w as usize).min(lx - 1);
                    out[y * w as usize + x] = vals[sy * lx + sx].max(0.0) as u64;
                }
            }
            labels.push((name.clone(), out));
        }
        Ok(Plane {
            width: w as u32,
            height: h as u32,
            channels,
            labels,
        })
    }

    /// The coarsest level whose plane still has at least `min_pixels` along
    /// its longest side (or level 0).
    pub fn level_for(&self, min_pixels: u64) -> usize {
        let mut best = 0;
        for i in 0..self.levels.len() {
            let (h, w) = self.plane_shape(i);
            if h.max(w) >= min_pixels {
                best = i;
            }
        }
        best
    }
}

/// CPU data for one plane.
#[derive(Debug, Clone)]
pub struct Plane {
    pub width: u32,
    pub height: u32,
    pub channels: Vec<Vec<f32>>,
    pub labels: Vec<(String, Vec<u64>)>,
}

/// One plane on the GPU.
pub struct GpuPlane {
    pub width: u32,
    pub height: u32,
    channels: Vec<(GpuVolume, SliceBinding)>,
    labels: Vec<(GpuVolume, SliceBinding)>,
}

/// What to show and how.
#[derive(Debug, Clone)]
pub struct ViewParams {
    /// Centre of the view in plane pixels.
    pub center: (f32, f32),
    /// Plane pixels per screen pixel.
    pub voxels_per_pixel: f32,
    pub show_labels: bool,
    pub label_opacity: f32,
    pub label_seed: u32,
    pub background: [f64; 3],
}

impl ViewParams {
    /// Fit a `width x height` plane into a `vw x vh` viewport.
    pub fn fit(width: u32, height: u32, vw: u32, vh: u32) -> ViewParams {
        let vpp = (width as f32 / vw as f32).max(height as f32 / vh as f32);
        ViewParams {
            center: (width as f32 / 2.0, height as f32 / 2.0),
            voxels_per_pixel: vpp,
            show_labels: true,
            label_opacity: 0.5,
            label_seed: 7,
            background: [0.0, 0.0, 0.0],
        }
    }
}

/// GPU pipelines for drawing planes into a target of a given format.
pub struct PlaneRenderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    channel: SliceRenderer,
    label: SliceRenderer,
    pub format: wgpu::TextureFormat,
}

impl PlaneRenderer {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        format: wgpu::TextureFormat,
    ) -> PlaneRenderer {
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::OVER,
        };
        let channel = SliceRenderer::with_target(
            &device,
            SliceTarget {
                format,
                blend: Some(additive),
                depth: None,
            },
            VolumeSample::Float,
            CHANNEL_WGSL,
        );
        let label = SliceRenderer::labels(&device, format);
        PlaneRenderer {
            device,
            queue,
            channel,
            label,
            format,
        }
    }

    /// Upload a plane.
    pub fn upload(&self, plane: &Plane) -> GpuPlane {
        let (w, h) = (plane.width, plane.height);
        let channels = plane
            .channels
            .iter()
            .map(|c| {
                let v = GpuVolume::from_texture(
                    &self.device,
                    &self.queue,
                    &Data3DTexture::from_f32(w, h, 1, c),
                );
                let b = self.channel.create_binding(&self.device, &v);
                (v, b)
            })
            .collect();
        let labels = plane
            .labels
            .iter()
            .map(|(_, l)| {
                let v = GpuVolume::from_texture(
                    &self.device,
                    &self.queue,
                    &Data3DTexture::from_u64(w, h, 1, l),
                );
                let b = self.label.create_binding(&self.device, &v);
                (v, b)
            })
            .collect();
        GpuPlane {
            width: w,
            height: h,
            channels,
            labels,
        }
    }

    /// Draw a plane into `target` (cleared to the background first).
    pub fn draw(
        &self,
        target: &wgpu::TextureView,
        size: (u32, u32),
        plane: &GpuPlane,
        channels: &[ChannelSettings],
        view: &ViewParams,
    ) {
        let rect = PixelRect {
            x: 0,
            y: 0,
            width: size.0,
            height: size.1,
        };
        let to_voxel = plane_to_voxel(
            Vector3::new(view.center.0, view.center.1, 0.5),
            Quaternion::identity(),
            view.voxels_per_pixel,
            (size.0 as f32, size.1 as f32),
            true,
        );
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let [r, g, b] = view.background;
        drop(fullscreen::begin_with(
            &mut encoder,
            target,
            fullscreen::Load::Clear,
            wgpu::Color { r, g, b, a: 1.0 },
            "ome-zarr clear",
        ));
        self.queue.submit([encoder.finish()]);
        // Separate uniform writes need separate submissions.
        for ((_, binding), ch) in plane.channels.iter().zip(channels) {
            if !ch.active {
                continue;
            }
            let mut params = SliceParams::window(to_voxel, ch.start, ch.end);
            params.params[1] = [ch.color[0], ch.color[1], ch.color[2], 1.0];
            self.channel.update(&self.queue, binding, &params);
            let mut encoder = self.device.create_command_encoder(&Default::default());
            self.channel.draw(&mut encoder, target, rect, binding);
            self.queue.submit([encoder.finish()]);
        }
        if view.show_labels {
            for (_, binding) in &plane.labels {
                self.label.update(
                    &self.queue,
                    binding,
                    &SliceParams::labels(to_voxel, view.label_seed, view.label_opacity),
                );
                let mut encoder = self.device.create_command_encoder(&Default::default());
                self.label.draw(&mut encoder, target, rect, binding);
                self.queue.submit([encoder.finish()]);
            }
        }
    }
}

/// Options for [`render_to_png`].
#[derive(Debug, Clone)]
pub struct RenderOptions {
    pub width: u32,
    pub height: u32,
    /// Plane index along z (level 0 index; default: middle).
    pub z: Option<u64>,
    /// Time point (default 0).
    pub t: u64,
    /// Pyramid level (default: the coarsest level that still fills the output).
    pub level: Option<usize>,
    pub show_labels: bool,
    pub label_opacity: f32,
}

impl Default for RenderOptions {
    fn default() -> Self {
        RenderOptions {
            width: 512,
            height: 512,
            z: None,
            t: 0,
            level: None,
            show_labels: true,
            label_opacity: 0.5,
        }
    }
}

/// Render an OME-Zarr image headless; returns RGBA8 pixels.
#[cfg(not(target_arch = "wasm32"))]
pub fn render_rgba(path: impl Into<StoreLike>, opts: &RenderOptions) -> Result<Vec<u8>> {
    let src = RenderSource::open(path)?;
    render_source_rgba(&src, opts)
}

/// Render an already opened source headless; returns RGBA8 pixels.
#[cfg(not(target_arch = "wasm32"))]
pub fn render_source_rgba(src: &RenderSource, opts: &RenderOptions) -> Result<Vec<u8>> {
    let gpu = threers::gpu::GpuContext::headless_blocking()
        .map_err(|e| Error::value(format!("no GPU adapter: {e}")))?;
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let renderer = PlaneRenderer::new(gpu.device.clone(), gpu.queue.clone(), format);
    let level = opts
        .level
        .unwrap_or_else(|| src.level_for(opts.width.max(opts.height) as u64));
    let level = level.min(src.levels.len() - 1);
    let z = opts.z.unwrap_or(src.size("z") / 2);
    let plane = src.plane(level, z, opts.t)?;
    let gp = renderer.upload(&plane);
    let mut view = ViewParams::fit(plane.width, plane.height, opts.width, opts.height);
    view.show_labels = opts.show_labels;
    view.label_opacity = opts.label_opacity;
    let target = threers::renderer::RenderTarget::new(&gpu.device, opts.width, opts.height, format);
    renderer.draw(
        &target.color_view,
        (opts.width, opts.height),
        &gp,
        &src.channels,
        &view,
    );
    Ok(readback::texture_bytes(
        &gpu.device,
        &gpu.queue,
        &target.color_texture,
    ))
}

/// Render an OME-Zarr image to a PNG file (headless).
#[cfg(not(target_arch = "wasm32"))]
pub fn render_to_png(
    path: impl Into<StoreLike>,
    output: &std::path::Path,
    opts: &RenderOptions,
) -> Result<()> {
    let rgba = render_rgba(path, opts)?;
    std::fs::write(output, threers::encode_png(opts.width, opts.height, &rgba))?;
    Ok(())
}
