//! Interactive native viewer: a window showing an OME-Zarr image with its
//! labels, rendered by threers.
//!
//! Controls: drag to pan, scroll to zoom, `Up`/`Down` (or `.`/`,`) step
//! through z, `]`/`[` through time, `1`-`9` toggle channels, `L` toggles
//! labels, `R` resets the view, `S` saves a screenshot, `Q`/`Esc` quits.

use std::collections::HashMap;
use std::sync::Arc;

use log::{info, warn};
use threers::wgpu;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

use super::{GpuPlane, PlaneRenderer, RenderSource, ViewParams};
use crate::error::{Error, Result};

/// View state, in level-0 pixel units.
#[derive(Debug, Clone)]
struct State {
    center: (f32, f32),
    /// Screen pixels per level-0 pixel.
    zoom: f32,
    z: u64,
    t: u64,
    show_labels: bool,
}

impl State {
    fn fit(src: &RenderSource, w: u32, h: u32) -> State {
        let (ph, pw) = src.plane_shape(0);
        let zoom = (w as f32 / pw as f32).min(h as f32 / ph as f32);
        State {
            center: (pw as f32 / 2.0, ph as f32 / 2.0),
            zoom,
            z: src.size("z") / 2,
            t: 0,
            show_labels: true,
        }
    }
}

/// Pyramid level for a zoom: the coarsest level whose pixels are no larger
/// than a screen pixel.
fn pick_level(src: &RenderSource, zoom: f32) -> (usize, f32) {
    let (_, w0) = src.plane_shape(0);
    let mut best = (0usize, 1.0f32);
    for l in 0..src.levels.len() {
        let (_, w) = src.plane_shape(l);
        let f = w0 as f32 / w as f32;
        if f * zoom <= 1.0 {
            best = (l, f);
        }
    }
    best
}

fn title(src: &RenderSource, s: &State, level: usize) -> String {
    let mut t = format!("{} — level {level}", src.name);
    if src.size("z") > 1 {
        t.push_str(&format!("  z {}/{}", s.z, src.size("z") - 1));
    }
    if src.size("t") > 1 {
        t.push_str(&format!("  t {}/{}", s.t, src.size("t") - 1));
    }
    t
}

/// The window and what draws into it, made once the event loop is running.
struct Gfx {
    window: Arc<Window>,
    gpu: threers::gpu::GpuContext,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    renderer: PlaneRenderer,
}

struct Viewer {
    title: String,
    src: RenderSource,
    gfx: Option<Gfx>,
    state: Option<State>,
    cache: HashMap<(usize, u64, u64), GpuPlane>,
    cursor: (f32, f32),
    dragging: Option<(f32, f32)>,
    error: Option<Error>,
}

impl Viewer {
    fn init(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        let attrs = Window::default_attributes()
            .with_title(self.title.clone())
            .with_inner_size(winit::dpi::LogicalSize::new(1024, 768));
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .map_err(|e| Error::value(format!("window: {e}")))?,
        );
        let (gpu, surface) = threers::gpu::GpuContext::with_surface_blocking(window.clone().into())
            .map_err(|e| Error::value(format!("gpu: {e}")))?;
        info!("adapter: {}", gpu.describe());
        let size = window.inner_size();
        let config = gpu.surface_config(&surface, size.width, size.height, true);
        surface.configure(&gpu.device, &config);
        let renderer = PlaneRenderer::new(gpu.device.clone(), gpu.queue.clone(), config.format);
        self.state = Some(State::fit(&self.src, config.width, config.height));
        self.gfx = Some(Gfx {
            window,
            gpu,
            surface,
            config,
            renderer,
        });
        Ok(())
    }
}

impl ApplicationHandler for Viewer {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        if let Err(e) = self.init(event_loop) {
            self.error = Some(e);
            event_loop.exit();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let (Some(gfx), Some(state)) = (self.gfx.as_mut(), self.state.as_mut()) else {
            return;
        };
        if window_id != gfx.window.id() {
            return;
        }
        let src = &mut self.src;
        let config = &mut gfx.config;
        let mut redraw = false;
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(s) => {
                config.width = s.width.max(1);
                config.height = s.height.max(1);
                gfx.surface.configure(&gfx.gpu.device, config);
                redraw = true;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let p = (position.x as f32, position.y as f32);
                if let Some(prev) = self.dragging {
                    state.center.0 -= (p.0 - prev.0) / state.zoom;
                    state.center.1 -= (p.1 - prev.1) / state.zoom;
                    self.dragging = Some(p);
                    redraw = true;
                }
                self.cursor = p;
            }
            WindowEvent::MouseInput {
                state: s,
                button: MouseButton::Left,
                ..
            } => {
                self.dragging = if s == ElementState::Pressed {
                    Some(self.cursor)
                } else {
                    None
                };
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let dy = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
                let factor = 1.15f32.powf(dy);
                let cursor = self.cursor;
                // zoom around the cursor
                let (cx, cy) = (config.width as f32 / 2.0, config.height as f32 / 2.0);
                let before = (
                    state.center.0 + (cursor.0 - cx) / state.zoom,
                    state.center.1 + (cursor.1 - cy) / state.zoom,
                );
                state.zoom = (state.zoom * factor).clamp(1e-4, 1e3);
                state.center = (
                    before.0 - (cursor.0 - cx) / state.zoom,
                    before.1 - (cursor.1 - cy) / state.zoom,
                );
                redraw = true;
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                let nz = src.size("z");
                let nt = src.size("t");
                match event.logical_key.as_ref() {
                    Key::Named(NamedKey::Escape) | Key::Character("q") => event_loop.exit(),
                    Key::Named(NamedKey::ArrowUp) | Key::Character(".") => {
                        state.z = (state.z + 1).min(nz - 1)
                    }
                    Key::Named(NamedKey::ArrowDown) | Key::Character(",") => {
                        state.z = state.z.saturating_sub(1)
                    }
                    Key::Character("]") => state.t = (state.t + 1).min(nt - 1),
                    Key::Character("[") => state.t = state.t.saturating_sub(1),
                    Key::Character("l") => state.show_labels = !state.show_labels,
                    Key::Character("r") => *state = State::fit(src, config.width, config.height),
                    Key::Character("+") | Key::Character("=") => state.zoom *= 1.25,
                    Key::Character("-") => state.zoom /= 1.25,
                    Key::Character("s") => {
                        let (level, _) = pick_level(src, state.zoom);
                        let opts = super::RenderOptions {
                            width: config.width,
                            height: config.height,
                            z: Some(state.z),
                            t: state.t,
                            level: Some(level),
                            show_labels: state.show_labels,
                            label_opacity: 0.5,
                        };
                        let name = format!(
                            "ome_zarr_screenshot_{}.png",
                            chrono::Utc::now().format("%Y%m%d_%H%M%S")
                        );
                        match super::render_source_rgba(src, &opts) {
                            Ok(rgba) => {
                                let _ = std::fs::write(
                                    &name,
                                    threers::encode_png(config.width, config.height, &rgba),
                                );
                                println!("saved {name}");
                            }
                            Err(e) => warn!("screenshot failed: {e}"),
                        }
                    }
                    Key::Character(c)
                        if c.len() == 1 && c.chars().all(|ch| ch.is_ascii_digit() && ch != '0') =>
                    {
                        let i = c.parse::<usize>().unwrap() - 1;
                        if let Some(ch) = src.channels.get_mut(i) {
                            ch.active = !ch.active;
                            println!(
                                "channel {} ({}) {}",
                                i + 1,
                                ch.label,
                                if ch.active { "on" } else { "off" }
                            );
                        }
                    }
                    _ => {}
                }
                redraw = true;
            }
            WindowEvent::RedrawRequested => {
                let (level, f) = pick_level(src, state.zoom);
                let key = (level, state.z, state.t);
                if let std::collections::hash_map::Entry::Vacant(e) = self.cache.entry(key) {
                    match src.plane(level, state.z, state.t) {
                        Ok(p) => {
                            e.insert(gfx.renderer.upload(&p));
                        }
                        Err(e) => {
                            warn!("could not load plane: {e}");
                            return;
                        }
                    }
                }
                let gp = &self.cache[&key];
                gfx.window.set_title(&title(src, state, level));
                let view = ViewParams {
                    center: (state.center.0 / f, state.center.1 / f),
                    voxels_per_pixel: 1.0 / (state.zoom * f),
                    show_labels: state.show_labels,
                    label_opacity: 0.5,
                    label_seed: 7,
                    background: [0.0, 0.0, 0.0],
                };
                match gfx.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(frame)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                        let view_tex = frame.texture.create_view(&Default::default());
                        gfx.renderer.draw(
                            &view_tex,
                            (config.width, config.height),
                            gp,
                            &src.channels,
                            &view,
                        );
                        gfx.gpu.queue.present(frame);
                    }
                    wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                        gfx.surface.configure(&gfx.gpu.device, config);
                        redraw = true;
                    }
                    e => log::error!("surface error: {e:?}"),
                }
            }
            _ => {}
        }
        if redraw {
            gfx.window.request_redraw();
        }
    }
}

/// Open a window showing the OME-Zarr image at `path`.
pub fn run(path: &str) -> Result<()> {
    let src = RenderSource::open(path)?;
    let event_loop = EventLoop::new().map_err(|e| Error::value(format!("event loop: {e}")))?;
    println!(
        "drag: pan  scroll: zoom  up/down: z  [ ]: t  1-9: channels  L: labels  R: reset  S: screenshot  Q: quit"
    );
    let mut viewer = Viewer {
        title: format!("ome_zarr — {path}"),
        src,
        gfx: None,
        state: None,
        cache: HashMap::new(),
        cursor: (0.0, 0.0),
        dragging: None,
        error: None,
    };
    event_loop
        .run_app(&mut viewer)
        .map_err(|e| Error::value(format!("event loop: {e}")))?;
    viewer.error.map_or(Ok(()), Err)
}
