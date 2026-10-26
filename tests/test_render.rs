//! GPU rendering through threers (skips without a GPU adapter).
#![cfg(all(feature = "render", not(target_arch = "wasm32")))]

use ome_zarr::render::{RenderOptions, RenderSource, render_source_rgba};

#[test]
fn render_coins_with_labels() {
    if threers::gpu::GpuContext::headless_blocking().is_err() {
        eprintln!("no GPU adapter; skipping");
        return;
    }
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("coins.zarr");
    ome_zarr::data::create_zarr_default(p.to_str().unwrap()).unwrap();
    let src = RenderSource::open(p.as_path()).unwrap();
    assert_eq!(src.channels.len(), 1);
    assert_eq!(src.channels[0].color, [1.0, 0.0, 0.0]);
    assert_eq!(src.labels.len(), 1);
    let opts = RenderOptions {
        width: 284,
        height: 203,
        level: Some(0),
        ..Default::default()
    };
    let with = render_source_rgba(&src, &opts).unwrap();
    let without = render_source_rgba(
        &src,
        &RenderOptions {
            show_labels: false,
            ..opts.clone()
        },
    )
    .unwrap();
    assert_eq!(with.len(), 284 * 203 * 4);
    // the red channel only: green and blue are zero without labels
    assert!(without.chunks(4).all(|p| p[1] == 0 && p[2] == 0));
    assert!(without.chunks(4).any(|p| p[0] > 100));
    // labels tint the segmented coins
    let tinted = with.chunks(4).filter(|p| p[1] > 0 || p[2] > 0).count();
    assert!(tinted > 1000, "{tinted}");
    // label 0 is transparent: background pixels are unchanged
    let (img, lbl) = ome_zarr::data::coins_arrays().unwrap();
    let (y, x) = (0, 0);
    assert_eq!(lbl[[y, x]], 0);
    let i = (y * 284 + x) * 4;
    assert_eq!(with[i..i + 4], without[i..i + 4]);
    let _ = img;
}
