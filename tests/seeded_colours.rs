//! Seeded label colours (alone in this test binary: the generator is
//! process-wide, as numpy's `np.random` is).

/// `np.random.seed(...)` + label colours: the same as ome-zarr-py's
/// (expected values recorded from numpy 2.5 / ome-zarr-py).
#[test]
fn seeded_label_colours_match_numpy() {
    use ome_zarr::utils::{Mt19937, np_random_seed};
    let mut mt = Mt19937::new(1234);
    let draws: Vec<i64> = (0..40).map(|_| mt.randint(0, 255)).collect();
    assert_eq!(
        draws,
        [
            47, 211, 38, 53, 204, 116, 152, 249, 143, 177, 23, 233, 154, 30, 171, 158, 236, 124,
            26, 118, 186, 120, 112, 220, 69, 80, 201, 127, 246, 254, 175, 50, 240, 251, 76, 37, 34,
            166, 250, 195
        ]
    );
    let mut mt = Mt19937::new(4294967295);
    assert_eq!(
        (0..5).map(|_| mt.randint(0, 255)).collect::<Vec<_>>(),
        [163, 34, 12, 71, 180]
    );
    np_random_seed(7);
    let data =
        ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[8, 8]), |i| ((i[0] * 8 + i[1]) % 5) as u8);
    let img = ome_zarr::classes::OmeZarrImage::from_axes(
        ome_zarr::array::LazyArray::from_ndarray(ome_zarr::array::NdArray::UInt8(data)),
        "yx",
    )
    .unwrap();
    let labels = ome_zarr::classes::OmeZarrLabels::from_image(img).unwrap();
    let colors: Vec<(i64, [u8; 4])> = labels
        .image_label
        .as_ref()
        .unwrap()
        .colors
        .as_ref()
        .unwrap()
        .iter()
        .map(|c| (c.label_value, c.rgba.unwrap()))
        .collect();
    assert_eq!(
        colors,
        [
            (0, [175, 196, 25, 255]),
            (1, [246, 67, 211, 255]),
            (2, [151, 103, 92, 255]),
            (3, [185, 142, 23, 255]),
            (4, [72, 89, 110, 255])
        ]
    );
}
