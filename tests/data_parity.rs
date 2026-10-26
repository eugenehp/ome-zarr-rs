//! Sample data parity: the coins segmentation must equal scikit-image's.
#![cfg(not(target_arch = "wasm32"))]

use ome_zarr::array::NdArray;
use ome_zarr::store::Store;
use ome_zarr::zarr::ZArray;

#[test]
fn coins_matches_skimage() {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/py/coins_reference.zarr");
    let s = Store::local(&p, true).unwrap();
    let want_img = ZArray::open(&s, "image").unwrap().read_all().unwrap();
    let want_lbl = ZArray::open(&s, "labels").unwrap().read_all().unwrap();
    let (img, lbl) = ome_zarr::data::coins_arrays().unwrap();
    assert_eq!(NdArray::UInt8(img.into_dyn()), want_img);
    let got = NdArray::Int32(lbl.into_dyn());
    assert_eq!(got.dtype(), want_lbl.dtype());
    assert!(got == want_lbl, "coins labels differ");
}
