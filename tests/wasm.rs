//! Runs on wasm32 (`scripts/test-wasm.sh`) as well as natively: in-memory
//! stores, every codec, pyramids, labels, the reader and transforms.

use ome_zarr::array::{DType, LazyArray, NdArray};
use ome_zarr::classes::{LabelsArg, OmeZarrImage, OmeZarrLabels, OmeZarrMultiscale, Version};
use ome_zarr::format::Format;
use ome_zarr::io::ZarrLocation;
use ome_zarr::reader::Reader;
use ome_zarr::scale::{Methods, ScaleFactors};
use ome_zarr::store::Store;
use ome_zarr::writer::{self, ImageOptions, StorageOptions};
use ome_zarr::zarr::{Compressor, Group, Mode};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

fn data(shape: &[usize]) -> NdArray {
    let n: usize = shape.iter().product();
    NdArray::from_f64_vec(
        DType::UInt16,
        shape,
        (0..n as u64)
            .map(|i| ((i * 2654435761) % 4096) as f64)
            .collect(),
    )
    .unwrap()
}

#[test]
fn memory_store_roundtrip_all_codecs() {
    for (fmt, comp) in [
        (
            Format::V04,
            Compressor::Zstd {
                level: 0,
                checksum: false,
            },
        ),
        (Format::V04, Compressor::blosc_zstd()),
        (Format::V05, Compressor::Gzip { level: 1 }),
        (Format::V05, Compressor::blosc_zstd()),
        (Format::V05, Compressor::None),
    ] {
        let store = Store::memory();
        let g = Group::open(&store, "", Mode::Write, Some(fmt.zarr_format())).unwrap();
        let img = data(&[2, 40, 50]);
        let opts = ImageOptions {
            fmt: Some(fmt),
            axes: Some("cyx".into()),
            scale_factors: Some(ScaleFactors::Ints(vec![2, 4])),
            storage_options: Some(
                StorageOptions {
                    chunks: Some(vec![16, 16]),
                    compressor: Some(comp.clone()),
                    ..Default::default()
                }
                .into(),
            ),
            ..Default::default()
        };
        writer::write_image(&LazyArray::from_ndarray(img.clone()), &g, &opts).unwrap();
        let nodes = Reader::new(ZarrLocation::open(&store).unwrap())
            .unwrap()
            .nodes()
            .unwrap();
        assert_eq!(nodes[0].data.len(), 3);
        assert_eq!(nodes[0].data[0].compute().unwrap(), img);
        assert_eq!(nodes[0].data[2].shape(), &[2, 10, 12]);
        assert_eq!(
            g.array("s1").unwrap().compressor(),
            if comp == Compressor::None {
                None
            } else {
                Some(&comp)
            }
        );
    }
}

#[test]
fn sharded_pyramid_with_labels() {
    let store = Store::memory();
    let g = Group::open(&store, "img", Mode::Write, Some(3)).unwrap();
    let lbl = NdArray::from_f64_vec(
        DType::UInt8,
        &[64, 64],
        (0..64 * 64).map(|i| ((i / 640) % 3) as f64).collect(),
    )
    .unwrap();
    let labels = OmeZarrLabels::new(
        OmeZarrImage::from_axes(LazyArray::from_ndarray(lbl.clone()), "yx")
            .unwrap()
            .with_name("cells"),
        None,
        Some(Methods::Nearest),
        true,
    )
    .unwrap();
    let ms = OmeZarrMultiscale::builder(
        OmeZarrImage::from_axes(LazyArray::from_ndarray(data(&[64, 64])), "yx").unwrap(),
    )
    .labels(LabelsArg::One(labels))
    .build()
    .unwrap();
    let so = StorageOptions {
        chunks: Some(vec![8, 8]),
        shards: Some(vec![32, 32]),
        ..Default::default()
    }
    .into();
    ms.to_ome_zarr(g.clone(), Some(&so), Version::V06, true, true)
        .unwrap();
    let back = OmeZarrMultiscale::from_ome_zarr(g).unwrap();
    assert_eq!(back.images().len(), 5);
    let l = back.label("cells").unwrap();
    assert_eq!(l.images()[0].data.compute().unwrap(), lbl);
    assert_eq!(
        l.image_label
            .as_ref()
            .unwrap()
            .colors
            .as_ref()
            .unwrap()
            .len(),
        3
    );
}

#[test]
fn transforms_graph() {
    use ome_zarr::transformnd::{Spaced, Transform, TransformGraph};
    let mut g = TransformGraph::new();
    let t = Spaced::new(Transform::Scale(vec![2.0, 3.0]), "a", "b");
    g.add_transform(t.clone()).unwrap();
    g.add_transform(t.invert().unwrap()).unwrap();
    assert_eq!(
        g.transform(&"b", &"a", &vec![vec![2.0, 3.0]]).unwrap(),
        vec![vec![1.0, 1.0]]
    );
}

/// A zipped fileset held in memory, as a web app would fetch it: written to
/// a memory store, zipped, then opened with `Store::zip_bytes` and read back.
#[test]
fn zip_bytes_store() {
    use std::io::Write;
    use zarrs_storage::{ListableStorageTraits, ReadableStorageTraits};
    let mem = Store::memory();
    let g = Group::open(&mem, "", Mode::Write, Some(3)).unwrap();
    let opts = ImageOptions {
        axes: Some("yx".into()),
        scale_factors: Some(ScaleFactors::Ints(vec![2])),
        ..Default::default()
    };
    writer::write_image(&LazyArray::from_ndarray(data(&[32, 32])), &g, &opts).unwrap();
    let mut buf = std::io::Cursor::new(Vec::new());
    {
        let mut z = zip::ZipWriter::new(&mut buf);
        for key in mem.inner().list().unwrap() {
            let bytes = mem.inner().get(&key).unwrap().unwrap();
            z.start_file(key.as_str(), zip::write::SimpleFileOptions::default())
                .unwrap();
            z.write_all(&bytes).unwrap();
        }
        z.finish().unwrap();
    }
    let store = Store::zip_bytes(buf.into_inner(), "image.ome.zarr.zip").unwrap();
    let loc = ZarrLocation::open(store.clone()).unwrap();
    let nodes = Reader::new(loc).unwrap().nodes().unwrap();
    assert_eq!(nodes[0].data.len(), 2);
    let a = ome_zarr::zarr::ZArray::open(&store, "s0")
        .unwrap()
        .read_all()
        .unwrap();
    assert!(a == data(&[32, 32]));
}

/// The numcodecs compressors (zlib, bz2, lz4, lzma: pure Rust or bundled C)
/// round-trip on every target.
#[test]
fn numcodecs_compressors_roundtrip() {
    for (id, args) in [
        ("zlib", vec![("level", serde_json::json!(3))]),
        ("bz2", vec![("level", serde_json::json!(5))]),
        ("lz4", vec![]),
        ("lzma", vec![]),
    ] {
        for fmt in [Format::V04, Format::V05] {
            let store = Store::memory();
            let g = Group::open(&store, "", Mode::Write, Some(fmt.zarr_format())).unwrap();
            let opts = ImageOptions {
                fmt: Some(fmt),
                axes: Some("yx".into()),
                scale_factors: Some(ScaleFactors::Ints(vec![2])),
                storage_options: Some(writer::StorageOptionsArg::All(StorageOptions {
                    compressor: Some(Compressor::numcodecs(id, &args).unwrap()),
                    ..Default::default()
                })),
                ..Default::default()
            };
            writer::write_image(&LazyArray::from_ndarray(data(&[32, 32])), &g, &opts).unwrap();
            let a = ome_zarr::zarr::ZArray::open(&store, "s0")
                .unwrap()
                .read_all()
                .unwrap();
            assert!(a == data(&[32, 32]), "{id} {fmt:?}");
        }
    }
}

/// float16 and complex images through the pyramid writer and back.
#[test]
fn float16_and_complex_images() {
    use num_complex::Complex;
    let re = data(&[16, 16]).to_f64();
    for (dtype, arr) in [
        (
            DType::Float16,
            NdArray::Float16(re.mapv(half::f16::from_f64)),
        ),
        (
            DType::Complex64,
            NdArray::Complex64(re.mapv(|v| Complex::new(v as f32, -(v as f32)))),
        ),
        (
            DType::Complex128,
            NdArray::Complex128(re.mapv(|v| Complex::new(v, 2.0 * v))),
        ),
    ] {
        for method in [Methods::Nearest, Methods::LocalMean] {
            let store = Store::memory();
            let g = Group::open(&store, "", Mode::Write, Some(3)).unwrap();
            let opts = ImageOptions {
                axes: Some("yx".into()),
                method: Some(method),
                scale_factors: Some(ScaleFactors::Ints(vec![2])),
                ..Default::default()
            };
            writer::write_image(&LazyArray::from_ndarray(arr.clone()), &g, &opts).unwrap();
            let s0 = ome_zarr::zarr::ZArray::open(&store, "s0")
                .unwrap()
                .read_all()
                .unwrap();
            assert!(s0 == arr, "{dtype}");
            let s1 = ome_zarr::zarr::ZArray::open(&store, "s1")
                .unwrap()
                .read_all()
                .unwrap();
            assert_eq!(s1.dtype(), dtype);
            assert_eq!(s1.shape(), vec![8, 8]);
        }
    }
}

/// pydantic's validation errors, numpy's seeded colours and argparse errors
/// need nothing platform-specific.
#[test]
fn validation_colours_and_cli_parsing() {
    let e = ome_zarr::models::pydantic::validate_model(
        "CoordinateSystem",
        &serde_json::json!({"name": "a", "axes": [{"name": "x"}, {"name": "x"}]}),
    )
    .unwrap_err();
    assert!(e.py_str().starts_with("1 validation error for CoordinateSystem\naxes\n  Value error, Duplicate values found in ['x', 'x']."));
    let mut mt = ome_zarr::utils::Mt19937::new(1234);
    assert_eq!(
        (0..4).map(|_| mt.randint(0, 255)).collect::<Vec<_>>(),
        [47, 211, 38, 53]
    );
    let exit = ome_zarr::cli::parser(false)
        .parse(&["info".into()])
        .unwrap_err();
    assert_eq!(exit.code, 2);
    assert!(
        exit.stderr
            .ends_with("ome_zarr info: error: the following arguments are required: path\n")
    );
}
