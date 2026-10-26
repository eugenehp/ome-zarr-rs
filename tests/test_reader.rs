//! Ports of ome-zarr-py's test_reader.py, test_node.py, test_starting_points.py
//! and test_io.py.
#![cfg(not(target_arch = "wasm32"))]

use ome_zarr::array::{DType, LazyArray, NdArray};
use ome_zarr::classes::{OmeZarrImage, OmeZarrMultiscale, Version};
use ome_zarr::data::create_zarr_default;
use ome_zarr::format::Format;
use ome_zarr::io::{StoreLike, ZarrLocation, parse_url, parse_url_r};
use ome_zarr::reader::{Node, Reader, Spec, SpecKind};
use ome_zarr::scale::ScaleFactors;
use ome_zarr::store::Store;
use ome_zarr::writer::{self, ImageOptions, add_metadata, get_metadata};
use ome_zarr::zarr::{ArraySpec, Group, JsonMap, Mode, store_lazy};
use serde_json::json;

fn setup() -> (tempfile::TempDir, String) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data");
    create_zarr_default(path.to_str().unwrap()).unwrap();
    (dir, path.to_string_lossy().to_string())
}

fn kinds(node: &Node) -> Vec<String> {
    let mut v: Vec<String> = node.specs.iter().map(|s| s.kind().to_string()).collect();
    v.sort();
    v
}

#[test]
fn reader_image_labels_label() {
    let (_d, path) = setup();
    for p in [
        path.clone(),
        format!("{path}/labels"),
        format!("{path}/labels/coins"),
    ] {
        let reader = Reader::new(parse_url_r(p.as_str()).unwrap().unwrap()).unwrap();
        assert_eq!(reader.nodes().unwrap().len(), 3, "{p}");
    }
}

#[test]
fn reader_omero() {
    let (_d, path) = setup();
    let nodes = Reader::new(parse_url_r(path.as_str()).unwrap().unwrap())
        .unwrap()
        .nodes()
        .unwrap();
    let omero = nodes[0].zarr.root_attrs()["omero"].clone();
    assert_eq!(omero["channels"].as_array().unwrap().len(), 1);
}

#[test]
fn read_v05() {
    let dir = tempfile::tempdir().unwrap();
    let img_path = dir.path().join("test_read_v05.zarr");
    let root = Group::open_url(img_path.to_str().unwrap(), Mode::Append, None).unwrap();
    let data: Vec<f64> = (0..10 * 128 * 128).map(|v| (v % 23) as f64).collect();
    let nd = NdArray::from_f64_vec(DType::UInt8, &[10, 128, 128], data).unwrap();
    let arr = root
        .create_array(
            "s0",
            &ArraySpec::new(&[10, 128, 128], DType::UInt8, &[10, 10, 10], 3),
            false,
        )
        .unwrap();
    store_lazy(&LazyArray::from_ndarray(nd.clone()), &arr).unwrap();
    root.set_attr(
        "ome",
        json!({"version": "0.5", "multiscales": [{
            "axes": [{"name": "z", "type": "space"}, {"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
            "datasets": [{"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 1]}]}]
        }]}),
    )
    .unwrap();
    let nodes = Reader::new(parse_url_r(img_path.as_path()).unwrap().unwrap())
        .unwrap()
        .nodes()
        .unwrap();
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].data[0].compute().unwrap(), nd);
    let ms = OmeZarrMultiscale::from_ome_zarr(img_path.as_path()).unwrap();
    assert_eq!(ms.images().len(), 1);
}

#[test]
fn node_image_labels_label() {
    let (_d, path) = setup();
    let node = Node::root(parse_url_r(path.as_str()).unwrap().unwrap()).unwrap();
    assert!(!node.data.is_empty() && !node.metadata.is_empty());
    assert_eq!(node.specs.len(), 2);
    assert!(matches!(node.specs[0], Spec::Multiscales(_)));

    let node = Node::root(parse_url_r(format!("{path}/labels")).unwrap().unwrap()).unwrap();
    assert!(node.data.is_empty() && node.metadata.is_empty());
    assert_eq!(node.specs.len(), 1);
    assert_eq!(node.specs[0].kind(), SpecKind::Labels);

    let node = Node::root(
        parse_url_r(format!("{path}/labels/coins"))
            .unwrap()
            .unwrap(),
    )
    .unwrap();
    assert!(!node.data.is_empty() && !node.metadata.is_empty());
    assert_eq!(node.specs[0].kind(), SpecKind::Label);
    assert_eq!(node.specs[1].kind(), SpecKind::Multiscales);
}

#[test]
fn starting_points() {
    let (_d, path) = setup();
    let node = Node::root(parse_url_r(path.as_str()).unwrap().unwrap()).unwrap();
    assert_eq!(kinds(&node), vec!["Multiscales", "OMERO"]);
    let node = Node::root(parse_url_r(format!("{path}/labels")).unwrap().unwrap()).unwrap();
    assert_eq!(kinds(&node), vec!["Labels"]);
    let node = Node::root(
        parse_url_r(format!("{path}/labels/coins"))
            .unwrap()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(kinds(&node), vec!["Label", "Multiscales"]);
}

fn plate(
    fmt: Format,
    rows: &[&str],
    cols: &[&str],
    wells: &[&str],
    fields: usize,
    dims: &[u64],
    axes: Option<&str>,
) -> (tempfile::TempDir, String) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data");
    let root =
        Group::open_url(path.to_str().unwrap(), Mode::Write, Some(fmt.zarr_format())).unwrap();
    writer::write_plate_metadata_simple(&root, rows, cols, wells, Some(fmt)).unwrap();
    for wp in wells {
        let well = root.require_group(wp).unwrap();
        let images: Vec<_> = (0..fields).map(|f| json!(f.to_string())).collect();
        writer::write_well_metadata(&well, &images, Some(fmt)).unwrap();
        for f in 0..fields {
            let image = well.require_group(&f.to_string()).unwrap();
            let data = LazyArray::zeros(dims, DType::Float64, None);
            let opts = ImageOptions {
                fmt: Some(fmt),
                axes: axes.map(|a| a.into()),
                ..Default::default()
            };
            writer::write_image(&data, &image, &opts).unwrap();
        }
    }
    (dir, path.to_string_lossy().to_string())
}

#[test]
fn minimal_plate() {
    let (_d, path) = plate(
        Format::V04,
        &["A"],
        &["1"],
        &["A/1"],
        1,
        &[1, 1, 1, 256, 256],
        None,
    );
    let node = Node::root(parse_url_r(path.as_str()).unwrap().unwrap()).unwrap();
    assert!(!node.data.is_empty() && !node.metadata.is_empty());
    assert_eq!(node.specs.len(), 1);
    let Spec::Plate(p) = &node.specs[0] else {
        panic!("not a plate")
    };
    assert_eq!(p.row_names, vec!["A"]);
    assert_eq!(p.col_names, vec!["1"]);
    assert_eq!(p.well_paths, vec!["A/1"]);
    assert_eq!((p.row_count, p.column_count), (1, 1));
    let node = Node::root(parse_url_r(format!("{path}/A/1")).unwrap().unwrap()).unwrap();
    assert_eq!(node.specs[0].kind(), SpecKind::Well);
    assert!(!node.data.is_empty());
}

#[test]
fn multiwells_plate() {
    let rows = ["A", "B", "C"];
    let cols = ["1", "2", "3", "4"];
    let wells = ["A/1", "A/2", "A/4", "B/2", "B/3", "C/1", "C/3", "C/4"];
    let (_d, path) = plate(
        Format::V04,
        &rows,
        &cols,
        &wells,
        3,
        &[1, 1, 1, 256, 256],
        None,
    );
    let node = Node::root(parse_url_r(path.as_str()).unwrap().unwrap()).unwrap();
    let Spec::Plate(p) = &node.specs[0] else {
        panic!("not a plate")
    };
    assert_eq!(p.well_paths, wells.to_vec());
    assert_eq!((p.row_count, p.column_count), (3, 4));
    // the stitched plate: 3 rows x 4 columns of 2x2 fields (3 fields -> 2x2 grid)
    assert_eq!(node.data[0].shape(), &[1, 1, 1, 3 * 256, 4 * 256]);
    for wp in wells {
        let n = Node::root(parse_url_r(format!("{path}/{wp}")).unwrap().unwrap()).unwrap();
        assert_eq!(n.specs[0].kind(), SpecKind::Well);
        assert_eq!(n.data[0].shape(), &[1, 1, 1, 512, 512]);
    }
    for wp in ["A/3", "B/1", "B/4", "C/2"] {
        assert!(parse_url_r(format!("{path}/{wp}")).unwrap().is_none());
    }
}

#[test]
fn plate_2d_5d() {
    for (axes, dims) in [
        ("yx", vec![256u64, 256]),
        ("tyx", vec![1, 256, 256]),
        ("zyx", vec![1, 256, 256]),
        ("cyx", vec![1, 256, 256]),
        ("czyx", vec![1, 1, 256, 256]),
        ("tzyx", vec![1, 1, 256, 256]),
        ("tcyx", vec![1, 1, 256, 256]),
    ] {
        let (_d, path) = plate(Format::V04, &["A"], &["1"], &["A/1"], 1, &dims, Some(axes));
        let node = Node::root(parse_url_r(path.as_str()).unwrap().unwrap()).unwrap();
        assert_eq!(node.specs[0].kind(), SpecKind::Plate, "{axes}");
        let node = Node::root(parse_url_r(format!("{path}/A/1")).unwrap().unwrap()).unwrap();
        assert_eq!(node.specs[0].kind(), SpecKind::Well, "{axes}");
    }
}

#[test]
fn io_parse_url() {
    let (_d, path) = setup();
    assert!(parse_url_r(path.as_str()).unwrap().is_some());
    assert!(
        parse_url_r(format!("{path}/does-not-exist"))
            .unwrap()
            .is_none()
    );
    assert!(ZarrLocation::open(path.as_str()).unwrap().exists());
    assert!(
        ZarrLocation::open(std::path::Path::new(&path))
            .unwrap()
            .exists()
    );
    let store = Store::local(&path, false).unwrap();
    assert!(ZarrLocation::open(store).unwrap().exists());
}

#[test]
fn io_no_overwrite() {
    let (_d, path) = setup();
    let loc = parse_url(path.as_str(), "w", Format::V05).unwrap().unwrap();
    let root = Group::open(loc.store(), "", Mode::Append, None).unwrap();
    let mut m = JsonMap::new();
    m.insert("extra".into(), json!("test_no_overwrite"));
    add_metadata(&root, &m, None).unwrap();
    let read_root = Group::open_url(&path, Mode::Read, None).unwrap();
    let attrs = get_metadata(&read_root).unwrap();
    assert_eq!(attrs["extra"], "test_no_overwrite");
    assert!(attrs.contains_key("multiscales"));
}

fn image64() -> NdArray {
    NdArray::from_f64_vec(
        DType::UInt16,
        &[64, 64],
        (0..64 * 64).map(|v| v as f64).collect(),
    )
    .unwrap()
}

#[test]
fn store_without_a_path() {
    let store = Store::memory();
    let g = Group::open(&store, "", Mode::Write, None).unwrap();
    let opts = ImageOptions {
        axes: Some("yx".into()),
        ..Default::default()
    };
    writer::write_image(&LazyArray::from_ndarray(image64()), &g, &opts).unwrap();
    let loc = ZarrLocation::open(&store).unwrap();
    assert!(loc.exists());
    assert!(loc.store().same(&store));
    let nodes = Reader::new(loc).unwrap().nodes().unwrap();
    assert!(!nodes.is_empty());
    assert_eq!(nodes[0].data[0].compute().unwrap(), image64());
}

#[test]
fn prefix_inside_a_store() {
    let store = Store::memory();
    let g = Group::open(&store, "images/img", Mode::Write, None).unwrap();
    let opts = ImageOptions {
        axes: Some("yx".into()),
        ..Default::default()
    };
    writer::write_image(&LazyArray::from_ndarray(image64()), &g, &opts).unwrap();
    let loc = ZarrLocation::new(
        StoreLike::StorePath(store.clone(), "images/img".into()),
        "r",
        Format::V05,
    )
    .unwrap();
    assert!(loc.exists());
    assert_eq!(loc.basename(), "img");
    let finest = loc.root_attrs()["multiscales"][0]["datasets"][0]["path"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(loc.load(&finest).unwrap().compute().unwrap(), image64());
    let child = loc.create(&finest).unwrap();
    let expect = ZarrLocation::new(
        StoreLike::StorePath(store.clone(), format!("images/img/{finest}")),
        "r",
        Format::V05,
    )
    .unwrap();
    assert!(child == expect);
    let other = ZarrLocation::new(
        StoreLike::StorePath(store.clone(), "images/other".into()),
        "r",
        Format::V05,
    )
    .unwrap();
    assert!(!other.exists());
}

#[test]
fn unrelated_stores_differ() {
    let a = ZarrLocation::open(Store::memory()).unwrap();
    let b = ZarrLocation::open(Store::memory()).unwrap();
    assert!(a != b);
}

#[test]
fn class_writer_on_a_store() {
    let store = Store::memory();
    let ms = OmeZarrMultiscale::builder(
        OmeZarrImage::from_axes(LazyArray::from_ndarray(image64()), "yx").unwrap(),
    )
    .scale_factors(Some(ScaleFactors::Ints(vec![])))
    .build()
    .unwrap();
    let g = Group::open(&store, "images/img", Mode::Write, None).unwrap();
    ms.to_ome_zarr(g, None, Version::V06, true, true).unwrap();
    let loc = ZarrLocation::new(
        StoreLike::StorePath(store.clone(), "images/img".into()),
        "r",
        Format::V05,
    )
    .unwrap();
    let nodes = Reader::new(loc).unwrap().nodes().unwrap();
    assert_eq!(nodes[0].data[0].compute().unwrap(), image64());
    let back = OmeZarrMultiscale::from_ome_zarr(
        Group::open(&store, "images/img", Mode::Read, None).unwrap(),
    )
    .unwrap();
    assert_eq!(back.images()[0].data.compute().unwrap(), image64());
}

#[test]
fn read_from_a_zip() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let directory = dir.path().join("img.zarr");
    let g = Group::open_url(directory.to_str().unwrap(), Mode::Write, None).unwrap();
    let opts = ImageOptions {
        axes: Some("yx".into()),
        ..Default::default()
    };
    writer::write_image(&LazyArray::from_ndarray(image64()), &g, &opts).unwrap();
    let archive = dir.path().join("img.zarr.zip");
    let mut zw = zip::ZipWriter::new(std::fs::File::create(&archive).unwrap());
    fn add(zw: &mut zip::ZipWriter<std::fs::File>, base: &std::path::Path, d: &std::path::Path) {
        for e in std::fs::read_dir(d).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                add(zw, base, &p);
            } else {
                let name = p
                    .strip_prefix(base)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                zw.start_file(name, zip::write::SimpleFileOptions::default())
                    .unwrap();
                zw.write_all(&std::fs::read(&p).unwrap()).unwrap();
            }
        }
    }
    add(&mut zw, &directory, &directory);
    zw.finish().unwrap();
    let store = Store::zip(&archive).unwrap();
    let loc = ZarrLocation::open(store).unwrap();
    assert!(loc.exists());
    let nodes = Reader::new(loc).unwrap().nodes().unwrap();
    assert_eq!(nodes[0].data[0].compute().unwrap(), image64());
}
