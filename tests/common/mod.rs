//! Helpers shared by the parity tests.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use ome_zarr::array::{DType, Element, IntoNdArray, NdArray};
use ome_zarr::store::Store;
use ome_zarr::zarr::ZArray;
use serde_json::Value;

/// `gen_data()` of scripts/gen_parity_fixtures.py: `((i * 2654435761) % m) - offset`, cast.
pub fn gen_data(shape: &[usize], dtype: DType, m: u64, offset: i64) -> NdArray {
    let n: usize = shape.iter().product();
    let vals: Vec<i64> = (0..n as u64)
        .map(|i| (i.wrapping_mul(2654435761) % m) as i64 - offset)
        .collect();
    let sh = ndarray::IxDyn(shape);
    ome_zarr::dtype_dispatch!(dtype, T => {
        let v: Vec<T> = match dtype {
            DType::Float32 => vals.iter().map(|&x| <T as Element>::from_f64((x as f64 / 7.0) as f32 as f64)).collect(),
            DType::Float64 => vals.iter().map(|&x| <T as Element>::from_f64(x as f64 / 7.0)).collect(),
            DType::Bool => vals.iter().map(|&x| <T as Element>::from_f64(if x % 2 == 1 { 1.0 } else { 0.0 })).collect(),
            _ => vals.iter().map(|&x| <T as Element>::from_i64(x)).collect(),
        };
        ndarray::ArrayD::from_shape_vec(sh, v).unwrap().into_nd()
    })
}

pub fn gen_u8(shape: &[usize]) -> NdArray {
    gen_data(shape, DType::UInt8, 251, 0)
}

pub fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/parity")
}

pub fn py_store(name: &str) -> PathBuf {
    fixtures().join("stores").join(name)
}

pub fn observations() -> Value {
    serde_json::from_str(&std::fs::read_to_string(fixtures().join("observations.json")).unwrap())
        .unwrap()
}

/// Remove random label colours before comparing.
pub fn mask(v: &mut Value) {
    match v {
        Value::Object(m) => {
            if m.contains_key("rgba") && m.contains_key("label-value") {
                m.insert("rgba".into(), Value::String("masked".into()));
            }
            for (_, x) in m.iter_mut() {
                mask(x);
            }
        }
        Value::Array(a) => a.iter_mut().for_each(mask),
        _ => {}
    }
}

pub fn walk(dir: &Path, base: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            walk(&p, base, out);
        } else {
            out.push(p.strip_prefix(base).unwrap().to_path_buf());
        }
    }
}

/// Compare two stores: the same files; every metadata document
/// byte-identical (with random label colours masked unless `exact_colors`);
/// every array's data equal. `masked_colors` lists label groups whose colours
/// are random in both.
pub fn compare_stores(py: &Path, rs: &Path) {
    let mut files = vec![];
    walk(py, py, &mut files);
    files.sort();
    let mut rs_files = vec![];
    walk(rs, rs, &mut rs_files);
    rs_files.sort();
    if files != rs_files {
        let only_py: Vec<_> = files
            .iter()
            .filter(|f| !rs_files.contains(f))
            .take(10)
            .collect();
        let only_rs: Vec<_> = rs_files
            .iter()
            .filter(|f| !files.contains(f))
            .take(10)
            .collect();
        panic!(
            "different files in {}: only python {only_py:?}, only rust {only_rs:?}",
            py.display()
        );
    }
    for f in &files {
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        if !matches!(
            name.as_str(),
            "zarr.json" | ".zattrs" | ".zgroup" | ".zarray"
        ) {
            continue;
        }
        let py_bytes = std::fs::read(py.join(f)).unwrap();
        let rs_bytes = std::fs::read(rs.join(f)).unwrap();
        let a: Value = serde_json::from_slice(&py_bytes).unwrap();
        let b: Value = serde_json::from_slice(&rs_bytes).unwrap();
        assert_eq!(
            ome_zarr::pyjson::dumps(&a, Some(2)).as_bytes(),
            py_bytes.as_slice(),
            "pyjson round trip of {}",
            f.display()
        );
        let (mut am, mut bm) = (a.clone(), b);
        mask(&mut am);
        mask(&mut bm);
        let (ta, tb) = (
            ome_zarr::pyjson::dumps(&am, Some(2)),
            ome_zarr::pyjson::dumps(&bm, Some(2)),
        );
        if ta != tb {
            let (la, lb): (Vec<&str>, Vec<&str>) = (ta.lines().collect(), tb.lines().collect());
            let i = la
                .iter()
                .zip(&lb)
                .position(|(x, y)| x != y)
                .unwrap_or(la.len().min(lb.len()));
            let ctx = |l: &[&str]| l[i.saturating_sub(4)..(i + 5).min(l.len())].join("\n");
            panic!(
                "metadata differs in {} ({}) at line {i}:\n--- python\n{}\n--- rust\n{}",
                f.display(),
                py.display(),
                ctx(&la),
                ctx(&lb)
            );
        }
        if am == a {
            assert!(py_bytes == rs_bytes, "file bytes differ in {}", f.display());
        }
        let is_array =
            name == ".zarray" || a.get("node_type").and_then(|n| n.as_str()) == Some("array");
        if is_array {
            let arr_path = f.parent().unwrap().to_string_lossy().to_string();
            let pa = ZArray::open(&Store::local(py, true).unwrap(), &arr_path)
                .unwrap()
                .read_all()
                .unwrap();
            let ra = ZArray::open(&Store::local(rs, true).unwrap(), &arr_path)
                .unwrap()
                .read_all()
                .unwrap();
            assert_eq!(pa.dtype(), ra.dtype(), "dtype {arr_path}");
            if pa != ra {
                if pa.dtype().is_complex() {
                    let (xr, xi) = ome_zarr::skimage::complex_parts(&pa);
                    let (yr, yi) = ome_zarr::skimage::complex_parts(&ra);
                    let bad = xr
                        .iter()
                        .zip(&yr)
                        .chain(xi.iter().zip(&yi))
                        .filter(|(a, b)| a != b && !(a.is_nan() && b.is_nan()))
                        .count();
                    assert!(
                        bad == 0,
                        "complex array data differs in {arr_path} ({}): {bad} parts",
                        py.display()
                    );
                }
                let (x, y) = (pa.to_f64_vec(), ra.to_f64_vec());
                let bad = x
                    .iter()
                    .zip(&y)
                    .filter(|(a, b)| a != b && !(a.is_nan() && b.is_nan()))
                    .count();
                if bad > 0 {
                    let first = x.iter().zip(&y).position(|(a, b)| a != b).unwrap();
                    panic!(
                        "array data differs in {arr_path} ({}): {bad}/{} values, first at {first}: {} vs {}",
                        py.display(),
                        x.len(),
                        x[first],
                        y[first]
                    );
                }
            }
        }
    }
}
