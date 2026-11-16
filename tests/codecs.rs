//! The pure-Rust blosc and zstd codecs (src/blosc.rs, src/codecs.rs) against
//! numcodecs (c-blosc, libzstd) and zarr-python
//! (scripts/gen_codec_fixtures.py): every chunk numcodecs wrote decodes, and
//! the arrays zarr-python wrote read. What the tests write here is left in
//! $CARGO_TARGET_TMPDIR/written/codecs for scripts/check_codecs.py.
#![cfg(feature = "io")]

use std::path::PathBuf;

use ome_zarr::array::{DType, NdArray};
use ome_zarr::blosc::{self, Compressor as Cname, Shuffle};
use ome_zarr::store::Store;
use ome_zarr::zarr::{ArraySpec, Compressor, ZArray};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/codecs")
}

fn written() -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("written/codecs");
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The `uint16` array of `array.npy` (C order, 64 x 48).
fn expected() -> Vec<u16> {
    let b = std::fs::read(fixtures().join("array.npy")).unwrap();
    let header = u16::from_le_bytes([b[8], b[9]]) as usize;
    b[10 + header..]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&c| u16::from_le_bytes(c))
        .collect()
}

fn values(a: &NdArray) -> Vec<u16> {
    match a {
        NdArray::UInt16(a) => a.iter().copied().collect(),
        other => panic!("{other:?}"),
    }
}

#[test]
fn numcodecs_chunks_decode() {
    let cases: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(fixtures().join("cases.json")).unwrap())
            .unwrap();
    for c in &cases {
        let file = c["file"].as_str().unwrap();
        let encoded = std::fs::read(fixtures().join(file)).unwrap();
        let raw = std::fs::read(fixtures().join(c["raw"].as_str().unwrap())).unwrap();
        let got = if file.starts_with("blosc") {
            blosc::decompress(&encoded).unwrap_or_else(|e| panic!("{file}: {e}"))
        } else {
            let mut out = Vec::new();
            zstd_rs::Decompressor::new()
                .decompress(&encoded, None, raw.len(), &mut out)
                .unwrap();
            out
        };
        assert!(got == raw, "{file}");
    }
    assert_eq!(cases.len(), 184);
}

#[test]
fn zarr_python_arrays_read() {
    let want = expected();
    for name in ["v2_blosc", "v2_zstd", "v3_blosc", "v3_zstd"] {
        let store = Store::local(fixtures().join(format!("{name}.zarr")), true).unwrap();
        let a = ZArray::open(&store, "").unwrap();
        assert_eq!(a.shape(), &[64, 48], "{name}");
        assert_eq!(values(&a.read_all().unwrap()), want, "{name}");
    }
}

#[test]
fn arrays_and_chunks_for_zarr_python() {
    let want = expected();
    let data = NdArray::from_f64_vec(
        DType::UInt16,
        &[64, 48],
        want.iter().map(|&v| v as f64).collect(),
    )
    .unwrap();
    let blosc = |cname: &str, shuffle: u8| Compressor::Blosc {
        cname: cname.into(),
        clevel: 5,
        shuffle,
        blocksize: 0,
    };
    for (name, format, compressor) in [
        ("v2_blosc_lz4", 2, blosc("lz4", 1)),
        ("v2_blosc_zstd_bit", 2, blosc("zstd", 2)),
        ("v3_blosc_zlib", 3, blosc("zlib", 1)),
        ("v3_blosc_lz4_bit", 3, blosc("lz4", 2)),
        (
            "v2_zstd",
            2,
            Compressor::Zstd {
                level: 5,
                checksum: false,
            },
        ),
        (
            "v3_zstd_checksum",
            3,
            Compressor::Zstd {
                level: 3,
                checksum: true,
            },
        ),
    ] {
        let path = written().join(format!("{name}.zarr"));
        std::fs::remove_dir_all(&path).ok();
        let store = Store::local(&path, false).unwrap();
        let mut spec = ArraySpec::new(&[64, 48], DType::UInt16, &[20, 16], format);
        spec.compressor = compressor;
        let a = ZArray::create(&store, "", &spec).unwrap();
        a.write_region(&[0..64, 0..48], &data).unwrap();
        // and back, through this crate
        let back = ZArray::open(&Store::local(&path, true).unwrap(), "").unwrap();
        assert_eq!(values(&back.read_all().unwrap()), want, "{name}");
    }
    // raw blosc chunks of every kind, for numcodecs
    let raw: Vec<u8> = (0..30_001u32)
        .flat_map(|i| ((i * 7) % 4099).to_le_bytes())
        .collect();
    std::fs::write(written().join("raw.bin"), &raw).unwrap();
    for (cname, code) in [
        ("blosclz", Cname::BloscLz),
        ("lz4", Cname::Lz4),
        ("zlib", Cname::Zlib),
        ("zstd", Cname::Zstd),
    ] {
        for (sh, s) in [
            ("none", Shuffle::None),
            ("byte", Shuffle::Byte),
            ("bit", Shuffle::Bit),
        ] {
            for blocksize in [0, 4096] {
                let c = blosc::compress(&raw, code, 5, s, 4, blocksize).unwrap();
                assert_eq!(blosc::decompress(&c).unwrap(), raw);
                std::fs::write(
                    written().join(format!("blosc_{cname}_{sh}_{blocksize}.bin")),
                    c,
                )
                .unwrap();
            }
        }
    }
}
