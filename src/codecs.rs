//! numcodecs compressors that zarrs does not provide, registered as zarrs
//! codec plugins so arrays written by zarr-python with them can be read and
//! written: `lz4` (numcodecs' block format: a little-endian `u32` size, then
//! an LZ4 block) and `lzma` (xz or "alone" streams). Pure Rust, so they work
//! on wasm too. Zarr v2 ids `lz4` / `lzma`; zarr v3 names `numcodecs.lz4` /
//! `numcodecs.lzma`.

use std::borrow::Cow;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use zarrs_codec::{
    ArrayBytesRaw, BytesRepresentation, BytesToBytesCodecTraits, Codec, CodecError,
    CodecMetadataOptions, CodecOptions, CodecPluginV2, CodecPluginV3, CodecTraits, CodecTraitsV2,
    CodecTraitsV3, PartialDecoderCapability, PartialEncoderCapability, RecommendedConcurrency,
};
use zarrs_metadata::Configuration;
use zarrs_metadata::v2::MetadataV2;
use zarrs_metadata::v3::MetadataV3;
use zarrs_plugin::{PluginCreateError, ZarrVersion};

fn codec_err(e: impl std::fmt::Display) -> CodecError {
    CodecError::Other(e.to_string())
}

fn configuration<T: Serialize>(c: &T) -> Option<Configuration> {
    match serde_json::to_value(c) {
        Ok(serde_json::Value::Object(m)) => Some(Configuration::from(m)),
        _ => None,
    }
}

// ---------------------------------------------------------------- lz4

/// numcodecs `LZ4(acceleration=1)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lz4Configuration {
    #[serde(default = "one")]
    pub acceleration: i64,
}

fn one() -> i64 {
    1
}

/// The numcodecs `lz4` codec.
#[derive(Debug, Clone)]
pub struct Lz4Codec {
    config: Lz4Configuration,
}

zarrs_plugin::impl_extension_aliases!(Lz4Codec,
    v3: "numcodecs.lz4", [],
    v2: "lz4", []
);

inventory::submit! { CodecPluginV3::new::<Lz4Codec>() }
inventory::submit! { CodecPluginV2::new::<Lz4Codec>() }

impl CodecTraitsV3 for Lz4Codec {
    fn create(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
        let config: Lz4Configuration = metadata.to_typed_configuration()?;
        Ok(Codec::BytesToBytes(Arc::new(Lz4Codec { config })))
    }
}

impl CodecTraitsV2 for Lz4Codec {
    fn create(metadata: &MetadataV2) -> Result<Codec, PluginCreateError> {
        let config: Lz4Configuration = metadata.to_typed_configuration()?;
        Ok(Codec::BytesToBytes(Arc::new(Lz4Codec { config })))
    }
}

impl CodecTraits for Lz4Codec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        configuration(&self.config)
    }
    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: false,
            partial_decode: false,
        }
    }
    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: false,
        }
    }
}

impl BytesToBytesCodecTraits for Lz4Codec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn BytesToBytesCodecTraits> {
        self as Arc<dyn BytesToBytesCodecTraits>
    }
    fn recommended_concurrency(
        &self,
        _r: &BytesRepresentation,
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }
    fn encode<'a>(
        &self,
        decoded: ArrayBytesRaw<'a>,
        _o: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        Ok(Cow::Owned(lz4_flex::block::compress_prepend_size(&decoded)))
    }
    fn decode<'a>(
        &self,
        encoded: ArrayBytesRaw<'a>,
        _r: &BytesRepresentation,
        _o: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        lz4_flex::block::decompress_size_prepended(&encoded)
            .map(Cow::Owned)
            .map_err(codec_err)
    }
    fn encoded_representation(&self, r: &BytesRepresentation) -> BytesRepresentation {
        r.size().map_or(BytesRepresentation::UnboundedSize, |n| {
            BytesRepresentation::BoundedSize(
                lz4_flex::block::get_maximum_output_size(n as usize) as u64 + 4,
            )
        })
    }
}

// ---------------------------------------------------------------- lzma

/// numcodecs `LZMA(format=1, check=-1, preset=None, filters=None)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LzmaConfiguration {
    /// 1 = xz, 2 = "alone" (.lzma), 3 = raw (unsupported).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub check: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filters: Option<serde_json::Value>,
}

/// The numcodecs `lzma` codec.
#[derive(Debug, Clone)]
pub struct LzmaCodec {
    config: LzmaConfiguration,
}

zarrs_plugin::impl_extension_aliases!(LzmaCodec,
    v3: "numcodecs.lzma", [],
    v2: "lzma", []
);

inventory::submit! { CodecPluginV3::new::<LzmaCodec>() }
inventory::submit! { CodecPluginV2::new::<LzmaCodec>() }

fn lzma_codec(config: LzmaConfiguration) -> Result<Codec, PluginCreateError> {
    match config.format.unwrap_or(1) {
        1 | 2 => {}
        f => {
            return Err(PluginCreateError::Other(format!(
                "lzma format {f} is not supported"
            )));
        }
    }
    if config.filters.as_ref().is_some_and(|f| !f.is_null()) {
        return Err(PluginCreateError::Other(
            "lzma filters are not supported".into(),
        ));
    }
    Ok(Codec::BytesToBytes(Arc::new(LzmaCodec { config })))
}

impl CodecTraitsV3 for LzmaCodec {
    fn create(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
        lzma_codec(metadata.to_typed_configuration()?)
    }
}

impl CodecTraitsV2 for LzmaCodec {
    fn create(metadata: &MetadataV2) -> Result<Codec, PluginCreateError> {
        lzma_codec(metadata.to_typed_configuration()?)
    }
}

impl CodecTraits for LzmaCodec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        configuration(&self.config)
    }
    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: false,
            partial_decode: false,
        }
    }
    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: false,
        }
    }
}

impl BytesToBytesCodecTraits for LzmaCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn BytesToBytesCodecTraits> {
        self as Arc<dyn BytesToBytesCodecTraits>
    }
    fn recommended_concurrency(
        &self,
        _r: &BytesRepresentation,
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }
    fn encode<'a>(
        &self,
        decoded: ArrayBytesRaw<'a>,
        _o: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        let mut input = std::io::Cursor::new(&decoded[..]);
        let mut out = vec![];
        match self.config.format.unwrap_or(1) {
            2 => lzma_rs::lzma_compress(&mut input, &mut out),
            _ => lzma_rs::xz_compress(&mut input, &mut out),
        }
        .map_err(codec_err)?;
        Ok(Cow::Owned(out))
    }
    fn decode<'a>(
        &self,
        encoded: ArrayBytesRaw<'a>,
        _r: &BytesRepresentation,
        _o: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        let mut input = std::io::BufReader::new(std::io::Cursor::new(&encoded[..]));
        let mut out = vec![];
        match self.config.format.unwrap_or(1) {
            2 => lzma_rs::lzma_decompress(&mut input, &mut out),
            _ => lzma_rs::xz_decompress(&mut input, &mut out),
        }
        .map_err(codec_err)?;
        Ok(Cow::Owned(out))
    }
    fn encoded_representation(&self, _r: &BytesRepresentation) -> BytesRepresentation {
        BytesRepresentation::UnboundedSize
    }
}
