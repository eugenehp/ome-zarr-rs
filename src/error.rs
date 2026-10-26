//! Error type shared by the whole crate.
//!
//! The variants mirror the Python exception classes raised by ome-zarr-py so that
//! callers can distinguish e.g. a `ValueError` from a `TypeError`.

use thiserror::Error;

/// Result alias used throughout the crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Errors raised by ome-zarr.
#[derive(Debug, Error)]
pub enum Error {
    /// Invalid value (Python `ValueError`).
    #[error("{0}")]
    Value(String),
    /// Invalid type (Python `TypeError`).
    #[error("{0}")]
    Type(String),
    /// Missing key (Python `KeyError`).
    #[error("{0}")]
    Key(String),
    /// Removed / deprecated functionality (Python `DeprecationWarning` raised as error).
    #[error("{0}")]
    Deprecation(String),
    /// Assertion failures (Python `AssertionError`), e.g. "not a zarr".
    #[error("{0}")]
    Assertion(String),
    /// Model validation failed (pydantic `ValidationError`).
    #[error("{0}")]
    Validation(String),
    /// Python `IndexError`.
    #[error("{0}")]
    Index(String),
    /// Python `RuntimeError` (e.g. scipy's "data type not supported").
    #[error("{0}")]
    Runtime(String),
    /// No array at a path (zarr-python's `ArrayNotFoundError`).
    #[error("{0}")]
    ArrayNotFound(String),
    /// No path between two nodes of a transform graph (networkx `NetworkXNoPath`).
    #[error("{0}")]
    NoPath(String),
    /// A transformation could not be converted into functional form.
    #[error("{msg}: {parsed}")]
    UnsupportedTransformation { msg: String, parsed: String },
    /// Operating system / IO errors (Python `OSError`).
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// JSON (de)serialisation errors.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Errors from the zarr storage layer.
    #[error("storage error: {0}")]
    Storage(String),
    /// Errors from the zarr array layer.
    #[error("zarr error: {0}")]
    Zarr(String),
    /// Feature not implemented.
    #[error("not implemented: {0}")]
    NotImplemented(String),
}

impl Error {
    pub fn value(msg: impl Into<String>) -> Self {
        Error::Value(msg.into())
    }
    pub fn type_err(msg: impl Into<String>) -> Self {
        Error::Type(msg.into())
    }
    pub fn key(msg: impl Into<String>) -> Self {
        Error::Key(msg.into())
    }
    pub fn unsupported(msg: impl Into<String>, parsed: impl std::fmt::Debug) -> Self {
        Error::UnsupportedTransformation {
            msg: msg.into(),
            parsed: format!("{parsed:?}"),
        }
    }
    /// The Python exception class this error corresponds to.
    pub fn py_type(&self) -> &'static str {
        match self {
            Error::Value(_) | Error::Json(_) | Error::Storage(_) | Error::Zarr(_) => "ValueError",
            Error::Type(_) => "TypeError",
            Error::Key(_) => "KeyError",
            Error::Deprecation(_) => "DeprecationWarning",
            Error::Assertion(_) => "AssertionError",
            Error::Validation(_) => "ValidationError",
            Error::NoPath(_) => "NetworkXNoPath",
            Error::ArrayNotFound(_) => "ArrayNotFoundError",
            Error::Runtime(_) => "RuntimeError",
            Error::Index(_) => "IndexError",
            Error::UnsupportedTransformation { .. } => "UnsupportedTransformation",
            Error::Io(_) => "OSError",
            Error::NotImplemented(_) => "NotImplementedError",
        }
    }

    /// `str(exception)` as Python prints it (`KeyError` quotes its message).
    pub fn py_str(&self) -> String {
        match self {
            Error::Key(m) => crate::pyjson::repr_str(m),
            Error::Io(e) => e.to_string(),
            Error::NotImplemented(m) => m.clone(),
            other => other.to_string(),
        }
    }

    /// True if this is a `ValueError`.
    pub fn is_value(&self) -> bool {
        matches!(self, Error::Value(_))
    }
    /// True if this is a `TypeError`.
    pub fn is_type(&self) -> bool {
        matches!(self, Error::Type(_))
    }
    /// True if this is a `KeyError`.
    pub fn is_key(&self) -> bool {
        matches!(self, Error::Key(_))
    }
}

impl From<zarrs_storage::StorageError> for Error {
    fn from(e: zarrs_storage::StorageError) -> Self {
        Error::Storage(e.to_string())
    }
}

impl From<zarrs::array::ArrayError> for Error {
    fn from(e: zarrs::array::ArrayError) -> Self {
        Error::Zarr(e.to_string())
    }
}

impl From<zarrs::array::ArrayCreateError> for Error {
    fn from(e: zarrs::array::ArrayCreateError) -> Self {
        Error::Zarr(e.to_string())
    }
}
