//! Arrays: dtypes, in-memory n-d arrays and lazy chunked arrays.

pub mod chunks;
pub mod dtype;
pub mod lazy;
pub mod nd;

pub use chunks::{Chunks, retuple};
pub use dtype::{DType, Element};
pub use lazy::{BlockFn, ChunkedSource, LazyArray, Region};
pub use nd::{FromNdArray, IntoNdArray, NdArray};
