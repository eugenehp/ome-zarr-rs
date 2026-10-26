//! Data parallelism: rayon on native targets, the same calls running
//! sequentially on wasm (where threads are not generally available).

#[cfg(not(target_arch = "wasm32"))]
pub use rayon::prelude::*;

#[cfg(target_arch = "wasm32")]
mod seq {
    /// `into_par_iter()` as a plain iterator.
    pub trait IntoParallelIterator: IntoIterator + Sized {
        fn into_par_iter(self) -> Self::IntoIter {
            self.into_iter()
        }
    }
    impl<T: IntoIterator> IntoParallelIterator for T {}

    /// `par_iter()` as a plain slice iterator.
    pub trait ParallelSlice<T> {
        fn par_iter(&self) -> std::slice::Iter<'_, T>;
    }
    impl<T> ParallelSlice<T> for [T] {
        fn par_iter(&self) -> std::slice::Iter<'_, T> {
            self.iter()
        }
    }

    /// `par_chunks_mut()` as plain mutable chunks.
    pub trait ParallelSliceMut<T> {
        fn par_chunks_mut(&mut self, n: usize) -> std::slice::ChunksMut<'_, T>;
    }
    impl<T> ParallelSliceMut<T> for [T] {
        fn par_chunks_mut(&mut self, n: usize) -> std::slice::ChunksMut<'_, T> {
            self.chunks_mut(n)
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub use seq::*;

/// `Send + Sync` on native targets; no bound on wasm (where zarrs' storage
/// traits are not `Send`/`Sync` either).
#[cfg(not(target_arch = "wasm32"))]
pub trait MaybeSendSync: Send + Sync {}
#[cfg(not(target_arch = "wasm32"))]
impl<T: Send + Sync + ?Sized> MaybeSendSync for T {}

#[cfg(target_arch = "wasm32")]
pub trait MaybeSendSync {}
#[cfg(target_arch = "wasm32")]
impl<T: ?Sized> MaybeSendSync for T {}
