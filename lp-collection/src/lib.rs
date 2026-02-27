//! Embedded/low-memory friendly collections.
//!
//! Chunked data structures that allocate in small chunks to reduce OOM risk
//! from heap fragmentation on constrained heaps (e.g. embedded, regalloc).

#![no_std]

extern crate alloc;

pub mod chunked_hashmap;
pub mod chunked_vec;

pub use chunked_hashmap::ChunkedHashMap;
pub use chunked_vec::ChunkedVec;
