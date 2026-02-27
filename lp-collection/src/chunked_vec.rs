//! Chunked vector for allocation-sensitive contexts (e.g. embedded).
//!
//! Allocates in small chunks instead of one large contiguous block to reduce
//! OOM risk from heap fragmentation when compiling on constrained heaps.
//! Shared by regalloc2 (fastalloc) and cranelift-codegen (VCode).

use alloc::vec;
use alloc::vec::Vec;
use core::ops::{Index, IndexMut};
use core::ptr;

/// Chunk size in elements. Keeps each allocation small (~1–2KB typical).
const CHUNK_SIZE: usize = 16;

/// A vector backed by multiple smaller allocations.
///
/// Uses `ceil(len/CHUNK_SIZE)` chunks instead of one large Vec to reduce
/// peak allocation size and improve success on fragmented heaps.
/// Provides O(1) index access; indexing assumes uniform layout (chunk i
/// covers indices i*CHUNK_SIZE..(i+1)*CHUNK_SIZE or to end).
#[derive(Clone, Default, Debug)]
pub struct ChunkedVec<T> {
    chunks: Vec<Vec<T>>,
    len: usize,
}

impl<T> ChunkedVec<T> {
    /// Create an empty ChunkedVec.
    pub fn new() -> Self {
        Self {
            chunks: Vec::new(),
            len: 0,
        }
    }

    /// Create with given length, filling with `default`.
    /// Used by fastalloc for pre-allocated VReg state.
    pub fn with_capacity_and_default(len: usize, default: T) -> Self
    where
        T: Clone,
    {
        let mut chunks = Vec::new();
        let mut remaining = len;
        while remaining > 0 {
            let chunk_len = remaining.min(CHUNK_SIZE);
            chunks.push(vec![default.clone(); chunk_len]);
            remaining -= chunk_len;
        }
        Self { chunks, len }
    }

    /// Create a ChunkedVec with capacity for at least `cap` elements.
    pub fn with_capacity(cap: usize) -> Self {
        let n_chunks = (cap + CHUNK_SIZE - 1) / CHUNK_SIZE;
        let mut chunks = Vec::with_capacity(n_chunks.max(1));
        if cap > 0 {
            let first_cap = cap.min(CHUNK_SIZE);
            chunks.push(Vec::with_capacity(first_cap));
        }
        Self { chunks, len: 0 }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn chunk_and_offset(&self, i: usize) -> (usize, usize) {
        (i / CHUNK_SIZE, i % CHUNK_SIZE)
    }

    pub fn push(&mut self, value: T) {
        let offset = self.len % CHUNK_SIZE;
        if offset == 0 && self.len > 0 {
            self.chunks.push(Vec::with_capacity(CHUNK_SIZE));
        }
        if offset == 0 && self.len == 0 && self.chunks.is_empty() {
            self.chunks.push(Vec::with_capacity(CHUNK_SIZE));
        }
        let last = self.chunks.len() - 1;
        self.chunks[last].push(value);
        self.len += 1;
    }

    /// Resize to `new_len`, filling new slots with `value` if growing.
    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        if new_len <= self.len {
            self.len = new_len;
            let keep_chunks = (new_len + CHUNK_SIZE - 1) / CHUNK_SIZE;
            if keep_chunks == 0 {
                self.chunks.clear();
                return;
            }
            self.chunks.truncate(keep_chunks);
            let last_offset = new_len % CHUNK_SIZE;
            let last_len = if last_offset == 0 && new_len > 0 {
                CHUNK_SIZE
            } else {
                last_offset
            };
            self.chunks[keep_chunks - 1].truncate(last_len);
            return;
        }
        while self.len < new_len {
            self.push(value.clone());
        }
    }

    /// Removes and returns the element at `index`, replacing it with the last element.
    /// Returns `None` if index is out of bounds.
    pub fn swap_remove(&mut self, index: usize) -> Option<T> {
        if index >= self.len {
            return None;
        }
        let last = self.len - 1;
        if index != last {
            let a = self.get_mut(index).expect("valid index") as *mut T;
            let b = self.get_mut(last).expect("valid index") as *mut T;
            unsafe { ptr::swap(a, b) };
        }
        let (ci, o) = self.chunk_and_offset(last);
        let chunk = self.chunks.get_mut(ci)?;
        let val = chunk.swap_remove(o);
        self.len -= 1;
        if chunk.is_empty() {
            self.chunks.pop();
        }
        Some(val)
    }

    pub fn get(&self, i: usize) -> Option<&T> {
        if i >= self.len {
            return None;
        }
        let (ci, o) = self.chunk_and_offset(i);
        self.chunks.get(ci).and_then(|c| c.get(o))
    }

    pub fn get_mut(&mut self, i: usize) -> Option<&mut T> {
        if i >= self.len {
            return None;
        }
        let (ci, o) = self.chunk_and_offset(i);
        self.chunks.get_mut(ci).and_then(|c| c.get_mut(o))
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.chunks.iter().flat_map(|c| c.iter())
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.chunks.iter_mut().flat_map(|c| c.iter_mut())
    }

    /// Reverse the sequence in place. O(n) time, no allocation.
    /// Uses element swap to preserve the chunk layout required by get/index.
    pub fn reverse(&mut self) {
        let len = self.len;
        for i in 0..len / 2 {
            let j = len - 1 - i;
            let a = self.get_mut(i).expect("valid index") as *mut T;
            let b = self.get_mut(j).expect("valid index") as *mut T;
            unsafe { ptr::swap(a, b) };
        }
    }
}

impl<T> Index<usize> for ChunkedVec<T> {
    type Output = T;

    #[inline]
    fn index(&self, i: usize) -> &Self::Output {
        self.get(i).expect("ChunkedVec index out of bounds")
    }
}

impl<T> IndexMut<usize> for ChunkedVec<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        self.get_mut(i).expect("ChunkedVec index out of bounds")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunked_vec_new() -> ChunkedVec<i32> {
        ChunkedVec::new()
    }

    #[test]
    fn push_and_len() {
        let mut v = chunked_vec_new();
        assert_eq!(v.len(), 0);
        v.push(1);
        v.push(2);
        v.push(3);
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], 1);
        assert_eq!(v[1], 2);
        assert_eq!(v[2], 3);
    }

    #[test]
    fn with_capacity_and_default() {
        let v = ChunkedVec::with_capacity_and_default(130, -1);
        assert_eq!(v.len(), 130);
        for i in 0..130 {
            assert_eq!(v[i], -1);
        }
    }

    #[test]
    fn get_and_get_mut() {
        let mut v = chunked_vec_new();
        v.push(10);
        v.push(20);
        assert_eq!(v.get(0), Some(&10));
        assert_eq!(v.get(1), Some(&20));
        assert_eq!(v.get(2), None);
        *v.get_mut(1).unwrap() = 99;
        assert_eq!(v[1], 99);
    }

    #[test]
    fn resize_grow() {
        let mut v = chunked_vec_new();
        v.resize(5, -1);
        assert_eq!(v.len(), 5);
        for i in 0..5 {
            assert_eq!(v[i], -1);
        }
    }

    #[test]
    fn resize_shrink() {
        let mut v = chunked_vec_new();
        v.resize(10, 0);
        v.resize(3, 0);
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn iter_yields_sequence() {
        let mut v = chunked_vec_new();
        for i in 0..10 {
            v.push(i as i32);
        }
        let collected: Vec<i32> = v.iter().copied().collect();
        assert_eq!(collected, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn reverse_preserves_layout_and_order() {
        let mut v = chunked_vec_new();
        for i in 0..130 {
            v.push(i as i32);
        }
        v.reverse();
        for i in 0..130 {
            assert_eq!(v[i], (129 - i) as i32, "index {i} after reverse");
        }
    }

    #[test]
    fn reverse_small_partial_chunk() {
        let mut v = chunked_vec_new();
        for i in 0..70 {
            v.push(i as i32);
        }
        v.reverse();
        for i in 0..70 {
            assert_eq!(v[i], (69 - i) as i32);
        }
    }

    #[test]
    fn with_capacity_then_many_pushes() {
        let mut v = ChunkedVec::with_capacity(600);
        for i in 0..600 {
            v.push(i as i32);
        }
        assert_eq!(v.len(), 600);
        assert_eq!(v[0], 0);
        assert_eq!(v[299], 299);
        assert_eq!(v[599], 599);
    }

    #[test]
    fn swap_remove() {
        let mut v = chunked_vec_new();
        for i in 0..5 {
            v.push(i as i32);
        }
        let removed = v.swap_remove(1);
        assert_eq!(removed, Some(1));
        assert_eq!(v.len(), 4);
        assert_eq!(v[0], 0);
        assert_eq!(v[1], 4);
        assert_eq!(v[2], 2);
        assert_eq!(v[3], 3);
    }
}
