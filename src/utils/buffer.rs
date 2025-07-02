use std::ffi::c_void;
use std::mem;
use std::slice;

/// Get a slice from a raw buffer pointer and element count
pub unsafe fn slice_from_raw_parts<'a, T>(data: *const c_void, count: usize) -> &'a [T] {
    slice::from_raw_parts(data as *const T, count)
}

/// Get a mutable slice from a raw buffer pointer and element count
pub unsafe fn slice_from_raw_parts_mut<'a, T>(data: *mut c_void, count: usize) -> &'a mut [T] {
    slice::from_raw_parts_mut(data as *mut T, count)
}

/// Calculate the size in bytes of a slice
pub fn byte_size_of_slice<T>(slice: &[T]) -> usize {
    mem::size_of::<T>() * slice.len()
}