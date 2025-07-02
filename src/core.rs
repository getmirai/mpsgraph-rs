//! Core utility functions and types for mpsgraph

use objc2::rc::Retained;
use objc2_foundation::{NSArray, NSNumber};

/// Trait for types that provide a custom default implementation
/// when working with NSObject-based types
pub trait ClassType {
    /// Get the class for the given type
    fn class<T>() -> *const objc2::runtime::AnyClass;
}

/// Create an NSArray from a slice of i32 values
pub fn create_ns_array_from_i32_slice(values: &[i32]) -> Retained<NSArray<NSNumber>> {
    // Create NSNumber objects
    let numbers: Vec<Retained<NSNumber>> = values
        .iter()
        .map(|&value| NSNumber::new_i32(value))
        .collect();

    // Convert to slice of references
    let number_refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();

    // Create NSArray from the NSNumber objects
    NSArray::from_slice(&number_refs)
}

/// Create an NSArray from a slice of i64 values
pub fn create_ns_array_from_i64_slice(values: &[i64]) -> Retained<NSArray<NSNumber>> {
    // Create NSNumber objects
    let numbers: Vec<Retained<NSNumber>> = values
        .iter()
        .map(|&value| NSNumber::new_i64(value))
        .collect();

    // Convert to slice of references
    let number_refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();

    // Create NSArray from the NSNumber objects
    NSArray::from_slice(&number_refs)
}

/// Create an NSArray from a slice of u64 values
pub fn create_ns_array_from_u64_slice(values: &[u64]) -> Retained<NSArray<NSNumber>> {
    // Create NSNumber objects
    let numbers: Vec<Retained<NSNumber>> = values
        .iter()
        .map(|&value| NSNumber::new_u64(value))
        .collect();

    // Convert to slice of references
    let number_refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();

    // Create NSArray from the NSNumber objects
    NSArray::from_slice(&number_refs)
}

/// Create an NSArray from a slice of pointers
pub fn create_ns_array_from_slice<T: 'static + objc2::Message>(
    objects: &[*const T],
) -> Retained<NSArray<T>> {
    unsafe {
        // Convert raw pointers to references
        let refs: Vec<&T> = objects.iter().map(|&p| &*p).collect();

        // Create array from references
        NSArray::from_slice(&refs)
    }
}

/// Re-export DataType from tensor module to avoid circular dependencies
pub use crate::tensor::DataType;
