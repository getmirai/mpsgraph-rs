//! Metal Performance Shaders Graph Tools - High-level utilities for the mpsgraph crate
//!
//! This crate provides high-level utilities and ergonomic APIs for working with
//! Apple's Metal Performance Shaders Graph framework through the mpsgraph crate.
//!
//! # Features
//!
//! - **Complete Re-export**: All functionality from the vanilla mpsgraph crate is re-exported
//! - **Enhanced Tensor Type**: A wrapper around mpsgraph's Tensor with operator overloading
//! - **Tensor Operations API**: Ergonomic, functional-style tensor operations
//! - **Utility Functions**: Convenience methods for common tensor operations
//! - **Tensor Creation Helpers**: Easy creation of tensors with different initialization patterns

// Re-export all of mpsgraph
pub use mpsgraph::*;

// Enhanced tensor type with operator overloading
pub mod tensor;

// Tensor operations module (additional functionality beyond vanilla mpsgraph)
pub mod tensor_ops;

/// Convenience prelude module with most commonly used items
pub mod prelude {
    // Re-export the entire mpsgraph prelude
    pub use mpsgraph::prelude::*;

    // Enhanced tensor type with operator overloading
    pub use crate::tensor::Tensor;

    // Tensor operations (our additional functionality)
    pub use crate::tensor_ops;
    pub use crate::tensor_ops::{
        abs, clip, exp, gelu, log, pow, relu, sigmoid, silu, sqrt, square, tanh, GraphExtensions,
        TensorOps, TensorScalarOps,
    };
}

#[cfg(test)]
mod tests;
