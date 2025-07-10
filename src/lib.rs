//! Modern Rust bindings for Apple's Metal Performance Shaders Graph framework
//!
//! This crate provides Rust bindings for Apple's MPS Graph framework
//! using modern `extern_class!` macros and automatic memory management.

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
compile_error!("MetalPerformanceShadersGraph is only available on macOS and iOS");

// Re-export objc2 crates for use by end users
pub use objc2;
pub use objc2_foundation;

// Prelude module for commonly used types and traits
pub mod prelude {
    pub use crate::core::{ClassType, DataType};
    pub use crate::data_types::ShapedType;
    pub use crate::device::Device;
    pub use crate::graph::{Graph, TensorDataScalar};
    pub use crate::operation::Operation;
    pub use crate::shape::Shape;
    pub use crate::tensor::Tensor;
    pub use crate::tensor_data::TensorData;

    // Operation traits now inherent on `Graph`; no trait re-exports needed.
    // control-flow helpers are inherent on Graph.

    // Re-export objc2 types that users would need
    pub use objc2::rc::Retained;
    pub use objc2_foundation::NSObject;

    pub use crate::matrix_multiplication_ops::ScaledDotProductAttentionExt;
}

// Core types
pub mod command_buffer;
pub mod core;
pub mod data_types;
pub mod device;
pub mod executable;
pub mod graph;
pub mod operation;
pub mod shape;
pub mod tensor;
pub mod tensor_data;

// Operation types
pub mod activation_ops;
pub mod arithmetic_ops;
pub mod call_ops;
pub mod convolution_ops;
pub mod convolution_transpose_ops;
pub mod cumulative_ops;
pub mod depthwise_convolution_ops;
pub mod fourier_transform_ops;
pub mod gather_ops;
pub mod gradient_ops;
pub mod im2col_ops;
pub mod loss_ops;
pub mod matrix_multiplication_ops;
pub mod memory_ops;
pub mod non_maximum_suppression_ops;
pub mod non_zero_ops;
pub mod normalization_ops;
pub mod one_hot_ops;
pub mod optimizer_ops;
pub mod pooling_ops;
pub mod quantization_ops;
pub mod random_ops;
pub mod reduction_ops;
pub mod resize_ops;
pub mod rnn_ops;
pub mod sample_grid_ops;
pub mod scatter_nd_ops;
pub mod sort_ops;
pub mod sparse_ops;
pub mod stencil_ops;
pub mod tensor_shape_ops;
pub mod top_k_ops;
pub mod utils;
// Additional modules will be added as they are implemented

// Re-export core types
pub use core::{ClassType, DataType};
pub use device::Device;
pub use executable::{
    CompilationDescriptor, DeploymentPlatform, Executable, ExecutableExecutionDescriptor,
    ExecutionDescriptor, ExecutionResult, ExecutionStage, Optimization, OptimizationProfile,
    SerializationDescriptor,
};
pub use graph::{Graph, TensorDataScalar};
pub use operation::Operation;
pub use shape::Shape;
pub use tensor::Tensor;
pub use tensor_data::TensorData;

// Arithmetic and cumulative helpers are inherent on `Graph` now – no trait re-export.
pub use command_buffer::{CommandBuffer, CommandBufferStatus};
pub use data_types::{DataTypeAttributeValue, ExecutionMode, ShapeDescriptor, ShapedType, Type};
pub use depthwise_convolution_ops::{
    DepthwiseConvolution2DOpDescriptor, DepthwiseConvolution3DOpDescriptor,
};
pub use fourier_transform_ops::{FFTDescriptor, FFTScalingMode};
pub use im2col_ops::ImToColOpDescriptor;
pub use loss_ops::LossReductionType;

pub use non_maximum_suppression_ops::NonMaximumSuppressionCoordinateMode;

pub use optimizer_ops::VariableOp;
pub use pooling_ops::{PoolingReturnIndicesMode, TensorNamedDataLayout};

pub use random_ops::{RandomDistribution, RandomNormalSamplingMethod, RandomOpDescriptor};

pub use rnn_ops::{GRUDescriptor, LSTMDescriptor, RNNActivation, SingleGateRNNDescriptor};

pub use sparse_ops::{CreateSparseOpDescriptor, SparseStorageType};
pub use stencil_ops::{BoundaryMode, ReductionMode, StencilOpDescriptor};

pub use utils::buffer;
pub use utils::tensor as tensor_utils;

#[cfg(test)]
mod tests {
    mod data_types_tests;
    mod executable_tests;
}

pub mod matrix_inverse_ops;

pub mod scalar_or_tensor;

pub use scalar_or_tensor::ScalarOrTensor;

pub use fourier_transform_ops::FFTAxesTensorExt;
pub use matrix_multiplication_ops::ScaledDotProductAttentionExt;
