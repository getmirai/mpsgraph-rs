#[macro_use]
extern crate objc2;

/// Rust bindings for Apple's Metal Performance Shaders Graph API.
///
/// This library provides idiomatic Rust bindings for the Metal Performance Shaders Graph API, 
/// which is part of Apple's Metal Performance Shaders framework. The API follows Rust naming
/// conventions while maintaining compatibility with the underlying Objective-C interfaces.

// Tests module (only included when running tests)
#[cfg(test)]
mod tests;

// Core modules
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

// Operation-specific modules
pub mod activation_ops;
pub mod arithmetic_ops;
pub mod convolution_ops;
pub mod convolution_transpose_ops;
pub mod depthwise_convolution_ops;
pub mod gradient_ops;
pub mod loss_ops;
pub mod matrix_inverse_ops;
pub mod matrix_ops;
pub mod normalization_ops;
pub mod optimizer_ops;
pub mod pooling_ops;
pub mod random_ops;
pub mod reduction_ops;
pub mod rnn_ops;
pub mod tensor_shape_ops;
// Temporarily disable control_flow_ops due to Block_copy issues
// pub mod control_flow_ops;
pub mod call_ops;
pub mod cumulative_ops;
pub mod fourier_transform_ops;
pub mod gather_ops;
pub mod im2col_ops;
pub mod linear_algebra_ops;
pub mod memory_ops;
pub mod non_maximum_suppression_ops;
pub mod non_zero_ops;
pub mod one_hot_ops;
pub mod quantization_ops;
pub mod resize_ops;
pub mod sample_grid_ops;
pub mod scatter_nd_ops;
pub mod sort_ops;
pub mod sparse_ops;
pub mod stencil_ops;
pub mod top_k_ops;

// Re-export most commonly used types
pub use command_buffer::MPSCommandBuffer;
pub use convolution_transpose_ops::{
    MPSGraphConvolution2DOpDescriptor, PaddingStyle, TensorNamedDataLayout,
};
pub use core::{
    MPSDataType, MPSGraphExecutionStage, MPSGraphOptimization, MPSGraphOptimizationProfile,
    MPSGraphOptions,
};
pub use data_types::{MPSGraphShapedType, MPSGraphType};
pub use depthwise_convolution_ops::{
    MPSGraphDepthwiseConvolution2DOpDescriptor, MPSGraphDepthwiseConvolution3DOpDescriptor,
};
pub use device::Device;
pub use executable::{
    CompilationDescriptor, MPSGraphExecutable, ExecutionDescriptor,
};
pub use graph::Graph;
pub use graph::TensorDataScalar;
pub use loss_ops::MPSGraphLossReductionType;
pub use operation::MPSGraphOperation;
pub use random_ops::{
    MPSGraphRandomDistribution, MPSGraphRandomNormalSamplingMethod, MPSGraphRandomOpDescriptor,
};
pub use rnn_ops::{
    MPSGraphGRUDescriptor, MPSGraphLSTMDescriptor, MPSGraphRNNActivation,
    MPSGraphSingleGateRNNDescriptor,
};
pub use shape::Shape;
pub use tensor::Tensor;
pub use tensor_data::TensorData;
// Note: gather_ops doesn't have any standalone structs or enums to re-export
pub use fourier_transform_ops::{MPSGraphFFTDescriptor, MPSGraphFFTScalingMode};
pub use im2col_ops::MPSGraphImToColOpDescriptor;
pub use non_maximum_suppression_ops::MPSGraphNonMaximumSuppressionCoordinateMode;
pub use resize_ops::{MPSGraphResizeMode, MPSGraphResizeNearestRoundingMode};
pub use sample_grid_ops::MPSGraphPaddingMode;
pub use scatter_nd_ops::MPSGraphScatterMode;
pub use sparse_ops::{MPSGraphCreateSparseOpDescriptor, MPSGraphSparseStorageType};
pub use stencil_ops::{MPSGraphReductionMode, MPSGraphStencilOpDescriptor};

/// Convenience prelude module with most commonly used items
pub mod prelude {
    pub use crate::command_buffer::MPSCommandBuffer;
    pub use crate::convolution_transpose_ops::{
        MPSGraphConvolution2DOpDescriptor, PaddingStyle, TensorNamedDataLayout,
    };
    pub use crate::core::{
        MPSDataType, MPSGraphExecutionStage, MPSGraphOptimization, MPSGraphOptimizationProfile,
        MPSGraphOptions,
    };
    pub use crate::data_types::{MPSGraphShapedType, MPSGraphType};
    pub use crate::depthwise_convolution_ops::{
        MPSGraphDepthwiseConvolution2DOpDescriptor, MPSGraphDepthwiseConvolution3DOpDescriptor,
    };
    pub use crate::device::Device;
    pub use crate::executable::{
        CompilationDescriptor, MPSGraphExecutable, ExecutionDescriptor,
    };
    pub use crate::graph::Graph;
    pub use crate::graph::TensorDataScalar;
    pub use crate::loss_ops::MPSGraphLossReductionType;
    pub use crate::operation::MPSGraphOperation;
    pub use crate::random_ops::{
        MPSGraphRandomDistribution, MPSGraphRandomNormalSamplingMethod, MPSGraphRandomOpDescriptor,
    };
    pub use crate::rnn_ops::{
        MPSGraphGRUDescriptor, MPSGraphLSTMDescriptor, MPSGraphRNNActivation,
        MPSGraphSingleGateRNNDescriptor,
    };
    pub use crate::shape::Shape;
    pub use crate::tensor::Tensor;
    pub use crate::tensor_data::TensorData;
    // No separate types to import from gather_ops
    pub use crate::fourier_transform_ops::{MPSGraphFFTDescriptor, MPSGraphFFTScalingMode};
    pub use crate::im2col_ops::MPSGraphImToColOpDescriptor;
    pub use crate::non_maximum_suppression_ops::MPSGraphNonMaximumSuppressionCoordinateMode;
    pub use crate::resize_ops::{MPSGraphResizeMode, MPSGraphResizeNearestRoundingMode};
    pub use crate::sample_grid_ops::MPSGraphPaddingMode;
    pub use crate::scatter_nd_ops::MPSGraphScatterMode;
    pub use crate::sparse_ops::{MPSGraphCreateSparseOpDescriptor, MPSGraphSparseStorageType};
    pub use crate::stencil_ops::{MPSGraphReductionMode, MPSGraphStencilOpDescriptor};
}
