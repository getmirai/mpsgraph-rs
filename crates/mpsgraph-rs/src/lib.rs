//! Modern Rust bindings for Apple's Metal Performance Shaders Graph framework
//! 
//! This crate provides Rust bindings for Apple's MPS Graph framework
//! using modern `extern_class!` macros and automatic memory management.

#[cfg(not(target_os = "macos"))]
compile_error!("MetalPerformanceShadersGraph is only available on macOS");

// Re-export objc2 crates for use by end users
pub use objc2;
pub use objc2_foundation;

// Prelude module for commonly used types and traits
pub mod prelude {
    pub use crate::core::{ClassType, DataType};
    pub use crate::device::Device;
    pub use crate::graph::{Graph, TensorDataScalar};
    pub use crate::shape::{Shape, ShapeHelper, ShapeExtensions, ShapeHelpers};
    pub use crate::tensor::Tensor;
    pub use crate::tensor_data::TensorData;
    pub use crate::operation::Operation;
    pub use crate::data_types::ShapedType;
    
    // Operation traits
    pub use crate::activation_ops::{GraphActivationOps, GraphActivationOpsExtension};
    pub use crate::arithmetic_ops::{GraphArithmeticOps, GraphArithmeticOpsExtension};
    pub use crate::call_ops::GraphCallOps;
    pub use crate::control_flow_ops::GraphControlFlowOps;
    pub use crate::gradient_ops::{GraphGradientOps, GraphGradientOpsExtension};
    pub use crate::reduction_ops::{GraphReductionOps, GraphReductionOpsExtension};
    pub use crate::tensor_shape_ops::{GraphTensorShapeOps, GraphTensorShapeOpsExtension};
    
    // Re-export objc2 types that users would need
    pub use objc2::rc::Retained;
    pub use objc2_foundation::NSObject;
}

// Core types
pub mod core;
pub mod device;
pub mod graph;
pub mod shape;
pub mod tensor;
pub mod tensor_data;
pub mod operation;
pub mod executable;
pub mod command_buffer;
pub mod data_types;

// Operation types
pub mod activation_ops;
pub mod arithmetic_ops;
pub mod call_ops;
pub mod control_flow_ops;
pub mod convolution_ops;
pub mod convolution_transpose_ops;
pub mod cumulative_ops;
pub mod depthwise_convolution_ops;
pub mod fourier_transform_ops;
pub mod gather_ops;
pub mod gradient_ops;
pub mod im2col_ops;
pub mod linear_algebra_ops;
pub mod loss_ops;
pub mod matrix_inverse_ops;
pub mod matrix_ops;
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
pub use core::{ClassType, create_ns_array_from_i64_slice, DataType};
pub use device::{Device, CustomDefault};
pub use graph::{Graph, TensorDataScalar};
pub use shape::{Shape, ShapeHelper, ShapeExtensions, ShapeHelpers};
pub use tensor::Tensor;
pub use tensor_data::TensorData;
pub use operation::Operation;
pub use executable::{
    CompilationDescriptor, 
    ExecutionDescriptor, 
    ExecutableExecutionDescriptor,
    SerializationDescriptor,
    Executable,
    Optimization, 
    OptimizationProfile,
    ExecutionStage,
    DeploymentPlatform,
    ExecutionResult
};

pub use command_buffer::{CommandBuffer, CommandBufferStatus};
pub use pooling_ops::{
    GraphPoolingOps,
    Pooling2DOpDescriptor,
    Pooling4DOpDescriptor,
    PoolingReturnIndicesMode,
    TensorNamedDataLayout,
    PaddingStyle
};
pub use quantization_ops::{GraphQuantizationOps, GraphQuantizationOpsExtension};
pub use convolution_ops::{
    Convolution2DOpDescriptor,
    Convolution3DOpDescriptor,
    PaddingMode,
    ConvolutionDataLayout,
    WeightsLayout
};
pub use depthwise_convolution_ops::{
    DepthwiseConvolution2DOpDescriptor,
    DepthwiseConvolution3DOpDescriptor
};
pub use stencil_ops::{
    StencilOpDescriptor,
    ReductionMode,
    BoundaryMode
};
pub use sparse_ops::{
    CreateSparseOpDescriptor,
    SparseStorageType
};
pub use im2col_ops::ImToColOpDescriptor;
pub use rnn_ops::{
    SingleGateRNNDescriptor,
    LSTMDescriptor,
    GRUDescriptor,
    RNNActivation
};
pub use random_ops::{
    RandomOpDescriptor,
    RandomDistribution,
    RandomNormalSamplingMethod
};
pub use optimizer_ops::VariableOp;
pub use data_types::{
    Type,
    ShapedType,
    DataTypeAttributeValue,
    ExecutionMode,
    ShapeDescriptor
};
pub use resize_ops::{
    GraphResizeOps,
    ResizeMode,
    ResizeNearestRoundingMode
};
pub use activation_ops::{GraphActivationOps, GraphActivationOpsExtension};
pub use arithmetic_ops::{GraphArithmeticOps, GraphArithmeticOpsExtension};
pub use call_ops::GraphCallOps;
pub use control_flow_ops::GraphControlFlowOps;
pub use convolution_transpose_ops::GraphConvolutionTransposeOps;
pub use cumulative_ops::GraphCumulativeOps;
pub use fourier_transform_ops::{FFTDescriptor, FFTScalingMode, GraphFourierTransformOps, GraphFourierTransformOpsExtension};
pub use gather_ops::{GraphGatherOps, GraphGatherOpsExtension};
pub use gradient_ops::{GraphGradientOps, GraphGradientOpsExtension};
pub use linear_algebra_ops::{GraphLinearAlgebraOps, GraphLinearAlgebraOpsExtension};
pub use loss_ops::{GraphLossOps, GraphLossOpsExtension, LossReductionType};
pub use matrix_inverse_ops::{GraphMatrixInverseOps, GraphMatrixInverseOpsExtension};
pub use matrix_ops::GraphMatrixOps;
pub use memory_ops::{GraphMemoryOps, GraphMemoryOpsExtension};
pub use non_maximum_suppression_ops::{GraphNonMaximumSuppressionOps, GraphNonMaximumSuppressionOpsExtension, NonMaximumSuppressionCoordinateMode};
pub use non_zero_ops::{GraphNonZeroOps, GraphNonZeroOpsExtension};
pub use one_hot_ops::{GraphOneHotOps, GraphOneHotOpsExtension};
pub use normalization_ops::GraphNormalizationOps;
pub use reduction_ops::GraphReductionOps;
pub use sample_grid_ops::{GraphSampleGridOps, GraphSampleGridOpsExtension, PaddingMode as SampleGridPaddingMode};
pub use scatter_nd_ops::{GraphScatterNdOps, GraphScatterNdOpsExtension, ScatterMode};
pub use sort_ops::{GraphSortOps, GraphSortOpsExtension};
pub use tensor_shape_ops::GraphTensorShapeOps;
pub use top_k_ops::{GraphTopKOps, GraphTopKOpsExtension};
pub use utils::buffer;
pub use utils::tensor as tensor_utils;

#[cfg(test)]
mod tests;