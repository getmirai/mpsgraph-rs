#[cfg(not(any(target_os = "macos", target_os = "ios")))]
compile_error!("MetalPerformanceShadersGraph is only available on macOS and iOS");

mod command_buffer;
mod core;
mod device;
mod executable;
mod graph;
mod ops;
// mod scalar_or_tensor;

// mod activation_ops;
// mod arithmetic_ops;
// mod call_ops;
// mod convolution_ops;
// mod convolution_transpose_ops;

pub use core::{GraphObject, Shape};
pub use device::{ComputeDevice, Device, DeviceType};
// pub use executable::{
//     CompilationDescriptor, DeploymentPlatform, Executable, ExecutableExecutionDescriptor,
//     ExecutionDescriptor, ExecutionResult, ExecutionStage, Optimization, OptimizationProfile,
//     SerializationDescriptor,
// };
// pub use graph::{Graph, TensorDataScalar};

pub use command_buffer::{CommandBuffer, CommandBufferStatus};
// pub use scalar_or_tensor::ScalarOrTensor;
