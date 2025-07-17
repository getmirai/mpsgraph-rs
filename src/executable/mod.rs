mod compilation_descriptor;
mod deployment_platform;
mod executable;
mod executable_execution_descriptor;
mod execution_descriptor;
mod execution_stage;
mod optimization;
mod optimization_profile;
mod serialization_descriptor;

use super::{Tensor, TensorData};
use objc2::rc::Retained;
use std::collections::HashMap;

pub use compilation_descriptor::CompilationDescriptor;
pub use deployment_platform::DeploymentPlatform;
pub use executable::Executable;
pub use executable_execution_descriptor::ExecutableExecutionDescriptor;
pub use execution_descriptor::ExecutionDescriptor;
pub use execution_stage::ExecutionStage;
pub use optimization::Optimization;
pub use optimization_profile::OptimizationProfile;
pub use serialization_descriptor::SerializationDescriptor;

/// Result type for graph execution
pub type ExecutionResult = HashMap<Retained<Tensor>, Retained<TensorData>>;
