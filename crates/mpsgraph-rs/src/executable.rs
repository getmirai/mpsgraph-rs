use metal::foreign_types::ForeignType;
use metal::SharedEvent;
use objc2::rc::Retained;
use objc2::runtime::{AnyClass, NSObject};
use objc2::{extern_class, msg_send, ClassType};
use objc2_foundation::{NSArray, NSObjectProtocol, NSString, NSURL};
use std::collections::HashMap;

use crate::tensor::Tensor;
use crate::tensor_data::TensorData;

/// Represents the optimization level for graph compilation
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u32)]
pub enum Optimization {
    /// Default optimization level
    Default = 0,
    /// Optimized for size
    Size = 1,
    /// Optimized for performance
    Performance = 2,
}

/// Represents the optimization profile for graph compilation
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u32)]
pub enum OptimizationProfile {
    /// Default optimization profile
    Default = 0,
    /// Profile optimized for size
    Size = 1,
    /// Profile optimized for performance
    Performance = 2,
}

/// Represents the stages of execution for a graph
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ExecutionStage {
    /// Execution is completed
    Completed = 0,
}

/// Represents the deployment platform for a graph
#[repr(u64)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DeploymentPlatform {
    /// macOS platform
    MacOS = 0,
    /// iOS platform
    IOS = 1,
    /// tvOS platform
    TVOS = 2,
    /// visionOS platform
    VisionOS = 3,
}

extern_class!(
    #[derive(Debug, PartialEq, Eq)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraphCompilationDescriptor"]
    pub struct CompilationDescriptor;
);

unsafe impl NSObjectProtocol for CompilationDescriptor {}

impl CompilationDescriptor {
    /// Create a new compilation descriptor
    pub fn new() -> Retained<Self> {
        unsafe {
            let class = Self::class();
            let alloc: *mut Self = msg_send![class, alloc];
            let obj_ptr: *mut Self = msg_send![alloc, init];
            Retained::from_raw(obj_ptr).unwrap()
        }
    }

    /// Set the optimization level
    pub fn set_optimization_level(&self, level: Optimization) {
        unsafe {
            let _: () = msg_send![self, setOptimizationLevel: level as u32];
        }
    }

    /// Set the optimization profile
    pub fn set_optimization_profile(&self, profile: OptimizationProfile) {
        unsafe {
            let _: () = msg_send![self, setOptimizationProfile: profile as u32];
        }
    }

    /// Set whether to debug compile
    pub fn set_debug_compile(&self, debug_compile: bool) {
        unsafe {
            let _: () = msg_send![self, setDebugCompile: debug_compile];
        }
    }
}

extern_class!(
    #[derive(Debug, PartialEq, Eq)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraphExecutionDescriptor"]
    pub struct ExecutionDescriptor;
);

unsafe impl NSObjectProtocol for ExecutionDescriptor {}

impl ExecutionDescriptor {
    /// Create a new execution descriptor
    pub fn new() -> Retained<Self> {
        unsafe {
            let class = Self::class();
            let alloc: *mut Self = msg_send![class, alloc];
            let obj_ptr: *mut Self = msg_send![alloc, init];
            Retained::from_raw(obj_ptr).unwrap()
        }
    }

    /// Set wait until completed flag
    pub fn set_wait_until_completed(&self, wait: bool) {
        unsafe {
            let _: () = msg_send![self, setWaitUntilCompleted: wait];
        }
    }

    /// Set the state of the execution descriptor to prefer synchronous execution
    pub fn prefer_synchronous_execution(&self) {
        self.set_wait_until_completed(true);
    }

    /// Set the state of the execution descriptor to prefer asynchronous execution
    pub fn prefer_asynchronous_execution(&self) {
        self.set_wait_until_completed(false);
    }

    /// Wait for a Metal shared event with a specific value before scheduling execution
    pub fn wait_for_event(&self, event: &SharedEvent, value: u64) {
        unsafe {
            let event_ptr = event.as_ptr() as *mut std::ffi::c_void;
            let _: () = msg_send![self, waitForEvent: event_ptr, value: value];
        }
    }

    /// Signal a Metal shared event with a value at a specific execution stage
    pub fn signal_event(&self, event: &SharedEvent, execution_stage: ExecutionStage, value: u64) {
        unsafe {
            let event_ptr = event.as_ptr() as *mut std::ffi::c_void;
            let _: () = msg_send![self, signalEvent: event_ptr, atExecutionEvent: execution_stage as u32, value: value];
        }
    }
}

extern_class!(
    #[derive(Debug, PartialEq, Eq)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraphExecutableSerializationDescriptor"]
    pub struct SerializationDescriptor;
);

unsafe impl NSObjectProtocol for SerializationDescriptor {}

impl SerializationDescriptor {
    /// Create a new serialization descriptor
    pub fn new() -> Retained<Self> {
        unsafe {
            let class = Self::class();
            let alloc: *mut Self = msg_send![class, alloc];
            let obj_ptr: *mut Self = msg_send![alloc, init];
            Retained::from_raw(obj_ptr).unwrap()
        }
    }

    /// Set append flag - if true, appends to existing file instead of overwriting
    pub fn set_append(&self, append: bool) {
        unsafe {
            let _: () = msg_send![self, setAppend: append];
        }
    }

    /// Set deployment platform
    pub fn set_deployment_platform(&self, platform: DeploymentPlatform) {
        unsafe {
            let _: () = msg_send![self, setDeploymentPlatform: platform as u64];
        }
    }

    /// Set minimum deployment target as a string (e.g., "13.0")
    pub fn set_minimum_deployment_target(&self, target: &str) {
        unsafe {
            let target_str = NSString::from_str(target);
            let _: () = msg_send![self, setMinimumDeploymentTarget: &*target_str];
        }
    }
}

extern_class!(
    #[derive(Debug, PartialEq, Eq)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraphExecutableExecutionDescriptor"]
    pub struct ExecutableExecutionDescriptor;
);

unsafe impl NSObjectProtocol for ExecutableExecutionDescriptor {}

impl ExecutableExecutionDescriptor {
    /// Create a new executable execution descriptor
    pub fn new() -> Retained<Self> {
        unsafe {
            let class = Self::class();
            let alloc: *mut Self = msg_send![class, alloc];
            let obj_ptr: *mut Self = msg_send![alloc, init];
            Retained::from_raw(obj_ptr).unwrap()
        }
    }

    /// Set wait until completed flag
    pub fn set_wait_until_completed(&self, wait: bool) {
        unsafe {
            let _: () = msg_send![self, setWaitUntilCompleted: wait];
        }
    }

    /// Set the state of the execution descriptor to prefer synchronous execution
    pub fn prefer_synchronous_execution(&self) {
        self.set_wait_until_completed(true);
    }

    /// Set the state of the execution descriptor to prefer asynchronous execution
    pub fn prefer_asynchronous_execution(&self) {
        self.set_wait_until_completed(false);
    }

    /// Wait for a Metal shared event with a specific value before scheduling execution
    pub fn wait_for_event(&self, event: &SharedEvent, value: u64) {
        unsafe {
            let event_ptr = event.as_ptr() as *mut std::ffi::c_void;
            let _: () = msg_send![self, waitForEvent: event_ptr, value: value];
        }
    }

    /// Signal a Metal shared event with a value at a specific execution stage
    pub fn signal_event(&self, event: &SharedEvent, execution_stage: ExecutionStage, value: u64) {
        unsafe {
            let event_ptr = event.as_ptr() as *mut std::ffi::c_void;
            let _: () = msg_send![self, signalEvent: event_ptr, atExecutionEvent: execution_stage as u32, value: value];
        }
    }
}

/// Result type for graph execution
pub type ExecutionResult = HashMap<Retained<Tensor>, Retained<TensorData>>;

extern_class!(
    #[derive(Debug, PartialEq, Eq)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraphExecutable"]
    pub struct Executable;
);

unsafe impl NSObjectProtocol for Executable {}

impl Executable {
    /// Create a new executable from a serialized package at the specified URL
    pub fn from_serialized_package(
        url_string: &str,
        compilation_descriptor: Option<&Retained<CompilationDescriptor>>,
    ) -> Option<Retained<Self>> {
        unsafe {
            // Convert URL string to file URL string
            let file_url_string = format!("file://{}", url_string);
            let url_str = NSString::from_str(&file_url_string); // Use file URL string
            let nsurl_class = AnyClass::get(c"NSURL").unwrap();
            let nsurl_ptr: *mut NSURL = msg_send![nsurl_class, URLWithString: &*url_str];

            if nsurl_ptr.is_null() {
                return None;
            }

            let nsurl = Retained::from_raw(nsurl_ptr).unwrap();

            // Initialize from URL
            let class = Self::class();
            let alloc: *mut Self = msg_send![class, alloc];

            // Get compilation descriptor or pass nil
            match compilation_descriptor {
                Some(desc) => {
                    let executable: *mut Self = msg_send![
                        alloc,
                        initWithMPSGraphPackageAtURL: &*nsurl,
                        compilationDescriptor: desc.as_ref() as &CompilationDescriptor
                    ];

                    if executable.is_null() {
                        None
                    } else {
                        Retained::from_raw(executable)
                    }
                }
                None => {
                    let executable: *mut Self = msg_send![
                        alloc,
                        initWithMPSGraphPackageAtURL: &*nsurl,
                        compilationDescriptor: std::ptr::null::<CompilationDescriptor>()
                    ];

                    if executable.is_null() {
                        None
                    } else {
                        Retained::from_raw(executable)
                    }
                }
            }
        }
    }

    /// Serializes the executable to a file URL
    pub fn serialize_to_url(
        &self,
        url_string: &str,
        descriptor: &Retained<SerializationDescriptor>,
    ) {
        unsafe {
            // Convert URL string to file URL string
            let file_url_string = format!("file://{}", url_string);
            let url_str = NSString::from_str(&file_url_string);
            let nsurl_class = AnyClass::get(c"NSURL").unwrap();
            let nsurl_ptr: *mut NSURL = msg_send![nsurl_class, URLWithString: &*url_str];

            if nsurl_ptr.is_null() {
                eprintln!("Error: Could not create NSURL from string: {}", url_string);
                return;
            }

            let nsurl = Retained::from_raw(nsurl_ptr).unwrap();

            // Serialize (returns void)
            let _: () = msg_send![
                self,
                serializeToMPSGraphPackageAtURL: &*nsurl,
                descriptor: descriptor.as_ref() as &SerializationDescriptor
            ];
        }
    }

    /// Specialize the executable for a device and input types.
    pub fn specialize_with_device(
        &self,
        device: Option<&Retained<crate::device::Device>>,
        input_types: &NSArray<crate::data_types::Type>,
        compilation_descriptor: Option<&Retained<CompilationDescriptor>>,
    ) {
        unsafe {
            let device_ptr = device.map_or(std::ptr::null(), |d| {
                d.as_ref() as *const crate::device::Device
            });
            let desc_ptr = compilation_descriptor.map_or(std::ptr::null(), |d| {
                d.as_ref() as *const CompilationDescriptor
            });

            let _: () = msg_send![
                self,
                specializeWithDevice: device_ptr,
                inputTypes: input_types,
                compilationDescriptor: desc_ptr
            ];
        }
    }

    /// Returns the feed tensors for the executable.
    pub fn feed_tensors(&self) -> Option<Vec<Retained<Tensor>>> {
        unsafe {
            let array_ptr: *mut NSArray<Tensor> = msg_send![self, feedTensors];
            if array_ptr.is_null() {
                None
            } else {
                // Work directly with the raw pointer
                let count: usize = msg_send![array_ptr, count];
                let mut vec = Vec::with_capacity(count);
                for i in 0..count {
                    // Get pointer, check null, create Retained from raw
                    let tensor_ptr: *mut Tensor = msg_send![array_ptr, objectAtIndex: i];
                    if !tensor_ptr.is_null() {
                        vec.push(Retained::from_raw(tensor_ptr).unwrap());
                    }
                }
                Some(vec)
            }
        }
    }

    /// Returns the target tensors for the executable.
    pub fn target_tensors(&self) -> Option<Vec<Retained<Tensor>>> {
        unsafe {
            let array_ptr: *mut NSArray<Tensor> = msg_send![self, targetTensors];
            if array_ptr.is_null() {
                None
            } else {
                // Work directly with the raw pointer
                let count: usize = msg_send![array_ptr, count];
                let mut vec = Vec::with_capacity(count);
                for i in 0..count {
                    // Get pointer, check null, create Retained from raw
                    let tensor_ptr: *mut Tensor = msg_send![array_ptr, objectAtIndex: i];
                    if !tensor_ptr.is_null() {
                        vec.push(Retained::from_raw(tensor_ptr).unwrap());
                    }
                }
                Some(vec)
            }
        }
    }

    /// Encodes the executable's commands to a command buffer.
    pub fn encode_to_command_buffer(
        &self,
        command_buffer: &Retained<crate::command_buffer::CommandBuffer>, // Use crate::CommandBuffer
        inputs: &[&Retained<TensorData>],
        results: Option<&[&Retained<TensorData>]>,
        execution_descriptor: Option<&Retained<ExecutableExecutionDescriptor>>,
    ) -> Option<Vec<Retained<TensorData>>> {
        unsafe {
            // Create NSArray for inputs
            let input_refs: Vec<&TensorData> = inputs.iter().map(|t| t.as_ref()).collect();
            let inputs_array = NSArray::from_slice(&input_refs);

            // Create NSArray for results (or null)
            let results_array_ptr: *const NSArray<TensorData> = match results {
                Some(res_slice) => {
                    let result_refs: Vec<&TensorData> =
                        res_slice.iter().map(|t| t.as_ref()).collect();
                    let array = NSArray::from_slice(&result_refs);
                    &*array
                }
                None => std::ptr::null(),
            };

            // Get descriptor pointer or null
            let desc_ptr = execution_descriptor.map_or(std::ptr::null(), |d| {
                d.as_ref() as *const ExecutableExecutionDescriptor
            });

            // Call encodeToCommandBuffer
            let returned_array_ptr: *mut NSArray<TensorData> = msg_send![
                self,
                encodeToCommandBuffer: command_buffer.as_ref() as &crate::command_buffer::CommandBuffer, // Explicit cast
                inputsArray: &*inputs_array,
                resultsArray: results_array_ptr,
                executionDescriptor: desc_ptr
            ];

            // Convert returned NSArray to Vec<Retained<TensorData>>
            if returned_array_ptr.is_null() {
                None
            } else {
                // Work directly with the raw pointer
                let count: usize = msg_send![returned_array_ptr, count];
                let mut vec = Vec::with_capacity(count);
                for i in 0..count {
                    // Get pointer, check null, create Retained from raw
                    let data_ptr: *mut TensorData = msg_send![returned_array_ptr, objectAtIndex: i];
                    if !data_ptr.is_null() {
                        vec.push(Retained::from_raw(data_ptr).unwrap());
                    }
                }
                Some(vec)
            }
        }
    }
}
