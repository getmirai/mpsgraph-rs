use crate::command_buffer::CommandBuffer;
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
    /// Runs the graph for the given feeds and returns the target tensor values,
    /// ensuring all target operations also executed.
    /// This call is asynchronous and will return immediately after finishing encoding.
    pub fn encode(
        self: &Self,
        command_buffer: &CommandBuffer,
        inputs: &[Retained<TensorData>],
        results: Option<&[Retained<TensorData>]>,
        execution_descriptor: Option<&ExecutableExecutionDescriptor>,
    ) -> Box<[Retained<TensorData>]> {
        unsafe {
            let command_buffer_ptr = command_buffer as *const CommandBuffer;
            let inputs_nsarray = NSArray::from_retained_slice(inputs);
            let inputs_nsarray_ptr = inputs_nsarray.as_ref() as *const NSArray<TensorData>;
            let results_optional_nsarray = results.map(|r| NSArray::from_retained_slice(r));
            let results_nsarray_ptr = results_optional_nsarray.map_or(std::ptr::null(), |r| {
                r.as_ref() as *const NSArray<TensorData>
            });
            let execution_descriptor_ptr =
                execution_descriptor.map_or(std::ptr::null(), |r| r as *const _);

            let results: Retained<NSArray<TensorData>> = msg_send![
                self,
                encodeToCommandBuffer: command_buffer_ptr,
                inputsArray: inputs_nsarray_ptr,
                resultsArray: results_nsarray_ptr,
                executionDescriptor: execution_descriptor_ptr,
            ];

            results.iter().collect()
        }
    }

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

    pub fn encode_to_command_buffer(
        &self,
        command_buffer: &Retained<crate::command_buffer::CommandBuffer>,
        inputs: &[&Retained<TensorData>],
        results: Option<&[&Retained<TensorData>]>,
        execution_descriptor: Option<&Retained<ExecutableExecutionDescriptor>>,
    ) -> Option<Vec<Retained<TensorData>>> {
        use objc2::rc::autoreleasepool;
        use objc2_foundation::NSArray;

        autoreleasepool(|_pool| {
            unsafe {
                // Use from_slice for inputs
                let input_refs: Vec<&TensorData> = inputs
                    .iter()
                    .map(|r: &&Retained<TensorData>| -> &TensorData { &**r })
                    .collect();

                // Create NSArray using from_slice with the Vec of references
                let inputs_array = NSArray::from_slice(&input_refs);

                // --- Results Array (Optional) ---
                let results_array_option: Option<Retained<NSArray<TensorData>>> =
                    results.map(|r_slice| {
                        let result_refs: Vec<&TensorData> = r_slice
                            .iter()
                            .map(|r: &&Retained<TensorData>| -> &TensorData { &**r })
                            .collect();
                        NSArray::from_slice(&result_refs)
                    });
                // Get a raw pointer (*const) or null for msg_send! Use Retained::as_ptr
                let results_array_ptr: *const NSArray<TensorData> = results_array_option
                    .as_ref()
                    .map_or(std::ptr::null(), |arr| Retained::as_ptr(arr));

                // --- Execution Descriptor (Optional) ---
                // Get a raw pointer (*const) or null for msg_send! Use Retained::as_ptr
                let desc_ptr: *const ExecutableExecutionDescriptor =
                    execution_descriptor.map_or(std::ptr::null(), |d| Retained::as_ptr(d));

                // --- Command Buffer Pointer ---
                // Use Retained::as_ptr to get the raw pointer
                let cmd_buffer_ptr: *const crate::command_buffer::CommandBuffer =
                    Retained::as_ptr(command_buffer);

                // --- Inputs Array Pointer ---
                // Use Retained::as_ptr to get the raw pointer
                let inputs_array_ptr: *const NSArray<TensorData> = Retained::as_ptr(&inputs_array);

                // --- Call Objective-C Method ---
                // Pass raw pointers for all arguments
                let result_ptr: *mut NSArray<TensorData> = msg_send![
                    self,
                    encodeToCommandBuffer: cmd_buffer_ptr,
                    inputsArray: inputs_array_ptr,
                    resultsArray: results_array_ptr,
                    executionDescriptor: desc_ptr
                ];

                // --- Handle Return Value ---
                if result_ptr.is_null() {
                    None
                } else {
                    // Attempt to take ownership of the potentially autoreleased NSArray*
                    if let Some(result_array) = Retained::retain_autoreleased(result_ptr) {
                        // Successfully retained the array, now process items
                        let mut output_vec = Vec::with_capacity(result_array.len());
                        for item in result_array.iter() {
                            // iter() yields &Retained<TensorData>
                            // Clone the item to transfer ownership to the output Vec
                            output_vec.push(item.clone());
                        }
                        Some(output_vec)
                    } else {
                        // retain_autoreleased failed, pointer was likely invalid
                        eprintln!("Warning: retain_autoreleased failed for non-null result_ptr in encode_to_command_buffer");
                        None
                    }
                }
            }
        })
    }
}
