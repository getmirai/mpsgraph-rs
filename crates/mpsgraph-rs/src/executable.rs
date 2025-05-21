use crate::command_buffer::CommandBuffer;
use metal::foreign_types::ForeignType;
use metal::SharedEvent;
use objc2::rc::Retained;
use objc2::runtime::NSObject;
use objc2::{extern_class, msg_send, ClassType, Message};
use objc2_foundation::{
    NSArray, NSDictionary, NSMutableDictionary, NSObjectProtocol, NSString, NSURL,
};
use std::collections::HashMap;
use std::path::Path;

use crate::device::MPSGraphComputeDevice;
use crate::tensor::Tensor;
use crate::tensor_data::TensorData;
use objc2::rc::autoreleasepool;

/// Represents the optimization level for graph compilation
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u64)]
pub enum Optimization {
    /// Graph performs core optimizations only
    Level0 = 0,
    /// Graph performs additional optimizations
    Level1 = 1,
}

/// Represents the optimization profile for graph compilation
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u64)]
pub enum OptimizationProfile {
    /// Profile optimized for performance
    Performance = 0,
    /// Profile optimized for power efficiency
    PowerEfficiency = 1,
}

/// Represents the stages of execution for a graph
#[repr(u64)]
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
            let _: () = msg_send![self, setOptimizationLevel: level as u64];
        }
    }

    /// Set the optimization profile
    pub fn set_optimization_profile(&self, profile: OptimizationProfile) {
        unsafe {
            let _: () = msg_send![self, setOptimizationProfile: profile as u64];
        }
    }

    /// Set whether to debug compile
    pub fn set_debug_compile(&self, debug_compile: bool) {
        unsafe {
            let _: () = msg_send![self, setDebugCompile: debug_compile];
        }
    }

    /// Get the callables map as a Rust HashMap
    pub fn get_callables(&self) -> HashMap<String, Retained<Executable>> {
        unsafe {
            let callables: *mut NSDictionary<NSString, Executable> = msg_send![self, callables];
            if callables.is_null() {
                return HashMap::new();
            }

            // Safely retain the dictionary
            let ns_dict = match Retained::retain_autoreleased(callables) {
                Some(dict) => dict,
                None => return HashMap::new(),
            };

            // Get the keys
            let keys_array: *mut NSArray<NSString> = msg_send![&*ns_dict, allKeys];
            if keys_array.is_null() {
                return HashMap::new();
            }

            let keys = match Retained::retain_autoreleased(keys_array) {
                Some(arr) => arr,
                None => return HashMap::new(),
            };

            // Create a HashMap and populate it
            let mut result = HashMap::with_capacity(keys.len());

            for i in 0..keys.len() {
                let key_ptr: *mut NSString = msg_send![&*keys, objectAtIndex: i];
                if key_ptr.is_null() {
                    continue;
                }

                // Get the key as a Rust string
                let ns_key = match Retained::retain_autoreleased(key_ptr) {
                    Some(key) => key,
                    None => continue,
                };

                let key_str = ns_key.to_string();

                // Get the corresponding executable
                let exec_ptr: *mut Executable = msg_send![&*ns_dict, objectForKey: &*ns_key];
                if exec_ptr.is_null() {
                    continue;
                }

                let executable = match Retained::retain_autoreleased(exec_ptr) {
                    Some(exec) => exec,
                    None => continue,
                };

                result.insert(key_str, executable);
            }

            result
        }
    }

    /// Set the callables map using a Rust HashMap
    pub fn set_callables(&self, callables: &HashMap<String, &Executable>) {
        if callables.is_empty() {
            // If the HashMap is empty, set the property to nil
            unsafe {
                let _: () = msg_send![self, setCallables: std::ptr::null::<NSDictionary<NSString, Executable>>()];
            }
            return;
        }

        // Create a mutable dictionary
        let mutable_dict = NSMutableDictionary::<NSString, Executable>::new();

        // Add each entry to the dictionary
        for (key, &exec_ref) in callables {
            let ns_key = NSString::from_str(key);
            unsafe {
                let _: () = msg_send![&*mutable_dict, setObject: exec_ref, forKey: &*ns_key];
            }
        }

        // Convert to immutable dictionary
        let immutable_dict = unsafe {
            let dict_ptr: *mut NSDictionary<NSString, Executable> = msg_send![&*mutable_dict, copy];
            Retained::from_raw(dict_ptr).unwrap()
        };

        // Set the property
        unsafe {
            let _: () = msg_send![self, setCallables: &*immutable_dict];
        }
    }

    /// Add a callable executable for a specific symbol name
    pub fn add_callable(&self, symbol_name: &str, executable: &Executable) {
        // Get the current callables
        let mut callables_retained_map = self.get_callables();

        // Add the new callable, retaining it for the map
        callables_retained_map.insert(symbol_name.to_string(), executable.retain());

        // Convert map values from Retained<Executable> to &Executable for set_callables
        let callables_refs_map: HashMap<String, &Executable> = callables_retained_map
            .iter()
            .map(|(k, v)| (k.clone(), v.as_ref()))
            .collect();

        // Update the callables property
        self.set_callables(&callables_refs_map);
    }

    /// Remove a callable executable for a specific symbol name
    pub fn remove_callable(&self, symbol_name: &str) {
        let mut callables_retained_map = self.get_callables();
        callables_retained_map.remove(symbol_name);

        let callables_refs_map: HashMap<String, &Executable> = callables_retained_map
            .iter()
            .map(|(k, v)| (k.clone(), v.as_ref()))
            .collect();
        self.set_callables(&callables_refs_map);
    }

    pub fn set_allowed_device(&self, devices: u64) {
        unsafe {
            let _: () = msg_send![self, setAllowedComputeDevices: devices as u64];
        }
    }

    pub fn set_compiler_options(&self, options: u64) {
        unsafe {
            let _: () = msg_send![self, setCompilerOptions: options];
        }
    }

    pub fn set_ane_compiler_spatial_splitting(&self, value: u64) {
        unsafe {
            let _: () = msg_send![self, setAneCompilerSpatialSplitting: value];
        }
    }

    pub fn set_enable_ane_fw_to_fw_signal(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setEnableANEFWToFWSignal: enable];
        }
    }

    pub fn set_enable_ane_late_latch(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setEnableANELateLatch: enable];
        }
    }

    pub fn set_print_ane_placement_analysis(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setPrintANEPlacementAnalysis: enable];
        }
    }

    pub fn set_preferred_device(&self, device: MPSGraphComputeDevice) {
        unsafe {
            let _: () = msg_send![self, setPreferredDevice: device.bits()];
        }
    }

    pub fn set_allowed_compute_devices(&self, devices: MPSGraphComputeDevice) {
        unsafe {
            let _: () = msg_send![self, setAllowedComputeDevices: devices.bits()];
        }
    }

    pub fn set_enable_parallel_encode(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setEnableParallelEncode: enable];
        }
    }

    pub fn set_maximum_number_of_parallel_encoding_regions(&self, value: u64) {
        unsafe {
            let _: () = msg_send![self, setMaximumNumberOfParallelEncodingRegions: value];
        }
    }

    pub fn set_minimum_number_of_ops_in_parallel_region(&self, value: u64) {
        unsafe {
            let _: () = msg_send![self, setMinimumNumberOfOpsInParallelRegion: value];
        }
    }

    pub fn set_enable_mlir_diagnostics(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setEnableMLIRDiagnostics: enable];
        }
    }

    pub fn set_enable_shape_equivalence(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setEnableShapeEquivalence: enable];
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
            let _: () = msg_send![self, signalEvent: event_ptr, atExecutionEvent: execution_stage as u64, value: value];
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
            // alloc/init returns a +1 retained object.
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
            let _: () = msg_send![self, signalEvent: event_ptr, atExecutionEvent: execution_stage as u64, value: value];
        }
    }

    pub fn set_enable_commit_and_continue(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setEnableCommitAndContinue: enable];
        }
    }

    pub fn set_simulate_ane_compile_failure(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setSimulateANECompileFailure: enable];
        }
    }

    pub fn set_simulate_ane_load_model_failure(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setSimulateANELoadModelFailure: enable];
        }
    }

    pub fn set_disable_synchronize_results(&self, disable: bool) {
        unsafe {
            let _: () = msg_send![self, setDisableSynchronizeResults: disable];
        }
    }

    pub fn set_disable_ane_caching(&self, disable: bool) {
        unsafe {
            let _: () = msg_send![self, setDisableANECaching: disable];
        }
    }

    pub fn set_disable_ane_fallback(&self, disable: bool) {
        unsafe {
            let _: () = msg_send![self, setDisableANEFallback: disable];
        }
    }

    pub fn set_enable_profiling_op_names(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setEnableProfilingOpNames: enable];
        }
    }

    pub fn set_brief_profiling_op_names(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setBriefProfilingOpNames: enable];
        }
    }

    pub fn set_break_up_metal_encoders(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setBreakUpMetalEncoders: enable];
        }
    }

    pub fn set_generate_runtime_execution_report(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setGenerateRuntimeExecutionReport: enable];
        }
    }

    pub fn set_maximum_number_of_encoding_threads(&self, value: u64) {
        unsafe {
            let _: () = msg_send![self, setMaximumNumberOfEncodingThreads: value];
        }
    }

    pub fn number_of_commits_by_mps_graph(&self) -> u64 {
        unsafe { msg_send![self, numberOfCommitsByMPSGraph] }
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
        path: &Path,
        compilation_descriptor: Option<&CompilationDescriptor>,
    ) -> Option<Retained<Self>> {
        unsafe {
            // Convert path to file URL string
            let path_str = path.to_str()?;

            // Create NSURL from path directly (returns Retained<NSURL>)
            let path_ns = NSString::from_str(path_str);
            let url = NSURL::fileURLWithPath(&path_ns);

            // Get compilation descriptor or pass nil
            let class = Self::class();
            let alloc: *mut Self = msg_send![class, alloc];

            match compilation_descriptor {
                Some(desc) => {
                    let executable: *mut Self = msg_send![
                        alloc,
                        initWithMPSGraphPackageAtURL: &*url,
                        compilationDescriptor: desc
                    ];

                    if executable.is_null() {
                        None
                    } else {
                        // This is an init method, which returns an object with +1 retain count
                        Retained::from_raw(executable)
                    }
                }
                None => {
                    let executable: *mut Self = msg_send![
                        alloc,
                        initWithMPSGraphPackageAtURL: &*url,
                        compilationDescriptor: std::ptr::null::<CompilationDescriptor>()
                    ];

                    if executable.is_null() {
                        None
                    } else {
                        // This is an init method, which returns an object with +1 retain count
                        Retained::from_raw(executable)
                    }
                }
            }
        }
    }

    pub fn serialize_to_url(&self, path: &Path, descriptor: &SerializationDescriptor) {
        unsafe {
            // Convert path to NSURL using fileURLWithPath
            if let Some(path_str) = path.to_str() {
                let path_ns = NSString::from_str(path_str);

                // fileURLWithPath returns Retained<NSURL>, not Option<Retained<NSURL>>
                let url = NSURL::fileURLWithPath(&path_ns);

                // Serialize (returns void)
                let _: () = msg_send![
                    self,
                    serializeToMPSGraphPackageAtURL: &*url,
                    descriptor: descriptor
                ];
            } else {
                eprintln!("Error: Could not convert path to string: {:?}", path);
            }
        }
    }

    /// Specialize the executable for a device and input types.
    pub fn specialize_with_device(
        &self,
        device: Option<&crate::device::Device>,
        input_types: &NSArray<crate::data_types::Type>,
        compilation_descriptor: Option<&CompilationDescriptor>,
    ) {
        unsafe {
            let device_ptr = device.map_or(std::ptr::null(), |d| d as *const crate::device::Device);
            let desc_ptr = compilation_descriptor
                .map_or(std::ptr::null(), |d| d as *const CompilationDescriptor);

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
                return None;
            }

            let array_opt = Retained::retain_autoreleased(array_ptr);
            if array_opt.is_none() {
                return None;
            }
            let array = array_opt.unwrap();

            let count = array.len();
            let mut vec = Vec::with_capacity(count);

            for i in 0..count {
                let tensor_ptr: *mut Tensor = msg_send![&*array, objectAtIndex: i];
                if let Some(tensor) = Retained::retain_autoreleased(tensor_ptr) {
                    vec.push(tensor);
                }
            }
            Some(vec)
        }
    }

    /// Returns the target tensors for the executable.
    pub fn target_tensors(&self) -> Option<Vec<Retained<Tensor>>> {
        unsafe {
            let array_ptr: *mut NSArray<Tensor> = msg_send![self, targetTensors];
            if array_ptr.is_null() {
                return None;
            }

            let array_opt = Retained::retain_autoreleased(array_ptr);
            if array_opt.is_none() {
                return None;
            }
            let array = array_opt.unwrap();

            let count = array.len();
            let mut vec = Vec::with_capacity(count);

            for i in 0..count {
                let tensor_ptr: *mut Tensor = msg_send![&*array, objectAtIndex: i];
                if let Some(tensor) = Retained::retain_autoreleased(tensor_ptr) {
                    vec.push(tensor);
                }
            }
            Some(vec)
        }
    }

    pub fn encode_to_command_buffer(
        &self,
        command_buffer: &CommandBuffer,
        inputs: &[&TensorData],
        results: Option<&[&TensorData]>,
        execution_descriptor: Option<&ExecutableExecutionDescriptor>,
    ) -> Option<Vec<Retained<TensorData>>> {
        autoreleasepool(|_pool| unsafe {
            let input_refs: Vec<&TensorData> = inputs.iter().map(|&r| r).collect();
            let inputs_array = NSArray::from_slice(&input_refs);

            let results_array_option: Option<Retained<NSArray<TensorData>>> =
                results.map(|r_slice| {
                    let result_refs: Vec<&TensorData> = r_slice.iter().map(|&r| r).collect();
                    NSArray::from_slice(&result_refs)
                });
            let results_array_ptr: *const NSArray<TensorData> = results_array_option
                .as_ref()
                .map_or(std::ptr::null(), |arr| Retained::as_ptr(arr));

            let desc_ptr: *const ExecutableExecutionDescriptor =
                execution_descriptor.map_or(std::ptr::null(), |d| d as *const _);

            let cmd_buffer_ptr: *const CommandBuffer = command_buffer as *const _;

            let inputs_array_ptr: *const NSArray<TensorData> = Retained::as_ptr(&inputs_array);

            let result_ptr: *mut NSArray<TensorData> = msg_send![
                self,
                encodeToCommandBuffer: cmd_buffer_ptr,
                inputsArray: inputs_array_ptr,
                resultsArray: results_array_ptr,
                executionDescriptor: desc_ptr
            ];

            if result_ptr.is_null() {
                None
            } else {
                if let Some(result_array) = Retained::retain_autoreleased(result_ptr) {
                    let mut output_vec = Vec::with_capacity(result_array.len());
                    for item in result_array.iter() {
                        output_vec.push(item.clone());
                    }
                    Some(output_vec)
                } else {
                    eprintln!("Warning: retain_autoreleased failed for non-null result_ptr in encode_to_command_buffer");
                    None
                }
            }
        })
    }

    /// Run the executable with a Metal command queue
    pub fn run_with_command_queue(
        &self,
        command_queue: &metal::CommandQueue,
        inputs: &[&TensorData],
        results: Option<&[&TensorData]>,
        execution_descriptor: Option<&ExecutableExecutionDescriptor>,
    ) -> Option<Vec<Retained<TensorData>>> {
        use objc2::rc::autoreleasepool;
        use objc2_foundation::NSArray;

        autoreleasepool(|_pool| unsafe {
            let input_refs: Vec<&TensorData> = inputs.iter().map(|&r| r).collect();
            let inputs_array = NSArray::from_slice(&input_refs);

            let results_array_option: Option<Retained<NSArray<TensorData>>> =
                results.map(|r_slice| {
                    let result_refs: Vec<&TensorData> = r_slice.iter().map(|&r| r).collect();
                    NSArray::from_slice(&result_refs)
                });
            let results_array_ptr: *const NSArray<TensorData> = results_array_option
                .as_ref()
                .map_or(std::ptr::null(), |arr| Retained::as_ptr(arr));

            let desc_ptr: *const ExecutableExecutionDescriptor =
                execution_descriptor.map_or(std::ptr::null(), |d| d as *const _);

            let cmd_queue_ptr = command_queue.as_ptr() as *mut std::ffi::c_void;

            let inputs_array_ptr: *const NSArray<TensorData> = Retained::as_ptr(&inputs_array);

            let result_ptr: *mut NSArray<TensorData> = msg_send![
                self,
                runWithMTLCommandQueue: cmd_queue_ptr,
                inputsArray: inputs_array_ptr,
                resultsArray: results_array_ptr,
                executionDescriptor: desc_ptr
            ];

            if result_ptr.is_null() {
                None
            } else {
                if let Some(result_array) = Retained::retain_autoreleased(result_ptr) {
                    let mut output_vec = Vec::with_capacity(result_array.len());
                    for item in result_array.iter() {
                        output_vec.push(item.clone());
                    }
                    Some(output_vec)
                } else {
                    eprintln!("Warning: retain_autoreleased failed for non-null result_ptr in run_with_command_queue");
                    None
                }
            }
        })
    }

    pub fn dump(&self) {
        unsafe {
            let _: () = msg_send![self, dump];
        }
    }
}
