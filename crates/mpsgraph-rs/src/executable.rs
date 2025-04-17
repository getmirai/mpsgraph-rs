use metal::{SharedEvent};
use metal::foreign_types::ForeignType;
use objc2::rc::Retained;
use objc2::{extern_class, msg_send, ClassType};
use objc2::runtime::{AnyClass, NSObject};
use objc2_foundation::{NSObjectProtocol, NSString, NSURL};
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
#[repr(u32)]
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

impl crate::CustomDefault for CompilationDescriptor {
    fn custom_default() -> Retained<Self> {
        Self::new()
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
    pub fn signal_event(
        &self,
        event: &SharedEvent,
        execution_stage: ExecutionStage,
        value: u64,
    ) {
        unsafe {
            let event_ptr = event.as_ptr() as *mut std::ffi::c_void;
            let _: () = msg_send![self, signalEvent: event_ptr, atExecutionEvent: execution_stage as u32, value: value];
        }
    }
}

impl crate::CustomDefault for ExecutionDescriptor {
    fn custom_default() -> Retained<Self> {
        Self::new()
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
            let _: () = msg_send![self, setDeploymentPlatform: platform as u32];
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

impl crate::CustomDefault for SerializationDescriptor {
    fn custom_default() -> Retained<Self> {
        Self::new()
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
    pub fn signal_event(
        &self,
        event: &SharedEvent,
        execution_stage: ExecutionStage,
        value: u64,
    ) {
        unsafe {
            let event_ptr = event.as_ptr() as *mut std::ffi::c_void;
            let _: () = msg_send![self, signalEvent: event_ptr, atExecutionEvent: execution_stage as u32, value: value];
        }
    }
}

impl crate::CustomDefault for ExecutableExecutionDescriptor {
    fn custom_default() -> Retained<Self> {
        Self::new()
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
        compilation_descriptor: Option<&CompilationDescriptor>,
    ) -> Option<Retained<Self>> {
        unsafe {
            // Convert URL to NSURL
            let url_str = NSString::from_str(url_string);
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
                        compilationDescriptor: desc
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
        descriptor: &SerializationDescriptor,
    ) -> bool {
        unsafe {
            // Convert URL to NSURL
            let url_str = NSString::from_str(url_string);
            let nsurl_class = AnyClass::get(c"NSURL").unwrap();
            let nsurl_ptr: *mut NSURL = msg_send![nsurl_class, URLWithString: &*url_str];
            
            if nsurl_ptr.is_null() {
                return false;
            }
            
            let nsurl = Retained::from_raw(nsurl_ptr).unwrap();
            
            // Serialize
            let result: bool = msg_send![
                self,
                serializeToMPSGraphPackageAtURL: &*nsurl,
                descriptor: descriptor
            ];
            
            result
        }
    }
}

impl crate::CustomDefault for Executable {
    fn custom_default() -> Retained<Self> {
        // Not a typical default case, a dummy implementation is returned
        // Users should use from_serialized_package or similar methods
        unsafe {
            let class = Self::class();
            let alloc: *mut Self = msg_send![class, alloc];
            let obj_ptr: *mut Self = msg_send![alloc, init];
            Retained::from_raw(obj_ptr).unwrap()
        }
    }
}