use objc2::rc::autoreleasepool;
use objc2::rc::{Allocated, Retained};
use objc2::runtime::NSObject;
use objc2::{extern_class, msg_send, ClassType};
use objc2_foundation::{NSArray, NSObjectProtocol, NSString, NSURL};
use std::path::Path;

use crate::command_buffer::CommandBuffer;
use crate::executable::{
    CompilationDescriptor, ExecutableExecutionDescriptor, SerializationDescriptor,
};
use crate::{DataType, Device, Tensor};

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
            let path_str = path.to_str()?;
            let path_ns = NSString::from_str(path_str);
            let url = NSURL::fileURLWithPath(&path_ns);
            let class = Self::class();

            let allocated: Allocated<Self> = msg_send![class, alloc];

            msg_send![
                allocated,
                initWithMPSGraphPackageAtURL: &*url,
                compilationDescriptor: compilation_descriptor.as_deref()
            ]
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
        device: Option<&Device>,
        input_types: &NSArray<DataType>,
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
            let array_opt: Option<Retained<NSArray<Tensor>>> = msg_send![self, feedTensors];
            match array_opt {
                Some(array) => {
                    let count = array.len();
                    let mut vec = Vec::with_capacity(count);
                    for i in 0..count {
                        let tensor_opt: Option<Retained<Tensor>> =
                            msg_send![&*array, objectAtIndex: i];
                        if let Some(tensor) = tensor_opt {
                            vec.push(tensor);
                        }
                    }
                    Some(vec)
                }
                None => None,
            }
        }
    }

    /// Returns the target tensors for the executable.
    pub fn target_tensors(&self) -> Option<Vec<Retained<Tensor>>> {
        unsafe {
            let array_opt: Option<Retained<NSArray<Tensor>>> = msg_send![self, targetTensors];
            match array_opt {
                Some(array) => {
                    let count = array.len();
                    let mut vec = Vec::with_capacity(count);
                    for i in 0..count {
                        let tensor_opt: Option<Retained<Tensor>> =
                            msg_send![&*array, objectAtIndex: i];
                        if let Some(tensor) = tensor_opt {
                            vec.push(tensor);
                        }
                    }
                    Some(vec)
                }
                None => None,
            }
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

            let result_array_opt: Option<Retained<NSArray<TensorData>>> = msg_send![
                self,
                encodeToCommandBuffer: cmd_buffer_ptr,
                inputsArray: inputs_array_ptr,
                resultsArray: results_array_ptr,
                executionDescriptor: desc_ptr
            ];

            match result_array_opt {
                Some(result_array) => {
                    let mut output_vec = Vec::with_capacity(result_array.len());
                    for item in result_array.iter() {
                        output_vec.push(item.clone());
                    }
                    Some(output_vec)
                }
                None => {
                    // eprintln! was here for warning, if desired, can be re-added.
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

            let result_array_opt: Option<Retained<NSArray<TensorData>>> = msg_send![
                self,
                runWithMTLCommandQueue: cmd_queue_ptr,
                inputsArray: inputs_array_ptr,
                resultsArray: results_array_ptr,
                executionDescriptor: desc_ptr
            ];

            match result_array_opt {
                Some(result_array) => {
                    let mut output_vec = Vec::with_capacity(result_array.len());
                    for item in result_array.iter() {
                        output_vec.push(item.clone());
                    }
                    Some(output_vec)
                }
                None => {
                    // eprintln! was here for warning, if desired, can be re-added.
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
