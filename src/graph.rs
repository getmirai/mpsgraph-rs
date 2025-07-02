use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::NSObject;
use objc2::{extern_class, msg_send, ClassType};
use objc2_foundation::{
    NSArray, NSData, NSDictionary, NSMutableDictionary, NSObjectProtocol, NSString,
};
use std::collections::HashMap;

use crate::command_buffer::CommandBuffer;
use crate::device::Device;
use crate::executable::{CompilationDescriptor, Executable, ExecutionDescriptor};
use crate::operation::Operation;
use crate::shape::Shape;
use crate::tensor::{DataType, Tensor};
use crate::tensor_data::TensorData;
use crate::ShapedType;

/// Trait for scalar types that can be used in Graph operations
/// This trait is used for both single scalar values and arrays of values
pub trait TensorDataScalar: Copy {
    /// Convert a scalar value to f64 for use with Objective-C scalar methods
    fn to_f64(&self) -> f64;
}

// Implement for common numeric types
impl TensorDataScalar for f32 {
    fn to_f64(&self) -> f64 {
        *self as f64
    }
}

impl TensorDataScalar for f64 {
    fn to_f64(&self) -> f64 {
        *self
    }
}

impl TensorDataScalar for i32 {
    fn to_f64(&self) -> f64 {
        *self as f64
    }
}

impl TensorDataScalar for i64 {
    fn to_f64(&self) -> f64 {
        *self as f64 // This may lose precision for large values
    }
}

impl TensorDataScalar for u32 {
    fn to_f64(&self) -> f64 {
        *self as f64
    }
}

impl TensorDataScalar for u64 {
    fn to_f64(&self) -> f64 {
        *self as f64 // This may lose precision for large values
    }
}

extern_class!(
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraph"]
    pub struct Graph;
);

unsafe impl NSObjectProtocol for Graph {}

impl Graph {
    /// Creates a new Graph
    pub fn new() -> Retained<Self> {
        unsafe {
            let class = Self::class();
            // 'new' is a class method that returns an object with +1 retain count.
            let obj: Retained<Self> = msg_send![class, new];
            obj
        }
    }

    pub fn placeholder(
        &self,
        data_type: DataType,
        shape: &Shape,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let tensor: Retained<Tensor> = msg_send![
                self,
                placeholderWithShape: shape.as_ptr(),
                dataType: data_type as u32,
                name: name_ptr
            ];
            tensor
        }
    }

    /// Execute the graph asynchronously with feeds and optional execution descriptor
    /// (Maps to ObjC runAsyncWithFeeds:targetTensors:targetOperations:executionDescriptor:)
    ///
    /// - Parameters:
    ///   - feeds: A dictionary mapping input tensors to their values
    ///   - output_tensors: An array of tensors whose values should be computed
    ///   - execution_descriptor: Optional descriptor controlling execution options
    ///
    /// - Returns: A dictionary mapping output tensors to their computed values
    pub fn run_async_with_feeds(
        &self,
        feeds: &HashMap<&Tensor, &TensorData>,
        output_tensors: &[&Tensor],
        execution_descriptor: Option<&ExecutionDescriptor>,
    ) -> HashMap<Retained<Tensor>, Retained<TensorData>> {
        unsafe {
            // Create NSMutableDictionary for feeds
            let dictionary_class = NSMutableDictionary::<Tensor, TensorData>::class();
            let dictionary: Retained<NSMutableDictionary<Tensor, TensorData>> =
                msg_send![dictionary_class, dictionaryWithCapacity: feeds.len()];

            // Add entries to dictionary
            for (tensor, data) in feeds {
                let _: () = msg_send![&*dictionary, setObject: data.as_ref() as &TensorData, forKey: tensor.as_ref() as &Tensor];
            }

            // Create NSArray for output tensors
            let output_refs: Vec<&Tensor> = output_tensors.iter().map(|t| t.as_ref()).collect();
            let output_array = NSArray::from_slice(&output_refs);

            // Get descriptor pointer or null
            let desc_ptr = execution_descriptor.map_or(std::ptr::null(), |d| {
                d.as_ref() as *const ExecutionDescriptor
            });

            // Always call the async version
            let results_dict_opt: Option<Retained<NSMutableDictionary<Tensor, TensorData>>> = msg_send![
                self,
                runAsyncWithFeeds: &*dictionary,
                targetTensors: &*output_array,
                targetOperations: std::ptr::null::<NSArray<Operation>>(), // Pass nil for operations
                executionDescriptor: desc_ptr // Pass pointer or null
            ];

            let results_dict = match results_dict_opt {
                Some(dict) => dict,
                None => return HashMap::new(), // Preserve original early return logic
            };

            // Convert NSDictionary to HashMap
            let mut result = HashMap::new();
            let keys: Retained<NSArray<Tensor>> = msg_send![&*results_dict, allKeys];
            let keys_count: usize = keys.len();

            for i in 0..keys_count {
                let key: Retained<Tensor> = msg_send![&*keys, objectAtIndex: i];
                let value: Retained<TensorData> = msg_send![&*results_dict, objectForKey: &*key];
                result.insert(key, value);
            }

            result
        }
    }

    /// Creates a constant tensor with the given raw bytes
    pub fn constant_with_data(
        &self,
        data: &[u8],
        shape: &Shape,
        data_type: DataType,
    ) -> Retained<Tensor> {
        unsafe {
            let ns_data = NSData::with_bytes(&data);
            let tensor: Retained<Tensor> = msg_send![
                self,
                constantWithData: &*ns_data,
                shape: shape.as_ptr(),
                dataType: data_type as u32
            ];
            tensor
        }
    }

    /// Creates a constant scalar tensor
    pub fn constant_with_scalar(&self, value: f64, data_type: DataType) -> Retained<Tensor> {
        unsafe {
            let tensor: Retained<Tensor> = msg_send![
                self,
                constantWithScalar: value,
                dataType: data_type as u32
            ];
            tensor
        }
    }

    /// Creates a constant tensor with scalar value and shape
    pub fn constant_with_filled_scalar(
        &self,
        value: f64,
        data_type: DataType,
        shape: &Shape,
    ) -> Retained<Tensor> {
        unsafe {
            let tensor: Retained<Tensor> = msg_send![
                self,
                constantWithScalar: value,
                shape: shape.as_ptr(),
                dataType: data_type as u32
            ];
            tensor
        }
    }

    /// Compiles the graph against a given set of feeds and targets
    ///
    /// - Parameters:
    ///   - device: Metal device to compile for
    ///   - feeds: A dictionary mapping input tensors to their values
    ///   - targets: An array of tensors whose values should be computed
    ///   - descriptor: Optional compilation descriptor
    ///
    /// - Returns: A compiled executable
    pub fn compile(
        &self,
        device: &Device,
        feeds: &HashMap<&Tensor, &ShapedType>,
        targets: &[&Tensor],
        descriptor: Option<&CompilationDescriptor>,
    ) -> Retained<Executable> {
        autoreleasepool(|_| unsafe {
            // Create immutable NSDictionary for feeds directly
            let keys: Vec<&Tensor> = feeds.keys().map(|k| k.as_ref()).collect();
            let values: Vec<&ShapedType> = feeds.values().map(|v| v.as_ref()).collect();
            let keys_array = NSArray::from_slice(&keys);
            let values_array = NSArray::from_slice(&values);

            // dictionaryWithObjects returns an autoreleased object, so we need to retain it
            let dictionary_class = NSDictionary::<Tensor, ShapedType>::class();
            let feeds_dict: Retained<NSDictionary<Tensor, ShapedType>> = msg_send![dictionary_class, dictionaryWithObjects: &*values_array, forKeys: &*keys_array];

            // Create NSArray for target tensors
            let targets_refs: Vec<&Tensor> = targets
                .iter()
                .map(|retained_tensor| retained_tensor.as_ref())
                .collect();
            let targets_array = NSArray::from_slice(&targets_refs);

            let desc_ptr = descriptor.map_or(std::ptr::null(), |d_ref| &**d_ref as *const _);

            // Compile the graph
            let executable: Retained<Executable> = msg_send![
                self,
                compileWithDevice: device as &Device,
                feeds: &*feeds_dict,
                targetTensors: &*targets_array,
                targetOperations: std::ptr::null::<NSArray<Operation>>(),
                compilationDescriptor: desc_ptr
            ];
            executable
        })
    }

    /// Compiles the graph against a given set of feeds, targets, and target operations
    ///
    /// - Parameters:
    ///   - device: Metal device to compile for
    ///   - feeds: A dictionary mapping input tensors to their values
    ///   - targets: An array of tensors whose values should be computed
    ///   - target_ops: An array of operations to be completed
    ///   - descriptor: Optional compilation descriptor
    ///
    /// - Returns: A compiled executable
    pub fn compile_with_targets_and_ops(
        &self,
        device: &Device,
        feeds: &HashMap<&Tensor, &ShapedType>,
        targets: &[&Tensor],
        target_ops: &[&Operation],
        descriptor: Option<&CompilationDescriptor>,
    ) -> Retained<Executable> {
        autoreleasepool(|_| unsafe {
            // Create immutable NSDictionary for feeds directly
            let keys: Vec<&Tensor> = feeds.keys().map(|k| k.as_ref()).collect();
            let values: Vec<&ShapedType> = feeds.values().map(|v| v.as_ref()).collect();
            let keys_array = NSArray::from_slice(&keys);
            let values_array = NSArray::from_slice(&values);

            // dictionaryWithObjects returns an autoreleased object, so we need to retain it
            let dictionary_class = NSDictionary::<Tensor, ShapedType>::class();
            let feeds_dict: Retained<NSDictionary<Tensor, ShapedType>> = msg_send![dictionary_class, dictionaryWithObjects: &*values_array, forKeys: &*keys_array];

            // Create NSArray for target tensors
            let targets_refs: Vec<&Tensor> = targets
                .iter()
                .map(|retained_tensor| retained_tensor.as_ref())
                .collect();
            let targets_array = NSArray::from_slice(&targets_refs);

            // Create NSArray for target operations
            let ops_refs: Vec<&Operation> = target_ops.iter().map(|op| op.as_ref()).collect();
            let ops_array = NSArray::from_slice(&ops_refs);

            let desc_ptr = descriptor.map_or(std::ptr::null(), |d_ref| &**d_ref as *const _);

            // Compile the graph
            let executable: Retained<Executable> = msg_send![
                self,
                compileWithDevice: device as &Device,
                feeds: &*feeds_dict,
                targetTensors: &*targets_array,
                targetOperations: &*ops_array,
                compilationDescriptor: desc_ptr
            ];
            executable
        })
    }

    /// Encodes the graph to a command buffer for execution
    ///
    /// - Parameters:
    ///   - command_buffer: Command buffer to encode operations into
    ///   - feeds: A dictionary mapping input tensors to their values
    ///   - target_tensors: Tensors whose values should be computed
    ///   - target_operations: Optional operations to be completed
    ///   - execution_descriptor: Optional execution descriptor
    ///
    /// - Returns: A dictionary mapping output tensors to their computed values
    pub fn encode_to_command_buffer(
        &self,
        command_buffer: &CommandBuffer,
        feeds: &HashMap<&Tensor, &TensorData>,
        target_tensors: Option<&[&Tensor]>,
        target_operations: Option<&[&Operation]>,
        execution_descriptor: Option<&ExecutionDescriptor>,
    ) -> HashMap<Retained<Tensor>, Retained<TensorData>> {
        autoreleasepool(|_| unsafe {
            // Create NSMutableDictionary for feeds
            let dictionary_class = NSMutableDictionary::<Tensor, TensorData>::class();
            let dictionary: Retained<NSMutableDictionary<Tensor, TensorData>> =
                msg_send![dictionary_class, dictionaryWithCapacity: feeds.len()];

            // Add entries to dictionary
            for (tensor, data) in feeds {
                let _: () = msg_send![&*dictionary, setObject: data.as_ref() as &TensorData, forKey: tensor.as_ref() as &Tensor];
            }

            // Create NSArray for target tensors if provided
            let targets_array_option = match target_tensors {
                Some(tensors) => {
                    let targets_refs: Vec<&Tensor> = tensors
                        .iter()
                        .map(|retained_tensor| retained_tensor.as_ref())
                        .collect();
                    Some(NSArray::from_slice(&targets_refs))
                }
                None => None,
            };

            let targets_array_ptr = targets_array_option
                .as_ref()
                .map_or(std::ptr::null(), |arr| &**arr as *const NSArray<Tensor>);

            // Create NSArray for target operations if provided
            let ops_array_option: Option<Retained<NSArray<Operation>>> = match target_operations {
                Some(ops_retained) => {
                    let ops_refs: Vec<&Operation> = ops_retained
                        .iter()
                        .map(|retained_op| retained_op.as_ref())
                        .collect();
                    Some(NSArray::from_slice(&ops_refs))
                }
                None => None,
            };

            let ops_array_ptr = ops_array_option
                .as_ref()
                .map_or(std::ptr::null(), |arr| &**arr as *const NSArray<Operation>);

            // Get descriptor pointer if provided
            let desc_ptr = execution_descriptor.map_or(std::ptr::null(), |d| {
                d.as_ref() as *const ExecutionDescriptor
            });

            // Encode the graph to the command buffer
            let results_dict_opt: Option<Retained<NSMutableDictionary<Tensor, TensorData>>> = msg_send![
                self,
                encodeToCommandBuffer: command_buffer as &CommandBuffer,
                feeds: &*dictionary,
                targetTensors: targets_array_ptr,
                targetOperations: ops_array_ptr,
                executionDescriptor: desc_ptr
            ];

            let results_dict = match results_dict_opt {
                Some(dict) => dict,
                None => return HashMap::new(),
            };

            let mut result = HashMap::new();
            let keys: Retained<NSArray<Tensor>> = msg_send![&*results_dict, allKeys];

            // Retain the keys array
            let keys_count = keys.len();

            for i in 0..keys_count {
                let key: Retained<Tensor> = msg_send![&*keys, objectAtIndex: i];
                let value: Retained<TensorData> = msg_send![&*results_dict, objectForKey: &*key];
                result.insert(key, value);
            }

            result
        })
    }

    /// Encodes the graph to a command buffer with a results dictionary
    ///
    /// - Parameters:
    ///   - command_buffer: Command buffer to encode operations into
    ///   - feeds: A dictionary mapping input tensors to their values
    ///   - target_operations: Optional operations to be completed
    ///   - results_dict: Dictionary mapping tensors to TensorData where results will be stored
    ///   - execution_descriptor: Optional execution descriptor
    pub fn encode_to_command_buffer_with_results(
        &self,
        command_buffer: &CommandBuffer,
        feeds: &HashMap<&Tensor, &TensorData>,
        target_operations: Option<&[&Operation]>,
        results_dict: &HashMap<&Tensor, &TensorData>,
        execution_descriptor: Option<&ExecutionDescriptor>,
    ) {
        autoreleasepool(|_| unsafe {
            // Create NSMutableDictionary for feeds
            let feeds_dictionary_class = NSMutableDictionary::<Tensor, TensorData>::class();
            let feeds_dictionary: Retained<NSMutableDictionary<Tensor, TensorData>> =
                msg_send![feeds_dictionary_class, dictionaryWithCapacity: feeds.len()];

            // feeds_dictionary is already Retained, no need for retain_autoreleased here
            // Removed: let feeds_dictionary = Retained::retain_autoreleased(feeds_dictionary).unwrap();

            // Add entries to feeds dictionary
            for (tensor, data) in feeds {
                let _: () = msg_send![&*feeds_dictionary, setObject: data.as_ref() as &TensorData, forKey: tensor.as_ref() as &Tensor];
            }

            // Create NSMutableDictionary for results
            let results_dictionary_class = NSMutableDictionary::<Tensor, TensorData>::class();
            let results_dictionary: Retained<NSMutableDictionary<Tensor, TensorData>> =
                msg_send![results_dictionary_class, dictionaryWithCapacity: results_dict.len()];

            // results_dictionary is already Retained, no need for retain_autoreleased here
            // Removed: let results_dictionary = Retained::retain_autoreleased(results_dictionary).unwrap();

            // Add entries to results dictionary
            for (tensor, data) in results_dict {
                let _: () = msg_send![&*results_dictionary, setObject: data.as_ref() as &TensorData, forKey: tensor.as_ref() as &Tensor];
            }

            // Create NSArray for target operations if provided
            let ops_array_option: Option<Retained<NSArray<Operation>>> = match target_operations {
                Some(ops_retained) => {
                    let ops_refs: Vec<&Operation> = ops_retained
                        .iter()
                        .map(|retained_op| retained_op.as_ref())
                        .collect();
                    Some(NSArray::from_slice(&ops_refs))
                }
                None => None,
            };

            let ops_array_ptr = ops_array_option
                .as_ref()
                .map_or(std::ptr::null(), |arr| &**arr as *const NSArray<Operation>);

            // Get descriptor pointer if provided
            let desc_ptr = execution_descriptor.map_or(std::ptr::null(), |d| {
                d.as_ref() as *const ExecutionDescriptor
            });

            // Encode the graph to the command buffer with results
            let _: () = msg_send![
                self,
                encodeToCommandBuffer: command_buffer as &CommandBuffer,
                feeds: &*feeds_dictionary,
                targetOperations: ops_array_ptr,
                resultsDictionary: &*results_dictionary,
                executionDescriptor: desc_ptr
            ];
        })
    }

    /// Creates a tensor with random uniform values
    pub fn random_uniform(
        &self,
        shape: &Shape,
        min: f32,
        max: f32,
        data_type: DataType,
        seed: u32,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let tensor: Retained<Tensor> = msg_send![
                self,
                randomUniformTensorWithShape: shape.as_ptr(),
                min: min,
                max: max,
                dataType: data_type as u64,
                seed: seed,
                name: name_ptr
            ];
            tensor
        }
    }

    /// Creates a tensor with random normal values
    pub fn random_normal(
        &self,
        shape: &Shape,
        mean: f32,
        std_dev: f32,
        data_type: DataType,
        seed: u32,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let tensor: Retained<Tensor> = msg_send![
                self,
                randomNormalTensorWithShape: shape.as_ptr(),
                mean: mean,
                standardDeviation: std_dev,
                dataType: data_type as u64,
                seed: seed,
                name: name_ptr
            ];
            tensor
        }
    }

    /// Adds two tensors element-wise
    pub fn add(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let tensor: Retained<Tensor> = msg_send![
                self,
                additionWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name_ptr
            ];
            tensor
        }
    }

    /// Multiplies two tensors element-wise
    pub fn multiply(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let tensor: Retained<Tensor> = msg_send![
                self,
                multiplicationWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name_ptr
            ];
            tensor
        }
    }

    /// Subtracts one tensor from another element-wise
    pub fn subtract(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let tensor: Retained<Tensor> = msg_send![
                self,
                subtractionWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name_ptr
            ];
            tensor
        }
    }

    /// Divides one tensor by another element-wise
    pub fn divide(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let tensor: Retained<Tensor> = msg_send![
                self,
                divisionWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name_ptr
            ];
            tensor
        }
    }

    /// Gets the placeholder tensors associated with the graph, returned as a Vec of retained Tensors.
    pub fn placeholder_tensors(&self) -> Vec<Retained<Tensor>> {
        unsafe {
            let array: Retained<NSArray<Tensor>> = msg_send![self, placeholderTensors];

            let count = array.len();
            let mut vec = Vec::with_capacity(count);
            for i in 0..count {
                // objectAtIndex: returns a borrowed reference, so we need to retain it.
                let retained_tensor: Retained<Tensor> = msg_send![&*array, objectAtIndex: i];
                vec.push(retained_tensor);
            }
            vec
        }
    }

    pub fn dump(&self) {
        unsafe {
            let _: () = msg_send![self, dump];
        }
    }
}
