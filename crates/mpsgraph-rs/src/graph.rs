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
            let obj_ptr: *mut Self = msg_send![class, new];
            Retained::from_raw(obj_ptr).unwrap()
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

            let tensor_ptr: *mut Tensor = msg_send![
                self,
                placeholderWithShape: shape.as_ptr(),
                dataType: data_type as u32,
                name: name_ptr
            ];

            if tensor_ptr.is_null() {
                panic!("Failed to create placeholder tensor");
            } else {
                Retained::retain_autoreleased(tensor_ptr).unwrap()
            }
        }
    }

    /// Performs matrix multiplication of two tensors
    pub fn matmul(
        &self,
        lhs: &Retained<Tensor>,
        rhs: &Retained<Tensor>,
        _transpose_lhs: bool,
        _transpose_rhs: bool,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result_ptr: *mut Tensor = msg_send![
                self,
                matrixMultiplicationWithPrimaryTensor: lhs.as_ref() as &Tensor,
                secondaryTensor: rhs.as_ref() as &Tensor,
                name: name_ptr
            ];

            if result_ptr.is_null() {
                panic!("Failed to create matrix multiplication tensor");
            } else {
                Retained::retain_autoreleased(result_ptr).unwrap()
            }
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
        feeds: &HashMap<&Retained<Tensor>, &Retained<TensorData>>,
        output_tensors: &[&Retained<Tensor>],
        execution_descriptor: Option<&Retained<ExecutionDescriptor>>,
    ) -> HashMap<Retained<Tensor>, Retained<TensorData>> {
        unsafe {
            // Create NSMutableDictionary for feeds
            let dictionary_class = NSMutableDictionary::<Tensor, TensorData>::class();
            let dictionary_ptr: *mut NSMutableDictionary<Tensor, TensorData> =
                msg_send![dictionary_class, dictionaryWithCapacity: feeds.len()];

            // Retain the dictionary since dictionaryWithCapacity returns an autoreleased object
            let dictionary = Retained::retain_autoreleased(dictionary_ptr).unwrap();

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
            let results_ptr: *mut NSMutableDictionary<Tensor, TensorData> = msg_send![
                self,
                runAsyncWithFeeds: &*dictionary,
                targetTensors: &*output_array,
                targetOperations: std::ptr::null::<NSArray<Operation>>(), // Pass nil for operations
                executionDescriptor: desc_ptr // Pass pointer or null
            ];

            // Check if results_ptr is null before proceeding
            if results_ptr.is_null() {
                // Handle error appropriately, maybe return empty map or panic
                // For now, let's return an empty map, though panic might be better
                // depending on expected behavior.
                return HashMap::new();
                // panic!("runAsyncWithFeeds returned null dictionary");
            }

            // Retain the results dictionary
            let results_dict = Retained::retain_autoreleased(results_ptr).unwrap();

            // Convert NSDictionary to HashMap
            let mut result = HashMap::new();
            let keys_ptr: *mut NSArray<Tensor> = msg_send![&*results_dict, allKeys];

            // Retain the keys array
            let keys = Retained::retain_autoreleased(keys_ptr).unwrap();
            let keys_count: usize = keys.len();

            for i in 0..keys_count {
                let key_ptr: *mut Tensor = msg_send![&*keys, objectAtIndex: i];
                let key = Retained::retain_autoreleased(key_ptr).unwrap();

                let value_ptr: *mut TensorData = msg_send![&*results_dict, objectForKey: &*key];
                let value = Retained::retain_autoreleased(value_ptr).unwrap();

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
            let tensor_ptr: *mut Tensor = msg_send![
                self,
                constantWithData: &*ns_data,
                shape: shape.as_ptr(),
                dataType: data_type as u32
            ];

            if tensor_ptr.is_null() {
                panic!("Failed to create constant tensor from raw bytes");
            } else {
                Retained::retain_autoreleased(tensor_ptr).unwrap()
            }
        }
    }

    /// Creates a constant scalar tensor
    pub fn constant_with_scalar(&self, value: f64, data_type: DataType) -> Retained<Tensor> {
        unsafe {
            let tensor_ptr: *mut Tensor = msg_send![
                self,
                constantWithScalar: value,
                dataType: data_type as u32
            ];

            if tensor_ptr.is_null() {
                panic!("Failed to create constant scalar tensor");
            } else {
                Retained::retain_autoreleased(tensor_ptr).unwrap()
            }
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
            let tensor_ptr: *mut Tensor = msg_send![
                self,
                constantWithScalar: value,
                shape: shape.as_ptr(),
                dataType: data_type as u32
            ];

            if tensor_ptr.is_null() {
                panic!("Failed to create constant scalar tensor with shape");
            } else {
                Retained::retain_autoreleased(tensor_ptr).unwrap()
            }
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
        device: &Retained<Device>,
        feeds: &HashMap<&Retained<Tensor>, &Retained<ShapedType>>,
        targets: &[&Retained<Tensor>],
        descriptor: Option<&Retained<CompilationDescriptor>>,
    ) -> Retained<Executable> {
        autoreleasepool(|_| unsafe {
            // Create immutable NSDictionary for feeds directly
            let keys: Vec<&Tensor> = feeds.keys().map(|k| k.as_ref()).collect();
            let values: Vec<&ShapedType> = feeds.values().map(|v| v.as_ref()).collect();
            let keys_array = NSArray::from_slice(&keys);
            let values_array = NSArray::from_slice(&values);

            // dictionaryWithObjects returns an autoreleased object, so we need to retain it
            let dictionary_class = NSDictionary::<Tensor, ShapedType>::class();
            let feeds_dict_ptr: *mut NSDictionary<Tensor, ShapedType> = msg_send![
                dictionary_class,
                dictionaryWithObjects: &*values_array,
                forKeys: &*keys_array
            ];
            let feeds_dict = Retained::retain_autoreleased(feeds_dict_ptr).unwrap();

            // Create NSArray for target tensors
            let targets_refs: Vec<&Tensor> = targets
                .iter()
                .map(|retained_tensor| retained_tensor.as_ref())
                .collect();
            let targets_array = NSArray::from_slice(&targets_refs);

            // Get descriptor pointer if provided
            let desc_ptr = descriptor.map_or(std::ptr::null(), |d| {
                d.as_ref() as *const CompilationDescriptor
            });

            // Compile the graph
            let executable_ptr: *mut Executable = msg_send![
                self,
                compileWithDevice: device.as_ref() as &Device,
                feeds: &*feeds_dict,
                targetTensors: &*targets_array,
                targetOperations: std::ptr::null::<NSArray<Operation>>(),
                compilationDescriptor: desc_ptr
            ];

            if executable_ptr.is_null() {
                panic!("Failed to compile graph");
            } else {
                Retained::retain_autoreleased(executable_ptr).unwrap()
            }
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
        device: &Retained<Device>,
        feeds: &HashMap<&Retained<Tensor>, &Retained<ShapedType>>,
        targets: &[&Retained<Tensor>],
        target_ops: &[&Retained<Operation>],
        descriptor: Option<&Retained<CompilationDescriptor>>,
    ) -> Retained<Executable> {
        autoreleasepool(|_| unsafe {
            // Create immutable NSDictionary for feeds directly
            let keys: Vec<&Tensor> = feeds.keys().map(|k| k.as_ref()).collect();
            let values: Vec<&ShapedType> = feeds.values().map(|v| v.as_ref()).collect();
            let keys_array = NSArray::from_slice(&keys);
            let values_array = NSArray::from_slice(&values);

            // dictionaryWithObjects returns an autoreleased object, so we need to retain it
            let dictionary_class = NSDictionary::<Tensor, ShapedType>::class();
            let feeds_dict_ptr: *mut NSDictionary<Tensor, ShapedType> = msg_send![
                dictionary_class,
                dictionaryWithObjects: &*values_array,
                forKeys: &*keys_array
            ];
            let feeds_dict = Retained::retain_autoreleased(feeds_dict_ptr).unwrap();

            // Create NSArray for target tensors
            let targets_refs: Vec<&Tensor> = targets
                .iter()
                .map(|retained_tensor| retained_tensor.as_ref())
                .collect();
            let targets_array = NSArray::from_slice(&targets_refs);

            // Create NSArray for target operations
            let ops_refs: Vec<&Operation> = target_ops.iter().map(|op| op.as_ref()).collect();
            let ops_array = NSArray::from_slice(&ops_refs);

            // Get descriptor pointer if provided
            let desc_ptr = descriptor.map_or(std::ptr::null(), |d| {
                d.as_ref() as *const CompilationDescriptor
            });

            // Compile the graph
            let executable_ptr: *mut Executable = msg_send![
                self,
                compileWithDevice: device.as_ref() as &Device,
                feeds: &*feeds_dict,
                targetTensors: &*targets_array,
                targetOperations: &*ops_array,
                compilationDescriptor: desc_ptr
            ];

            if executable_ptr.is_null() {
                panic!("Failed to compile graph with targets and operations");
            } else {
                Retained::retain_autoreleased(executable_ptr).unwrap()
            }
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
        command_buffer: &Retained<CommandBuffer>,
        feeds: &HashMap<&Retained<Tensor>, &Retained<TensorData>>,
        target_tensors: Option<&[&Retained<Tensor>]>,
        target_operations: Option<&[Retained<Operation>]>,
        execution_descriptor: Option<&Retained<ExecutionDescriptor>>,
    ) -> HashMap<Retained<Tensor>, Retained<TensorData>> {
        autoreleasepool(|_| unsafe {
            // Create NSMutableDictionary for feeds
            let dictionary_class = NSMutableDictionary::<Tensor, TensorData>::class();
            let dictionary_ptr: *mut NSMutableDictionary<Tensor, TensorData> =
                msg_send![dictionary_class, dictionaryWithCapacity: feeds.len()];

            // Retain the dictionary since dictionaryWithCapacity returns an autoreleased object
            let dictionary = Retained::retain_autoreleased(dictionary_ptr).unwrap();

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
            let results_ptr: *mut NSMutableDictionary<Tensor, TensorData> = msg_send![
                self,
                encodeToCommandBuffer: command_buffer.as_ref() as &CommandBuffer,
                feeds: &*dictionary,
                targetTensors: targets_array_ptr,
                targetOperations: ops_array_ptr,
                executionDescriptor: desc_ptr
            ];

            // Check if results_ptr is null
            if results_ptr.is_null() {
                // Consider how to handle this - return empty or panic?
                return HashMap::new();
                // panic!("encodeToCommandBuffer returned null dictionary");
            }

            // Convert NSMutableDictionary to HashMap - retain the results dictionary
            let results_dict = Retained::retain_autoreleased(results_ptr).unwrap();

            let mut result = HashMap::new();
            let keys_ptr: *mut NSArray<Tensor> = msg_send![&*results_dict, allKeys];

            // Retain the keys array
            let keys = Retained::retain_autoreleased(keys_ptr).unwrap();
            let keys_count = keys.len();

            for i in 0..keys_count {
                let key_ptr: *mut Tensor = msg_send![&*keys, objectAtIndex: i];
                let key = Retained::retain_autoreleased(key_ptr).unwrap();

                let value_ptr: *mut TensorData = msg_send![&*results_dict, objectForKey: &*key];
                let value = Retained::retain_autoreleased(value_ptr).unwrap();

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
        command_buffer: &Retained<CommandBuffer>,
        feeds: &HashMap<&Retained<Tensor>, &Retained<TensorData>>,
        target_operations: Option<&[Retained<Operation>]>,
        results_dict: &HashMap<&Retained<Tensor>, &Retained<TensorData>>,
        execution_descriptor: Option<&Retained<ExecutionDescriptor>>,
    ) {
        autoreleasepool(|_| unsafe {
            // Create NSMutableDictionary for feeds
            let feeds_dictionary_class = NSMutableDictionary::<Tensor, TensorData>::class();
            let feeds_ptr: *mut NSMutableDictionary<Tensor, TensorData> =
                msg_send![feeds_dictionary_class, dictionaryWithCapacity: feeds.len()];

            // Retain the feeds dictionary
            let feeds_dictionary = Retained::retain_autoreleased(feeds_ptr).unwrap();

            // Add entries to feeds dictionary
            for (tensor, data) in feeds {
                let _: () = msg_send![&*feeds_dictionary, setObject: data.as_ref() as &TensorData, forKey: tensor.as_ref() as &Tensor];
            }

            // Create NSMutableDictionary for results
            let results_dictionary_class = NSMutableDictionary::<Tensor, TensorData>::class();
            let results_ptr: *mut NSMutableDictionary<Tensor, TensorData> =
                msg_send![results_dictionary_class, dictionaryWithCapacity: results_dict.len()];

            // Retain the results dictionary
            let results_dictionary = Retained::retain_autoreleased(results_ptr).unwrap();

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
                encodeToCommandBuffer: command_buffer.as_ref() as &CommandBuffer,
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

            let tensor_ptr: *mut Tensor = msg_send![
                self,
                randomUniformTensorWithShape: shape.as_ptr(),
                min: min,
                max: max,
                dataType: data_type as u64,
                seed: seed,
                name: name_ptr
            ];

            if tensor_ptr.is_null() {
                panic!("Failed to create random uniform tensor");
            } else {
                Retained::retain_autoreleased(tensor_ptr).unwrap()
            }
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

            let tensor_ptr: *mut Tensor = msg_send![
                self,
                randomNormalTensorWithShape: shape.as_ptr(),
                mean: mean,
                standardDeviation: std_dev,
                dataType: data_type as u64,
                seed: seed,
                name: name_ptr
            ];

            if tensor_ptr.is_null() {
                panic!("Failed to create random normal tensor");
            } else {
                Retained::retain_autoreleased(tensor_ptr).unwrap()
            }
        }
    }

    /// Adds two tensors element-wise
    pub fn add(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let tensor_ptr: *mut Tensor = msg_send![
                self,
                additionWithPrimaryTensor: primary.as_ref() as &Tensor,
                secondaryTensor: secondary.as_ref() as &Tensor,
                name: name_ptr
            ];

            if tensor_ptr.is_null() {
                panic!("Failed to create addition tensor");
            } else {
                Retained::retain_autoreleased(tensor_ptr).unwrap()
            }
        }
    }

    /// Multiplies two tensors element-wise
    pub fn multiply(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let tensor_ptr: *mut Tensor = msg_send![
                self,
                multiplicationWithPrimaryTensor: primary.as_ref() as &Tensor,
                secondaryTensor: secondary.as_ref() as &Tensor,
                name: name_ptr
            ];

            if tensor_ptr.is_null() {
                panic!("Failed to create multiplication tensor");
            } else {
                Retained::retain_autoreleased(tensor_ptr).unwrap()
            }
        }
    }

    /// Subtracts one tensor from another element-wise
    pub fn subtract(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let tensor_ptr: *mut Tensor = msg_send![
                self,
                subtractionWithPrimaryTensor: primary.as_ref() as &Tensor,
                secondaryTensor: secondary.as_ref() as &Tensor,
                name: name_ptr
            ];

            if tensor_ptr.is_null() {
                panic!("Failed to create subtraction tensor");
            } else {
                Retained::retain_autoreleased(tensor_ptr).unwrap()
            }
        }
    }

    /// Divides one tensor by another element-wise
    pub fn divide(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let tensor_ptr: *mut Tensor = msg_send![
                self,
                divisionWithPrimaryTensor: primary.as_ref() as &Tensor,
                secondaryTensor: secondary.as_ref() as &Tensor,
                name: name_ptr
            ];

            if tensor_ptr.is_null() {
                panic!("Failed to create division tensor");
            } else {
                Retained::retain_autoreleased(tensor_ptr).unwrap()
            }
        }
    }

    /// Gets the placeholder tensors associated with the graph, returned as a Vec of retained Tensors.
    pub fn placeholder_tensors(&self) -> Vec<Retained<Tensor>> {
        unsafe {
            let array_ptr: *mut NSArray<Tensor> = msg_send![self, placeholderTensors];
            let array = Retained::retain_autoreleased(array_ptr).unwrap();

            let count = array.len();
            let mut vec = Vec::with_capacity(count);
            for i in 0..count {
                // objectAtIndex: returns a borrowed reference, so we need to retain it.
                let tensor_ptr: *mut Tensor = msg_send![&*array, objectAtIndex: i];
                // It's crucial to retain each object obtained from the array.
                let retained_tensor = Retained::retain_autoreleased(tensor_ptr).unwrap();
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
