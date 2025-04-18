use objc2::rc::Retained;
use objc2::runtime::NSObject;
use objc2::{extern_class, msg_send, ClassType};
use objc2_foundation::{NSArray, NSData, NSMutableDictionary, NSObjectProtocol, NSString};
use std::collections::HashMap;

use crate::command_buffer::CommandBuffer;
use crate::device::Device;
use crate::executable::{CompilationDescriptor, Executable, ExecutionDescriptor};
use crate::operation::Operation;
use crate::shape::Shape;
use crate::shape::{ShapeExtensions, ShapeHelper};
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
            let obj: Retained<Self> = msg_send![class, new];
            obj
        }
    }

    /// Creates a placeholder tensor with the given data type and shape
    pub fn placeholder(&self, data_type: DataType, shape: &Shape) -> Option<Retained<Tensor>> {
        unsafe {
            // Use null for the name parameter
            let tensor_ptr: *mut Tensor = msg_send![
                self,
                placeholderWithShape: shape,
                dataType: data_type as u32,
                name: std::ptr::null::<NSString>()
            ];

            if tensor_ptr.is_null() {
                None
            } else {
                Retained::from_raw(tensor_ptr)
            }
        }
    }

    /// Creates a placeholder tensor with the given data type, shape, and name
    pub fn placeholder_with_name(
        &self,
        data_type: DataType,
        shape: &Shape,
        name: &str,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = NSString::from_str(name);

            let tensor_ptr: *mut Tensor = msg_send![
                self,
                placeholderWithShape: shape,
                dataType: data_type as u32,
                name: &*name_ns
            ];

            if tensor_ptr.is_null() {
                None
            } else {
                Retained::from_raw(tensor_ptr)
            }
        }
    }

    /// Performs matrix multiplication of two tensors
    pub fn matmul(
        &self,
        lhs: &Tensor,
        rhs: &Tensor,
        _transpose_lhs: bool,
        _transpose_rhs: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let result_ptr: *mut Tensor = match name {
                Some(name_str) => {
                    let name_ns = NSString::from_str(name_str);
                    msg_send![
                        self,
                        matrixMultiplicationWithPrimaryTensor: lhs,
                        secondaryTensor: rhs,
                        name: &*name_ns
                    ]
                }
                None => {
                    msg_send![
                        self,
                        matrixMultiplicationWithPrimaryTensor: lhs,
                        secondaryTensor: rhs,
                        name: std::ptr::null::<NSString>()
                    ]
                }
            };

            if result_ptr.is_null() {
                None
            } else {
                Retained::from_raw(result_ptr)
            }
        }
    }

    /// Execute the graph with feeds and get results
    ///
    /// - Parameters:
    ///   - feeds: A dictionary mapping input tensors to their values
    ///   - output_tensors: An array of tensors whose values should be computed
    ///
    /// - Returns: A dictionary mapping output tensors to their computed values
    pub fn run_with_feeds(
        &self,
        feeds: &HashMap<&Tensor, &TensorData>,
        output_tensors: &[&Tensor],
    ) -> HashMap<Retained<Tensor>, Retained<TensorData>> {
        self.run_with_feeds_and_descriptor(feeds, output_tensors, None)
    }

    /// Execute the graph with feeds and execution descriptor
    ///
    /// - Parameters:
    ///   - feeds: A dictionary mapping input tensors to their values
    ///   - output_tensors: An array of tensors whose values should be computed
    ///   - execution_descriptor: Optional descriptor controlling execution options
    ///
    /// - Returns: A dictionary mapping output tensors to their computed values
    pub fn run_with_feeds_and_descriptor(
        &self,
        feeds: &HashMap<&Tensor, &TensorData>,
        output_tensors: &[&Tensor],
        execution_descriptor: Option<&ExecutionDescriptor>,
    ) -> HashMap<Retained<Tensor>, Retained<TensorData>> {
        unsafe {
            // Create NSMutableDictionary for feeds
            let dictionary_class = NSMutableDictionary::<Tensor, TensorData>::class();
            let dictionary_ptr: *mut NSMutableDictionary<Tensor, TensorData> =
                msg_send![dictionary_class, dictionaryWithCapacity: feeds.len()];

            // Add entries to dictionary
            for (tensor, data) in feeds {
                let _: () = msg_send![dictionary_ptr, setObject: *data, forKey: *tensor];
            }

            // Create NSArray for output tensors
            let output_array = NSArray::from_slice(output_tensors);

            // Run the graph
            let results_ptr: *mut NSMutableDictionary<Tensor, TensorData> =
                match execution_descriptor {
                    Some(desc) => {
                        msg_send![
                            self,
                            runWithFeeds: dictionary_ptr,
                            targetTensors: &*output_array,
                            targetOperations: std::ptr::null::<NSArray<Operation>>(),
                            executionDescriptor: desc
                        ]
                    }
                    None => {
                        msg_send![
                            self,
                            runWithFeeds: dictionary_ptr,
                            targetTensors: &*output_array,
                            targetOperations: std::ptr::null::<NSArray<Operation>>(),
                            executionDescriptor: std::ptr::null::<ExecutionDescriptor>()
                        ]
                    }
                };

            let _results_dict = Retained::from_raw(results_ptr).unwrap();

            // Convert NSDictionary to HashMap
            let mut result = HashMap::new();
            let keys: *mut NSArray<Tensor> = msg_send![results_ptr, allKeys];

            let keys_count: usize = msg_send![keys, count];

            for i in 0..keys_count {
                let key_ptr: *const Tensor = msg_send![keys, objectAtIndex: i];
                let key = Retained::from_raw(key_ptr as *mut _).unwrap();

                let value_ptr: *const TensorData = msg_send![results_ptr, objectForKey: key_ptr];
                let value = Retained::from_raw(value_ptr as *mut _).unwrap();

                result.insert(key, value);
            }

            result
        }
    }

    /// Creates a constant tensor with the given data
    /// Adds two tensors element-wise
    pub fn add(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let tensor_ptr: *mut Tensor = msg_send![
                self,
                additionWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name_ptr
            ];

            if tensor_ptr.is_null() {
                None
            } else {
                Retained::from_raw(tensor_ptr)
            }
        }
    }

    /// Multiplies two tensors element-wise
    pub fn multiply(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let tensor_ptr: *mut Tensor = msg_send![
                self,
                multiplicationWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name_ptr
            ];

            if tensor_ptr.is_null() {
                None
            } else {
                Retained::from_raw(tensor_ptr)
            }
        }
    }

    /// Subtracts one tensor from another element-wise
    pub fn subtract(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let tensor_ptr: *mut Tensor = msg_send![
                self,
                subtractionWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name_ptr
            ];

            if tensor_ptr.is_null() {
                None
            } else {
                Retained::from_raw(tensor_ptr)
            }
        }
    }

    /// Divides one tensor by another element-wise
    pub fn divide(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let tensor_ptr: *mut Tensor = msg_send![
                self,
                divisionWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name_ptr
            ];

            if tensor_ptr.is_null() {
                None
            } else {
                Retained::from_raw(tensor_ptr)
            }
        }
    }

    /// Creates a constant tensor with scalar value and shape
    pub fn constant_scalar_with_shape(
        &self,
        value: f64,
        data_type: DataType,
        shape: &Shape,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let tensor_ptr: *mut Tensor = msg_send![
                self,
                constantWithScalar: value,
                dataType: data_type as u64,
                shape: shape,
                name: name_ptr
            ];

            if tensor_ptr.is_null() {
                None
            } else {
                Retained::from_raw(tensor_ptr)
            }
        }
    }

    /// Creates a constant scalar tensor
    pub fn constant_scalar(
        &self,
        value: f64,
        data_type: DataType,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let tensor_ptr: *mut Tensor = msg_send![
                self,
                constantWithScalar: value,
                dataType: data_type as u64,
                name: name_ptr
            ];

            if tensor_ptr.is_null() {
                None
            } else {
                Retained::from_raw(tensor_ptr)
            }
        }
    }

    /// Creates a constant tensor with array values and shape
    pub fn constant_with_shape(
        &self,
        values: &[f64],
        data_type: DataType,
        shape: &Shape,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        // Create TensorData from values
        let dims = shape.dimensions();
        let data = TensorData::from_bytes(values, &dims, data_type);

        // Create constant with data
        self.constant_with_data(&data, Some(shape), name)
    }

    /// Creates a constant tensor with array values, inferring shape from array dimensions
    pub fn constant(
        &self,
        values: &[f64],
        shape_dimensions: &[i64],
        data_type: DataType,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        // Create shape
        let shape = ShapeHelper::from_dimensions(shape_dimensions);

        // Create constant with shape
        self.constant_with_shape(values, data_type, &shape, name)
    }

    /// Creates a constant tensor with the given data
    pub fn constant_with_data(
        &self,
        data: &Retained<TensorData>,
        shape: Option<&Shape>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);
            
            let shape_ptr = shape.map_or(std::ptr::null(), |s| s as *const _);
            
            let tensor_ptr: *mut Tensor = msg_send![
                self,
                constantWithData: &**data,
                shape: shape_ptr,
                name: name_ptr
            ];
            
            if tensor_ptr.is_null() {
                None
            } else {
                Retained::from_raw(tensor_ptr)
            }
        }
    }

    /// Creates a constant tensor with the given raw bytes
    pub fn constant_with_raw_data(
        &self,
        data: &[u8],
        shape: &Shape,
        data_type: DataType,
    ) -> Option<Retained<Tensor>> {
        // Create TensorData from the raw bytes
        let tensor_data = unsafe {
            // Get the default Metal device
            let device = metal::Device::system_default().expect("No Metal device found");

            // Create NSData with our data
            let ns_data = NSData::with_bytes(data);

            // Create MPSGraphDevice from MTLDevice
            let mps_device = Device::with_device(&device);

            // Create the TensorData with NSData
            let tensor_data_class = TensorData::class();
            let data_type_val = data_type as u32;

            let alloc: *mut TensorData = msg_send![tensor_data_class, alloc];
            let tensor_data: *mut TensorData = msg_send![alloc,
                initWithDevice:&*mps_device,
                data:&*ns_data,
                shape:shape,
                dataType:data_type_val
            ];

            if tensor_data.is_null() {
                return None;
            }

            Retained::from_raw(tensor_data).unwrap()
        };

        // Use the existing constant_with_data method with the created TensorData
        self.constant_with_data(&tensor_data, Some(shape), None)
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
        feeds: &HashMap<Retained<Tensor>, Retained<ShapedType>>,
        targets: &[&Retained<Tensor>],
        descriptor: Option<&CompilationDescriptor>,
    ) -> Option<Retained<Executable>> {
        unsafe {
            // Create NSMutableDictionary for feeds
            let dictionary_class = NSMutableDictionary::<Tensor, ShapedType>::class();
            let dictionary_ptr: *mut NSMutableDictionary<Tensor, ShapedType> =
                msg_send![dictionary_class, dictionaryWithCapacity: feeds.len()];

            // Add entries to dictionary
            for (tensor, shaped_type) in feeds {
                // Get raw pointers to the inner Objective-C objects
                let tensor_ptr = tensor.as_ref() as *const Tensor;
                let shape_ptr = shaped_type.as_ref() as *const ShapedType;

                // Create temporary references for message sending
                let tensor_ref: &Tensor = &*tensor_ptr;
                let shape_ref: &ShapedType = &*shape_ptr;

                let _: () = msg_send![dictionary_ptr, setObject: shape_ref, forKey: tensor_ref];
            }

            // Create NSArray for target tensors
            // Need to convert &[&Retained<Tensor>] to a slice of &Tensor for NSArray::from_slice
            let targets_refs: Vec<&Tensor> = targets
                .iter()
                .map(|retained_tensor| retained_tensor.as_ref())
                .collect();
            let targets_array = NSArray::from_slice(&targets_refs);

            // Get descriptor pointer if provided
            let desc_ptr = match descriptor {
                Some(desc) => desc as *const _,
                None => std::ptr::null(),
            };

            // Compile the graph
            let executable_ptr: *mut Executable = msg_send![
                self,
                compileWithDevice: device,
                feeds: dictionary_ptr,
                targetTensors: &*targets_array,
                targetOperations: std::ptr::null::<NSArray<Operation>>(),
                compilationDescriptor: desc_ptr
            ];

            if executable_ptr.is_null() {
                None
            } else {
                Retained::from_raw(executable_ptr)
            }
        }
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
        feeds: &HashMap<&Retained<Tensor>, &Retained<ShapedType>>,
        targets: &[&Retained<Tensor>],
        target_ops: &[&Operation],
        descriptor: Option<&CompilationDescriptor>,
    ) -> Option<Retained<Executable>> {
        unsafe {
            // Create NSMutableDictionary for feeds
            let dictionary_class = NSMutableDictionary::<Tensor, ShapedType>::class();
            let dictionary_ptr: *mut NSMutableDictionary<Tensor, ShapedType> =
                msg_send![dictionary_class, dictionaryWithCapacity: feeds.len()];

            // Add entries to dictionary
            for (tensor, shaped_type) in feeds {
                // Get raw pointers to the inner Objective-C objects
                let tensor_ptr = tensor.as_ref() as *const Tensor;
                let shape_ptr = shaped_type.as_ref() as *const ShapedType;

                // Create temporary references for message sending
                let tensor_ref: &Tensor = &*tensor_ptr;
                let shape_ref: &ShapedType = &*shape_ptr;

                let _: () = msg_send![dictionary_ptr, setObject: shape_ref, forKey: tensor_ref];
            }

            // Create NSArray for target tensors
            // Need to convert &[&Retained<Tensor>] to a slice of &Tensor for NSArray::from_slice
            let targets_refs: Vec<&Tensor> = targets
                .iter()
                .map(|retained_tensor| retained_tensor.as_ref())
                .collect();
            let targets_array = NSArray::from_slice(&targets_refs);

            // Create NSArray for target operations
            let ops_array = NSArray::from_slice(target_ops);

            // Get descriptor pointer if provided
            let desc_ptr = match descriptor {
                Some(desc) => desc as *const _,
                None => std::ptr::null(),
            };

            // Compile the graph
            let executable_ptr: *mut Executable = msg_send![
                self,
                compileWithDevice: device,
                feeds: dictionary_ptr,
                targetTensors: &*targets_array,
                targetOperations: &*ops_array,
                compilationDescriptor: desc_ptr
            ];

            if executable_ptr.is_null() {
                None
            } else {
                Retained::from_raw(executable_ptr)
            }
        }
    }

    /// Compiles the graph against a given set of feeds, targets, and target operations
    ///
    /// This version accepts Retained references, which is the preferred approach.
    ///
    /// - Parameters:
    ///   - device: Metal device to compile for
    ///   - feeds: A dictionary mapping input tensors to their values
    ///   - targets: An array of tensors whose values should be computed
    ///   - target_ops: An array of operations to be completed
    ///   - descriptor: Optional compilation descriptor
    ///
    /// - Returns: A compiled executable

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
        feeds: &HashMap<&Retained<Tensor>, &Retained<TensorData>>,
        target_tensors: Option<&[&Retained<Tensor>]>,
        target_operations: Option<&[&Operation]>,
        execution_descriptor: Option<&ExecutionDescriptor>,
    ) -> HashMap<Retained<Tensor>, Retained<TensorData>> {
        unsafe {
            // Create NSMutableDictionary for feeds
            let dictionary_class = NSMutableDictionary::<Tensor, TensorData>::class();
            let dictionary_ptr: *mut NSMutableDictionary<Tensor, TensorData> =
                msg_send![dictionary_class, dictionaryWithCapacity: feeds.len()];

            // Add entries to dictionary
            for (tensor, data) in feeds {
                // Get raw pointers to the inner Objective-C objects
                let tensor_ptr = tensor.as_ref() as *const Tensor;
                let data_ptr = data.as_ref() as *const TensorData;

                // Create temporary references for message sending
                let tensor_ref: &Tensor = &*tensor_ptr;
                let data_ref: &TensorData = &*data_ptr;

                let _: () = msg_send![dictionary_ptr, setObject: data_ref, forKey: tensor_ref];
            }

            // Create NSArray for target tensors if provided
            let targets_array_ptr = match target_tensors {
                Some(tensors) => {
                    // Need to convert &[&Retained<Tensor>] to a slice of &Tensor for NSArray::from_slice
                    let targets_refs: Vec<&Tensor> = tensors
                        .iter()
                        .map(|retained_tensor| retained_tensor.as_ref())
                        .collect();
                    let array = NSArray::from_slice(&targets_refs);
                    &*array as *const _
                }
                None => std::ptr::null(),
            };

            // Create NSArray for target operations if provided
            let ops_array_ptr = match target_operations {
                Some(ops) => {
                    let array = NSArray::from_slice(ops);
                    &*array as *const _
                }
                None => std::ptr::null(),
            };

            // Get descriptor pointer if provided
            let desc_ptr = match execution_descriptor {
                Some(desc) => desc as *const _,
                None => std::ptr::null(),
            };

            // Encode the graph to the command buffer
            let results_ptr: *mut NSMutableDictionary<Tensor, TensorData> = msg_send![
                self,
                encodeToCommandBuffer: command_buffer,
                feeds: dictionary_ptr,
                targetTensors: targets_array_ptr,
                targetOperations: ops_array_ptr,
                executionDescriptor: desc_ptr
            ];

            // Convert NSMutableDictionary to HashMap
            let _results_dict = Retained::from_raw(results_ptr).unwrap();

            let mut result = HashMap::new();
            let keys: *mut NSArray<Tensor> = msg_send![results_ptr, allKeys];

            let keys_count: usize = msg_send![keys, count];

            for i in 0..keys_count {
                let key_ptr: *const Tensor = msg_send![keys, objectAtIndex: i];
                let key = Retained::from_raw(key_ptr as *mut _).unwrap();

                let value_ptr: *const TensorData = msg_send![results_ptr, objectForKey: key_ptr];
                let value = Retained::from_raw(value_ptr as *mut _).unwrap();

                result.insert(key, value);
            }

            result
        }
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
        feeds: &HashMap<&Retained<Tensor>, &Retained<TensorData>>,
        target_operations: Option<&[&Operation]>,
        results_dict: &HashMap<&Retained<Tensor>, &Retained<TensorData>>,
        execution_descriptor: Option<&ExecutionDescriptor>,
    ) {
        unsafe {
            // Create NSMutableDictionary for feeds
            let feeds_dictionary_class = NSMutableDictionary::<Tensor, TensorData>::class();
            let feeds_ptr: *mut NSMutableDictionary<Tensor, TensorData> =
                msg_send![feeds_dictionary_class, dictionaryWithCapacity: feeds.len()];

            // Add entries to feeds dictionary
            for (tensor, data) in feeds {
                // Get raw pointers to the inner Objective-C objects
                let tensor_ptr = tensor.as_ref() as *const Tensor;
                let data_ptr = data.as_ref() as *const TensorData;

                // Create temporary references for message sending
                let tensor_ref: &Tensor = &*tensor_ptr;
                let data_ref: &TensorData = &*data_ptr;

                let _: () = msg_send![feeds_ptr, setObject: data_ref, forKey: tensor_ref];
            }

            // Create NSMutableDictionary for results
            let results_dictionary_class = NSMutableDictionary::<Tensor, TensorData>::class();
            let results_ptr: *mut NSMutableDictionary<Tensor, TensorData> =
                msg_send![results_dictionary_class, dictionaryWithCapacity: results_dict.len()];

            // Add entries to results dictionary
            for (tensor, data) in results_dict {
                // Get raw pointers to the inner Objective-C objects
                let tensor_ptr = tensor.as_ref() as *const Tensor;
                let data_ptr = data.as_ref() as *const TensorData;

                // Create temporary references for message sending
                let tensor_ref: &Tensor = &*tensor_ptr;
                let data_ref: &TensorData = &*data_ptr;

                let _: () = msg_send![results_ptr, setObject: data_ref, forKey: tensor_ref];
            }

            // Create NSArray for target operations if provided
            let ops_array_ptr = match target_operations {
                Some(ops) => {
                    let array = NSArray::from_slice(ops);
                    &*array as *const _
                }
                None => std::ptr::null(),
            };

            // Get descriptor pointer if provided
            let desc_ptr = match execution_descriptor {
                Some(desc) => desc as *const _,
                None => std::ptr::null(),
            };

            // Encode the graph to the command buffer with results
            let _: () = msg_send![
                self,
                encodeToCommandBuffer: command_buffer,
                feeds: feeds_ptr,
                targetOperations: ops_array_ptr,
                resultsDictionary: results_ptr,
                executionDescriptor: desc_ptr
            ];
        }
    }

    /// Execute the graph with feeds on a specific device
    ///
    /// - Parameters:
    ///   - device: Metal device to run on
    ///   - feeds: A dictionary mapping input tensors to their values
    ///   - output_tensors: Tensors whose values should be computed
    ///   - execution_descriptor: Optional execution descriptor
    ///
    /// - Returns: A dictionary mapping output tensors to their computed values
    pub fn run_with_feeds_on_device(
        &self,
        device: &Device,
        feeds: &HashMap<&Tensor, &TensorData>,
        output_tensors: &[&Tensor],
        execution_descriptor: Option<&ExecutionDescriptor>,
    ) -> HashMap<Retained<Tensor>, Retained<TensorData>> {
        unsafe {
            // Create NSMutableDictionary for feeds
            let dictionary_class = NSMutableDictionary::<Tensor, TensorData>::class();
            let dictionary_ptr: *mut NSMutableDictionary<Tensor, TensorData> =
                msg_send![dictionary_class, dictionaryWithCapacity: feeds.len()];

            // Add entries to dictionary
            for (tensor, data) in feeds {
                let _: () = msg_send![dictionary_ptr, setObject: *data, forKey: *tensor];
            }

            // Create NSArray for output tensors
            let output_array = NSArray::from_slice(output_tensors);

            // Run the graph on device
            let results_ptr: *mut NSMutableDictionary<Tensor, TensorData> =
                match execution_descriptor {
                    Some(desc) => {
                        msg_send![
                            self,
                            runWithFeeds: dictionary_ptr,
                            targetTensors: &*output_array,
                            targetOperations: std::ptr::null::<NSArray<Operation>>(),
                            onDevice: device,
                            executionDescriptor: desc
                        ]
                    }
                    None => {
                        msg_send![
                            self,
                            runWithFeeds: dictionary_ptr,
                            targetTensors: &*output_array,
                            targetOperations: std::ptr::null::<NSArray<Operation>>(),
                            onDevice: device,
                            executionDescriptor: std::ptr::null::<ExecutionDescriptor>()
                        ]
                    }
                };

            let _results_dict = Retained::from_raw(results_ptr).unwrap();

            // Convert NSDictionary to HashMap
            let mut result = HashMap::new();
            let keys: *mut NSArray<Tensor> = msg_send![results_ptr, allKeys];

            let keys_count: usize = msg_send![keys, count];

            for i in 0..keys_count {
                let key_ptr: *const Tensor = msg_send![keys, objectAtIndex: i];
                let key = Retained::from_raw(key_ptr as *mut _).unwrap();

                let value_ptr: *const TensorData = msg_send![results_ptr, objectForKey: key_ptr];
                let value = Retained::from_raw(value_ptr as *mut _).unwrap();

                result.insert(key, value);
            }

            result
        }
    }

    /// Execute the graph with feeds and target operations on a specific device
    ///
    /// - Parameters:
    ///   - device: Metal device to run on
    ///   - feeds: A dictionary mapping input tensors to their values
    ///   - output_tensors: Tensors whose values should be computed
    ///   - target_operations: Operations to be executed
    ///   - execution_descriptor: Optional execution descriptor
    ///
    /// - Returns: A dictionary mapping output tensors to their computed values
    /// Creates a tensor with random uniform values
    pub fn random_uniform(
        &self,
        shape: &Shape,
        min: f32,
        max: f32,
        data_type: DataType,
        seed: u32,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let tensor_ptr: *mut Tensor = msg_send![
                self,
                randomUniformTensorWithShape: shape,
                min: min,
                max: max,
                dataType: data_type as u64,
                seed: seed,
                name: name_ptr
            ];

            if tensor_ptr.is_null() {
                None
            } else {
                Retained::from_raw(tensor_ptr)
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
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let tensor_ptr: *mut Tensor = msg_send![
                self,
                randomNormalTensorWithShape: shape,
                mean: mean,
                standardDeviation: std_dev,
                dataType: data_type as u64,
                seed: seed,
                name: name_ptr
            ];

            if tensor_ptr.is_null() {
                None
            } else {
                Retained::from_raw(tensor_ptr)
            }
        }
    }

    pub fn run_with_feeds_and_ops_on_device(
        &self,
        device: &Device,
        feeds: &HashMap<&Tensor, &TensorData>,
        output_tensors: &[&Tensor],
        target_operations: &[&Operation],
        execution_descriptor: Option<&ExecutionDescriptor>,
    ) -> HashMap<Retained<Tensor>, Retained<TensorData>> {
        unsafe {
            // Create NSMutableDictionary for feeds
            let dictionary_class = NSMutableDictionary::<Tensor, TensorData>::class();
            let dictionary_ptr: *mut NSMutableDictionary<Tensor, TensorData> =
                msg_send![dictionary_class, dictionaryWithCapacity: feeds.len()];

            // Add entries to dictionary
            for (tensor, data) in feeds {
                let _: () = msg_send![dictionary_ptr, setObject: *data, forKey: *tensor];
            }

            // Create NSArray for output tensors
            let output_array = NSArray::from_slice(output_tensors);

            // Create NSArray for target operations
            let ops_array = NSArray::from_slice(target_operations);

            // Run the graph on device with operations
            let results_ptr: *mut NSMutableDictionary<Tensor, TensorData> =
                match execution_descriptor {
                    Some(desc) => {
                        msg_send![
                            self,
                            runWithFeeds: dictionary_ptr,
                            targetTensors: &*output_array,
                            targetOperations: &*ops_array,
                            onDevice: device,
                            executionDescriptor: desc
                        ]
                    }
                    None => {
                        msg_send![
                            self,
                            runWithFeeds: dictionary_ptr,
                            targetTensors: &*output_array,
                            targetOperations: &*ops_array,
                            onDevice: device,
                            executionDescriptor: std::ptr::null::<ExecutionDescriptor>()
                        ]
                    }
                };

            let _results_dict = Retained::from_raw(results_ptr).unwrap();

            // Convert NSDictionary to HashMap
            let mut result = HashMap::new();
            let keys: *mut NSArray<Tensor> = msg_send![results_ptr, allKeys];

            let keys_count: usize = msg_send![keys, count];

            for i in 0..keys_count {
                let key_ptr: *const Tensor = msg_send![keys, objectAtIndex: i];
                let key = Retained::from_raw(key_ptr as *mut _).unwrap();

                let value_ptr: *const TensorData = msg_send![results_ptr, objectForKey: key_ptr];
                let value = Retained::from_raw(value_ptr as *mut _).unwrap();

                result.insert(key, value);
            }

            result
        }
    }
}

// Implement CustomDefault for Graph
impl crate::CustomDefault for Graph {
    fn custom_default() -> Retained<Self> {
        Self::new()
    }
}
