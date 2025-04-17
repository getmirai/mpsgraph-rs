use objc2::rc::Retained;
use objc2::{extern_class, ClassType, msg_send};
use objc2::runtime::NSObject;
use objc2_foundation::{NSArray, NSMutableDictionary, NSObjectProtocol, NSString};
use std::collections::HashMap;

use crate::tensor::{Tensor, DataType};
use crate::shape::Shape;
use crate::tensor_data::TensorData;
use crate::operation::Operation;
use crate::executable::{
    Executable, 
    ExecutionDescriptor, 
    CompilationDescriptor
};
use crate::command_buffer::CommandBuffer;
use crate::device::Device;
use crate::shape::{ShapeExtensions, ShapeHelper};

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
    pub fn placeholder(
        &self,
        data_type: DataType,
        shape: &Shape,
    ) -> Option<Retained<Tensor>> {
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
                },
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
            let dictionary_ptr: *mut NSMutableDictionary<Tensor, TensorData> = msg_send![dictionary_class, dictionaryWithCapacity: feeds.len()];
            
            // Add entries to dictionary
            for (tensor, data) in feeds {
                let _: () = msg_send![dictionary_ptr, setObject: *data, forKey: *tensor];
            }
            
            // Create NSArray for output tensors
            let output_array = NSArray::from_slice(output_tensors);
            
            // Run the graph
            let results_ptr: *mut NSMutableDictionary<Tensor, TensorData> = match execution_descriptor {
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
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);
            
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
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);
            
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
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);
            
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
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);
            
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
    pub fn constant_scalar_with_shape<T: TensorDataScalar>(
        &self,
        value: T,
        data_type: DataType,
        shape: &Shape,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let value_f64 = value.to_f64();
            
            let tensor_ptr: *mut Tensor = match name {
                Some(name_str) => {
                    let name_ns = NSString::from_str(name_str);
                    msg_send![
                        self,
                        constantWithScalar: value_f64,
                        dataType: data_type as u64,
                        shape: shape,
                        name: &*name_ns
                    ]
                },
                None => {
                    msg_send![
                        self,
                        constantWithScalar: value_f64,
                        dataType: data_type as u64,
                        shape: shape,
                        name: std::ptr::null::<NSString>()
                    ]
                }
            };
            
            if tensor_ptr.is_null() {
                None
            } else {
                Retained::from_raw(tensor_ptr)
            }
        }
    }
    
    /// Creates a constant scalar tensor
    pub fn constant_scalar<T: TensorDataScalar>(
        &self,
        value: T,
        data_type: DataType,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let value_f64 = value.to_f64();
            
            let tensor_ptr: *mut Tensor = match name {
                Some(name_str) => {
                    let name_ns = NSString::from_str(name_str);
                    msg_send![
                        self,
                        constantWithScalar: value_f64,
                        dataType: data_type as u64,
                        name: &*name_ns
                    ]
                },
                None => {
                    msg_send![
                        self,
                        constantWithScalar: value_f64,
                        dataType: data_type as u64,
                        name: std::ptr::null::<NSString>()
                    ]
                }
            };
            
            if tensor_ptr.is_null() {
                None
            } else {
                Retained::from_raw(tensor_ptr)
            }
        }
    }
    
    /// Creates a constant tensor with array values and shape
    pub fn constant_with_shape<T: TensorDataScalar>(
        &self,
        values: &[T],
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
    pub fn constant<T: TensorDataScalar>(
        &self,
        values: &[T],
        shape_dimensions: &[usize],
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
        data: &TensorData,
        shape: Option<&Shape>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            match (shape, name) {
                (Some(s), Some(n)) => {
                    let name_ns = NSString::from_str(n);
                    let tensor_ptr: *mut Tensor = msg_send![
                        self, 
                        constantWithData: data,
                        shape: s,
                        name: &*name_ns
                    ];
                    if tensor_ptr.is_null() { None } else { Retained::from_raw(tensor_ptr) }
                },
                (Some(s), None) => {
                    let tensor_ptr: *mut Tensor = msg_send![
                        self, 
                        constantWithData: data,
                        shape: s,
                        name: std::ptr::null::<NSString>()
                    ];
                    if tensor_ptr.is_null() { None } else { Retained::from_raw(tensor_ptr) }
                },
                (None, Some(n)) => {
                    let name_ns = NSString::from_str(n);
                    let tensor_ptr: *mut Tensor = msg_send![
                        self, 
                        constantWithData: data,
                        shape: std::ptr::null::<Shape>(),
                        name: &*name_ns
                    ];
                    if tensor_ptr.is_null() { None } else { Retained::from_raw(tensor_ptr) }
                },
                (None, None) => {
                    let tensor_ptr: *mut Tensor = msg_send![
                        self, 
                        constantWithData: data,
                        shape: std::ptr::null::<Shape>(),
                        name: std::ptr::null::<NSString>()
                    ];
                    if tensor_ptr.is_null() { None } else { Retained::from_raw(tensor_ptr) }
                }
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
        device: &Device,
        feeds: &HashMap<&Tensor, &TensorData>,
        targets: &[&Tensor],
        descriptor: Option<&CompilationDescriptor>,
    ) -> Option<Retained<Executable>> {
        unsafe {
            // Create NSMutableDictionary for feeds
            let dictionary_class = NSMutableDictionary::<Tensor, TensorData>::class();
            let dictionary_ptr: *mut NSMutableDictionary<Tensor, TensorData> = msg_send![dictionary_class, dictionaryWithCapacity: feeds.len()];
            
            // Add entries to dictionary
            for (tensor, data) in feeds {
                let _: () = msg_send![dictionary_ptr, setObject: *data, forKey: *tensor];
            }
            
            // Create NSArray for target tensors
            let targets_array = NSArray::from_slice(targets);
            
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
        feeds: &HashMap<&Tensor, &TensorData>,
        targets: &[&Tensor],
        target_ops: &[&Operation],
        descriptor: Option<&CompilationDescriptor>,
    ) -> Option<Retained<Executable>> {
        unsafe {
            // Create NSMutableDictionary for feeds
            let dictionary_class = NSMutableDictionary::<Tensor, TensorData>::class();
            let dictionary_ptr: *mut NSMutableDictionary<Tensor, TensorData> = msg_send![dictionary_class, dictionaryWithCapacity: feeds.len()];
            
            // Add entries to dictionary
            for (tensor, data) in feeds {
                let _: () = msg_send![dictionary_ptr, setObject: *data, forKey: *tensor];
            }
            
            // Create NSArray for target tensors
            let targets_array = NSArray::from_slice(targets);
            
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
        unsafe {
            // Create NSMutableDictionary for feeds
            let dictionary_class = NSMutableDictionary::<Tensor, TensorData>::class();
            let dictionary_ptr: *mut NSMutableDictionary<Tensor, TensorData> = msg_send![dictionary_class, dictionaryWithCapacity: feeds.len()];
            
            // Add entries to dictionary
            for (tensor, data) in feeds {
                let _: () = msg_send![dictionary_ptr, setObject: *data, forKey: *tensor];
            }
            
            // Create NSArray for target tensors if provided
            let targets_array_ptr = match target_tensors {
                Some(tensors) => {
                    let array = NSArray::from_slice(tensors);
                    &*array as *const _
                },
                None => std::ptr::null(),
            };
            
            // Create NSArray for target operations if provided
            let ops_array_ptr = match target_operations {
                Some(ops) => {
                    let array = NSArray::from_slice(ops);
                    &*array as *const _
                },
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
        feeds: &HashMap<&Tensor, &TensorData>,
        target_operations: Option<&[&Operation]>,
        results_dict: &HashMap<&Tensor, &TensorData>,
        execution_descriptor: Option<&ExecutionDescriptor>,
    ) {
        unsafe {
            // Create NSMutableDictionary for feeds
            let feeds_dictionary_class = NSMutableDictionary::<Tensor, TensorData>::class();
            let feeds_ptr: *mut NSMutableDictionary<Tensor, TensorData> = msg_send![feeds_dictionary_class, dictionaryWithCapacity: feeds.len()];
            
            // Add entries to feeds dictionary
            for (tensor, data) in feeds {
                let _: () = msg_send![feeds_ptr, setObject: *data, forKey: *tensor];
            }
            
            // Create NSMutableDictionary for results
            let results_dictionary_class = NSMutableDictionary::<Tensor, TensorData>::class();
            let results_ptr: *mut NSMutableDictionary<Tensor, TensorData> = msg_send![results_dictionary_class, dictionaryWithCapacity: results_dict.len()];
            
            // Add entries to results dictionary
            for (tensor, data) in results_dict {
                let _: () = msg_send![results_ptr, setObject: *data, forKey: *tensor];
            }
            
            // Create NSArray for target operations if provided
            let ops_array_ptr = match target_operations {
                Some(ops) => {
                    let array = NSArray::from_slice(ops);
                    &*array as *const _
                },
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
            let dictionary_ptr: *mut NSMutableDictionary<Tensor, TensorData> = msg_send![dictionary_class, dictionaryWithCapacity: feeds.len()];
            
            // Add entries to dictionary
            for (tensor, data) in feeds {
                let _: () = msg_send![dictionary_ptr, setObject: *data, forKey: *tensor];
            }
            
            // Create NSArray for output tensors
            let output_array = NSArray::from_slice(output_tensors);
            
            // Run the graph on device
            let results_ptr: *mut NSMutableDictionary<Tensor, TensorData> = match execution_descriptor {
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
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);
            
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
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);
            
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
            let dictionary_ptr: *mut NSMutableDictionary<Tensor, TensorData> = msg_send![dictionary_class, dictionaryWithCapacity: feeds.len()];
            
            // Add entries to dictionary
            for (tensor, data) in feeds {
                let _: () = msg_send![dictionary_ptr, setObject: *data, forKey: *tensor];
            }
            
            // Create NSArray for output tensors
            let output_array = NSArray::from_slice(output_tensors);
            
            // Create NSArray for target operations
            let ops_array = NSArray::from_slice(target_operations);
            
            // Run the graph on device with operations
            let results_ptr: *mut NSMutableDictionary<Tensor, TensorData> = match execution_descriptor {
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