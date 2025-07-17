use objc2::rc::{autoreleasepool, Allocated, Retained};
use objc2::runtime::NSObject;
use objc2::{extern_class, extern_conformance, extern_methods, msg_send, ClassType};
use objc2_foundation::{
    NSArray, NSData, NSDictionary, NSMutableDictionary, NSObjectProtocol, NSString,
};
use std::collections::HashMap;

use super::{GraphOptions, TensorShapedTypeHashMap};
use crate::command_buffer::CommandBuffer;
use crate::device::Device;
use crate::executable::{CompilationDescriptor, Executable, ExecutionDescriptor};
use crate::operation::Operation;
use crate::GraphObject;
use crate::Shape;
use crate::ShapedType;
use crate::{DataType, Tensor, TensorData};

extern_class!(
    /// The optimized representation of a compute graph of operations and tensors.
    ///
    /// An MPSGraph is a symbolic representation of operations to be utilized to execute compute graphs on a device.
    ///
    /// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph?language=objc)
    #[unsafe(super(GraphObject, NSObject))]
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[name = "MPSGraph"]
    pub struct Graph;
);

extern_conformance!(
    unsafe impl NSObjectProtocol for Graph {}
);

impl Graph {
    extern_methods!(
        /// Options for the graph.
        ///
        /// The default value is `MPSGraphOptionsDefault`.
        #[unsafe(method(options))]
        #[unsafe(method_family = none)]
        pub unsafe fn options(&self) -> GraphOptions;

        /// Setter for [`options`][Self::options].
        #[unsafe(method(setOptions:))]
        #[unsafe(method_family = none)]
        pub unsafe fn setOptions(&self, options: GraphOptions);

        /// Creates a new graph to insert nodes in.
        #[unsafe(method(new))]
        #[unsafe(method_family = new)]
        pub unsafe fn new() -> Retained<Self>;

        /// Initialize an MPSGraph to insert nodes in.
        #[unsafe(method(init))]
        #[unsafe(method_family = init)]
        pub unsafe fn init(this: Allocated<Self>) -> Retained<Self>;

        #[cfg(all(
            feature = "MPSGraphOperation",
            feature = "MPSGraphTensor",
            feature = "MPSGraphTensorData"
        ))]
        /// Runs the graph for the given feeds and returns the target tensor values, ensuring all target operations also executed.
        ///
        /// This call blocks until execution has completed.
        ///
        /// - Parameters:
        /// - feeds: Feeds dictionary for the placeholder tensors.
        /// - targetTensors: Tensors for which the caller wishes MPSGraphTensorData to be returned.
        /// - targetOperations: Operations to be completed at the end of the run.
        /// - Returns: A valid MPSGraphTensor : MPSGraphTensorData dictionary with results synchronized to the CPU memory.
        #[unsafe(method(runWithFeeds:targetTensors:targetOperations:))]
        #[unsafe(method_family = none)]
        pub unsafe fn runWithFeeds_targetTensors_targetOperations(
            &self,
            feeds: &MPSGraphTensorDataDictionary,
            target_tensors: &NSArray<MPSGraphTensor>,
            target_operations: Option<&NSArray<MPSGraphOperation>>,
        ) -> Retained<MPSGraphTensorDataDictionary>;

        #[cfg(all(
            feature = "MPSGraphOperation",
            feature = "MPSGraphTensor",
            feature = "MPSGraphTensorData"
        ))]
        /// Runs the graph for the given feeds and returns the target tensor values, ensuring all target operations also executed.
        ///
        /// This call blocks until execution has completed.
        ///
        /// - Parameters:
        /// - commandQueue: CommandQueue passed to exectute the graph on.
        /// - feeds: Feeds dictionary for the placeholder tensors.
        /// - targetTensors: Tensors for which the caller wishes MPSGraphTensorData to be returned.
        /// - targetOperations: Operations to be completed at the end of the run.
        /// - Returns: A valid MPSGraphTensor : MPSGraphTensorData dictionary with results synchronized to the CPU memory.
        #[unsafe(method(runWithMTLCommandQueue:feeds:targetTensors:targetOperations:))]
        #[unsafe(method_family = none)]
        pub unsafe fn runWithMTLCommandQueue_feeds_targetTensors_targetOperations(
            &self,
            command_queue: &ProtocolObject<dyn MTLCommandQueue>,
            feeds: &MPSGraphTensorDataDictionary,
            target_tensors: &NSArray<MPSGraphTensor>,
            target_operations: Option<&NSArray<MPSGraphOperation>>,
        ) -> Retained<MPSGraphTensorDataDictionary>;

        #[cfg(all(
            feature = "MPSGraphOperation",
            feature = "MPSGraphTensor",
            feature = "MPSGraphTensorData"
        ))]
        /// Runs the graph for the given feeds and returns the target tensor values in the results dictionary provided by the user.
        ///
        /// It also ensures all target operations also executed. This call blocks until execution has completed.
        ///
        /// - Parameters:
        /// - commandQueue: CommandQueue passed to exectute the graph on.
        /// - feeds: Feeds dictionary for the placeholder tensors.
        /// - targetOperations: Operations to be completed at the end of the run.
        /// - resultsDictionary: MPSGraphTensors dictionary passed by user, these will be filled with graph output data.
        #[unsafe(method(runWithMTLCommandQueue:feeds:targetOperations:resultsDictionary:))]
        #[unsafe(method_family = none)]
        pub unsafe fn runWithMTLCommandQueue_feeds_targetOperations_resultsDictionary(
            &self,
            command_queue: &ProtocolObject<dyn MTLCommandQueue>,
            feeds: &MPSGraphTensorDataDictionary,
            target_operations: Option<&NSArray<MPSGraphOperation>>,
            results_dictionary: &MPSGraphTensorDataDictionary,
        );

        #[cfg(all(
            feature = "MPSGraphOperation",
            feature = "MPSGraphTensor",
            feature = "MPSGraphTensorData"
        ))]
        /// Runs the graph for the given feeds and returns the target tensor values, ensuring all target operations also executed.
        ///
        /// This call is asynchronous and will return immediately if a completionHandler is set.
        ///
        /// - Parameters:
        /// - feeds: Feeds dictionary for the placeholder tensors.
        /// - targetTensors: Tensors for which the caller wishes MPSGraphTensorData to be returned.
        /// - targetOperations: Operations to be completed at the end of the run.
        /// - executionDescriptor: ExecutionDescriptor to be passed in and used.
        /// - Returns: A valid MPSGraphTensor : MPSGraphTensorData dictionary with results synchronized to the CPU memory.
        #[unsafe(method(runAsyncWithFeeds:targetTensors:targetOperations:executionDescriptor:))]
        #[unsafe(method_family = none)]
        pub unsafe fn runAsyncWithFeeds_targetTensors_targetOperations_executionDescriptor(
            &self,
            feeds: &MPSGraphTensorDataDictionary,
            target_tensors: &NSArray<MPSGraphTensor>,
            target_operations: Option<&NSArray<MPSGraphOperation>>,
            execution_descriptor: Option<&MPSGraphExecutionDescriptor>,
        ) -> Retained<MPSGraphTensorDataDictionary>;

        #[cfg(all(
            feature = "MPSGraphOperation",
            feature = "MPSGraphTensor",
            feature = "MPSGraphTensorData"
        ))]
        /// Runs the graph for the given feeds and returns the target tensor values, ensuring all target operations also executed.
        ///
        /// This call is asynchronous and will return immediately if a completionHandler is set.
        ///
        /// - Parameters:
        /// - commandQueue: CommandQueue passed to exectute the graph on.
        /// - feeds: Feeds dictionary for the placeholder tensors.
        /// - targetTensors: Tensors for which the caller wishes MPSGraphTensorData to be returned.
        /// - targetOperations: Operations to be completed at the end of the run.
        /// - executionDescriptor: ExecutionDescriptor to be passed in and used.
        /// - Returns: A valid MPSGraphTensor : MPSGraphTensorData dictionary with results synchronized to the CPU memory if MPSGraphOptionsSynchronizeResults set.
        #[unsafe(method(runAsyncWithMTLCommandQueue:feeds:targetTensors:targetOperations:executionDescriptor:))]
        #[unsafe(method_family = none)]
        pub unsafe fn runAsyncWithMTLCommandQueue_feeds_targetTensors_targetOperations_executionDescriptor(
            &self,
            command_queue: &ProtocolObject<dyn MTLCommandQueue>,
            feeds: &MPSGraphTensorDataDictionary,
            target_tensors: &NSArray<MPSGraphTensor>,
            target_operations: Option<&NSArray<MPSGraphOperation>>,
            execution_descriptor: Option<&MPSGraphExecutionDescriptor>,
        ) -> Retained<MPSGraphTensorDataDictionary>;

        #[cfg(all(
            feature = "MPSGraphOperation",
            feature = "MPSGraphTensor",
            feature = "MPSGraphTensorData"
        ))]
        /// Encodes the graph for the given feeds to returns the target tensor values in the results dictionary provided by the user.
        ///
        /// It ensures all target operations also executed. This call is asynchronous and will return immediately if a completionHandler is set.
        ///
        /// - Parameters:
        /// - commandQueue: CommandQueue passed to exectute the graph on.
        /// - feeds: Feeds dictionary for the placeholder tensors.
        /// - targetOperations: Operations to be completed at the end of the run.
        /// - resultsDictionary: MPSGraphTensors dictionary passed by user, these will be filled with graph output data.
        /// - executionDescriptor: ExecutionDescriptor to be passed in and used.
        #[unsafe(method(runAsyncWithMTLCommandQueue:feeds:targetOperations:resultsDictionary:executionDescriptor:))]
        #[unsafe(method_family = none)]
        pub unsafe fn runAsyncWithMTLCommandQueue_feeds_targetOperations_resultsDictionary_executionDescriptor(
            &self,
            command_queue: &ProtocolObject<dyn MTLCommandQueue>,
            feeds: &MPSGraphTensorDataDictionary,
            target_operations: Option<&NSArray<MPSGraphOperation>>,
            results_dictionary: &MPSGraphTensorDataDictionary,
            execution_descriptor: Option<&MPSGraphExecutionDescriptor>,
        );

        #[cfg(all(
            feature = "MPSGraphOperation",
            feature = "MPSGraphTensor",
            feature = "MPSGraphTensorData",
            feature = "objc2-metal-performance-shaders"
        ))]
        /// Encodes the graph for the given feeds to returns the target tensor values, ensuring all target operations also executed.
        ///
        /// This call is asynchronous and will return immediately if a completionHandler is set.
        ///
        /// - Parameters:
        /// - commandBuffer: commandBuffer passed to exectute the graph on, it is an MPSCommandBuffer, commitAndContinue might be called, please don't rely on underlying MTLCommandBuffer to remain uncommitted.
        /// - feeds: Feeds dictionary for the placeholder tensors.
        /// - targetTensors: Tensors for which the caller wishes MPSGraphTensorData to be returned.
        /// - targetOperations: Operations to be completed at the end of the run.
        /// - executionDescriptor: ExecutionDescriptor to be passed in and used.
        /// - Returns: A valid MPSGraphTensor : MPSGraphTensorData dictionary with results synchronized to the CPU memory if MPSGraphOptionsSynchronizeResults set.
        #[unsafe(method(encodeToCommandBuffer:feeds:targetTensors:targetOperations:executionDescriptor:))]
        #[unsafe(method_family = none)]
        pub unsafe fn encodeToCommandBuffer_feeds_targetTensors_targetOperations_executionDescriptor(
            &self,
            command_buffer: &MPSCommandBuffer,
            feeds: &MPSGraphTensorDataDictionary,
            target_tensors: &NSArray<MPSGraphTensor>,
            target_operations: Option<&NSArray<MPSGraphOperation>>,
            execution_descriptor: Option<&MPSGraphExecutionDescriptor>,
        ) -> Retained<MPSGraphTensorDataDictionary>;

        #[cfg(all(
            feature = "MPSGraphOperation",
            feature = "MPSGraphTensor",
            feature = "MPSGraphTensorData",
            feature = "objc2-metal-performance-shaders"
        ))]
        /// Encodes the graph for the given feeds to returns the target tensor values in the results dictionary provided by the user.
        ///
        /// It ensures all target operations also executed. This call is asynchronous and will return immediately if a completionHandler is set.
        ///
        /// - Parameters:
        /// - commandBuffer: commandBuffer passed to execute the graph on, commitAndContinue might be called, please don't rely on underlying MTLCommandBuffer to remain uncommitted.
        /// - feeds: Feeds dictionary for the placeholder tensors.
        /// - targetOperations: Operations to be completed at the end of the run.
        /// - resultsDictionary: MPSGraphTensors dictionary passed by user, these will be filled with graph output data.
        /// - executionDescriptor: ExecutionDescriptor to be passed in and used.
        #[unsafe(method(encodeToCommandBuffer:feeds:targetOperations:resultsDictionary:executionDescriptor:))]
        #[unsafe(method_family = none)]
        pub unsafe fn encodeToCommandBuffer_feeds_targetOperations_resultsDictionary_executionDescriptor(
            &self,
            command_buffer: &MPSCommandBuffer,
            feeds: &MPSGraphTensorDataDictionary,
            target_operations: Option<&NSArray<MPSGraphOperation>>,
            results_dictionary: &MPSGraphTensorDataDictionary,
            execution_descriptor: Option<&MPSGraphExecutionDescriptor>,
        );
    );
}

impl Graph {
    /// Compiles the graph for the given feeds to returns the target tensor values, ensuring all target operations would be executed.
    ///
    /// This call blocks until execution has completed. The compilation descriptor helps specialize the executable returned.
    ///
    /// - Parameters:
    /// - device: MPSGraph device to optimize for.
    /// - feeds: Feeds dictionary for the placeholder tensors.
    /// - targetTensors: Tensors for which the caller wishes MPSGraphTensorData to be returned.
    /// - targetOperations: Operations to be completed at the end of the run.
    /// - compilationDescriptor: compilation descriptor to set different compilation parameters.
    /// - Returns: A valid MPSGraphExecutable object
    #[unsafe(method(compileWithDevice:feeds:targetTensors:targetOperations:compilationDescriptor:))]
    #[unsafe(method_family = none)]
    pub unsafe fn compileWithDevice_feeds_targetTensors_targetOperations_compilationDescriptor(
        &self,
        device: Option<&MPSGraphDevice>,
        feeds: &MPSGraphTensorShapedTypeDictionary,
        target_tensors: &NSArray<MPSGraphTensor>,
        target_operations: Option<&NSArray<MPSGraphOperation>>,
        compilation_descriptor: Option<&MPSGraphCompilationDescriptor>,
    ) -> Retained<MPSGraphExecutable>;

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
        feeds: &TensorShapedTypeHashMap,
        targets: &[&Tensor],
        target_operations: Option<&[&Operation]>,
        descriptor: Option<&CompilationDescriptor>,
    ) -> Retained<Executable> {
        autoreleasepool(|_| unsafe {
            let feeds_keys: Vec<&Tensor> = feeds.keys().copied().collect();
            let feeds_values: Vec<&ShapedType> = feeds.values().copied().collect();
            let feeds_dict = NSDictionary::from_slices(&feeds_keys, &feeds_values);
            let targets_array = NSArray::from_slice(targets);
            let target_operations_array = target_operations.map(|ops| NSArray::from_slice(ops));
            let executable: Retained<Executable> = msg_send![
                self,
                compileWithDevice: device as &Device,
                feeds: &*feeds_dict,
                targetTensors: &*targets_array,
                targetOperations: target_operations_array.as_ref().map(|ops| &**ops),
                compilationDescriptor: descriptor
            ];
            executable
        })
    }

    pub fn placeholder(
        &self,
        data_type: DataType,
        shape: &Shape,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            msg_send![
                self,
                placeholderWithShape: &*shape,
                dataType: data_type as u32,
                name: name_ptr
            ]
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
                shape: shape,
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
            msg_send![
                self,
                constantWithScalar: value,
                shape: shape,
                dataType: data_type as u32
            ]
        }
    }

    /// Creates a variable tensor backed by the provided data.
    ///
    /// * `data`      – Raw values for the tensor (any scalar type that is `Copy`).
    /// * `shape`     – Final tensor shape (must be static).
    /// * `data_type` – MPS data type of the tensor contents.
    /// * `name`      – Optional operation name shown in Graph debugging output.
    pub fn variable<T: Copy>(
        &self,
        data: &[T],
        shape: &Shape,
        data_type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let bytes =
                std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data));

            self.variable_with_bytes(bytes, shape, data_type, name)
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
                randomUniformTensorWithShape: shape,
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
                randomNormalTensorWithShape: shape,
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

            msg_send![
                self,
                additionWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name_ptr
            ]
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

    /// Boxed slice of all the placeholder tensors.
    pub fn placeholder_tensors(&self) -> Box<[Retained<Tensor>]> {
        unsafe {
            let ns_array: Retained<NSArray<Tensor>> = msg_send![self, placeholderTensors];
            ns_array.to_vec().into_boxed_slice()
        }
    }

    pub fn dump(&self) {
        unsafe {
            let _: () = msg_send![self, dump];
        }
    }
}
