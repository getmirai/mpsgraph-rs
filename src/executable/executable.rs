use metal::{foreign_types::ForeignType, CommandQueue};
use objc2::rc::autoreleasepool;
use objc2::rc::{Allocated, Retained};
use objc2::runtime::NSObject;
use objc2::{extern_class, extern_conformance, extern_methods, msg_send, ClassType};
use objc2_foundation::{NSArray, NSObjectProtocol, NSString, NSURL};
use std::path::Path;

use crate::{
    CommandBuffer, CompilationDescriptor, Device, ExecutableExecutionDescriptor,
    ExecutableSerializationDescriptor, GraphObject, GraphOptions, GraphType, ShapedType, Tensor,
    TensorData,
};

extern_class!(
    /// The compiled representation of a compute graph executable.
    ///
    /// An `MPSGraphExecutable` is a compiled graph for specific feeds for specific target tensors and target operations.
    ///
    /// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphexecutable?language=objc)
    #[unsafe(super(GraphObject, NSObject))]
    #[derive(Debug, PartialEq, Eq, Hash)]
    pub struct Executable;
);

extern_conformance!(
    unsafe impl NSObjectProtocol for Executable {}
);

impl Executable {
    extern_methods!(
        #[unsafe(method(init))]
        #[unsafe(method_family = init)]
        pub unsafe fn init(this: Allocated<Self>) -> Retained<Self>;

        #[unsafe(method(new))]
        #[unsafe(method_family = new)]
        pub unsafe fn new() -> Retained<Self>;

        /// Options for the graph executable.
        #[unsafe(method(options))]
        #[unsafe(method_family = none)]
        pub unsafe fn options(&self) -> GraphOptions;

        /// Setter for [`options`][Self::options].
        #[unsafe(method(setOptions:))]
        #[unsafe(method_family = none)]
        pub unsafe fn set_options(&self, options: GraphOptions);

        /// Specialize the executable and optimize it.
        ///
        /// Use this method to choose when specialization happens, else it occurs at encode time automatically.
        ///
        /// - Parameters:
        /// - device: optional device to compile with.
        /// - inputTypes: input types expected to be passed to the executable.
        /// - compilationDescriptor: compilation descriptor to be used to specialize, since the executable was created with a compilationDescriptor already this one overrides those settings to the extent it can.
        #[unsafe(method(specializeWithDevice:inputTypes:compilationDescriptor:))]
        #[unsafe(method_family = none)]
        pub unsafe fn specialize(
            &self,
            device: Option<&Device>,
            input_types: &NSArray<GraphType>,
            compilation_descriptor: Option<&CompilationDescriptor>,
        );

        /// Get output shapes for a specialized executable.
        ///
        /// In case specialization has not been done yet then calling this function will specialize for the given input shapes.
        ///
        /// - Parameters:
        /// - device: Optional MPSGraph device to compile with
        /// - inputTypes: Input types expected to be passed to the executable.
        /// - compilationDescriptor: CompilationDescriptor to be used to specialize, since the executable was created with a compilationDescriptor already this one overrides those settings to the extent it can.
        #[unsafe(method(getOutputTypesWithDevice:inputTypes:compilationDescriptor:))]
        #[unsafe(method_family = none)]
        pub unsafe fn get_output_types(
            &self,
            device: Option<&Device>,
            input_types: &NSArray<GraphType>,
            compilation_descriptor: Option<&CompilationDescriptor>,
        ) -> Option<Retained<NSArray<ShapedType>>>;

        #[cfg(feature = "MPSGraphTensorData")]
        /// Runs the graph for the given feeds and returns the target tensor values, ensuring all target operations also executed.
        /// This call is asynchronous and will return immediately.
        ///
        /// - Parameters:
        /// - commandQueue: CommandQueue passed to exectute the graph on.
        /// - inputsArray: Feeds tensorData for the placeholder tensors, same order as arguments of main function.
        /// - resultsArray: Tensors for which the caller wishes MPSGraphTensorData to be returned.
        /// - executionDescriptor: ExecutionDescriptor to be passed in and used.
        /// - Returns: A valid MPSGraphTensorData array with results synchronized to the CPU memory if MPSGraphOptionsSynchronizeResults set.
        #[unsafe(method(runAsyncWithMTLCommandQueue:inputsArray:resultsArray:executionDescriptor:))]
        #[unsafe(method_family = none)]
        pub unsafe fn runAsyncWithMTLCommandQueue_inputsArray_resultsArray_executionDescriptor(
            &self,
            command_queue: &ProtocolObject<dyn MTLCommandQueue>,
            inputs_array: &NSArray<MPSGraphTensorData>,
            results_array: Option<&NSArray<MPSGraphTensorData>>,
            execution_descriptor: Option<&MPSGraphExecutableExecutionDescriptor>,
        ) -> Retained<NSArray<MPSGraphTensorData>>;

        #[cfg(all(
            feature = "MPSGraphTensorData",
            feature = "objc2-metal-performance-shaders"
        ))]
        /// Runs the graph for the given feeds and returns the target tensor values, ensuring all target operations also executed.
        /// This call is asynchronous and will return immediately after finishing encoding.
        ///
        /// - Parameters:
        /// - commandBuffer: CommandBuffer passed to exectute the graph on, commitAndContinue might be called, please don't rely on underlying MTLCommandBuffer to remain uncommitted
        /// - inputsArray: Feeds tensorData for the placeholder tensors, same order as arguments of main function
        /// - resultsArray: Tensors for which the caller wishes MPSGraphTensorData to be returned
        /// - executionDescriptor: ExecutionDescriptor to be passed in and used,
        /// - Returns: A valid MPSGraphTensorData array with results synchronized to the CPU memory if MPSGraphOptionsSynchronizeResults set.
        #[unsafe(method(encodeToCommandBuffer:inputsArray:resultsArray:executionDescriptor:))]
        #[unsafe(method_family = none)]
        pub unsafe fn encodeToCommandBuffer_inputsArray_resultsArray_executionDescriptor(
            &self,
            command_buffer: &MPSCommandBuffer,
            inputs_array: &NSArray<MPSGraphTensorData>,
            results_array: Option<&NSArray<MPSGraphTensorData>>,
            execution_descriptor: Option<&MPSGraphExecutableExecutionDescriptor>,
        ) -> Retained<NSArray<MPSGraphTensorData>>;

        /// Serialize the MPSGraph executable at the provided url.
        ///
        /// - Parameters:
        /// - url: The URL where to serialize the MPSGraph executable.
        /// - descriptor: The descriptor to be used to serialize the graph.
        #[unsafe(method(serializeToMPSGraphPackageAtURL:descriptor:))]
        #[unsafe(method_family = none)]
        pub unsafe fn serializeToMPSGraphPackageAtURL_descriptor(
            &self,
            url: &NSURL,
            descriptor: Option<&MPSGraphExecutableSerializationDescriptor>,
        );

        #[cfg(feature = "MPSGraph")]
        /// Initialize the executable with the Metal Performance Shaders Graph package at the provided URL.
        ///
        /// - Parameters:
        /// - mpsgraphPackageURL: The URL where to read the serialized MPSGraphExecutable.
        /// - compilationDescriptor: Compilation descriptor to be used to specialize, since the executable was created with a compilationDescriptor already this one overrides those settings to the extent it can.
        #[unsafe(method(initWithMPSGraphPackageAtURL:compilationDescriptor:))]
        #[unsafe(method_family = init)]
        pub unsafe fn initWithMPSGraphPackageAtURL_compilationDescriptor(
            this: Allocated<Self>,
            mpsgraph_package_url: &NSURL,
            compilation_descriptor: Option<&MPSGraphCompilationDescriptor>,
        ) -> Retained<Self>;

        #[cfg(feature = "MPSGraph")]
        /// Initialize the executable with the Core ML model package at the provided URL.
        ///
        /// - Parameters:
        /// - coreMLPackageURL: The URL where to read the Core ML model package.
        /// - compilationDescriptor: Compilation descriptor to be used to specialize, since the executable was created with a compilationDescriptor already this one overrides those settings to the extent it can.
        #[unsafe(method(initWithCoreMLPackageAtURL:compilationDescriptor:))]
        #[unsafe(method_family = init)]
        pub unsafe fn initWithCoreMLPackageAtURL_compilationDescriptor(
            this: Allocated<Self>,
            core_ml_package_url: &NSURL,
            compilation_descriptor: Option<&MPSGraphCompilationDescriptor>,
        ) -> Retained<Self>;
    );
}

impl Executable {
    /// Tensors fed to the graph, can be used to order the inputs when executable is created with a graph.
    pub fn feed_tensors(&self) -> Option<Box<[Retained<Tensor>]>> {
        unsafe {
            let array_opt: Option<Retained<NSArray<Tensor>>> = msg_send![self, feedTensors];
            array_opt.map(|array| array.to_vec().into_boxed_slice())
        }
    }

    /// Tensors targeted by the graph, can be used to order the outputs when executable was created with a graph.
    pub fn target_tensors(&self) -> Option<Box<[Retained<Tensor>]>> {
        unsafe {
            let array_opt: Option<Retained<NSArray<Tensor>>> = msg_send![self, targetTensors];
            array_opt.map(|array| array.to_vec().into_boxed_slice())
        }
    }

    /// Runs the graph for the given feeds and returns the target tensor values, ensuring all target operations also executed.
    ///
    /// This call is synchronous and will return on completion of execution.
    ///
    /// - Parameters:
    /// - commandQueue: CommandQueue passed to exectute the graph on.
    /// - inputsArray: Feeds tensorData for the placeholder tensors, same order as arguments of main function.
    /// - resultsArray: Results tensorData for which the caller wishes MPSGraphTensorData to be returned.
    /// - Returns: A valid MPSGraphTensorData array with results synchronized to the CPU memory if MPSGraphOptionsSynchronizeResults set.
    pub fn run_with_command_queue(
        &self,
        command_queue: &CommandQueue,
        inputs: &[&TensorData],
        results: Option<&[&TensorData]>,
        execution_descriptor: Option<&ExecutableExecutionDescriptor>,
    ) -> Box<[Retained<TensorData>]> {
        autoreleasepool(|_pool| unsafe {
            let cmd_queue_ptr = command_queue.as_ptr() as *mut std::ffi::c_void;
            let inputs_array = NSArray::from_slice(&inputs);
            let results_array = results.map(|r_slice| NSArray::from_slice(&r_slice));
            let result: Retained<NSArray<TensorData>> = msg_send![
                self,
                runWithMTLCommandQueue: cmd_queue_ptr,
                inputsArray: &*inputs_array,
                resultsArray: results_array.as_deref(),
                executionDescriptor: execution_descriptor.as_deref()
            ];
            result.to_vec().into_boxed_slice()
        })
    }

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

    pub fn serialize_to_url(&self, path: &Path, descriptor: &ExecutableSerializationDescriptor) {
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

    pub fn dump(&self) {
        unsafe {
            let _: () = msg_send![self, dump];
        }
    }
}
