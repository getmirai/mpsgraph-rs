use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::{NSArray, NSString};

use crate::data_types::ShapedType;
use crate::graph::Graph;
use crate::tensor::Tensor;

impl Graph {
    /// Creates an operation which invokes another executable.
    ///
    /// - Parameters:
    ///   - symbol_name: The unique identifier used to find the executable in the `MPSGraphCompilationDescriptor.callables` directory.
    ///   - input_tensors: The tensors which are passed as inputs to the executable being invoked.
    ///   - output_types: The expected return types of the executable being invoked.
    ///   - name: name of operation.
    /// - Returns: A boxed slice of valid `Tensor` objects representing the return tensors of the invoked executable.
    pub fn call_symbol_name(
        &self,
        symbol_name: &str,
        input_tensors: &[&Tensor],
        output_types: &[&ShapedType],
        name: Option<&str>,
    ) -> Box<[Retained<Tensor>]> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let symbol_name_ns = NSString::from_str(symbol_name);
            let input_tensors_array = NSArray::from_slice(input_tensors);
            let output_types_array = NSArray::from_slice(output_types);

            let result_nsarray: Retained<NSArray<Tensor>> = msg_send![
                self,
                callSymbolName: &*symbol_name_ns,
                inputTensors: &*input_tensors_array,
                outputTypes: &*output_types_array,
                name: name_ptr,
            ];

            result_nsarray.to_vec().into_boxed_slice()
        }
    }
}
