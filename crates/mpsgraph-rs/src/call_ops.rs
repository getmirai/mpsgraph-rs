use objc2::rc::Retained;
use objc2::msg_send;
use objc2_foundation::{NSArray, NSString};

use crate::graph::Graph;
use crate::tensor::Tensor;
use crate::data_types::ShapedType;
use crate::utils::block_wrapper::convert_nsarray_to_vec;

/// Trait for call operations on Graph
pub trait GraphCallOps {
    /// Creates an operation which invokes another executable.
    ///
    /// # Arguments
    ///
    /// * `symbol_name` - The unique identifier used to find the executable in the CompilationDescriptor.callables directory
    /// * `input_tensors` - The tensors which are passed as inputs to the executable being invoked
    /// * `output_types` - The expected return types of the executable being invoked
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A vector of Tensor objects representing the return tensors of the invoked executable
    fn call(
        &self,
        symbol_name: &str,
        input_tensors: &[&Tensor],
        output_types: &[&ShapedType],
        name: Option<&str>,
    ) -> Vec<Retained<Tensor>>;
}

impl GraphCallOps for Graph {
    fn call(
        &self,
        symbol_name: &str,
        input_tensors: &[&Tensor],
        output_types: &[&ShapedType],
        name: Option<&str>,
    ) -> Vec<Retained<Tensor>> {
        unsafe {
            // Convert name to NSString if provided
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);
            
            // Convert symbol_name to NSString
            let symbol_name_ns = NSString::from_str(symbol_name);
            
            // Create NSArray of input tensors
            let input_tensors_array = NSArray::from_slice(input_tensors);
            
            // Create NSArray of output types
            let output_types_array = NSArray::from_slice(output_types);
            
            // Call the Objective-C method and get the result array
            let result_array: *mut NSArray<Tensor> = msg_send![
                self,
                callSymbolName: &*symbol_name_ns,
                inputTensors: &*input_tensors_array,
                outputTypes: &*output_types_array,
                name: name_ptr,
            ];
            
            // Convert the result array to a Vec of Retained<Tensor>
            convert_nsarray_to_vec(result_array)
        }
    }
}