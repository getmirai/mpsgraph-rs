//! Subgraph calling helper exposed directly on `Graph`.

use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::{NSArray, NSString};

use crate::data_types::ShapedType;
use crate::graph::Graph;
use crate::tensor::Tensor;
use crate::utils::block_wrapper::convert_nsarray_to_vec;

impl Graph {
    /// Invoke another compiled executable by symbol name.
    ///
    /// * `symbol_name` – Identifier of the executable in the compilation descriptor.
    /// * `input_tensors` – Inputs passed to the executable.
    /// * `output_types` – Expected shapes/dtypes of the outputs.
    /// * `name` – Optional debug name.
    ///
    /// Returns the tensors produced by the callee.
    pub fn call(
        &self,
        symbol_name: &str,
        input_tensors: &[&Tensor],
        output_types: &[&ShapedType],
        name: Option<&str>,
    ) -> Vec<Retained<Tensor>> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let symbol_name_ns = NSString::from_str(symbol_name);
            let input_tensors_array = NSArray::from_slice(input_tensors);
            let output_types_array = NSArray::from_slice(output_types);

            let result_array_ptr: *mut NSArray<Tensor> = msg_send![
                self,
                callSymbolName: &*symbol_name_ns,
                inputTensors: &*input_tensors_array,
                outputTypes: &*output_types_array,
                name: name_ptr,
            ];

            convert_nsarray_to_vec(result_array_ptr)
        }
    }
}
