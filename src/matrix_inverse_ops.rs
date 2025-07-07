use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::tensor::Tensor;

impl Graph {
    /// Computes the inverse of an input tensor.
    ///
    /// The framework computes the inverse of a square matrix by calling LU decomposition and LU solver.
    /// All dimensions after the first 2 are treated as batch dimensions and the inverse for each batch is computed.
    /// Results are undefined for ill conditioned matrices.
    ///
    /// # Parameters
    ///
    /// * `input` - The input tensor.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A valid `MPSGraphTensor` object containing the inverse of the input tensor.
    pub fn inverse(&self, input: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let tensor: Retained<Tensor> = msg_send![self,
                inverseOfTensor: input,
                name: name_ptr
            ];
            tensor
        }
    }
}
