use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::tensor::Tensor;

/// Trait defining NonZero operations for a Graph

/// Implementation of NonZero operations for Graph
impl Graph {
    /// Computes the indices of the non-zero elements of the input tensor.
    ///
    /// The indices are returned as a two-dimensional tensor of size `[number_of_nonzeros, input_rank]`.
    /// Each row in the result contains indices of a nonzero elements in input.
    /// For example:
    /// ```text
    /// tensor = [[ 1,  0, 3],
    ///           [ 0, 10, 0]]
    /// indices = [[ 0, 0],
    ///            [ 0, 2],
    ///            [ 1, 1]]
    /// ```
    ///
    /// # Parameters
    ///
    /// * `tensor` - An MPSGraphTensor of which to compute the non-zero indices.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A valid MPSGraphTensor containing indices in signed int32 data type.
    pub fn non_zero_indices(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: Retained<Tensor> = msg_send![
                self,
                nonZeroIndicesOfTensor: tensor,
                name: name_ptr
            ];
            result
        }
    }
}
