use objc2::rc::Retained;
use objc2::msg_send;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::tensor::Tensor;

/// Trait defining NonZero operations for a Graph
pub trait GraphNonZeroOps {
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
    /// * `tensor` - An Tensor of which to compute the non-zero indices.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A valid Tensor containing indices in signed int32 data type.
    fn non_zero_indices(&self, tensor: &Tensor, name: Option<&str>) -> Option<Retained<Tensor>>;
}

/// Implementation of NonZero operations for Graph
impl GraphNonZeroOps for Graph {
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
    /// * `tensor` - An Tensor of which to compute the non-zero indices.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A valid Tensor containing indices in signed int32 data type.
    fn non_zero_indices(&self, tensor: &Tensor, name: Option<&str>) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self, 
                nonZeroIndicesOfTensor: tensor,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }
}

/// Extension trait for easier access to NonZero operations
pub trait GraphNonZeroOpsExtension {
    /// Get access to non-zero operations
    fn non_zero_ops(&self) -> &dyn GraphNonZeroOps;
}

impl GraphNonZeroOpsExtension for Graph {
    fn non_zero_ops(&self) -> &dyn GraphNonZeroOps {
        self
    }
}