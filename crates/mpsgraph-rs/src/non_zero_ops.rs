use crate::core::{AsRawObject, NSString};
use crate::graph::Graph;
use crate::tensor::Tensor;
use objc2::msg_send;
use objc2::runtime::AnyObject;

/// NonZero operations for Graph
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
    /// - Parameters:
    ///   - tensor: An Tensor of which to compute the non-zero indices.
    ///   - name: The name for the operation.
    /// - Returns: A valid Tensor containing indices in signed int32 data type.
    pub fn non_zero_indices(&self, tensor: &Tensor, name: Option<&str>) -> Tensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let result: *mut AnyObject = msg_send![self.0, nonZeroIndicesOfTensor: tensor.0,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }
}
