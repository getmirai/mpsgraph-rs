use objc2::rc::Retained;
use objc2::msg_send;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::tensor::Tensor;

/// Trait defining matrix inverse operations for a Graph
pub trait GraphMatrixInverseOps {
    /// Creates a matrix inverse operation
    ///
    /// This function computes the matrix inverse of a tensor.
    /// The input tensor must be a square matrix or a batch of square matrices.
    /// If the input is a batch of matrices, the inverse operation is applied to each matrix independently.
    ///
    /// # Parameters
    /// - `x`: The input tensor containing matrix/matrices to invert
    /// - `name`: Optional name for the operation
    ///
    /// # Returns
    /// A new tensor containing the inverted matrix/matrices
    fn inverse(&self, x: &Tensor, name: Option<&str>) -> Option<Retained<Tensor>>;
}

/// Implementation of matrix inverse operations for Graph
impl GraphMatrixInverseOps for Graph {
    /// Creates a matrix inverse operation
    ///
    /// This function computes the matrix inverse of a tensor.
    /// The input tensor must be a square matrix or a batch of square matrices.
    /// If the input is a batch of matrices, the inverse operation is applied to each matrix independently.
    ///
    /// # Parameters
    /// - `x`: The input tensor containing matrix/matrices to invert
    /// - `name`: Optional name for the operation
    ///
    /// # Returns
    /// A new tensor containing the inverted matrix/matrices
    fn inverse(&self, x: &Tensor, name: Option<&str>) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![self, inverseOfTensor: x, name: name_ptr];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }
}

/// Extension trait for easier access to matrix inverse operations
pub trait GraphMatrixInverseOpsExtension {
    /// Get access to matrix inverse operations
    fn matrix_inverse_ops(&self) -> &dyn GraphMatrixInverseOps;
}

impl GraphMatrixInverseOpsExtension for Graph {
    fn matrix_inverse_ops(&self) -> &dyn GraphMatrixInverseOps {
        self
    }
}