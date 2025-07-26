use crate::{Graph, ScalarOrTensor, ShapeOrTensor, ShapedType, Tensor};
use objc2::{extern_methods, msg_send, rc::Retained};
use objc2_foundation::NSString;

impl Graph {
    /// Creates a flatten2D operation and returns the result tensor.
    ///
    /// Flattens dimensions before `axis` to `result[0]` and dimensions starting
    /// from `axis` to `result[1]` and returns a rank-2 tensor as result.
    ///
    /// - Parameters:
    /// - tensor: The tensor to be flattened.
    /// - axis: The axis around which to flatten.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object.
    pub fn flatten_2d<'a>(
        &self,
        tensor: &Tensor,
        axis: ScalarOrTensor<'a, i64>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match axis {
            ScalarOrTensor::Scalar(axis) => unsafe {
                msg_send![
                    self,
                    flatten2DTensor: tensor,
                    axis: axis,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            ScalarOrTensor::Tensor(axis) => unsafe {
                msg_send![
                    self,
                    flatten2DTensor: tensor,
                    axisTensor: axis,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }
}
