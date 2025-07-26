use crate::{Graph, ShapeOrTensor, Tensor};
use objc2::{extern_methods, msg_send, rc::Retained};
use objc2_foundation::NSString;

impl Graph {
    /// Creates an expand-dimensions operation and returns the result tensor.
    ///
    /// Expands the tensor, inserting a dimension with size 1 at the specified axis.
    ///
    /// - Parameters:
    /// - tensor: The input tensor.
    /// - axis: The axis to expand.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object.
    pub fn expand_dims_of_tensor(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![self, expandDimsOfTensor: tensor, axis: axis, name: name.map(NSString::from_str).as_deref()]
        }
    }

    /// Creates an expand-dimensions operation and returns the result tensor.
    ///
    /// Expands the tensor, inserting dimensions with size 1 at specified axes.
    ///
    /// - Parameters:
    /// - tensor: The input tensor.
    /// - axes: The axes to expand.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object.
    pub fn expand_dims_of_tensor_axes<'a>(
        &self,
        tensor: &Tensor,
        axes: ShapeOrTensor<'a>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match axes {
            ShapeOrTensor::Shape(axes) => {
                let axes_ns_array = &**axes;
                unsafe {
                    msg_send![self, expandDimsOfTensor: tensor, axes: axes_ns_array, name: name.map(NSString::from_str).as_deref()]
                }
            }
            ShapeOrTensor::Tensor(axes) => unsafe {
                msg_send![self, expandDimsOfTensor: tensor, axesTensor: axes, name: name.map(NSString::from_str).as_deref()]
            },
        }
    }
}
