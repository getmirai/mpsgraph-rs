use objc2::{msg_send, rc::Retained};
use objc2_foundation::NSString;

use crate::{Graph, ShapeOrTensor, Tensor};

impl Graph {
    /// Creates a reverse operation and returns the result tensor.
    ///
    /// Reverses a tensor on given axes.
    /// Semantics based on [TensorFlow reverse op](https://www.tensorflow.org/api_docs/python/tf/reverse).
    ///
    /// - Parameters:
    /// - tensor: The tensor to be reversed.
    /// - axes: A tensor or nsarray of scalars that specifies axes to be reversed (Axes must be unique and within normal axis range).
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object.
    pub fn reverse_tensor_with_axes<'a>(
        &self,
        tensor: &Tensor,
        axes: ShapeOrTensor<'a>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match axes {
            ShapeOrTensor::Shape(axes) => {
                let axes_ns_array = &**axes;
                unsafe {
                    msg_send![self, reverseTensor: tensor, axes: axes_ns_array, name: name.map(NSString::from_str).as_deref()]
                }
            }
            ShapeOrTensor::Tensor(tensor) => unsafe {
                msg_send![self, reverseTensor: tensor, axesTensor: tensor, name: name.map(NSString::from_str).as_deref()]
            },
        }
    }

    /// Creates a reverse operation and returns the result tensor.
    ///
    /// Reverses a tensor on all axes.
    /// Semantics based on [TensorFlow reverse op](https://www.tensorflow.org/api_docs/python/tf/reverse).
    ///
    /// - Parameters:
    /// - tensor: The tensor to be reversed.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object.
    pub fn reverse_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![self, reverseTensor: tensor, name: name.map(NSString::from_str).as_deref()]
        }
    }
}
