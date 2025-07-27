use super::AxesOrTensor;
use crate::{ns_number_array_from_slice, Graph, Tensor};
use objc2::{msg_send, rc::Retained};
use objc2_foundation::NSString;

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
        axes: AxesOrTensor<'a>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match axes {
            AxesOrTensor::Axes(axes) => unsafe {
                msg_send![
                    self,
                    reverseTensor: tensor,
                    axes: &*ns_number_array_from_slice(axes),
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            AxesOrTensor::Tensor(tensor) => unsafe {
                msg_send![
                    self,
                    reverseTensor: tensor,
                    axesTensor: tensor,
                    name: name.map(NSString::from_str).as_deref(),
                ]
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
            msg_send![
                self,
                reverseTensor: tensor,
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }
}
