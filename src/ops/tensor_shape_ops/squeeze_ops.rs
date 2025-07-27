use crate::{ns_number_array_from_slice, AxesOrTensor, Graph, Tensor};
use objc2::{extern_methods, msg_send, rc::Retained};
use objc2_foundation::NSString;

impl Graph {
    /// Creates a squeeze operation and returns the result tensor.
    ///
    /// Squeezes the tensor, removing all dimensions with size 1.
    ///
    /// - Parameters:
    /// - tensor: The input tensor.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object.
    pub fn squeeze_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                squeezeTensor: tensor,
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }

    /// Creates a squeeze operation and returns the result tensor.
    ///
    /// Squeezes the tensor, removing a dimension with size 1 at the specified axis.
    /// The size of the input tensor must be 1 at the specified axis.
    ///
    /// - Parameters:
    /// - tensor: The input tensor.
    /// - axis: The axis to squeeze.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object.
    pub fn squeeze_tensor_axis(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![self, squeezeTensor: tensor, axis: axis, name: name.map(NSString::from_str).as_deref()]
        }
    }

    /// Creates a squeeze operation and returns the result tensor.
    ///
    /// Squeezes the tensor, removing dimensions with size 1 at specified axes.
    /// The size of the input tensor must be 1 at all specified axes.
    ///
    /// - Parameters:
    /// - tensor: The input tensor.
    /// - axes: The axes to squeeze.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object
    pub fn squeeze_tensor_axes<'a>(
        &self,
        tensor: &Tensor,
        axes: AxesOrTensor<'a>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match axes {
            AxesOrTensor::Axes(axes) => unsafe {
                msg_send![
                    self,
                    squeezeTensor: tensor,
                    axes: &*ns_number_array_from_slice(axes),
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            AxesOrTensor::Tensor(axes) => unsafe {
                msg_send![
                    self,
                    squeezeTensor: tensor,
                    axesTensor: axes,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }
}
