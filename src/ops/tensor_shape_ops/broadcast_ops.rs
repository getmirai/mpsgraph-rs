use crate::{ns_number_array_from_slice, Graph, ShapeOrTensor, Tensor};
use objc2::{msg_send, rc::Retained};
use objc2_foundation::NSString;

impl Graph {
    /// Creates a broadcast operation and returns the result tensor.
    ///
    /// Broadcasts values inside the tensor, starting from the trailing dimensions, to give it the correct shape.
    /// This is equivalent to the broadcasting for arithmetic operations when operands have different shapes.
    ///
    /// - Parameters:
    /// - tensor: The tensor to be broadcasted
    /// - shape: The shape of the result tensor.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object
    pub fn broadcast_tensor<'a>(
        &self,
        tensor: &Tensor,
        shape: ShapeOrTensor<'a>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match shape {
            ShapeOrTensor::Shape(shape) => unsafe {
                msg_send![
                    self,
                    broadcastTensor: tensor,
                    toShape: &*ns_number_array_from_slice(shape),
                    name: name.map(NSString::from_str).as_deref()
                ]
            },
            ShapeOrTensor::Tensor(shape) => unsafe {
                msg_send![
                    self,
                    broadcastTensor: tensor,
                    toShapeTensor: shape,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }
}
