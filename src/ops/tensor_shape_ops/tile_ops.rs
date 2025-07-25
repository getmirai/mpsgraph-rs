use super::StartEndStrideScalarsOrTensors;
use crate::{Graph, Shape, ShapeOrTensor, ShapedType, Tensor};
use objc2::{extern_methods, msg_send, rc::Retained};
use objc2_foundation::{NSArray, NSNumber, NSString};

impl Graph {
    /// Creates a tile operation and returns the result tensor.
    ///
    /// Creates a tensor which contains multiple copies of the input tensor along each dimension of the tensor.
    ///
    /// - Parameters:
    /// - tensor: The input tensor
    /// - multiplier: An array of numbers that specifies how many copies per dimension MPSGraph produces.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object.
    pub fn tile_tensor(
        &self,
        tensor: &Tensor,
        multiplier: &Shape,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                tileTensor: tensor,
                withMultiplier: multiplier,
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }

    /// Creates a tile gradient operation and returns the result tensor.
    ///
    /// - Parameters:
    /// - incomingGradientTensor: The input gradient tensor.
    /// - sourceTensor: The input tensor of the forward pass.
    /// - multiplier: An array of numbers that specifies how many copies per dimension MPSGraph produced in the forward pass.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object.
    pub fn tile_gradient(
        &self,
        incoming_gradient_tensor: &Tensor,
        source_tensor: &Tensor,
        multiplier: &Shape,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                tileGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                sourceTensor: source_tensor,
                withMultiplier: multiplier,
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }
}
