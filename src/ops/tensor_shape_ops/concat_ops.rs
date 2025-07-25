use super::StartEndStrideScalarsOrTensors;
use crate::{Graph, ShapeOrTensor, ShapedType, Tensor};
use objc2::{extern_methods, msg_send, rc::Retained};
use objc2_foundation::{NSArray, NSNumber, NSString};

impl Graph {
    /// Creates a concatenation operation and returns the result tensor.
    ///
    /// Concatenates two input tensors along the specified dimension. Tensors must be broadcast
    /// compatible along all other dimensions, and have the same datatype.
    ///
    /// - Parameters:
    /// - tensor: The first tensor to concatenate.
    /// - tensor2: The second tensor to concatenate.
    /// - dimensionIndex: The dimension to concatenate across, must be in range: `-rank
    /// <
    /// = dimension
    /// <
    /// rank`.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object.
    pub fn concat_tensor(
        &self,
        tensor: &Tensor,
        tensor2: &Tensor,
        dimension_index: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                concatTensor: tensor,
                withTensor: tensor2,
                dimension: dimension_index,
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }

    /// Creates a concatenation operation and returns the result tensor.
    ///
    /// Concatenates all input tensors along specified dimension. All inputs must be broadcast
    /// compatible along all other dimensions, and have the same type.
    ///
    /// When interleave is specified, all tensors will be interleaved. To interleave, make sure to provide broadcast
    /// compatible inputs along the specified dimension as well.
    ///
    /// For example:
    /// ```md
    /// operand0 = [1, 2, 3]
    /// operand1 = [4, 5, 6]
    /// concat([operand0, operand1], axis = 0, interleave = YES) = [1, 4, 2, 5, 3, 6]
    /// ```
    ///
    /// - Parameters:
    /// - tensors: The tensors to concatenate.
    /// - dimensionIndex: The dimension to concatenate across, must be in range: `-rank
    /// <
    /// = dimension
    /// <
    /// rank`.
    /// - interleave: A boolean value that specifies whether the operation interleaves input tensors.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object.
    pub fn concat_tensors(
        &self,
        tensors: &[&Tensor],
        dimension_index: i64,
        interleave: bool,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        let tensors_array = NSArray::from_slice(tensors);
        if interleave {
            unsafe {
                msg_send![
                    self,
                    concatTensors: &*tensors_array,
                    dimension: dimension_index,
                    interleave: interleave,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            }
        } else {
            unsafe {
                msg_send![
                    self,
                    concatTensors: &*tensors_array,
                    dimension: dimension_index,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            }
        }
    }
}
