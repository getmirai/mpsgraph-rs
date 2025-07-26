use crate::{Graph, ShapeOrTensor, ShapedType, Tensor};
use objc2::{extern_methods, msg_send, rc::Retained};
use objc2_foundation::{NSArray, NSNumber, NSString};

impl Graph {
    /// Creates a slice operation and returns the result tensor.
    ///
    /// - Parameters:
    /// - tensor: The tensor to be sliced.
    /// - dimensionIndex: The dimension to slice.
    /// - start: The starting index of the slice, can be negative to count from the end of the tensor dimension.
    /// - length: The length of the slice.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object.
    pub fn slice(
        &self,
        tensor: &Tensor,
        dimension_index: u64,
        start: i64,
        length: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                sliceTensor: tensor,
                dimension: dimension_index,
                start: start,
                length: length,
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }

    /// Creates a slice operation and returns the result tensor.
    ///
    /// Slices a tensor starting from `startTensor`, stopping short before `startTensor + endTensor` stepping
    /// a single pace between each value. Semantics based on
    /// [TensorFlow Strided Slice Op](https://www.tensorflow.org/api_docs/python/tf/strided_slice).
    ///
    /// - Parameters:
    /// - tensor: The Tensor to be sliced.
    /// - startTensor: The tensor that specifies the starting points for each dimension.
    /// - sizeTensor: The tensor that specifies the size of the result for each dimension.
    /// - squeezeMask: A bitmask that indicates dimensions the operation will squeeze out from the result.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object.
    pub fn slice_with_tensors(
        &self,
        tensor: &Tensor,
        start_tensor: &Tensor,
        size_tensor: &Tensor,
        squeeze_mask: u32,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                sliceTensor: tensor,
                startTensor: start_tensor,
                sizeTensor: size_tensor,
                squeezeMask: squeeze_mask,
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }
}
