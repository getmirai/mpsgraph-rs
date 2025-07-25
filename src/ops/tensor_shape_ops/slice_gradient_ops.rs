use super::StartEndStrideScalarsOrTensors;
use crate::{Graph, ShapeOrTensor, ShapedType, Tensor};
use objc2::{extern_methods, msg_send, rc::Retained};
use objc2_foundation::{NSArray, NSNumber, NSString};

impl Graph {
    /// Creates a strided-slice gradient operation and returns the result tensor.
    ///
    /// - Parameters:
    /// - inputGradientTensor: The input gradient.
    /// - fwdInShapeTensor: The shape of the forward pass input, that is the shape of the gradient output.
    /// - starts: An array of numbers that specify the starting points for each dimension.
    /// - ends: An array of numbers that specify the ending points for each dimension.
    /// - strides: An array of numbers that specify the strides for each dimension.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object
    pub fn slice_gradient(
        &self,
        input_gradient_tensor: &Tensor,
        fwd_in_shape_tensor: &Tensor,
        starts: &[u64],
        ends: &[u64],
        strides: &[u64],
        name: Option<&str>,
    ) -> Retained<Tensor> {
        let starts = starts
            .iter()
            .map(|x| NSNumber::new_u64(*x))
            .collect::<Box<[Retained<NSNumber>]>>();
        let ends = ends
            .iter()
            .map(|x| NSNumber::new_u64(*x))
            .collect::<Box<[Retained<NSNumber>]>>();
        let strides = strides
            .iter()
            .map(|x| NSNumber::new_u64(*x))
            .collect::<Box<[Retained<NSNumber>]>>();
        let starts_array = NSArray::from_retained_slice(&starts);
        let ends_array = NSArray::from_retained_slice(&ends);
        let strides_array = NSArray::from_retained_slice(&strides);
        unsafe {
            msg_send![
                self,
                sliceGradientTensor: input_gradient_tensor,
                fwdInShapeTensor: fwd_in_shape_tensor,
                starts: &*starts_array,
                ends: &*ends_array,
                strides: &*strides_array,
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }

    /// Creates a strided-slice gradient operation and returns the result tensor.
    ///
    /// - Parameters:
    /// - inputGradientTensor: The input gradient.
    /// - fwdInShapeTensor: The shape of the forward pass input, that is the shape of the gradient output.
    /// - start_end_stride: An enum of numbers or tensors that specify the starting points for each dimension.
    /// - startMask: A bitmask that indicates dimensions whose `starts` values the operation should ignore.
    /// - endMask: A bitmask that indicates dimensions whose `ends` values the operation should ignore.
    /// - squeezeMask: A bitmask that indicates dimensions the operation will squeeze out from the result.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object
    pub fn slice_gradient_with_masks<'a>(
        &self,
        input_gradient_tensor: &Tensor,
        fwd_in_shape_tensor: &Tensor,
        start_end_stride: StartEndStrideScalarsOrTensors<'a>,
        start_mask: u32,
        end_mask: u32,
        squeeze_mask: u32,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match start_end_stride {
            StartEndStrideScalarsOrTensors::Scalars {
                starts,
                ends,
                strides,
            } => {
                let starts = starts
                    .iter()
                    .map(|x| NSNumber::new_u64(*x))
                    .collect::<Box<[Retained<NSNumber>]>>();
                let ends = ends
                    .iter()
                    .map(|x| NSNumber::new_u64(*x))
                    .collect::<Box<[Retained<NSNumber>]>>();
                let strides = strides
                    .iter()
                    .map(|x| NSNumber::new_u64(*x))
                    .collect::<Box<[Retained<NSNumber>]>>();
                let starts_array = NSArray::from_retained_slice(&starts);
                let ends_array = NSArray::from_retained_slice(&ends);
                let strides_array = NSArray::from_retained_slice(&strides);
                unsafe {
                    msg_send![
                        self,
                        sliceGradientTensor: input_gradient_tensor,
                        fwdInShapeTensor: fwd_in_shape_tensor,
                        starts: &*starts_array,
                        ends: &*ends_array,
                        strides: &*strides_array,
                        startMask: start_mask,
                        endMask: end_mask,
                        squeezeMask: squeeze_mask,
                        name: name.map(NSString::from_str).as_deref(),
                    ]
                }
            }
            StartEndStrideScalarsOrTensors::Tensors {
                start_tensor,
                end_tensor,
                stride_tensor,
            } => unsafe {
                msg_send![
                    self,
                    sliceGradientTensor: input_gradient_tensor,
                    fwdInShapeTensor: fwd_in_shape_tensor,
                    startTensor: start_tensor,
                    endTensor: end_tensor,
                    strideTensor: stride_tensor,
                    startMask: start_mask,
                    endMask: end_mask,
                    squeezeMask: squeeze_mask,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }

    /// Creates a slice gradient operation and returns the result tensor.
    ///
    /// - Parameters:
    /// - inputGradientTensor: The input gradient.
    /// - fwdInShapeTensor: The shape of the forward pass input, that is the shape of the gradient output.
    /// - startTensor: The tensor that specifies the starting points for each dimension.
    /// - sizeTensor: The tensor that specifies the size of the forward result for each dimension.
    /// - squeezeMask: A bitmask that indicates dimensions the operation will squeeze out from the result.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object
    pub fn slice_gradient_start_tensor_size_tensor_squeeze_mask(
        &self,
        input_gradient_tensor: &Tensor,
        fwd_in_shape_tensor: &Tensor,
        start_tensor: &Tensor,
        size_tensor: &Tensor,
        squeeze_mask: u32,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                sliceGradientTensor: input_gradient_tensor,
                fwdInShapeTensor: fwd_in_shape_tensor,
                startTensor: start_tensor,
                sizeTensor: size_tensor,
                squeezeMask: squeeze_mask,
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }
}
