use super::StartEndStrideScalarsOrTensors;
use crate::{Graph, ShapeOrTensor, ShapedType, Tensor};
use objc2::{extern_methods, msg_send, rc::Retained};
use objc2_foundation::{NSArray, NSNumber, NSString};

impl Graph {
    /// Creates a strided-slice operation and returns the result tensor.
    ///
    /// Slices a tensor starting from `starts`, stopping short before `ends` stepping
    /// `strides` paces between each value. Semantics based on
    /// [TensorFlow Strided Slice Op](https://www.tensorflow.org/api_docs/python/tf/strided_slice).
    ///
    /// - Parameters:
    /// - tensor: The tensor to be sliced.
    /// - starts: An array of numbers that specify the starting points for each dimension.
    /// - ends: An array of numbers that specify the ending points for each dimension.
    /// - strides: An array of numbers that specify the strides for each dimension.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object.
    pub fn strided_slice(
        &self,
        tensor: &Tensor,
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
                sliceTensor: tensor,
                starts: &*starts_array,
                ends: &*ends_array,
                strides: &*strides_array,
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }

    /// Creates a strided-slice operation and returns the result tensor.
    ///
    /// Slices a tensor starting from `starts`, stopping short before `ends` stepping
    /// `strides` paces between each value. Semantics based on
    /// [TensorFlow Strided Slice Op](https://www.tensorflow.org/api_docs/python/tf/strided_slice).
    ///
    /// - Parameters:
    /// - tensor: The Tensor to be sliced.
    /// - start_end_stride: An enum of numbers or tensors that specify the starting points for each dimension.
    /// - startMask: A bitmask that indicates dimensions whose `starts` values the operation should ignore.
    /// - endMask: A bitmask that indicates dimensions whose `ends` values the operation should ignore.
    /// - squeezeMask: A bitmask that indicates dimensions the operation will squeeze out from the result.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object.
    pub fn strided_slice_with_masks<'a>(
        &self,
        tensor: &Tensor,
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
                    sliceTensor: tensor,
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
                    sliceTensor: tensor,
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
}
