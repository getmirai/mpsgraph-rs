use super::StartEndStrideScalarsOrTensors;
use crate::{Graph, ShapeOrTensor, ShapedType, Tensor};
use objc2::{extern_methods, msg_send, rc::Retained};
use objc2_foundation::{NSArray, NSNumber, NSString};

impl Graph {
    /// Creates a strided-slice update operation with zero masks and returns the result tensor.
    ///
    /// - Parameters:
    /// - dataTensor: The large tensor that will receive the update.
    /// - updateTensor: The tensor with the new values that will replace values in the dataTensor.
    /// - startsTensor: A Tensor that contains an array of numbers that specify the starting points for each dimension.
    /// - endsTensor: A Tensor that contains an array of numbers that specify the ending points for each dimension.
    /// - stridesTensor: A Tensor that contains an array of numbers that specify the strides for each dimension.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object
    pub fn slice_update<'a>(
        &self,
        data_tensor: &Tensor,
        update_tensor: &Tensor,
        start_end_stride: StartEndStrideScalarsOrTensors<'a>,
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
                        sliceUpdateDataTensor: data_tensor,
                        updateTensor: update_tensor,
                        starts: &*starts_array,
                        ends: &*ends_array,
                        strides: &*strides_array,
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
                    sliceUpdateDataTensor: data_tensor,
                    updateTensor: update_tensor,
                    startsTensor: start_tensor,
                    endsTensor: end_tensor,
                    stridesTensor: stride_tensor,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }

    /// Creates a strided-slice update operation and returns the result tensor.
    ///
    /// - Parameters:
    /// - dataTensor: The large tensor that will receive the update.
    /// - updateTensor: The tensor with the new values that will replace values in the dataTensor.
    /// - startsTensor: A Tensor that contains an array of numbers that specify the starting points for each dimension.
    /// - endsTensor: A Tensor that contains an array of numbers that specify the ending points for each dimension.
    /// - stridesTensor: A Tensor that contains an array of numbers that specify the strides for each dimension.
    /// - startMask: A bitmask that indicates dimensions whose `starts` values the operation should ignore.
    /// - endMask: A bitmask that indicates dimensions whose `ends` values the operation should ignore.
    /// - squeezeMask: A bitmask that indicates dimensions the operation will squeeze out from the result.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object
    pub fn slice_update_with_masks<'a>(
        &self,
        data_tensor: &Tensor,
        update_tensor: &Tensor,
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
                        sliceUpdateDataTensor: data_tensor,
                        updateTensor: update_tensor,
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
                    sliceUpdateDataTensor: data_tensor,
                    updateTensor: update_tensor,
                    startsTensor: start_tensor,
                    endsTensor: end_tensor,
                    stridesTensor: stride_tensor,
                    startMask: start_mask,
                    endMask: end_mask,
                    squeezeMask: squeeze_mask,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }
}
