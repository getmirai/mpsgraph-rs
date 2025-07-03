use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::{NSArray, NSString};

use crate::core::{create_ns_array_from_i32_slice, create_ns_array_from_i64_slice};
use crate::graph::Graph;
use crate::tensor::DataType;
use crate::tensor::Tensor;
use crate::Shape;

// NOTE: The public helper trait `GraphTensorShapeOps` has been fully inlined
// into inherent methods on `Graph`.  The old trait (and its extension) has
// been removed.

impl Graph {
    pub fn transpose(
        &self,
        x: &Tensor,
        permutation: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let permutation_array = create_ns_array_from_i64_slice(permutation);
            msg_send![self, transposeTensor: x, permutation: &*permutation_array, name: name_ptr]
        }
    }
    pub fn reshape(&self, x: &Tensor, shape: &Shape, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, reshapeTensor: x, withShape: shape.as_ptr(), name: name_ptr]
        }
    }
    pub fn reshape_with_tensor(
        &self,
        x: &Tensor,
        shape_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, reshapeTensor: x, withShapeTensor: shape_tensor, name: name_ptr]
        }
    }
    pub fn flatten2d(&self, x: &Tensor, axis: i64, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, flatten2DTensor: x, axis: axis, name: name_ptr]
        }
    }
    pub fn broadcast(&self, x: &Tensor, shape: &Shape, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, broadcastTensor: x, toShape: shape.as_ptr(), name: name_ptr]
        }
    }
    pub fn shape_of(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, shapeOfTensor: x, name: name_ptr]
        }
    }
    pub fn cast(&self, x: &Tensor, data_type: DataType, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, castTensor: x, toType: data_type as u32, name: name_ptr]
        }
    }
    pub fn stack(&self, tensors: &[&Tensor], axis: i64, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let tensor_array = NSArray::from_slice(tensors);
            msg_send![self, stackTensors: &*tensor_array, axis: axis, name: name_ptr]
        }
    }
    pub fn split(
        &self,
        x: &Tensor,
        num_splits: usize,
        axis: i64,
        name: Option<&str>,
    ) -> Vec<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let result_array_opt: Option<Retained<NSArray<Tensor>>> = msg_send![
                self,
                splitTensor: x,
                numSplits: num_splits,
                axis: axis,
                name: name_ptr
            ];
            result_array_opt.map_or(Vec::new(), |arr| {
                let count = arr.len();
                let mut tensors = Vec::with_capacity(count);
                for i in 0..count {
                    let tensor: Retained<Tensor> = msg_send![&*arr, objectAtIndex: i];
                    tensors.push(tensor);
                }
                tensors
            })
        }
    }
    pub fn slice(
        &self,
        x: &Tensor,
        dimension: usize,
        start: i64,
        length: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, sliceTensor: x, dimension: dimension, start: start, length: length, name: name_ptr]
        }
    }
    pub fn strided_slice(
        &self,
        x: &Tensor,
        starts: &[i32],
        ends: &[i32],
        strides: &[i32],
        start_mask: u32,
        end_mask: u32,
        squeeze_mask: u32,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let starts_array = create_ns_array_from_i32_slice(starts);
            let ends_array = create_ns_array_from_i32_slice(ends);
            let strides_array = create_ns_array_from_i32_slice(strides);
            msg_send![
                self, sliceTensor: x, starts: &*starts_array, ends: &*ends_array, strides: &*strides_array,
                startMask: start_mask, endMask: end_mask, squeezeMask: squeeze_mask, name: name_ptr
            ]
        }
    }
    pub fn strided_slice_with_tensors(
        &self,
        x: &Tensor,
        start_tensor: &Tensor,
        end_tensor: &Tensor,
        stride_tensor: &Tensor,
        start_mask: u32,
        end_mask: u32,
        squeeze_mask: u32,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self, sliceTensor: x, startTensor: start_tensor, endTensor: end_tensor, strideTensor: stride_tensor,
                startMask: start_mask, endMask: end_mask, squeezeMask: squeeze_mask, name: name_ptr
            ]
        }
    }
    pub fn slice_with_tensors(
        &self,
        x: &Tensor,
        start_tensor: &Tensor,
        size_tensor: &Tensor,
        squeeze_mask: u32,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self, sliceTensor: x, startTensor: start_tensor, sizeTensor: size_tensor,
                squeezeMask: squeeze_mask, name: name_ptr
            ]
        }
    }
    pub fn slice_with_arrays(
        &self,
        x: &Tensor,
        starts: &[i64],
        ends: &[i64],
        strides: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let starts_array = create_ns_array_from_i64_slice(starts);
            let ends_array = create_ns_array_from_i64_slice(ends);
            let strides_array = create_ns_array_from_i64_slice(strides);
            msg_send![self, sliceTensor: x, starts: &*starts_array, ends: &*ends_array, strides: &*strides_array, name: name_ptr]
        }
    }
    pub fn slice_gradient(
        &self,
        input_gradient: &Tensor,
        fwd_in_shape: &Tensor,
        starts: &[i64],
        ends: &[i64],
        strides: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let starts_array = create_ns_array_from_i64_slice(starts);
            let ends_array = create_ns_array_from_i64_slice(ends);
            let strides_array = create_ns_array_from_i64_slice(strides);
            msg_send![
                self, sliceGradientTensor: input_gradient, fwdInShapeTensor: fwd_in_shape,
                starts: &*starts_array, ends: &*ends_array, strides: &*strides_array, name: name_ptr
            ]
        }
    }
    pub fn slice_gradient_with_masks(
        &self,
        input_gradient: &Tensor,
        fwd_in_shape: &Tensor,
        starts: &[i64],
        ends: &[i64],
        strides: &[i64],
        start_mask: u32,
        end_mask: u32,
        squeeze_mask: u32,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let starts_array = create_ns_array_from_i64_slice(starts);
            let ends_array = create_ns_array_from_i64_slice(ends);
            let strides_array = create_ns_array_from_i64_slice(strides);
            msg_send![
                self, sliceGradientTensor: input_gradient, fwdInShapeTensor: fwd_in_shape,
                starts: &*starts_array, ends: &*ends_array, strides: &*strides_array,
                startMask: start_mask, endMask: end_mask, squeezeMask: squeeze_mask, name: name_ptr
            ]
        }
    }
    pub fn slice_gradient_with_tensors(
        &self,
        input_gradient: &Tensor,
        fwd_in_shape: &Tensor,
        start_tensor: &Tensor,
        end_tensor: &Tensor,
        stride_tensor: &Tensor,
        start_mask: u32,
        end_mask: u32,
        squeeze_mask: u32,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self, sliceGradientTensor: input_gradient, fwdInShapeTensor: fwd_in_shape,
                startTensor: start_tensor, endTensor: end_tensor, strideTensor: stride_tensor,
                startMask: start_mask, endMask: end_mask, squeezeMask: squeeze_mask, name: name_ptr
            ]
        }
    }
    pub fn slice_gradient_with_size_tensor(
        &self,
        input_gradient: &Tensor,
        fwd_in_shape: &Tensor,
        start_tensor: &Tensor,
        size_tensor: &Tensor,
        squeeze_mask: u32,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self, sliceGradientTensor: input_gradient, fwdInShapeTensor: fwd_in_shape,
                startTensor: start_tensor, sizeTensor: size_tensor, squeezeMask: squeeze_mask, name: name_ptr
            ]
        }
    }
    pub fn slice_update(
        &self,
        data: &Tensor,
        update: &Tensor,
        starts: &[i64],
        ends: &[i64],
        strides: &[i64],
        start_mask: u32,
        end_mask: u32,
        squeeze_mask: u32,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let starts_array = create_ns_array_from_i64_slice(starts);
            let ends_array = create_ns_array_from_i64_slice(ends);
            let strides_array = create_ns_array_from_i64_slice(strides);
            msg_send![
                self, sliceUpdateDataTensor: data, updateTensor: update,
                starts: &*starts_array, ends: &*ends_array, strides: &*strides_array,
                startMask: start_mask, endMask: end_mask, squeezeMask: squeeze_mask, name: name_ptr
            ]
        }
    }
    pub fn slice_update_with_tensors(
        &self,
        data: &Tensor,
        update: &Tensor,
        starts_tensor: &Tensor,
        ends_tensor: &Tensor,
        strides_tensor: &Tensor,
        start_mask: u32,
        end_mask: u32,
        squeeze_mask: u32,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self, sliceUpdateDataTensor: data, updateTensor: update,
                startsTensor: starts_tensor, endsTensor: ends_tensor, stridesTensor: strides_tensor,
                startMask: start_mask, endMask: end_mask, squeezeMask: squeeze_mask, name: name_ptr
            ]
        }
    }
    pub fn slice_update_zero_masks(
        &self,
        data: &Tensor,
        update: &Tensor,
        starts: &[i64],
        ends: &[i64],
        strides: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let starts_array = create_ns_array_from_i64_slice(starts);
            let ends_array = create_ns_array_from_i64_slice(ends);
            let strides_array = create_ns_array_from_i64_slice(strides);
            msg_send![
                self, sliceUpdateDataTensor: data, updateTensor: update,
                starts: &*starts_array, ends: &*ends_array, strides: &*strides_array, name: name_ptr
            ]
        }
    }
    pub fn slice_update_zero_masks_with_tensors(
        &self,
        data: &Tensor,
        update: &Tensor,
        starts_tensor: &Tensor,
        ends_tensor: &Tensor,
        strides_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self, sliceUpdateDataTensor: data, updateTensor: update,
                startsTensor: starts_tensor, endsTensor: ends_tensor, stridesTensor: strides_tensor, name: name_ptr
            ]
        }
    }
    pub fn squeeze(&self, x: &Tensor, axes: &[i64], name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let axes_array = create_ns_array_from_i64_slice(axes);
            msg_send![self, squeezeTensor: x, axes: &*axes_array, name: name_ptr]
        }
    }
    pub fn squeeze_all(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, squeezeTensor: x, name: name_ptr] // Assuming nil axes means all
        }
    }
    pub fn squeeze_axis(&self, x: &Tensor, axis: i64, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, squeezeTensor: x, axis: axis, name: name_ptr]
        }
    }
    pub fn squeeze_with_tensor(
        &self,
        x: &Tensor,
        axes_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, squeezeTensor: x, axesTensor: axes_tensor, name: name_ptr]
        }
    }
    pub fn expand_dims(&self, x: &Tensor, axis: i64, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, expandDimsOfTensor: x, axis: axis, name: name_ptr]
        }
    }
    pub fn expand_dims_axes(
        &self,
        x: &Tensor,
        axes: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let axes_array = create_ns_array_from_i64_slice(axes);
            msg_send![self, expandDimsOfTensor: x, axes: &*axes_array, name: name_ptr]
        }
    }
    pub fn expand_dims_with_tensor(
        &self,
        x: &Tensor,
        axes_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, expandDimsOfTensor: x, axesTensor: axes_tensor, name: name_ptr]
        }
    }
    pub fn tile(&self, x: &Tensor, multiples: &[i64], name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let multiplier_array = create_ns_array_from_i64_slice(multiples);
            msg_send![self, tileTensor: x, withMultiplier: &*multiplier_array, name: name_ptr]
        }
    }
    pub fn pad(
        &self,
        x: &Tensor,
        padding: &[i64],
        constant: f64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let padding_array = create_ns_array_from_i64_slice(padding);
            msg_send![self, padTensor: x, paddings: &*padding_array, constantValue: constant, name: name_ptr]
        }
    }
    pub fn space_to_depth(
        &self,
        x: &Tensor,
        block_size: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, spaceToDepthWithTensor: x, blockSize: block_size, name: name_ptr]
        }
    }
    pub fn depth_to_space(
        &self,
        x: &Tensor,
        block_size: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, depthToSpaceWithTensor: x, blockSize: block_size, name: name_ptr]
        }
    }
    pub fn reverse(&self, x: &Tensor, axes: &[i64], name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let axes_array = create_ns_array_from_i64_slice(axes);
            msg_send![self, reverseTensor: x, axes: &*axes_array, name: name_ptr]
        }
    }
    pub fn concat(
        &self,
        x: &Tensor,
        y: &Tensor,
        dimension: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, concatTensor: x, withTensor: y, dimension: dimension, name: name_ptr]
        }
    }
    pub fn concat_tensors(
        &self,
        tensors: &[&Tensor],
        dimension: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let tensor_array = NSArray::from_slice(tensors);
            msg_send![self, concatTensors: &*tensor_array, dimension: dimension, name: name_ptr]
        }
    }
    pub fn coordinate_along_axis(
        &self,
        axis: i64,
        shape: &Shape,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, coordinateAlongAxis: axis, shape: shape.as_ptr(), name: name_ptr]
        }
    }
    pub fn coordinate_along_axis_tensor(
        &self,
        axis_tensor: &Tensor,
        shape: &Shape,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, coordinateAlongAxisTensor: axis_tensor, withShape: shape.as_ptr(), name: name_ptr]
        }
    }
    pub fn coordinate_along_axis_with_shape_tensor(
        &self,
        axis: i64,
        shape_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, coordinateAlongAxisWithShapeTensor: axis, shapeTensor: shape_tensor, name: name_ptr]
        }
    }
    pub fn coordinate_along_axis_tensor_with_shape_tensor(
        &self,
        axis_tensor: &Tensor,
        shape_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, coordinateAlongAxisTensorWithShapeTensor: axis_tensor, shapeTensor: shape_tensor, name: name_ptr]
        }
    }
}
