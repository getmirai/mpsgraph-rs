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
    /// Creates a transpose operation and returns the result tensor.
    ///
    /// Permutes the dimensions of the input tensor according to `permutation`.
    /// The length of `permutation` must equal the rank of `x` and encode a valid
    /// permutation of the dimensions.
    ///
    /// * `x` – The tensor to be transposed.
    /// * `permutation` – A slice describing the new order of the dimensions.
    /// * `name` – Optional name for the operation.
    ///
    /// Returns the transposed tensor.
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
    /// Creates a reshape operation and returns the result tensor.
    ///
    /// Reshapes the input tensor `x` to match `shape`. The total number of
    /// elements must remain the same. Shape entries may include `-1` to denote
    /// a dimension that should be inferred.
    pub fn reshape(&self, x: &Tensor, shape: &Shape, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, reshapeTensor: x, withShape: shape.as_ptr(), name: name_ptr]
        }
    }
    /// Variant of `reshape` where the target shape is provided as a tensor.
    /// The tensor must be 1-D and contain `i32` or `i64` values.
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
    /// Creates a `flatten2D` operation and returns a rank-2 tensor.
    ///
    /// Dimensions before `axis` are collapsed into the first output dimension,
    /// and dimensions starting at `axis` are collapsed into the second.
    pub fn flatten2d(&self, x: &Tensor, axis: i64, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, flatten2DTensor: x, axis: axis, name: name_ptr]
        }
    }
    /// Creates a broadcast operation that matches `x` to `shape`.
    /// Broadcasting semantics follow those of arithmetic ops – trailing
    /// dimensions are matched first.
    pub fn broadcast(&self, x: &Tensor, shape: &Shape, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, broadcastTensor: x, toShape: shape.as_ptr(), name: name_ptr]
        }
    }
    /// Returns a rank-1 tensor containing the (static) shape of `x`.
    pub fn shape_of(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, shapeOfTensor: x, name: name_ptr]
        }
    }
    /// Creates a cast operation and returns the result tensor.
    ///
    /// Converts the elements of `x` to `data_type` without changing the
    /// underlying values (subject to precision).
    pub fn cast(&self, x: &Tensor, data_type: DataType, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, castTensor: x, toType: data_type as u32, name: name_ptr]
        }
    }
    /// Stacks `tensors` along `axis`, returning a tensor of rank `rank+1`.
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
    /// Splits `x` into `num_splits` tensors of equal size along `axis`.
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
    /// Creates a basic slice operation extracting a contiguous segment along a
    /// single `dimension`.
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
    /// Creates a strided slice operation analogous to TensorFlow's
    /// `tf.strided_slice`.
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
    /// Strided slice where start/end/stride are provided as tensors.
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
    /// Creates a transpose operation swapping two dimensions.
    /// Corresponds to Objective-C selector `transposeTensor:dimension:withDimension:name:`.
    pub fn transpose_dims(
        &self,
        x: &Tensor,
        dim0: usize,
        dim1: usize,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, transposeTensor: x, dimension: dim0, withDimension: dim1, name: name_ptr]
        }
    }
    /// Concatenates `tensors` along `dimension`, optionally `interleave`-ing inputs.
    pub fn concat_tensors_interleave(
        &self,
        tensors: &[&Tensor],
        dimension: i64,
        interleave: bool,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let tensor_array = NSArray::from_slice(tensors);
            msg_send![
                self,
                concatTensors: &*tensor_array,
                dimension: dimension,
                interleave: interleave,
                name: name_ptr
            ]
        }
    }
    /// Tile gradient counterpart for `tileTensor`.
    pub fn tile_gradient(
        &self,
        incoming_gradient: &Tensor,
        source_tensor: &Tensor,
        multiplier: &Shape,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                tileGradientWithIncomingGradientTensor: incoming_gradient,
                sourceTensor: source_tensor,
                withMultiplier: multiplier.as_ptr(),
                name: name_ptr
            ]
        }
    }
    /// Padding gradient counterpart for `padTensor`.
    pub fn pad_gradient(
        &self,
        incoming_gradient: &Tensor,
        source_tensor: &Tensor,
        padding_mode: u32,
        left_padding: &Shape,
        right_padding: &Shape,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                padGradientWithIncomingGradientTensor: incoming_gradient,
                sourceTensor: source_tensor,
                paddingMode: padding_mode,
                leftPadding: left_padding.as_ptr(),
                rightPadding: right_padding.as_ptr(),
                name: name_ptr
            ]
        }
    }
    /// Broadcast variant where the target shape is provided as a tensor.
    pub fn broadcast_with_shape_tensor(
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
            msg_send![self, broadcastTensor: x, toShapeTensor: shape_tensor, name: name_ptr]
        }
    }
    /// flatten2D variant where `axis` is provided as a tensor.
    pub fn flatten2d_axis_tensor(
        &self,
        x: &Tensor,
        axis_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, flatten2DTensor: x, axisTensor: axis_tensor, name: name_ptr]
        }
    }
    /// Reverse tensor along axes specified by a tensor.
    pub fn reverse_with_axes_tensor(
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
            msg_send![self, reverseTensor: x, axesTensor: axes_tensor, name: name_ptr]
        }
    }
    /// Reverse tensor on all axes.
    pub fn reverse_all(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, reverseTensor: x, name: name_ptr]
        }
    }
    /// Splits tensor with explicit `split_sizes` array.
    pub fn split_with_sizes(
        &self,
        x: &Tensor,
        split_sizes: &[i64],
        axis: i64,
        name: Option<&str>,
    ) -> Vec<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let split_sizes_array = create_ns_array_from_i64_slice(split_sizes);
            let result_array_opt: Option<Retained<NSArray<Tensor>>> = msg_send![
                self,
                splitTensor: x,
                splitSizes: &*split_sizes_array,
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
    /// Split tensor where `split_sizes` provided as a tensor.
    pub fn split_with_sizes_tensor(
        &self,
        x: &Tensor,
        split_sizes_tensor: &Tensor,
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
                splitSizesTensor: split_sizes_tensor,
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
    /// Re-interprets the tensor element type without changing underlying data.
    pub fn reinterpret_cast(
        &self,
        x: &Tensor,
        data_type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, reinterpretCastTensor: x, toType: data_type as u32, name: name_ptr]
        }
    }
    /// Space-to-Depth 2-D.
    pub fn space_to_depth_2d(
        &self,
        x: &Tensor,
        width_axis: usize,
        height_axis: usize,
        depth_axis: usize,
        block_size: usize,
        pixel_shuffle: bool,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                spaceToDepth2DTensor: x,
                widthAxis: width_axis,
                heightAxis: height_axis,
                depthAxis: depth_axis,
                blockSize: block_size,
                usePixelShuffleOrder: pixel_shuffle,
                name: name_ptr
            ]
        }
    }
    /// Space-to-Depth 2-D variant with axis tensors.
    pub fn space_to_depth_2d_tensor(
        &self,
        x: &Tensor,
        width_axis_tensor: &Tensor,
        height_axis_tensor: &Tensor,
        depth_axis_tensor: &Tensor,
        block_size: usize,
        pixel_shuffle: bool,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                spaceToDepth2DTensor: x,
                widthAxisTensor: width_axis_tensor,
                heightAxisTensor: height_axis_tensor,
                depthAxisTensor: depth_axis_tensor,
                blockSize: block_size,
                usePixelShuffleOrder: pixel_shuffle,
                name: name_ptr
            ]
        }
    }
    /// Depth-to-Space 2-D.
    pub fn depth_to_space_2d(
        &self,
        x: &Tensor,
        width_axis: usize,
        height_axis: usize,
        depth_axis: usize,
        block_size: usize,
        pixel_shuffle: bool,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                depthToSpace2DTensor: x,
                widthAxis: width_axis,
                heightAxis: height_axis,
                depthAxis: depth_axis,
                blockSize: block_size,
                usePixelShuffleOrder: pixel_shuffle,
                name: name_ptr
            ]
        }
    }
    /// Depth-to-Space 2-D variant with axis tensors.
    pub fn depth_to_space_2d_tensor(
        &self,
        x: &Tensor,
        width_axis_tensor: &Tensor,
        height_axis_tensor: &Tensor,
        depth_axis_tensor: &Tensor,
        block_size: usize,
        pixel_shuffle: bool,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                depthToSpace2DTensor: x,
                widthAxisTensor: width_axis_tensor,
                heightAxisTensor: height_axis_tensor,
                depthAxisTensor: depth_axis_tensor,
                blockSize: block_size,
                usePixelShuffleOrder: pixel_shuffle,
                name: name_ptr
            ]
        }
    }
    /// Space-to-Batch operation.
    pub fn space_to_batch(
        &self,
        x: &Tensor,
        spatial_axes: &[i64],
        batch_axis: i64,
        block_dimensions: &[i64],
        pixel_shuffle: bool,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let spatial_axes_array = create_ns_array_from_i64_slice(spatial_axes);
            let block_dims_array = create_ns_array_from_i64_slice(block_dimensions);
            msg_send![
                self,
                spaceToBatchTensor: x,
                spatialAxes: &*spatial_axes_array,
                batchAxis: batch_axis,
                blockDimensions: &*block_dims_array,
                usePixelShuffleOrder: pixel_shuffle,
                name: name_ptr
            ]
        }
    }
    /// Space-to-Batch variant with tensor parameters.
    pub fn space_to_batch_tensor(
        &self,
        x: &Tensor,
        spatial_axes_tensor: &Tensor,
        batch_axis_tensor: &Tensor,
        block_dimensions_tensor: &Tensor,
        pixel_shuffle: bool,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                spaceToBatchTensor: x,
                spatialAxesTensor: spatial_axes_tensor,
                batchAxisTensor: batch_axis_tensor,
                blockDimensionsTensor: block_dimensions_tensor,
                usePixelShuffleOrder: pixel_shuffle,
                name: name_ptr
            ]
        }
    }
    /// Batch-to-Space operation.
    pub fn batch_to_space(
        &self,
        x: &Tensor,
        spatial_axes: &[i64],
        batch_axis: i64,
        block_dimensions: &[i64],
        pixel_shuffle: bool,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let spatial_axes_array = create_ns_array_from_i64_slice(spatial_axes);
            let block_dims_array = create_ns_array_from_i64_slice(block_dimensions);
            msg_send![
                self,
                batchToSpaceTensor: x,
                spatialAxes: &*spatial_axes_array,
                batchAxis: batch_axis,
                blockDimensions: &*block_dims_array,
                usePixelShuffleOrder: pixel_shuffle,
                name: name_ptr
            ]
        }
    }
    /// Batch-to-Space variant with tensor parameters.
    pub fn batch_to_space_tensor(
        &self,
        x: &Tensor,
        spatial_axes_tensor: &Tensor,
        batch_axis_tensor: &Tensor,
        block_dimensions_tensor: &Tensor,
        pixel_shuffle: bool,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                batchToSpaceTensor: x,
                spatialAxesTensor: spatial_axes_tensor,
                batchAxisTensor: batch_axis_tensor,
                blockDimensionsTensor: block_dimensions_tensor,
                usePixelShuffleOrder: pixel_shuffle,
                name: name_ptr
            ]
        }
    }
}
