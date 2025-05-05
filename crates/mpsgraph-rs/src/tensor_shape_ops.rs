use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::core::create_ns_array_from_i64_slice;
use crate::core::create_ns_array_from_slice;
use crate::graph::Graph;
use crate::tensor::DataType;
use crate::tensor::Tensor;
use crate::Shape;

/// Trait for tensor shape operations on Graph
pub trait GraphTensorShapeOps {
    /// Creates a transpose operation to permute the dimensions of a tensor
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `permutation` - The permutation to apply to the tensor's dimensions
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn transpose(
        &self,
        x: &Retained<Tensor>,
        permutation: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a reshape operation
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `shape` - New shape for the tensor
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn reshape(&self, x: &Retained<Tensor>, shape: &Shape, name: Option<&str>) -> Retained<Tensor>;

    /// Creates a reshape operation using a shape tensor
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `shape_tensor` - Tensor specifying the new shape
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn reshape_with_tensor(
        &self,
        x: &Retained<Tensor>,
        shape_tensor: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a flatten2D operation
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `axis` - Axis to flatten at
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn flatten2d(&self, x: &Retained<Tensor>, axis: i64, name: Option<&str>) -> Retained<Tensor>;

    /// Creates a broadcast operation
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `shape` - Target shape to broadcast to
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn broadcast(
        &self,
        x: &Retained<Tensor>,
        shape: &Shape,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a shape-of operation
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object containing the shape
    fn shape_of(&self, x: &Retained<Tensor>, name: Option<&str>) -> Retained<Tensor>;

    /// Creates a cast operation to change tensor data type
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `data_type` - Target data type
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object with the new data type
    fn cast(
        &self,
        x: &Retained<Tensor>,
        data_type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a stack operation to combine tensors along a new axis
    ///
    /// # Arguments
    ///
    /// * `tensors` - Array of tensors to stack
    /// * `axis` - Axis along which to stack
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn stack(
        &self,
        tensors: &[&Retained<Tensor>],
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a split operation to split a tensor into multiple parts
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `num_splits` - Number of splits to create
    /// * `axis` - Axis along which to split
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A vector of Tensor objects
    fn split(
        &self,
        x: &Retained<Tensor>,
        num_splits: u64,
        axis: i64,
        name: Option<&str>,
    ) -> Vec<Retained<Tensor>>;

    /// Creates a slice operation to get a portion of a tensor
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `dimension` - The dimension to slice
    /// * `start` - The starting index of the slice
    /// * `length` - The length of the slice
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn slice(
        &self,
        x: &Retained<Tensor>,
        dimension: usize,
        start: i64,
        length: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a strided slice operation to get a more flexible slice of a tensor
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `starts` - Starting indices for each dimension
    /// * `ends` - Ending indices for each dimension
    /// * `strides` - Strides for each dimension
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn slice_with_arrays(
        &self,
        x: &Retained<Tensor>,
        starts: &[i64],
        ends: &[i64],
        strides: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a strided slice operation to get a more flexible slice of a tensor
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `starts` - Starting indices for each dimension
    /// * `ends` - Ending indices for each dimension
    /// * `strides` - Strides for each dimension
    /// * `begin_mask` - Mask for start indices (0/1 per dimension)
    /// * `end_mask` - Mask for end indices (0/1 per dimension)
    /// * `shrink_axis_mask` - Mask for dimensions to shrink (0/1 per dimension)
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn strided_slice(
        &self,
        x: &Retained<Tensor>,
        starts: &[i64],
        ends: &[i64],
        strides: &[i64],
        begin_mask: i64,
        end_mask: i64,
        shrink_axis_mask: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a strided slice operation using tensor parameters
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `start_tensor` - Tensor containing starting indices for each dimension
    /// * `end_tensor` - Tensor containing ending indices for each dimension
    /// * `stride_tensor` - Tensor containing strides for each dimension
    /// * `begin_mask` - Mask for start indices (0/1 per dimension)
    /// * `end_mask` - Mask for end indices (0/1 per dimension)
    /// * `shrink_axis_mask` - Mask for dimensions to shrink (0/1 per dimension)
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn strided_slice_with_tensors(
        &self,
        x: &Retained<Tensor>,
        start_tensor: &Retained<Tensor>,
        end_tensor: &Retained<Tensor>,
        stride_tensor: &Retained<Tensor>,
        begin_mask: i64,
        end_mask: i64,
        shrink_axis_mask: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a slice operation using tensor parameters for start and size
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `start_tensor` - Tensor containing starting indices for each dimension
    /// * `size_tensor` - Tensor containing sizes for each dimension
    /// * `shrink_axis_mask` - Mask for dimensions to shrink (0/1 per dimension)
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn slice_with_tensors(
        &self,
        x: &Retained<Tensor>,
        start_tensor: &Retained<Tensor>,
        size_tensor: &Retained<Tensor>,
        shrink_axis_mask: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a strided slice gradient operation
    ///
    /// # Arguments
    ///
    /// * `input_gradient` - The input gradient tensor
    /// * `fwd_in_shape` - The shape of the forward pass input
    /// * `starts` - Starting indices for each dimension
    /// * `ends` - Ending indices for each dimension
    /// * `strides` - Strides for each dimension
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn slice_gradient(
        &self,
        input_gradient: &Retained<Tensor>,
        fwd_in_shape: &Retained<Tensor>,
        starts: &[i64],
        ends: &[i64],
        strides: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a strided slice gradient operation
    ///
    /// # Arguments
    ///
    /// * `input_gradient` - The input gradient tensor
    /// * `fwd_in_shape` - The shape of the forward pass input
    /// * `starts` - Starting indices for each dimension
    /// * `ends` - Ending indices for each dimension
    /// * `strides` - Strides for each dimension
    /// * `begin_mask` - Mask for start indices (0/1 per dimension)
    /// * `end_mask` - Mask for end indices (0/1 per dimension)
    /// * `shrink_axis_mask` - Mask for dimensions to shrink (0/1 per dimension)
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn slice_gradient_with_masks(
        &self,
        input_gradient: &Retained<Tensor>,
        fwd_in_shape: &Retained<Tensor>,
        starts: &[i64],
        ends: &[i64],
        strides: &[i64],
        begin_mask: i64,
        end_mask: i64,
        shrink_axis_mask: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a strided slice gradient operation with tensor parameters
    ///
    /// # Arguments
    ///
    /// * `input_gradient` - The input gradient tensor
    /// * `fwd_in_shape` - The shape of the forward pass input
    /// * `start_tensor` - Tensor containing starting indices for each dimension
    /// * `end_tensor` - Tensor containing ending indices for each dimension
    /// * `stride_tensor` - Tensor containing strides for each dimension
    /// * `begin_mask` - Mask for start indices (0/1 per dimension)
    /// * `end_mask` - Mask for end indices (0/1 per dimension)
    /// * `shrink_axis_mask` - Mask for dimensions to shrink (0/1 per dimension)
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn slice_gradient_with_tensors(
        &self,
        input_gradient: &Retained<Tensor>,
        fwd_in_shape: &Retained<Tensor>,
        start_tensor: &Retained<Tensor>,
        end_tensor: &Retained<Tensor>,
        stride_tensor: &Retained<Tensor>,
        begin_mask: i64,
        end_mask: i64,
        shrink_axis_mask: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a slice gradient operation with tensor parameters for start and size
    ///
    /// # Arguments
    ///
    /// * `input_gradient` - The input gradient tensor
    /// * `fwd_in_shape` - The shape of the forward pass input
    /// * `start_tensor` - Tensor containing starting indices for each dimension
    /// * `size_tensor` - Tensor containing sizes for each dimension
    /// * `shrink_axis_mask` - Mask for dimensions to shrink (0/1 per dimension)
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn slice_gradient_with_size_tensor(
        &self,
        input_gradient: &Retained<Tensor>,
        fwd_in_shape: &Retained<Tensor>,
        start_tensor: &Retained<Tensor>,
        size_tensor: &Retained<Tensor>,
        shrink_axis_mask: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a strided slice update operation
    ///
    /// # Arguments
    ///
    /// * `data` - The large tensor that will receive the update
    /// * `update` - The tensor with new values that will replace values in data
    /// * `starts` - Starting indices for each dimension
    /// * `ends` - Ending indices for each dimension
    /// * `strides` - Strides for each dimension
    /// * `begin_mask` - Mask for start indices (0/1 per dimension)
    /// * `end_mask` - Mask for end indices (0/1 per dimension)
    /// * `shrink_axis_mask` - Mask for dimensions to shrink (0/1 per dimension)
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn slice_update(
        &self,
        data: &Retained<Tensor>,
        update: &Retained<Tensor>,
        starts: &[i64],
        ends: &[i64],
        strides: &[i64],
        begin_mask: i64,
        end_mask: i64,
        shrink_axis_mask: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a strided slice update operation with tensor parameters
    ///
    /// # Arguments
    ///
    /// * `data` - The large tensor that will receive the update
    /// * `update` - The tensor with new values that will replace values in data
    /// * `starts_tensor` - Tensor containing starting indices for each dimension
    /// * `ends_tensor` - Tensor containing ending indices for each dimension
    /// * `strides_tensor` - Tensor containing strides for each dimension
    /// * `begin_mask` - Mask for start indices (0/1 per dimension)
    /// * `end_mask` - Mask for end indices (0/1 per dimension)
    /// * `shrink_axis_mask` - Mask for dimensions to shrink (0/1 per dimension)
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn slice_update_with_tensors(
        &self,
        data: &Retained<Tensor>,
        update: &Retained<Tensor>,
        starts_tensor: &Retained<Tensor>,
        ends_tensor: &Retained<Tensor>,
        strides_tensor: &Retained<Tensor>,
        begin_mask: i64,
        end_mask: i64,
        shrink_axis_mask: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a strided slice update operation with zero masks
    ///
    /// # Arguments
    ///
    /// * `data` - The large tensor that will receive the update
    /// * `update` - The tensor with new values that will replace values in data
    /// * `starts` - Starting indices for each dimension
    /// * `ends` - Ending indices for each dimension
    /// * `strides` - Strides for each dimension
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn slice_update_zero_masks(
        &self,
        data: &Retained<Tensor>,
        update: &Retained<Tensor>,
        starts: &[i64],
        ends: &[i64],
        strides: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a strided slice update operation with zero masks and tensor parameters
    ///
    /// # Arguments
    ///
    /// * `data` - The large tensor that will receive the update
    /// * `update` - The tensor with new values that will replace values in data
    /// * `starts_tensor` - Tensor containing starting indices for each dimension
    /// * `ends_tensor` - Tensor containing ending indices for each dimension
    /// * `strides_tensor` - Tensor containing strides for each dimension
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn slice_update_zero_masks_with_tensors(
        &self,
        data: &Retained<Tensor>,
        update: &Retained<Tensor>,
        starts_tensor: &Retained<Tensor>,
        ends_tensor: &Retained<Tensor>,
        strides_tensor: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a squeeze operation to remove dimensions of size 1 at specified axes
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `axes` - Axes to squeeze
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn squeeze(&self, x: &Retained<Tensor>, axes: &[i64], name: Option<&str>) -> Retained<Tensor>;

    /// Creates a squeeze operation to remove all dimensions of size 1
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn squeeze_all(&self, x: &Retained<Tensor>, name: Option<&str>) -> Retained<Tensor>;

    /// Creates a squeeze operation to remove a dimension of size 1 at the specified axis
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `axis` - The axis to squeeze
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn squeeze_axis(&self, x: &Retained<Tensor>, axis: i64, name: Option<&str>)
        -> Retained<Tensor>;

    /// Creates a squeeze operation to remove dimensions with size 1 specified by a tensor
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `axes_tensor` - The tensor containing the axes to squeeze
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn squeeze_with_tensor(
        &self,
        x: &Retained<Tensor>,
        axes_tensor: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates an expand_dims operation to insert a dimension of size 1 at the specified axis
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `axis` - The axis to expand
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn expand_dims(&self, x: &Retained<Tensor>, axis: i64, name: Option<&str>) -> Retained<Tensor>;

    /// Creates an expand_dims operation to insert dimensions of size 1 at specified axes
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `axes` - Axes at which to insert new dimensions
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn expand_dims_axes(
        &self,
        x: &Retained<Tensor>,
        axes: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates an expand_dims operation to insert dimensions with size 1 specified by a tensor
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `axes_tensor` - The tensor containing the axes to expand
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn expand_dims_with_tensor(
        &self,
        x: &Retained<Tensor>,
        axes_tensor: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a tile operation to repeat a tensor along specified dimensions
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `multiples` - Number of times to repeat in each dimension
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn tile(&self, x: &Retained<Tensor>, multiples: &[i64], name: Option<&str>)
        -> Retained<Tensor>;

    /// Creates a pad operation to pad a tensor
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `padding` - Padding specification
    /// * `constant` - Value to use for padding
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn pad(
        &self,
        x: &Retained<Tensor>,
        padding: &[i64],
        constant: f32,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a space-to-depth operation
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `block_size` - Size of spatial blocks
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn space_to_depth(
        &self,
        x: &Retained<Tensor>,
        block_size: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a depth-to-space operation
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `block_size` - Size of spatial blocks
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn depth_to_space(
        &self,
        x: &Retained<Tensor>,
        block_size: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a reverse operation to reverse a tensor along specified dimensions
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `axes` - Axes along which to reverse
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn reverse(&self, x: &Retained<Tensor>, axes: &[i64], name: Option<&str>) -> Retained<Tensor>;

    /// Creates a concatenation operation to concatenate two tensors
    ///
    /// # Arguments
    ///
    /// * `x` - The first tensor to concatenate
    /// * `y` - The second tensor to concatenate
    /// * `dimension` - The dimension to concatenate across
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn concat(
        &self,
        x: &Retained<Tensor>,
        y: &Retained<Tensor>,
        dimension: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a concatenation operation to concatenate multiple tensors
    ///
    /// # Arguments
    ///
    /// * `tensors` - The tensors to concatenate
    /// * `dimension` - The dimension to concatenate across
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn concat_tensors(
        &self,
        tensors: &[&Retained<Tensor>],
        dimension: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a coordinate tensor with values set to the coordinate along the specified axis
    ///
    /// # Arguments
    ///
    /// * `axis` - The coordinate axis to set values to
    /// * `shape` - The shape of the result tensor
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn coordinate_along_axis(
        &self,
        axis: i64,
        shape: &Shape,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a coordinate tensor with values set to the coordinate along the specified axis tensor
    ///
    /// # Arguments
    ///
    /// * `axis_tensor` - Tensor specifying the coordinate axis
    /// * `shape` - The shape of the result tensor
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn coordinate_along_axis_tensor(
        &self,
        axis_tensor: &Retained<Tensor>,
        shape: &Shape,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a coordinate tensor with values set to the coordinate along the specified axis using a shape tensor
    ///
    /// # Arguments
    ///
    /// * `axis` - The coordinate axis to set values to
    /// * `shape_tensor` - Tensor specifying the shape of the result
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn coordinate_along_axis_with_shape_tensor(
        &self,
        axis: i64,
        shape_tensor: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a coordinate tensor with values set to the coordinate along the specified axis tensor using a shape tensor
    ///
    /// # Arguments
    ///
    /// * `axis_tensor` - Tensor specifying the coordinate axis
    /// * `shape_tensor` - Tensor specifying the shape of the result
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn coordinate_along_axis_tensor_with_shape_tensor(
        &self,
        axis_tensor: &Retained<Tensor>,
        shape_tensor: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor>;
}

impl GraphTensorShapeOps for Graph {
    fn transpose(
        &self,
        x: &Retained<Tensor>,
        permutation: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let permutation_array = create_ns_array_from_i64_slice(permutation);
            let permutation_ptr = &*permutation_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                transposeTensor: &**x,
                dimension: permutation_ptr,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create transpose operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn slice(
        &self,
        x: &Retained<Tensor>,
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

            let result: *mut Tensor = msg_send![
                self,
                sliceTensor: &**x,
                dimension: dimension,
                start: start,
                length: length,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create slice operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn slice_with_arrays(
        &self,
        x: &Retained<Tensor>,
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

            // Create NSArrays from slices
            let starts_array = create_ns_array_from_i64_slice(starts);
            let starts_ptr = &*starts_array as *const _;

            let ends_array = create_ns_array_from_i64_slice(ends);
            let ends_ptr = &*ends_array as *const _;

            let strides_array = create_ns_array_from_i64_slice(strides);
            let strides_ptr = &*strides_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                sliceTensor: &**x,
                starts: starts_ptr,
                ends: ends_ptr,
                strides: strides_ptr,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create slice_with_arrays operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn strided_slice(
        &self,
        x: &Retained<Tensor>,
        starts: &[i64],
        ends: &[i64],
        strides: &[i64],
        begin_mask: i64,
        end_mask: i64,
        shrink_axis_mask: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            // Create NSArrays from slices
            let starts_array = create_ns_array_from_i64_slice(starts);
            let starts_ptr = &*starts_array as *const _;

            let ends_array = create_ns_array_from_i64_slice(ends);
            let ends_ptr = &*ends_array as *const _;

            let strides_array = create_ns_array_from_i64_slice(strides);
            let strides_ptr = &*strides_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                sliceTensor: &**x,
                starts: starts_ptr,
                ends: ends_ptr,
                strides: strides_ptr,
                startMask: begin_mask as u32,
                endMask: end_mask as u32,
                squeezeMask: shrink_axis_mask as u32,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create strided_slice operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn strided_slice_with_tensors(
        &self,
        x: &Retained<Tensor>,
        start_tensor: &Retained<Tensor>,
        end_tensor: &Retained<Tensor>,
        stride_tensor: &Retained<Tensor>,
        begin_mask: i64,
        end_mask: i64,
        shrink_axis_mask: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                sliceTensor: &**x,
                startTensor: &**start_tensor,
                endTensor: &**end_tensor,
                strideTensor: &**stride_tensor,
                startMask: begin_mask as u32,
                endMask: end_mask as u32,
                squeezeMask: shrink_axis_mask as u32,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create strided_slice_with_tensors operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn slice_with_tensors(
        &self,
        x: &Retained<Tensor>,
        start_tensor: &Retained<Tensor>,
        size_tensor: &Retained<Tensor>,
        shrink_axis_mask: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                sliceTensor: &**x,
                startTensor: &**start_tensor,
                sizeTensor: &**size_tensor,
                squeezeMask: shrink_axis_mask as u32,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create slice_with_tensors operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn reshape(&self, x: &Retained<Tensor>, shape: &Shape, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                reshapeTensor: &**x,
                withShape: shape.as_ptr(),
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create reshape operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn reshape_with_tensor(
        &self,
        x: &Retained<Tensor>,
        shape_tensor: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                reshapeTensor: &**x,
                withShapeTensor: &**shape_tensor,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create reshape_with_tensor operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn flatten2d(&self, x: &Retained<Tensor>, axis: i64, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                flatten2DTensor: &**x,
                axis: axis,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create flatten2d operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn broadcast(
        &self,
        x: &Retained<Tensor>,
        shape: &Shape,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                broadcastTensor: &**x,
                toShape: shape.as_ptr(),
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create broadcast operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn shape_of(&self, x: &Retained<Tensor>, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                shapeOfTensor: &**x,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create shape_of operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn cast(
        &self,
        x: &Retained<Tensor>,
        data_type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                castTensor: &**x,
                toType: data_type as u32,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create cast operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn stack(
        &self,
        tensors: &[&Retained<Tensor>],
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            // Create array of tensors
            let tensor_ptrs: Vec<*const Tensor> =
                tensors.iter().map(|t| &***t as *const Tensor).collect();

            let tensor_array = create_ns_array_from_slice(&tensor_ptrs);
            let tensor_array_ptr = &*tensor_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                stackTensors: tensor_array_ptr,
                axis: axis,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create stack operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn split(
        &self,
        x: &Retained<Tensor>,
        num_splits: u64,
        axis: i64,
        name: Option<&str>,
    ) -> Vec<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result_array: *mut objc2_foundation::NSArray<Tensor> = msg_send![
                self,
                splitTensor: &**x,
                numSplits: num_splits,
                axis: axis,
                name: name_ptr
            ];

            if result_array.is_null() {
                return Vec::new();
            }

            // Convert NSArray to Vec<Retained<Tensor>>
            let count = (*result_array).count();
            let mut tensors = Vec::with_capacity(count);

            for i in 0..count {
                let tensor: *mut Tensor = msg_send![result_array, objectAtIndex: i];
                if !tensor.is_null() {
                    tensors.push(Retained::from_raw(tensor).unwrap());
                }
            }

            tensors
        }
    }

    fn squeeze(&self, x: &Retained<Tensor>, axes: &[i64], name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let axes_array = create_ns_array_from_i64_slice(axes);
            let axes_ptr = &*axes_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                squeezeTensor: &**x,
                axes: axes_ptr,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create squeeze operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn expand_dims(&self, x: &Retained<Tensor>, axis: i64, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                expandDimsOfTensor: &**x,
                axis: axis,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create expand_dims operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn expand_dims_axes(
        &self,
        x: &Retained<Tensor>,
        axes: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let axes_array = create_ns_array_from_i64_slice(axes);
            let axes_ptr = &*axes_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                expandDimsOfTensor: &**x,
                axes: axes_ptr,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create expand_dims_axes operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn tile(
        &self,
        x: &Retained<Tensor>,
        multiples: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let multiples_array = create_ns_array_from_i64_slice(multiples);
            let multiples_ptr = &*multiples_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                tileTensor: &**x,
                withMultiples: multiples_ptr,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create tile operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn pad(
        &self,
        x: &Retained<Tensor>,
        padding: &[i64],
        constant: f32,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let padding_array = create_ns_array_from_i64_slice(padding);
            let padding_ptr = &*padding_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                padTensor: &**x,
                paddings: padding_ptr,
                constantValue: constant as f64,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create pad operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn space_to_depth(
        &self,
        x: &Retained<Tensor>,
        block_size: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                spaceToDepthWithTensor: &**x,
                blockSize: block_size,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create space_to_depth operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn depth_to_space(
        &self,
        x: &Retained<Tensor>,
        block_size: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                depthToSpaceWithTensor: &**x,
                blockSize: block_size,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create depth_to_space operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn reverse(&self, x: &Retained<Tensor>, axes: &[i64], name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let axes_array = create_ns_array_from_i64_slice(axes);
            let axes_ptr = &*axes_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                reverseTensor: &**x,
                axes: axes_ptr,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create reverse operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn slice_gradient(
        &self,
        input_gradient: &Retained<Tensor>,
        fwd_in_shape: &Retained<Tensor>,
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

            // Create NSArrays from slices
            let starts_array = create_ns_array_from_i64_slice(starts);
            let starts_ptr = &*starts_array as *const _;

            let ends_array = create_ns_array_from_i64_slice(ends);
            let ends_ptr = &*ends_array as *const _;

            let strides_array = create_ns_array_from_i64_slice(strides);
            let strides_ptr = &*strides_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                sliceGradientTensor: &**input_gradient,
                fwdInShapeTensor: &**fwd_in_shape,
                starts: starts_ptr,
                ends: ends_ptr,
                strides: strides_ptr,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create slice_gradient operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn slice_gradient_with_masks(
        &self,
        input_gradient: &Retained<Tensor>,
        fwd_in_shape: &Retained<Tensor>,
        starts: &[i64],
        ends: &[i64],
        strides: &[i64],
        begin_mask: i64,
        end_mask: i64,
        shrink_axis_mask: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            // Create NSArrays from slices
            let starts_array = create_ns_array_from_i64_slice(starts);
            let starts_ptr = &*starts_array as *const _;

            let ends_array = create_ns_array_from_i64_slice(ends);
            let ends_ptr = &*ends_array as *const _;

            let strides_array = create_ns_array_from_i64_slice(strides);
            let strides_ptr = &*strides_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                sliceGradientTensor: &**input_gradient,
                fwdInShapeTensor: &**fwd_in_shape,
                starts: starts_ptr,
                ends: ends_ptr,
                strides: strides_ptr,
                startMask: begin_mask as u32,
                endMask: end_mask as u32,
                squeezeMask: shrink_axis_mask as u32,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create slice_gradient_with_masks operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn slice_gradient_with_tensors(
        &self,
        input_gradient: &Retained<Tensor>,
        fwd_in_shape: &Retained<Tensor>,
        start_tensor: &Retained<Tensor>,
        end_tensor: &Retained<Tensor>,
        stride_tensor: &Retained<Tensor>,
        begin_mask: i64,
        end_mask: i64,
        shrink_axis_mask: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                sliceGradientTensor: &**input_gradient,
                fwdInShapeTensor: &**fwd_in_shape,
                startTensor: &**start_tensor,
                endTensor: &**end_tensor,
                strideTensor: &**stride_tensor,
                startMask: begin_mask as u32,
                endMask: end_mask as u32,
                squeezeMask: shrink_axis_mask as u32,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create slice_gradient_with_tensors operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn slice_gradient_with_size_tensor(
        &self,
        input_gradient: &Retained<Tensor>,
        fwd_in_shape: &Retained<Tensor>,
        start_tensor: &Retained<Tensor>,
        size_tensor: &Retained<Tensor>,
        shrink_axis_mask: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                sliceGradientTensor: &**input_gradient,
                fwdInShapeTensor: &**fwd_in_shape,
                startTensor: &**start_tensor,
                sizeTensor: &**size_tensor,
                squeezeMask: shrink_axis_mask as u32,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create slice_gradient_with_size_tensor operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn slice_update(
        &self,
        data: &Retained<Tensor>,
        update: &Retained<Tensor>,
        starts: &[i64],
        ends: &[i64],
        strides: &[i64],
        begin_mask: i64,
        end_mask: i64,
        shrink_axis_mask: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            // Create NSArrays from slices
            let starts_array = create_ns_array_from_i64_slice(starts);
            let starts_ptr = &*starts_array as *const _;

            let ends_array = create_ns_array_from_i64_slice(ends);
            let ends_ptr = &*ends_array as *const _;

            let strides_array = create_ns_array_from_i64_slice(strides);
            let strides_ptr = &*strides_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                sliceUpdateDataTensor: &**data,
                updateTensor: &**update,
                starts: starts_ptr,
                ends: ends_ptr,
                strides: strides_ptr,
                startMask: begin_mask as u32,
                endMask: end_mask as u32,
                squeezeMask: shrink_axis_mask as u32,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create slice_update operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn slice_update_with_tensors(
        &self,
        data: &Retained<Tensor>,
        update: &Retained<Tensor>,
        starts_tensor: &Retained<Tensor>,
        ends_tensor: &Retained<Tensor>,
        strides_tensor: &Retained<Tensor>,
        begin_mask: i64,
        end_mask: i64,
        shrink_axis_mask: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                sliceUpdateDataTensor: &**data,
                updateTensor: &**update,
                startsTensor: &**starts_tensor,
                endsTensor: &**ends_tensor,
                stridesTensor: &**strides_tensor,
                startMask: begin_mask as u32,
                endMask: end_mask as u32,
                squeezeMask: shrink_axis_mask as u32,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create slice_update_with_tensors operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn slice_update_zero_masks(
        &self,
        data: &Retained<Tensor>,
        update: &Retained<Tensor>,
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

            // Create NSArrays from slices
            let starts_array = create_ns_array_from_i64_slice(starts);
            let starts_ptr = &*starts_array as *const _;

            let ends_array = create_ns_array_from_i64_slice(ends);
            let ends_ptr = &*ends_array as *const _;

            let strides_array = create_ns_array_from_i64_slice(strides);
            let strides_ptr = &*strides_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                sliceUpdateDataTensor: &**data,
                updateTensor: &**update,
                starts: starts_ptr,
                ends: ends_ptr,
                strides: strides_ptr,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create slice_update_zero_masks operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn slice_update_zero_masks_with_tensors(
        &self,
        data: &Retained<Tensor>,
        update: &Retained<Tensor>,
        starts_tensor: &Retained<Tensor>,
        ends_tensor: &Retained<Tensor>,
        strides_tensor: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                sliceUpdateDataTensor: &**data,
                updateTensor: &**update,
                startsTensor: &**starts_tensor,
                endsTensor: &**ends_tensor,
                stridesTensor: &**strides_tensor,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create slice_update_zero_masks_with_tensors operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn squeeze_all(&self, x: &Retained<Tensor>, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                squeezeTensor: &**x,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create squeeze_all operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn squeeze_axis(
        &self,
        x: &Retained<Tensor>,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                squeezeTensor: &**x,
                axis: axis,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create squeeze_axis operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn squeeze_with_tensor(
        &self,
        x: &Retained<Tensor>,
        axes_tensor: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                squeezeTensor: &**x,
                axesTensor: &**axes_tensor,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create squeeze_with_tensor operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn expand_dims_with_tensor(
        &self,
        x: &Retained<Tensor>,
        axes_tensor: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                expandDimsOfTensor: &**x,
                axesTensor: &**axes_tensor,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create expand_dims_with_tensor operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn concat(
        &self,
        x: &Retained<Tensor>,
        y: &Retained<Tensor>,
        dimension: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                concatTensor: &**x,
                withTensor: &**y,
                dimension: dimension,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create concat operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn concat_tensors(
        &self,
        tensors: &[&Retained<Tensor>],
        dimension: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            // Create NSArray from tensors
            let tensor_ptrs: Vec<*const Tensor> =
                tensors.iter().map(|t| &***t as *const Tensor).collect();

            let tensor_array = create_ns_array_from_slice(&tensor_ptrs);
            let tensor_array_ptr = &*tensor_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                concatTensors: tensor_array_ptr,
                dimension: dimension,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create concat_tensors operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn coordinate_along_axis(
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

            let result: *mut Tensor = msg_send![
                self,
                coordinateAlongAxis: axis,
                shape: shape.as_ptr(),
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create coordinate_along_axis operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn coordinate_along_axis_tensor(
        &self,
        axis_tensor: &Retained<Tensor>,
        shape: &Shape,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                coordinateAlongAxisTensor: &**axis_tensor,
                withShape: shape.as_ptr(),
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create coordinate_along_axis_tensor operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn coordinate_along_axis_with_shape_tensor(
        &self,
        axis: i64,
        shape_tensor: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                coordinateAlongAxisWithShapeTensor: axis,
                shapeTensor: &**shape_tensor,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create coordinate_along_axis_with_shape_tensor operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn coordinate_along_axis_tensor_with_shape_tensor(
        &self,
        axis_tensor: &Retained<Tensor>,
        shape_tensor: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                coordinateAlongAxisTensorWithShapeTensor: &**axis_tensor,
                shapeTensor: &**shape_tensor,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create coordinate_along_axis_tensor_with_shape_tensor operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }
}

/// Extension trait providing a method for Graph to access tensor shape operations
pub trait GraphTensorShapeOpsExtension {
    /// Access tensor shape operations for this graph
    fn tensor_shape_ops(&self) -> &dyn GraphTensorShapeOps;
}

impl GraphTensorShapeOpsExtension for Graph {
    fn tensor_shape_ops(&self) -> &dyn GraphTensorShapeOps {
        self
    }
}
