use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::shape::Shape;
use crate::tensor::Tensor;

/// Scatter operation mode
#[repr(i64)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ScatterMode {
    /// Add values
    Add = 0,
    /// Subtract values
    Sub = 1,
    /// Multiply values
    Mul = 2,
    /// Divide values
    Div = 3,
    /// Take minimum value
    Min = 4,
    /// Take maximum value
    Max = 5,
    /// Set value (overwrite)
    Set = 6,
}

/// Trait for performing scatter and scatter ND operations on a graph
pub trait GraphScatterNdOps {
    /// Creates a ScatterND operation and returns the result tensor.
    ///
    /// Scatters the slices in updates_tensor to the result tensor along the indices in indices_tensor.
    ///
    /// # Arguments
    ///
    /// * `updates_tensor` - Tensor containing slices to be inserted into the result tensor
    /// * `indices_tensor` - Tensor containing the result indices to insert slices at
    /// * `shape` - The shape of the result tensor
    /// * `batch_dimensions` - The number of batch dimensions
    /// * `mode` - The type of update to use on the destination
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object or None if error
    fn scatter_nd(
        &self,
        updates_tensor: &Tensor,
        indices_tensor: &Tensor,
        shape: &Shape,
        batch_dimensions: usize,
        mode: ScatterMode,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a ScatterND operation with Add mode.
    ///
    /// # Arguments
    ///
    /// * `updates_tensor` - Tensor containing slices to be inserted into the result tensor
    /// * `indices_tensor` - Tensor containing the result indices to insert slices at
    /// * `shape` - The shape of the result tensor
    /// * `batch_dimensions` - The number of batch dimensions
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object or None if error
    fn scatter_nd_add(
        &self,
        updates_tensor: &Tensor,
        indices_tensor: &Tensor,
        shape: &Shape,
        batch_dimensions: usize,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a ScatterND operation with a data tensor as base.
    ///
    /// Scatters the slices in updates_tensor to the result tensor along the indices in indices_tensor, on top of data_tensor.
    ///
    /// # Arguments
    ///
    /// * `data_tensor` - Tensor containing initial values of same shape as result tensor
    /// * `updates_tensor` - Tensor containing slices to be inserted into the result tensor
    /// * `indices_tensor` - Tensor containing the result indices to insert slices at
    /// * `batch_dimensions` - The number of batch dimensions
    /// * `mode` - The type of update to use on the destination
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object or None if error
    fn scatter_nd_with_data(
        &self,
        data_tensor: &Tensor,
        updates_tensor: &Tensor,
        indices_tensor: &Tensor,
        batch_dimensions: usize,
        mode: ScatterMode,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a Scatter operation and returns the result tensor.
    ///
    /// Scatters the values in updates_tensor to the result tensor along the indices in indices_tensor.
    ///
    /// # Arguments
    ///
    /// * `updates_tensor` - Tensor containing values to be inserted into the result tensor
    /// * `indices_tensor` - Tensor containing the result indices to insert values at
    /// * `shape` - The shape of the result tensor
    /// * `axis` - The axis of the result tensor to scatter values along
    /// * `mode` - The type of update to use on the destination
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object or None if error
    fn scatter(
        &self,
        updates_tensor: &Tensor,
        indices_tensor: &Tensor,
        shape: &Shape,
        axis: i64,
        mode: ScatterMode,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a Scatter operation with a data tensor as base.
    ///
    /// Scatters the values in updates_tensor to the result tensor along the indices in indices_tensor, on top of data_tensor.
    ///
    /// # Arguments
    ///
    /// * `data_tensor` - Tensor containing initial values of same shape as result tensor
    /// * `updates_tensor` - Tensor containing values to be inserted into the result tensor
    /// * `indices_tensor` - Tensor containing the result indices to insert values at
    /// * `axis` - The axis of the result tensor to scatter values along
    /// * `mode` - The type of update to use on the destination
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object or None if error
    fn scatter_with_data(
        &self,
        data_tensor: &Tensor,
        updates_tensor: &Tensor,
        indices_tensor: &Tensor,
        axis: i64,
        mode: ScatterMode,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a ScatterAlongAxis operation.
    ///
    /// Scatter values from `updates_tensor` along the specified `axis` at indices in `indices_tensor` into a result tensor.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis to scatter to
    /// * `updates_tensor` - The input tensor to scatter values from
    /// * `indices_tensor` - Int32 or Int64 tensor used to index the result tensor
    /// * `shape` - The shape of the result tensor
    /// * `mode` - The type of update to use
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object or None if error
    fn scatter_along_axis(
        &self,
        axis: i64,
        updates_tensor: &Tensor,
        indices_tensor: &Tensor,
        shape: &Shape,
        mode: ScatterMode,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a ScatterAlongAxis operation with a data tensor as base.
    ///
    /// Scatter values from `updates_tensor` along the specified `axis` at indices in `indices_tensor` onto `data_tensor`.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis to scatter to
    /// * `data_tensor` - The input tensor to scatter values onto
    /// * `updates_tensor` - The input tensor to scatter values from
    /// * `indices_tensor` - Int32 or Int64 tensor used to index the result tensor
    /// * `mode` - The type of update to use
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object or None if error
    fn scatter_along_axis_with_data(
        &self,
        axis: i64,
        data_tensor: &Tensor,
        updates_tensor: &Tensor,
        indices_tensor: &Tensor,
        mode: ScatterMode,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;
}

/// Implementation of scatter and scatter ND operations for Graph
impl GraphScatterNdOps for Graph {
    fn scatter_nd(
        &self,
        updates_tensor: &Tensor,
        indices_tensor: &Tensor,
        shape: &Shape,
        batch_dimensions: usize,
        mode: ScatterMode,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                scatterNDWithUpdatesTensor: updates_tensor,
                indicesTensor: indices_tensor,
                shape: shape.as_ptr(),
                batchDimensions: batch_dimensions,
                mode: mode as i64,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn scatter_nd_add(
        &self,
        updates_tensor: &Tensor,
        indices_tensor: &Tensor,
        shape: &Shape,
        batch_dimensions: usize,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                scatterNDWithUpdatesTensor: updates_tensor,
                indicesTensor: indices_tensor,
                shape: shape.as_ptr(),
                batchDimensions: batch_dimensions,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn scatter_nd_with_data(
        &self,
        data_tensor: &Tensor,
        updates_tensor: &Tensor,
        indices_tensor: &Tensor,
        batch_dimensions: usize,
        mode: ScatterMode,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                scatterNDWithDataTensor: data_tensor,
                updatesTensor: updates_tensor,
                indicesTensor: indices_tensor,
                batchDimensions: batch_dimensions,
                mode: mode as i64,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn scatter(
        &self,
        updates_tensor: &Tensor,
        indices_tensor: &Tensor,
        shape: &Shape,
        axis: i64,
        mode: ScatterMode,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                scatterWithUpdatesTensor: updates_tensor,
                indicesTensor: indices_tensor,
                shape: shape.as_ptr(),
                axis: axis,
                mode: mode as i64,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn scatter_with_data(
        &self,
        data_tensor: &Tensor,
        updates_tensor: &Tensor,
        indices_tensor: &Tensor,
        axis: i64,
        mode: ScatterMode,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                scatterWithDataTensor: data_tensor,
                updatesTensor: updates_tensor,
                indicesTensor: indices_tensor,
                axis: axis,
                mode: mode as i64,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn scatter_along_axis(
        &self,
        axis: i64,
        updates_tensor: &Tensor,
        indices_tensor: &Tensor,
        shape: &Shape,
        mode: ScatterMode,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                scatterAlongAxis: axis,
                withUpdatesTensor: updates_tensor,
                indicesTensor: indices_tensor,
                shape: shape.as_ptr(),
                mode: mode as i64,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn scatter_along_axis_with_data(
        &self,
        axis: i64,
        data_tensor: &Tensor,
        updates_tensor: &Tensor,
        indices_tensor: &Tensor,
        mode: ScatterMode,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                scatterAlongAxis: axis,
                withDataTensor: data_tensor,
                updatesTensor: updates_tensor,
                indicesTensor: indices_tensor,
                mode: mode as i64,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }
}

/// Extension trait for easier access to scatter and scatter ND operations
pub trait GraphScatterNdOpsExtension {
    /// Get access to scatter and scatter ND operations
    fn scatter_nd_ops(&self) -> &dyn GraphScatterNdOps;
}

impl GraphScatterNdOpsExtension for Graph {
    fn scatter_nd_ops(&self) -> &dyn GraphScatterNdOps {
        self
    }
}
