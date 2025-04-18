use objc2::rc::Retained;
use objc2::msg_send;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::tensor::Tensor;

/// Trait for gathering operations on a graph
pub trait GraphGatherOps {
    /// Creates a GatherND operation and returns the result tensor.
    ///
    /// Gathers the slices in updatesTensor to the result tensor along the indices in indicesTensor.
    ///
    /// # Arguments
    ///
    /// * `updates_tensor` - Tensor containing slices to be inserted into the result tensor.
    /// * `indices_tensor` - Tensor containing the updates indices to read slices from
    /// * `batch_dimensions` - The number of batch dimensions
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn gather_nd(
        &self,
        updates_tensor: &Retained<Tensor>,
        indices_tensor: &Retained<Tensor>,
        batch_dimensions: usize,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a Gather operation and returns the result tensor.
    ///
    /// Gathers the values in updatesTensor to the result tensor along the indices in indicesTensor.
    ///
    /// # Arguments
    ///
    /// * `updates_tensor` - Tensor containing slices to be inserted into the result tensor.
    /// * `indices_tensor` - Tensor containing the updates indices to read slices from
    /// * `axis` - The dimension on which to perform the gather
    /// * `batch_dimensions` - The number of batch dimensions
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn gather(
        &self,
        updates_tensor: &Retained<Tensor>,
        indices_tensor: &Retained<Tensor>,
        axis: usize,
        batch_dimensions: usize,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a GatherAlongAxis operation and returns the result tensor.
    ///
    /// Gather values from `updates_tensor` along the specified `axis` at indices in `indices_tensor`.
    /// The shape of `updates_tensor` and `indices_tensor` must match except at `axis`.
    /// The shape of the result tensor is equal to the shape of `indices_tensor`.
    /// If an index is out of bounds of the `updates_tensor` along `axis` a 0 is inserted.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis to gather from. Negative values wrap around
    /// * `updates_tensor` - The input tensor to gather values from
    /// * `indices_tensor` - Int32 or Int64 tensor used to index `updates_tensor`
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn gather_along_axis(
        &self,
        axis: isize,
        updates_tensor: &Retained<Tensor>,
        indices_tensor: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a GatherAlongAxis operation using an axis tensor and returns the result tensor.
    ///
    /// Gather values from `updates_tensor` along the specified `axis_tensor` at indices in `indices_tensor`.
    /// The shape of `updates_tensor` and `indices_tensor` must match except at the axis specified.
    /// The shape of the result tensor is equal to the shape of `indices_tensor`.
    /// If an index is out of bounds of the `updates_tensor` along axis a 0 is inserted.
    ///
    /// # Arguments
    ///
    /// * `axis_tensor` - Scalar Int32 tensor. The axis to gather from. Negative values wrap around
    /// * `updates_tensor` - The input tensor to gather values from
    /// * `indices_tensor` - Int32 or Int64 tensor used to index `updates_tensor`
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn gather_along_axis_tensor(
        &self,
        axis_tensor: &Retained<Tensor>,
        updates_tensor: &Retained<Tensor>,
        indices_tensor: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor>;
}

/// Implementation of gather operations for Graph
impl GraphGatherOps for Graph {
    fn gather_nd(
        &self,
        updates_tensor: &Retained<Tensor>,
        indices_tensor: &Retained<Tensor>,
        batch_dimensions: usize,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self, 
                gatherNDWithUpdatesTensor: &**updates_tensor,
                indicesTensor: &**indices_tensor,
                batchDimensions: batch_dimensions,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create GatherND operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn gather(
        &self,
        updates_tensor: &Retained<Tensor>,
        indices_tensor: &Retained<Tensor>,
        axis: usize,
        batch_dimensions: usize,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self, 
                gatherWithUpdatesTensor: &**updates_tensor,
                indicesTensor: &**indices_tensor,
                axis: axis,
                batchDimensions: batch_dimensions,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create Gather operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn gather_along_axis(
        &self,
        axis: isize,
        updates_tensor: &Retained<Tensor>,
        indices_tensor: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,

                gatherAlongAxis: axis,
                withUpdatesTensor: &**updates_tensor,
                indicesTensor: &**indices_tensor,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create GatherAlongAxis operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn gather_along_axis_tensor(
        &self,
        axis_tensor: &Retained<Tensor>,
        updates_tensor: &Retained<Tensor>,
        indices_tensor: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self, 
                gatherAlongAxisTensor: &**axis_tensor,
                withUpdatesTensor: &**updates_tensor,
                indicesTensor: &**indices_tensor,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create GatherAlongAxisTensor operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }
}

/// Extension trait for easier access to gather operations
pub trait GraphGatherOpsExtension {
    /// Get access to gather operations
    fn gather_ops(&self) -> &dyn GraphGatherOps;
}

impl GraphGatherOpsExtension for Graph {
    fn gather_ops(&self) -> &dyn GraphGatherOps {
        self
    }
}