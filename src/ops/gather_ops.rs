use crate::{DataType, Graph, ScalarOrTensor, Tensor};
use objc2::{msg_send, rc::Retained};
use objc2_foundation::NSString;

/// GatherOps.
impl Graph {
    /// Creates a Gather operation and returns the result tensor.
    ///
    /// Gathers the values in updatesTensor to the result tensor along the indices in indicesTensor.
    /// The gather is defined as
    /// ```md
    /// B = batchDims
    /// U = updates.rank
    /// P = res.rank
    /// Q = inds.rank
    /// res[p_{0},...p_{axis-1}, i_{B},...,i_{Q}, ...,p_{axis+1},...,p{U-1}] =
    /// updates[p_{0},...p_{axis-1}, indices[p_{0},...,p_{B-1},i_{B},...,i_{Q}, ...,p_{axis+1},...,p{U-1}]
    /// ```
    /// The tensors have the following shape requirements
    /// ```md
    /// P = Q-B + U-1
    /// indices.shape[0:B] = updates.shape[0:B] = res.shape[0:B]
    /// res.shape[0:axis] = updates.shape[0:axis]
    /// res.shape[axis:axis+Q-B] = indices.shape[B:]
    /// res.shape[axis+1+Q-B:] = updates.shape[axis+1:]
    /// ```
    ///
    /// - Parameters:
    /// - updatesTensor: Tensor containing slices to be inserted into the result tensor.
    /// - indicesTensor: Tensor containg the updates indices to read slices from
    /// - axis: The dimension on which to perform the gather
    /// - batchDimensions: The number of batch dimensions
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object
    pub fn gather_with_updates(
        &self,
        updates_tensor: &Tensor,
        indices_tensor: &Tensor,
        axis: i64,
        batch_dimensions: u64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                gatherWithUpdatesTensor: updates_tensor,
                indicesTensor: indices_tensor,
                axis: axis,
                batchDimensions: batch_dimensions,
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }

    /// Creates a GatherAlongAxis operation and returns the result tensor.
    ///
    /// Gather values from `updatesTensor` along the specified `axis` at indices in `indicesTensor`.
    /// The shape of `updatesTensor` and `indicesTensor` must match except at `axis`.
    /// The shape of the result tensor is equal to the shape of `indicesTensor`.
    /// If an index is out of bounds of the `updatesTensor` along `axis` a 0 is inserted.
    ///
    /// - Parameters:
    /// - axis: The axis scalar or [`DataType::Int32`] tensor to gather from. Negative values wrap around
    /// - updatesTensor: The input tensor to gather values from
    /// - indicesTensor: Int32 or Int64 tensor used to index `updatesTensor`
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object
    pub fn gather_along_axis<'a>(
        &self,
        axis: ScalarOrTensor<'a, i64>,
        updates_tensor: &Tensor,
        indices_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match axis {
            ScalarOrTensor::Scalar(axis) => unsafe {
                msg_send![
                    self,
                    gatherAlongAxis: axis,
                    withUpdatesTensor: updates_tensor,
                    indicesTensor: indices_tensor,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            ScalarOrTensor::Tensor(axis) => unsafe {
                msg_send![
                    self,
                    gatherAlongAxisTensor: axis,
                    withUpdatesTensor: updates_tensor,
                    indicesTensor: indices_tensor,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }
}
