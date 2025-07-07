use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::shape::Shape;
use crate::tensor::Tensor;

/// Scatter operation update modes matching `MPSGraphScatterMode`.
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

/// Scatter and ScatterND operations for [`Graph`], adapted from
/// `MPSGraphScatterNDOps.h`.
impl Graph {
    /// Creates a **ScatterND** operation and returns the result tensor.
    ///
    /// Scatters slices from `updates_tensor` into a newly-created tensor of
    /// shape `shape` at positions given by `indices_tensor`. Collisions are
    /// resolved according to `mode` (see [`ScatterMode`]). Slices not written
    /// by `indices_tensor` are set to zero.
    ///
    /// Shape constraints (using notation from the header):
    /// ```text
    /// B = batchDimensions
    /// U = updates.rank − B
    /// P = result.rank − B
    /// Q = indices.rank − B
    /// K = indices.shape[-1]
    ///
    /// K ≤ P
    /// U = (P−K) + Q − 1
    /// indices.shape[0:Q−1] = updates.shape[0:Q−1]
    /// updates.shape[Q:U]    = result.shape[K:P]
    /// ```
    pub fn scatter_nd(
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
            msg_send![
                self,
                scatterNDWithUpdatesTensor: updates_tensor,
                indicesTensor: indices_tensor,
                shape: shape.as_ptr(),
                batchDimensions: batch_dimensions,
                mode: mode as i64,
                name: name_ptr
            ]
        }
    }

    /// Convenience wrapper for **ScatterND** with `mode = ScatterMode::Add`.
    /// Equivalent to calling [`scatter_nd`] with `mode == Add`.
    pub fn scatter_nd_add(
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
            msg_send![
                self,
                scatterNDWithUpdatesTensor: updates_tensor,
                indicesTensor: indices_tensor,
                shape: shape.as_ptr(),
                batchDimensions: batch_dimensions,
                name: name_ptr
            ]
        }
    }

    /// Creates a **ScatterND** operation on top of an existing `data_tensor`.
    ///
    /// The result tensor is initialised with `data_tensor`; updates at positions
    /// specified by `indices_tensor` are then combined with `updates_tensor`
    /// according to `mode`.
    pub fn scatter_nd_with_data(
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
            msg_send![
                self,
                scatterNDWithDataTensor: data_tensor,
                updatesTensor: updates_tensor,
                indicesTensor: indices_tensor,
                batchDimensions: batch_dimensions,
                mode: mode as i64,
                name: name_ptr
            ]
        }
    }

    /// Creates a rank-preserving **Scatter** (1-D indices) operation.
    ///
    /// Scatters slices from `updates_tensor` into a new tensor of shape `shape`
    /// along dimension `axis` at indices provided by `indices_tensor`.
    /// The update rule is:
    /// ```text
    /// U = updates.rank
    /// P = result.rank (== U)
    /// result[i₀ … i_{axis−1}, indices[i_axis], i_{axis+1} … i_{U−1}] =
    ///        updates[i₀ … i_{axis−1}, i_axis, i_{axis+1} … i_{U−1}]
    /// ```
    /// Shape requirements:
    /// ```text
    /// indices.rank = 1
    /// updates.shape[0:axis]   == result.shape[0:axis]   except at `axis`
    /// updates.shape[axis]     == indices.shape[0]
    /// updates.shape[axis+1:U] == result.shape[axis+1:P]
    /// ```
    /// Collisions are combined according to `mode`.
    pub fn scatter(
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
            msg_send![
                self,
                scatterWithUpdatesTensor: updates_tensor,
                indicesTensor: indices_tensor,
                shape: shape.as_ptr(),
                axis: axis,
                mode: mode as i64,
                name: name_ptr
            ]
        }
    }

    /// **Scatter** variant that updates an existing `data_tensor` instead of
    /// starting from zeros.
    pub fn scatter_with_data(
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
            msg_send![
                self,
                scatterWithDataTensor: data_tensor,
                updatesTensor: updates_tensor,
                indicesTensor: indices_tensor,
                axis: axis,
                mode: mode as i64,
                name: name_ptr
            ]
        }
    }

    /// Creates a **ScatterAlongAxis** operation.
    ///
    /// Values from `updates_tensor` are scattered into a result tensor of shape
    /// `shape` along dimension `axis` using indices in `indices_tensor`.
    /// Requirements:
    /// – `updates_tensor` and `indices_tensor` must have identical shapes.
    /// – `shape` must match those shapes except at `axis`.
    /// – Out-of-bounds indices are ignored.
    ///   The result tensor is initialised with the identity element implied by
    ///   `mode` (e.g. 0 for Add).
    pub fn scatter_along_axis(
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
            msg_send![
                self,
                scatterAlongAxis: axis,
                withUpdatesTensor: updates_tensor,
                indicesTensor: indices_tensor,
                shape: shape.as_ptr(),
                mode: mode as i64,
                name: name_ptr
            ]
        }
    }

    /// **ScatterAlongAxis** variant that scatters onto an existing
    /// `data_tensor` instead of starting from an initial value determined by
    /// `mode`.
    pub fn scatter_along_axis_with_data(
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
            msg_send![
                self,
                scatterAlongAxis: axis,
                withDataTensor: data_tensor,
                updatesTensor: updates_tensor,
                indicesTensor: indices_tensor,
                mode: mode as i64,
                name: name_ptr
            ]
        }
    }

    /// **ScatterAlongAxis** where the axis is supplied as a scalar tensor.
    pub fn scatter_along_axis_tensor(
        &self,
        axis_tensor: &Tensor,
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
            msg_send![
                self,
                scatterAlongAxisTensor: axis_tensor,
                withUpdatesTensor: updates_tensor,
                indicesTensor: indices_tensor,
                shape: shape.as_ptr(),
                mode: mode as i64,
                name: name_ptr
            ]
        }
    }

    /// **ScatterAlongAxis** variant with `axis_tensor` and existing
    /// `data_tensor`.
    pub fn scatter_along_axis_tensor_with_data(
        &self,
        axis_tensor: &Tensor,
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
            msg_send![
                self,
                scatterAlongAxisTensor: axis_tensor,
                withDataTensor: data_tensor,
                updatesTensor: updates_tensor,
                indicesTensor: indices_tensor,
                mode: mode as i64,
                name: name_ptr
            ]
        }
    }
}
