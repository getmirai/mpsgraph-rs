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


/// Implementation of scatter and scatter ND operations for Graph
impl Graph {
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
}

