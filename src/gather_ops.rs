//! Gather helpers are now inherent methods on `Graph`.

use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::tensor::Tensor;

impl Graph {
    pub fn gather_nd(
        &self,
        updates_tensor: &Tensor,
        indices_tensor: &Tensor,
        batch_dimensions: usize,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self,
                gatherNDWithUpdatesTensor: updates_tensor,
                indicesTensor: indices_tensor,
                batchDimensions: batch_dimensions,
                name: name_ptr]
        }
    }

    pub fn gather(
        &self,
        updates_tensor: &Tensor,
        indices_tensor: &Tensor,
        axis: usize,
        batch_dimensions: usize,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self,
                gatherWithUpdatesTensor: updates_tensor,
                indicesTensor: indices_tensor,
                axis: axis as isize,
                batchDimensions: batch_dimensions,
                name: name_ptr]
        }
    }

    pub fn gather_along_axis(
        &self,
        axis: isize,
        updates_tensor: &Tensor,
        indices_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self,
                gatherAlongAxis: axis,
                withUpdatesTensor: updates_tensor,
                indicesTensor: indices_tensor,
                name: name_ptr]
        }
    }

    pub fn gather_along_axis_tensor(
        &self,
        axis_tensor: &Tensor,
        updates_tensor: &Tensor,
        indices_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self,
                gatherAlongAxisTensor: axis_tensor,
                withUpdatesTensor: updates_tensor,
                indicesTensor: indices_tensor,
                name: name_ptr]
        }
    }
}
