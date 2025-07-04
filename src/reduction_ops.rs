use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::core::create_ns_array_from_i64_slice;
use crate::graph::Graph;
use crate::tensor::Tensor;

/// Trait for reduction operations on Graph


impl Graph {
    pub fn reduction_sum_with_axis(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, reductionSumWithTensor: tensor, axis: axis, name: name_ptr]
        }
    }

    pub fn reduction_sum_with_axes(
        &self,
        tensor: &Tensor,
        axes: Option<&[i64]>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let axes_ptr = axes.map_or(std::ptr::null(), |a| {
                &*create_ns_array_from_i64_slice(a) as *const _
            });
            msg_send![self, reductionSumWithTensor: tensor, axes: axes_ptr, name: name_ptr]
        }
    }

    pub fn reduction_maximum_with_axis(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, reductionMaximumWithTensor: tensor, axis: axis, name: name_ptr]
        }
    }

    pub fn reduction_maximum_with_axes(
        &self,
        tensor: &Tensor,
        axes: Option<&[i64]>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let axes_ptr = axes.map_or(std::ptr::null(), |a| {
                &*create_ns_array_from_i64_slice(a) as *const _
            });
            msg_send![self, reductionMaximumWithTensor: tensor, axes: axes_ptr, name: name_ptr]
        }
    }

    pub fn reduction_minimum_with_axis(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, reductionMinimumWithTensor: tensor, axis: axis, name: name_ptr]
        }
    }

    pub fn reduction_minimum_with_axes(
        &self,
        tensor: &Tensor,
        axes: Option<&[i64]>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let axes_ptr = axes.map_or(std::ptr::null(), |a| {
                &*create_ns_array_from_i64_slice(a) as *const _
            });
            msg_send![self, reductionMinimumWithTensor: tensor, axes: axes_ptr, name: name_ptr]
        }
    }

    pub fn reduction_product_with_axis(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, reductionProductWithTensor: tensor, axis: axis, name: name_ptr]
        }
    }

    pub fn reduction_product_with_axes(
        &self,
        tensor: &Tensor,
        axes: Option<&[i64]>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let axes_ptr = axes.map_or(std::ptr::null(), |a| {
                &*create_ns_array_from_i64_slice(a) as *const _
            });
            msg_send![self, reductionProductWithTensor: tensor, axes: axes_ptr, name: name_ptr]
        }
    }

    pub fn reduction_maximum_propagate_nan_with_axis(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, reductionMaximumWithPropagateNaNWithTensor: tensor, axis: axis, name: name_ptr]
        }
    }

    pub fn reduction_maximum_propagate_nan_with_axes(
        &self,
        tensor: &Tensor,
        axes: Option<&[i64]>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let axes_ptr = axes.map_or(std::ptr::null(), |a| {
                &*create_ns_array_from_i64_slice(a) as *const _
            });
            msg_send![self, reductionMaximumWithPropagateNaNWithTensor: tensor, axes: axes_ptr, name: name_ptr]
        }
    }

    pub fn reduction_minimum_propagate_nan_with_axis(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, reductionMinimumWithPropagateNaNWithTensor: tensor, axis: axis, name: name_ptr]
        }
    }

    pub fn reduction_minimum_propagate_nan_with_axes(
        &self,
        tensor: &Tensor,
        axes: Option<&[i64]>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let axes_ptr = axes.map_or(std::ptr::null(), |a| {
                &*create_ns_array_from_i64_slice(a) as *const _
            });
            msg_send![self, reductionMinimumWithPropagateNaNWithTensor: tensor, axes: axes_ptr, name: name_ptr]
        }
    }

    pub fn reduction_and_with_axis(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, reductionANDWithTensor: tensor, axis: axis, name: name_ptr]
        }
    }

    pub fn reduction_and_with_axes(
        &self,
        tensor: &Tensor,
        axes: Option<&[i64]>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let axes_ptr = axes.map_or(std::ptr::null(), |a| {
                &*create_ns_array_from_i64_slice(a) as *const _
            });
            msg_send![self, reductionANDWithTensor: tensor, axes: axes_ptr, name: name_ptr]
        }
    }

    pub fn reduction_or_with_axis(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, reductionORWithTensor: tensor, axis: axis, name: name_ptr]
        }
    }

    pub fn reduction_or_with_axes(
        &self,
        tensor: &Tensor,
        axes: Option<&[i64]>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let axes_ptr = axes.map_or(std::ptr::null(), |a| {
                &*create_ns_array_from_i64_slice(a) as *const _
            });
            msg_send![self, reductionORWithTensor: tensor, axes: axes_ptr, name: name_ptr]
        }
    }

    pub fn reduction_xor_with_axis(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, reductionXORWithTensor: tensor, axis: axis, name: name_ptr]
        }
    }

    pub fn reduction_xor_with_axes(
        &self,
        tensor: &Tensor,
        axes: Option<&[i64]>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let axes_ptr = axes.map_or(std::ptr::null(), |a| {
                &*create_ns_array_from_i64_slice(a) as *const _
            });
            msg_send![self, reductionXORWithTensor: tensor, axes: axes_ptr, name: name_ptr]
        }
    }

    pub fn reduction_arg_maximum_with_axis(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, reductionArgMaximumWithTensor: tensor, axis: axis, name: name_ptr]
        }
    }

    pub fn reduction_arg_minimum_with_axis(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, reductionArgMinimumWithTensor: tensor, axis: axis, name: name_ptr]
        }
    }
}

