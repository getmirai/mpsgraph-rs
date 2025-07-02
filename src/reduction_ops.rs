use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::core::create_ns_array_from_i64_slice;
use crate::graph::Graph;
use crate::tensor::Tensor;

/// Trait for reduction operations on Graph
pub trait GraphReductionOps {
    /// Creates a reduction sum operation along a single axis
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis` - Axis along which to perform the reduction
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn reduction_sum_with_axis(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a reduction sum operation along multiple axes
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axes` - Optional list of axes along which to perform the reduction (None for all axes)
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn reduction_sum_with_axes(
        &self,
        tensor: &Tensor,
        axes: Option<&[i64]>,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a reduction maximum operation along a single axis
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis` - Axis along which to perform the reduction
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn reduction_maximum_with_axis(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a reduction maximum operation along multiple axes
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axes` - Optional list of axes along which to perform the reduction (None for all axes)
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn reduction_maximum_with_axes(
        &self,
        tensor: &Tensor,
        axes: Option<&[i64]>,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a reduction minimum operation along a single axis
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis` - Axis along which to perform the reduction
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn reduction_minimum_with_axis(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a reduction minimum operation along multiple axes
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axes` - Optional list of axes along which to perform the reduction (None for all axes)
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn reduction_minimum_with_axes(
        &self,
        tensor: &Tensor,
        axes: Option<&[i64]>,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a reduction product operation along a single axis
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis` - Axis along which to perform the reduction
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn reduction_product_with_axis(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a reduction product operation along multiple axes
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axes` - Optional list of axes along which to perform the reduction (None for all axes)
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn reduction_product_with_axes(
        &self,
        tensor: &Tensor,
        axes: Option<&[i64]>,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a reduction maximum propagate NaN operation along a single axis
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis` - Axis along which to perform the reduction
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn reduction_maximum_propagate_nan_with_axis(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a reduction maximum propagate NaN operation along multiple axes
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axes` - Optional list of axes along which to perform the reduction (None for all axes)
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn reduction_maximum_propagate_nan_with_axes(
        &self,
        tensor: &Tensor,
        axes: Option<&[i64]>,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a reduction minimum propagate NaN operation along a single axis
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis` - Axis along which to perform the reduction
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn reduction_minimum_propagate_nan_with_axis(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a reduction minimum propagate NaN operation along multiple axes
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axes` - Optional list of axes along which to perform the reduction (None for all axes)
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn reduction_minimum_propagate_nan_with_axes(
        &self,
        tensor: &Tensor,
        axes: Option<&[i64]>,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a reduction AND operation along a single axis
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis` - Axis along which to perform the reduction
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn reduction_and_with_axis(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a reduction AND operation along multiple axes
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axes` - Optional list of axes along which to perform the reduction (None for all axes)
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn reduction_and_with_axes(
        &self,
        tensor: &Tensor,
        axes: Option<&[i64]>,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a reduction OR operation along a single axis
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis` - Axis along which to perform the reduction
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn reduction_or_with_axis(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a reduction OR operation along multiple axes
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axes` - Optional list of axes along which to perform the reduction (None for all axes)
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn reduction_or_with_axes(
        &self,
        tensor: &Tensor,
        axes: Option<&[i64]>,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a reduction XOR operation along a single axis
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis` - Axis along which to perform the reduction
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn reduction_xor_with_axis(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a reduction XOR operation along multiple axes
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axes` - Optional list of axes along which to perform the reduction (None for all axes)
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn reduction_xor_with_axes(
        &self,
        tensor: &Tensor,
        axes: Option<&[i64]>,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a reduction argmax operation along a single axis
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis` - Axis along which to perform the reduction
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object containing the indices of maximum values
    fn reduction_arg_maximum_with_axis(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a reduction argmin operation along a single axis
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis` - Axis along which to perform the reduction
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object containing the indices of minimum values
    fn reduction_arg_minimum_with_axis(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;
}

impl GraphReductionOps for Graph {
    fn reduction_sum_with_axis(
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

    fn reduction_sum_with_axes(
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

    fn reduction_maximum_with_axis(
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

    fn reduction_maximum_with_axes(
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

    fn reduction_minimum_with_axis(
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

    fn reduction_minimum_with_axes(
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

    fn reduction_product_with_axis(
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

    fn reduction_product_with_axes(
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

    fn reduction_maximum_propagate_nan_with_axis(
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

    fn reduction_maximum_propagate_nan_with_axes(
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

    fn reduction_minimum_propagate_nan_with_axis(
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

    fn reduction_minimum_propagate_nan_with_axes(
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

    fn reduction_and_with_axis(
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

    fn reduction_and_with_axes(
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

    fn reduction_or_with_axis(
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

    fn reduction_or_with_axes(
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

    fn reduction_xor_with_axis(
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

    fn reduction_xor_with_axes(
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

    fn reduction_arg_maximum_with_axis(
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

    fn reduction_arg_minimum_with_axis(
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

/// Extension trait providing a method for Graph to access reduction operations
pub trait GraphReductionOpsExtension {
    /// Access reduction operations for this graph
    fn reduction_ops(&self) -> &dyn GraphReductionOps;
}

impl GraphReductionOpsExtension for Graph {
    fn reduction_ops(&self) -> &dyn GraphReductionOps {
        self
    }
}
