use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::tensor::Tensor;

/// Sort and Arg-Sort operations for [`Graph`], adapted from
/// `MPSGraphSortOps.h`.
///
/// All functions sort elements along a chosen `axis` and return either the
/// sorted tensor (`sort*`) or the corresponding indices tensor (`arg_sort*`).
/// Negative axis values wrap around.
///
/// When `descending` is `true`, the result is ordered from largest to smallest;
/// otherwise ascending order is used.
///
/// `*_with_axis_tensor` variants accept `axis_tensor` (scalar `int32/int64`)
/// instead of a Rust `axis` argument.
///
/// The `arg_sort*` family always returns an `int32` tensor of indices.
impl Graph {
    /// Sorts elements of `tensor` along dimension `axis`.
    /// Returns the sorted tensor in ascending or descending order.
    pub fn sort(
        &self,
        tensor: &Tensor,
        axis: i64,
        descending: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                sortWithTensor: tensor,
                axis: axis,
                descending: descending,
                name: name_ptr
            ]
        }
    }

    /// Convenience wrapper for [`sort`] with `descending = false`.
    pub fn sort_ascending(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                sortWithTensor: tensor,
                axis: axis,
                name: name_ptr
            ]
        }
    }

    /// Sorts with the axis supplied as a scalar tensor.
    pub fn sort_with_axis_tensor(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        descending: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                sortWithTensor: tensor,
                axisTensor: axis_tensor,
                descending: descending,
                name: name_ptr
            ]
        }
    }

    /// Ascending variant of [`sort_with_axis_tensor`].
    pub fn sort_ascending_with_axis_tensor(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                sortWithTensor: tensor,
                axisTensor: axis_tensor,
                name: name_ptr
            ]
        }
    }

    /// Returns the indices that would sort `tensor` along `axis`.
    pub fn arg_sort(
        &self,
        tensor: &Tensor,
        axis: i64,
        descending: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                argSortWithTensor: tensor,
                axis: axis,
                descending: descending,
                name: name_ptr
            ]
        }
    }

    /// Convenience wrapper for [`arg_sort`] with `descending = false`.
    pub fn arg_sort_ascending(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                argSortWithTensor: tensor,
                axis: axis,
                name: name_ptr
            ]
        }
    }

    /// Arg-sort with axis provided as a tensor.
    pub fn arg_sort_with_axis_tensor(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        descending: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                argSortWithTensor: tensor,
                axisTensor: axis_tensor,
                descending: descending,
                name: name_ptr
            ]
        }
    }

    /// Ascending variant of [`arg_sort_with_axis_tensor`].
    pub fn arg_sort_ascending_with_axis_tensor(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                argSortWithTensor: tensor,
                axisTensor: axis_tensor,
                name: name_ptr
            ]
        }
    }
}
