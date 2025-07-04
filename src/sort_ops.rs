use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::tensor::Tensor;

/// Inherent implementation of sort operations for Graph
impl Graph {
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
