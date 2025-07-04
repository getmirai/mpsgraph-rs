use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::core::DataType;
use crate::graph::Graph;
use crate::tensor::Tensor;

/// One-hot operations for Graph


/// Implementation of one-hot operations for Graph
impl Graph {
    pub fn one_hot(
        &self,
        indices_tensor: &Tensor,
        depth: usize,
        axis: usize,
        data_type: DataType,
        on_value: f64,
        off_value: f64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                oneHotWithIndicesTensor: indices_tensor,
                depth: depth,
                axis: axis,
                dataType: data_type as u64,
                onValue: on_value,
                offValue: off_value,
                name: name_ptr
            ]
        }
    }

    pub fn one_hot_default_axis(
        &self,
        indices_tensor: &Tensor,
        depth: usize,
        data_type: DataType,
        on_value: f64,
        off_value: f64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                oneHotWithIndicesTensor: indices_tensor,
                depth: depth,
                dataType: data_type as u64,
                onValue: on_value,
                offValue: off_value,
                name: name_ptr
            ]
        }
    }

    pub fn one_hot_default_values(
        &self,
        indices_tensor: &Tensor,
        depth: usize,
        axis: usize,
        data_type: DataType,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                oneHotWithIndicesTensor: indices_tensor,
                depth: depth,
                axis: axis,
                dataType: data_type as u64,
                name: name_ptr
            ]
        }
    }

    pub fn one_hot_simple(
        &self,
        indices_tensor: &Tensor,
        depth: usize,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                oneHotWithIndicesTensor: indices_tensor,
                depth: depth,
                name: name_ptr
            ]
        }
    }

    pub fn one_hot_default_axis_values(
        &self,
        indices_tensor: &Tensor,
        depth: usize,
        data_type: DataType,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                oneHotWithIndicesTensor: indices_tensor,
                depth: depth,
                dataType: data_type as u64,
                name: name_ptr
            ]
        }
    }

    pub fn one_hot_default_type_values(
        &self,
        indices_tensor: &Tensor,
        depth: usize,
        axis: usize,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                oneHotWithIndicesTensor: indices_tensor,
                depth: depth,
                axis: axis,
                name: name_ptr
            ]
        }
    }
}

