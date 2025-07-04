use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::{NSArray, NSString};

use crate::graph::Graph;
use crate::tensor::Tensor;

/// Trait defining TopK operations for a Graph


/// Implementation of TopK operations for Graph
impl Graph {
    pub fn top_k(
        &self,
        source: &Tensor,
        k: usize,
        name: Option<&str>,
    ) -> Option<(Retained<Tensor>, Retained<Tensor>)> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result_array_opt: Option<Retained<NSArray<Tensor>>> = msg_send![
                self,
                topKWithSourceTensor: source,
                k: k,
                name: name_ptr
            ];

            result_array_opt.and_then(|array| {
                if array.count() == 2 {
                    let values: Option<Retained<Tensor>> = msg_send![&*array, objectAtIndex: 0u64];
                    let indices: Option<Retained<Tensor>> = msg_send![&*array, objectAtIndex: 1u64];
                    match (values, indices) {
                        (Some(v), Some(i)) => Some((v, i)),
                        _ => None,
                    }
                } else {
                    None
                }
            })
        }
    }

    pub fn top_k_axis(
        &self,
        source: &Tensor,
        axis: isize,
        k: usize,
        name: Option<&str>,
    ) -> Option<(Retained<Tensor>, Retained<Tensor>)> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let result_array_opt: Option<Retained<NSArray<Tensor>>> = msg_send![
                self,
                topKWithSourceTensor: source,
                axis: axis,
                k: k,
                name: name_ptr
            ];
            result_array_opt.and_then(|array| {
                if array.count() == 2 {
                    let values: Option<Retained<Tensor>> = msg_send![&*array, objectAtIndex: 0u64];
                    let indices: Option<Retained<Tensor>> = msg_send![&*array, objectAtIndex: 1u64];
                    match (values, indices) {
                        (Some(v), Some(i)) => Some((v, i)),
                        _ => None,
                    }
                } else {
                    None
                }
            })
        }
    }

    pub fn bottom_k_axis(
        &self,
        source: &Tensor,
        axis: isize,
        k: usize,
        name: Option<&str>,
    ) -> Option<(Retained<Tensor>, Retained<Tensor>)> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let result_array_opt: Option<Retained<NSArray<Tensor>>> = msg_send![
                self,
                bottomKWithSourceTensor: source,
                axis: axis,
                k: k,
                name: name_ptr
            ];
            result_array_opt.and_then(|array| {
                if array.count() == 2 {
                    let values: Option<Retained<Tensor>> = msg_send![&*array, objectAtIndex: 0u64];
                    let indices: Option<Retained<Tensor>> = msg_send![&*array, objectAtIndex: 1u64];
                    match (values, indices) {
                        (Some(v), Some(i)) => Some((v, i)),
                        _ => None,
                    }
                } else {
                    None
                }
            })
        }
    }

    pub fn top_k_with_tensor(
        &self,
        source: &Tensor,
        k_tensor: &Tensor,
        name: Option<&str>,
    ) -> Option<(Retained<Tensor>, Retained<Tensor>)> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let result_array_opt: Option<Retained<NSArray<Tensor>>> = msg_send![
                self,
                topKWithSourceTensor: source,
                kTensor: k_tensor,
                name: name_ptr
            ];
            result_array_opt.and_then(|array| {
                if array.count() == 2 {
                    let values: Option<Retained<Tensor>> = msg_send![&*array, objectAtIndex: 0u64];
                    let indices: Option<Retained<Tensor>> = msg_send![&*array, objectAtIndex: 1u64];
                    match (values, indices) {
                        (Some(v), Some(i)) => Some((v, i)),
                        _ => None,
                    }
                } else {
                    None
                }
            })
        }
    }

    pub fn top_k_with_axis_tensor(
        &self,
        source: &Tensor,
        axis_tensor: &Tensor,
        k_tensor: &Tensor,
        name: Option<&str>,
    ) -> Option<(Retained<Tensor>, Retained<Tensor>)> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let result_array_opt: Option<Retained<NSArray<Tensor>>> = msg_send![
                self,
                topKWithSourceTensor: source,
                axisTensor: axis_tensor,
                kTensor: k_tensor,
                name: name_ptr
            ];
            result_array_opt.and_then(|array| {
                if array.count() == 2 {
                    let values: Option<Retained<Tensor>> = msg_send![&*array, objectAtIndex: 0u64];
                    let indices: Option<Retained<Tensor>> = msg_send![&*array, objectAtIndex: 1u64];
                    match (values, indices) {
                        (Some(v), Some(i)) => Some((v, i)),
                        _ => None,
                    }
                } else {
                    None
                }
            })
        }
    }

    pub fn bottom_k_with_axis_tensor(
        &self,
        source: &Tensor,
        axis_tensor: &Tensor,
        k_tensor: &Tensor,
        name: Option<&str>,
    ) -> Option<(Retained<Tensor>, Retained<Tensor>)> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let result_array_opt: Option<Retained<NSArray<Tensor>>> = msg_send![
                self,
                bottomKWithSourceTensor: source,
                axisTensor: axis_tensor,
                kTensor: k_tensor,
                name: name_ptr
            ];
            result_array_opt.and_then(|array| {
                if array.count() == 2 {
                    let values: Option<Retained<Tensor>> = msg_send![&*array, objectAtIndex: 0u64];
                    let indices: Option<Retained<Tensor>> = msg_send![&*array, objectAtIndex: 1u64];
                    match (values, indices) {
                        (Some(v), Some(i)) => Some((v, i)),
                        _ => None,
                    }
                } else {
                    None
                }
            })
        }
    }

    pub fn top_k_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        k: usize,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                topKWithGradientTensor: gradient,
                source: source,
                k: k,
                name: name_ptr
            ]
        }
    }

    pub fn top_k_gradient_axis(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        axis: isize,
        k: usize,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                topKWithGradientTensor: gradient,
                source: source,
                axis: axis,
                k: k,
                name: name_ptr
            ]
        }
    }

    pub fn bottom_k_gradient_axis(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        axis: isize,
        k: usize,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                bottomKWithGradientTensor: gradient,
                source: source,
                axis: axis,
                k: k,
                name: name_ptr
            ]
        }
    }
}

