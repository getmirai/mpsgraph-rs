use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::{NSArray, NSString};

use crate::graph::Graph;
use crate::tensor::Tensor;

impl Graph {
    /// Creates a Top-K operation and returns the values and indices tensors.
    ///
    /// Finds the `k` largest values along the minor (last) dimension of `source`.
    /// `source` must contain at least `k` elements along this dimension. The first
    /// element of the returned tuple corresponds to the top values, and the
    /// second element corresponds to the indices of those values.
    ///
    /// * `source` – Tensor containing source data.
    /// * `k` – Number of largest values to return.
    /// * `name` – Optional name for the operation.
    ///
    /// Returns `Some((values, indices))` on success, or `None` if the underlying
    /// Objective-C call did not return the expected two tensors.
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

    /// Creates a Top-K operation along a specified axis and returns the values
    /// and indices tensors.
    ///
    /// Finds the `k` largest values along dimension `axis` of `source`. The
    /// tensor must contain at least `k` elements along that dimension.
    ///
    /// * `source` – Tensor containing source data.
    /// * `axis` – Dimension along which to compute the Top-K values.
    /// * `k` – Number of largest values to return.
    /// * `name` – Optional name for the operation.
    ///
    /// Returns `Some((values, indices))` on success, or `None` otherwise.
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

    /// Creates a Bottom-K operation along a specified axis and returns the
    /// values and indices tensors.
    ///
    /// Finds the `k` smallest values along dimension `axis` of `source`. The
    /// tensor must contain at least `k` elements along that dimension.
    ///
    /// * `source` – Tensor containing source data.
    /// * `axis` – Dimension along which to compute the Bottom-K values.
    /// * `k` – Number of smallest values to return.
    /// * `name` – Optional name for the operation.
    ///
    /// Returns `Some((values, indices))` on success, or `None` otherwise.
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

    /// Creates a Top-K operation where `k` is provided as a tensor and returns
    /// the values and indices tensors.
    ///
    /// * `source` – Tensor containing source data.
    /// * `k_tensor` – Tensor specifying the number of largest values to return.
    /// * `name` – Optional name for the operation.
    ///
    /// Returns `Some((values, indices))` on success, or `None` otherwise.
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

    /// Creates a Top-K operation where both the axis and `k` are provided as
    /// tensors and returns the values and indices tensors.
    ///
    /// * `source` – Tensor containing source data.
    /// * `axis_tensor` – Tensor specifying the dimension along which to compute
    ///   the Top-K values.
    /// * `k_tensor` – Tensor specifying the number of largest values to return.
    /// * `name` – Optional name for the operation.
    ///
    /// Returns `Some((values, indices))` on success, or `None` otherwise.
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

    /// Creates a Bottom-K operation where both the axis and `k` are provided as
    /// tensors and returns the values and indices tensors.
    ///
    /// * `source` – Tensor containing source data.
    /// * `axis_tensor` – Tensor specifying the dimension along which to compute
    ///   the Bottom-K values.
    /// * `k_tensor` – Tensor specifying the number of smallest values to return.
    /// * `name` – Optional name for the operation.
    ///
    /// Returns `Some((values, indices))` on success, or `None` otherwise.
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

    /// Creates a Top-K gradient operation and returns the gradient tensor.
    ///
    /// Computes the gradient of the Top-K operation with respect to `source`.
    ///
    /// * `gradient` – Incoming gradient tensor.
    /// * `source` – Tensor containing source data.
    /// * `k` – Number of largest values considered in the forward pass.
    /// * `name` – Optional name for the operation.
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

    /// Creates a Top-K gradient operation along a specific axis and returns the
    /// gradient tensor.
    ///
    /// * `gradient` – Incoming gradient tensor.
    /// * `source` – Tensor containing source data.
    /// * `axis` – Dimension along which the Top-K values were computed.
    /// * `k` – Number of largest values considered in the forward pass.
    /// * `name` – Optional name for the operation.
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

    /// Creates a Bottom-K gradient operation along a specific axis and returns
    /// the gradient tensor.
    ///
    /// * `gradient` – Incoming gradient tensor.
    /// * `source` – Tensor containing source data.
    /// * `axis` – Dimension along which the Bottom-K values were computed.
    /// * `k` – Number of smallest values considered in the forward pass.
    /// * `name` – Optional name for the operation.
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

    /// Creates a Top-K gradient operation where `k` is provided as a tensor and
    /// returns the gradient tensor.
    ///
    /// * `gradient` – Incoming gradient tensor.
    /// * `source` – Tensor containing source data.
    /// * `k_tensor` – Tensor specifying the number of largest values considered
    ///   in the forward pass.
    /// * `name` – Optional name for the operation.
    pub fn top_k_gradient_with_tensor(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        k_tensor: &Tensor,
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
                kTensor: k_tensor,
                name: name_ptr
            ]
        }
    }

    /// Creates a Top-K gradient operation where both the axis and `k` are
    /// provided as tensors and returns the gradient tensor.
    ///
    /// * `gradient` – Incoming gradient tensor.
    /// * `source` – Tensor containing source data.
    /// * `axis_tensor` – Tensor specifying the dimension along which the
    ///   Top-K values were computed.
    /// * `k_tensor` – Tensor specifying the number of largest values considered
    ///   in the forward pass.
    /// * `name` – Optional name for the operation.
    pub fn top_k_gradient_with_axis_tensor(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        axis_tensor: &Tensor,
        k_tensor: &Tensor,
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
                axisTensor: axis_tensor,
                kTensor: k_tensor,
                name: name_ptr
            ]
        }
    }

    /// Creates a Bottom-K gradient operation where both the axis and `k` are
    /// provided as tensors and returns the gradient tensor.
    ///
    /// * `gradient` – Incoming gradient tensor.
    /// * `source` – Tensor containing source data.
    /// * `axis_tensor` – Tensor specifying the dimension along which the
    ///   Bottom-K values were computed.
    /// * `k_tensor` – Tensor specifying the number of smallest values considered
    ///   in the forward pass.
    /// * `name` – Optional name for the operation.
    pub fn bottom_k_gradient_with_axis_tensor(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        axis_tensor: &Tensor,
        k_tensor: &Tensor,
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
                axisTensor: axis_tensor,
                kTensor: k_tensor,
                name: name_ptr
            ]
        }
    }
}
