use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::{NSArray, NSString};

use crate::graph::Graph;
use crate::tensor::Tensor;

/// Trait defining TopK operations for a Graph
pub trait GraphTopKOps {
    /// Finds the k largest values along the minor dimension of the input.
    ///
    /// The source must have at least k elements along its minor dimension.
    /// Returns a tuple of tensors: (values, indices) where:
    /// - values: The top k values
    /// - indices: The indices of those values
    ///
    /// # Parameters
    ///
    /// * `source` - Tensor containing source data
    /// * `k` - The number of largest values to return
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A tuple (values, indices) of Tensor objects or None if error
    fn top_k(
        &self,
        source: &Tensor,
        k: usize,
        name: Option<&str>,
    ) -> Option<(Retained<Tensor>, Retained<Tensor>)>;

    /// Finds the k largest values along the specified axis of the input.
    ///
    /// The source must have at least k elements along the specified axis.
    ///
    /// # Parameters
    ///
    /// * `source` - Tensor containing source data
    /// * `axis` - The dimension along which to compute the TopK values
    /// * `k` - The number of largest values to return
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A tuple (values, indices) of Tensor objects or None if error
    fn top_k_axis(
        &self,
        source: &Tensor,
        axis: isize,
        k: usize,
        name: Option<&str>,
    ) -> Option<(Retained<Tensor>, Retained<Tensor>)>;

    /// Finds the k smallest values along the specified axis of the input.
    ///
    /// The source must have at least k elements along the specified axis.
    ///
    /// # Parameters
    ///
    /// * `source` - Tensor containing source data
    /// * `axis` - The dimension along which to compute the BottomK values
    /// * `k` - The number of smallest values to return
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A tuple (values, indices) of Tensor objects or None if error
    fn bottom_k_axis(
        &self,
        source: &Tensor,
        axis: isize,
        k: usize,
        name: Option<&str>,
    ) -> Option<(Retained<Tensor>, Retained<Tensor>)>;

    /// Finds the k largest values using tensors for parameters.
    ///
    /// # Parameters
    ///
    /// * `source` - Tensor containing source data
    /// * `k_tensor` - Tensor containing the value of k
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A tuple (values, indices) of Tensor objects or None if error
    fn top_k_with_tensor(
        &self,
        source: &Tensor,
        k_tensor: &Tensor,
        name: Option<&str>,
    ) -> Option<(Retained<Tensor>, Retained<Tensor>)>;

    /// Finds the k largest values using tensors for axis and k parameters.
    ///
    /// # Parameters
    ///
    /// * `source` - Tensor containing source data
    /// * `axis_tensor` - Tensor containing the axis along which to compute TopK
    /// * `k_tensor` - Tensor containing the value of k
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A tuple (values, indices) of Tensor objects or None if error
    fn top_k_with_axis_tensor(
        &self,
        source: &Tensor,
        axis_tensor: &Tensor,
        k_tensor: &Tensor,
        name: Option<&str>,
    ) -> Option<(Retained<Tensor>, Retained<Tensor>)>;

    /// Finds the k smallest values using tensors for axis and k parameters.
    ///
    /// # Parameters
    ///
    /// * `source` - Tensor containing source data
    /// * `axis_tensor` - Tensor containing the axis along which to compute BottomK
    /// * `k_tensor` - Tensor containing the value of k
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A tuple (values, indices) of Tensor objects or None if error
    fn bottom_k_with_axis_tensor(
        &self,
        source: &Tensor,
        axis_tensor: &Tensor,
        k_tensor: &Tensor,
        name: Option<&str>,
    ) -> Option<(Retained<Tensor>, Retained<Tensor>)>;

    /// Computes the gradient for a TopK operation.
    ///
    /// # Parameters
    ///
    /// * `gradient` - Tensor containing the incoming gradient
    /// * `source` - Tensor containing source data
    /// * `k` - The number of largest values used in the forward pass
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object or None if error
    fn top_k_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        k: usize,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Computes the gradient for a TopK operation with specified axis.
    ///
    /// # Parameters
    ///
    /// * `gradient` - Tensor containing the incoming gradient
    /// * `source` - Tensor containing source data
    /// * `axis` - The dimension along which TopK was computed
    /// * `k` - The number of largest values used in the forward pass
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object or None if error
    fn top_k_gradient_axis(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        axis: isize,
        k: usize,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Computes the gradient for a BottomK operation with specified axis.
    ///
    /// # Parameters
    ///
    /// * `gradient` - Tensor containing the incoming gradient
    /// * `source` - Tensor containing source data
    /// * `axis` - The dimension along which BottomK was computed
    /// * `k` - The number of smallest values used in the forward pass
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object or None if error
    fn bottom_k_gradient_axis(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        axis: isize,
        k: usize,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;
}

/// Implementation of TopK operations for Graph
impl GraphTopKOps for Graph {
    fn top_k(
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

    fn top_k_axis(
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

    fn bottom_k_axis(
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

    fn top_k_with_tensor(
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

    fn top_k_with_axis_tensor(
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

    fn bottom_k_with_axis_tensor(
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

    fn top_k_gradient(
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

    fn top_k_gradient_axis(
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

    fn bottom_k_gradient_axis(
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

/// Extension trait for easier access to TopK operations
pub trait GraphTopKOpsExtension {
    /// Get access to TopK operations
    fn top_k_ops(&self) -> &dyn GraphTopKOps;
}

impl GraphTopKOpsExtension for Graph {
    fn top_k_ops(&self) -> &dyn GraphTopKOps {
        self
    }
}
