use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::tensor::Tensor;

/// Trait defining cumulative operations for a Graph
pub trait GraphCumulativeOps {
    /// Computes the cumulative sum of the input tensor along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis` - The tensor dimension where you compute the cumulative operation
    /// * `exclusive` - If true, perform the exclusive cumulative operation, and the first element will be equal to zero
    /// * `reverse` - If true, reverse the direction of the cumulative operation along the specified axis
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn cumulative_sum(
        &self,
        tensor: &Tensor,
        axis: i64,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Computes the cumulative sum of the input tensor along the specified axis using an axis tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis_tensor` - The tensor containing the axis to compute the cumulative operation on
    /// * `exclusive` - If true, perform the exclusive cumulative operation, and the first element will be equal to zero
    /// * `reverse` - If true, reverse the direction of the cumulative operation along the specified axis
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn cumulative_sum_with_axis_tensor(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Computes the cumulative sum of the input tensor along the specified axis (simplified version).
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis` - The tensor dimension where you compute the cumulative operation
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn cumulative_sum_simple(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Computes the cumulative product of the input tensor along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis` - The tensor dimension where you compute the cumulative operation
    /// * `exclusive` - If true, perform the exclusive cumulative operation, and the first element will be equal to one
    /// * `reverse` - If true, reverse the direction of the cumulative operation along the specified axis
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn cumulative_product(
        &self,
        tensor: &Tensor,
        axis: i64,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Computes the cumulative minimum of the input tensor along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis` - The tensor dimension where you compute the cumulative operation
    /// * `exclusive` - If true, perform the exclusive cumulative operation, and the first element will be equal to the largest value of the tensor data type
    /// * `reverse` - If true, reverse the direction of the cumulative operation along the specified axis
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn cumulative_minimum(
        &self,
        tensor: &Tensor,
        axis: i64,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Computes the cumulative maximum of the input tensor along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis` - The tensor dimension where you compute the cumulative operation
    /// * `exclusive` - If true, perform the exclusive cumulative operation, and the first element will be equal to the lowest value of the tensor data type
    /// * `reverse` - If true, reverse the direction of the cumulative operation along the specified axis
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn cumulative_maximum(
        &self,
        tensor: &Tensor,
        axis: i64,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Computes the cumulative product of the input tensor along the axis specified by a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis_tensor` - The tensor containing the axis to compute the cumulative operation on
    /// * `exclusive` - If true, perform the exclusive cumulative operation, and the first element will be equal to one
    /// * `reverse` - If true, reverse the direction of the cumulative operation along the specified axis
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn cumulative_product_with_axis_tensor(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Computes the cumulative product of the input tensor along the specified axis (simplified version).
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis` - The tensor dimension where you compute the cumulative operation
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn cumulative_product_simple(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Computes the cumulative minimum of the input tensor along the axis specified by a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis_tensor` - The tensor containing the axis to compute the cumulative operation on
    /// * `exclusive` - If true, perform the exclusive cumulative operation, and the first element will be equal to the largest value of the tensor data type
    /// * `reverse` - If true, reverse the direction of the cumulative operation along the specified axis
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn cumulative_minimum_with_axis_tensor(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Computes the cumulative minimum of the input tensor along the specified axis (simplified version).
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis` - The tensor dimension where you compute the cumulative operation
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn cumulative_minimum_simple(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Computes the cumulative maximum of the input tensor along the axis specified by a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis_tensor` - The tensor containing the axis to compute the cumulative operation on
    /// * `exclusive` - If true, perform the exclusive cumulative operation, and the first element will be equal to the lowest value of the tensor data type
    /// * `reverse` - If true, reverse the direction of the cumulative operation along the specified axis
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn cumulative_maximum_with_axis_tensor(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Computes the cumulative maximum of the input tensor along the specified axis (simplified version).
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis` - The tensor dimension where you compute the cumulative operation
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn cumulative_maximum_simple(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;
}

/// Implementation of cumulative operations for Graph
impl GraphCumulativeOps for Graph {
    fn cumulative_sum(
        &self,
        tensor: &Tensor,
        axis: i64,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                cumulativeSumWithTensor: tensor,
                axis: axis,
                exclusive: exclusive,
                reverse: reverse,
                name: name_ptr,
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::retain_autoreleased(result).unwrap())
            }
        }
    }

    fn cumulative_sum_with_axis_tensor(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                cumulativeSumWithTensor: tensor,
                axisTensor: axis_tensor,
                exclusive: exclusive,
                reverse: reverse,
                name: name_ptr,
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::retain_autoreleased(result).unwrap())
            }
        }
    }

    fn cumulative_sum_simple(
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

            let result: *mut Tensor = msg_send![
                self,
                cumulativeSumWithTensor: tensor,
                axis: axis,
                name: name_ptr,
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::retain_autoreleased(result).unwrap())
            }
        }
    }

    fn cumulative_product(
        &self,
        tensor: &Tensor,
        axis: i64,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                cumulativeProductWithTensor: tensor,
                axis: axis,
                exclusive: exclusive,
                reverse: reverse,
                name: name_ptr,
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::retain_autoreleased(result).unwrap())
            }
        }
    }

    fn cumulative_minimum(
        &self,
        tensor: &Tensor,
        axis: i64,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                cumulativeMinimumWithTensor: tensor,
                axis: axis,
                exclusive: exclusive,
                reverse: reverse,
                name: name_ptr,
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::retain_autoreleased(result).unwrap())
            }
        }
    }

    fn cumulative_maximum(
        &self,
        tensor: &Tensor,
        axis: i64,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                cumulativeMaximumWithTensor: tensor,
                axis: axis,
                exclusive: exclusive,
                reverse: reverse,
                name: name_ptr,
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::retain_autoreleased(result).unwrap())
            }
        }
    }

    fn cumulative_product_with_axis_tensor(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                cumulativeProductWithTensor: tensor,
                axisTensor: axis_tensor,
                exclusive: exclusive,
                reverse: reverse,
                name: name_ptr,
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::retain_autoreleased(result).unwrap())
            }
        }
    }

    fn cumulative_product_simple(
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

            let result: *mut Tensor = msg_send![
                self,
                cumulativeProductWithTensor: tensor,
                axis: axis,
                name: name_ptr,
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::retain_autoreleased(result).unwrap())
            }
        }
    }

    fn cumulative_minimum_with_axis_tensor(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                cumulativeMinimumWithTensor: tensor,
                axisTensor: axis_tensor,
                exclusive: exclusive,
                reverse: reverse,
                name: name_ptr,
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::retain_autoreleased(result).unwrap())
            }
        }
    }

    fn cumulative_minimum_simple(
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

            let result: *mut Tensor = msg_send![
                self,
                cumulativeMinimumWithTensor: tensor,
                axis: axis,
                name: name_ptr,
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::retain_autoreleased(result).unwrap())
            }
        }
    }

    fn cumulative_maximum_with_axis_tensor(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                cumulativeMaximumWithTensor: tensor,
                axisTensor: axis_tensor,
                exclusive: exclusive,
                reverse: reverse,
                name: name_ptr,
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::retain_autoreleased(result).unwrap())
            }
        }
    }

    fn cumulative_maximum_simple(
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

            let result: *mut Tensor = msg_send![
                self,
                cumulativeMaximumWithTensor: tensor,
                axis: axis,
                name: name_ptr,
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::retain_autoreleased(result).unwrap())
            }
        }
    }
}
