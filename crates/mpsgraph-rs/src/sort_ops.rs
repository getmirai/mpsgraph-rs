use objc2::rc::Retained;
use objc2::msg_send;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::tensor::Tensor;

/// Trait for performing sort operations on a graph
pub trait GraphSortOps {
    /// Sorts the elements of the input tensor along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis` - The tensor dimension over which to sort the tensor
    /// * `descending` - If true, reverse the sort direction
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object or None if error
    fn sort(
        &self,
        tensor: &Tensor,
        axis: i64,
        descending: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Sorts the elements of the input tensor along the specified axis (in ascending order).
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis` - The tensor dimension over which to sort the tensor
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object or None if error
    fn sort_ascending(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Sorts the elements of the input tensor along an axis specified by a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis_tensor` - A scalar tensor that specifies the dimension over which to sort
    /// * `descending` - If true, reverse the sort direction
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object or None if error
    fn sort_with_axis_tensor(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        descending: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Sorts the elements of the input tensor along an axis specified by a tensor (in ascending order).
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis_tensor` - A scalar tensor that specifies the dimension over which to sort
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object or None if error
    fn sort_ascending_with_axis_tensor(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Computes the indices that sort the elements of the input tensor along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis` - The tensor dimension over which to sort the tensor
    /// * `descending` - If true, reverse the sort direction
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object with 32-bit integer data type or None if error
    fn arg_sort(
        &self,
        tensor: &Tensor,
        axis: i64,
        descending: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Computes the indices that sort the elements of the input tensor along the specified axis (in ascending order).
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis` - The tensor dimension over which to sort the tensor
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object with 32-bit integer data type or None if error
    fn arg_sort_ascending(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Computes the indices that sort the elements of the input tensor along an axis specified by a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis_tensor` - A scalar tensor that specifies the dimension over which to sort
    /// * `descending` - If true, reverse the sort direction
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object with 32-bit integer data type or None if error
    fn arg_sort_with_axis_tensor(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        descending: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Computes the indices that sort the elements of the input tensor along an axis specified by a tensor (in ascending order).
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axis_tensor` - A scalar tensor that specifies the dimension over which to sort
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object with 32-bit integer data type or None if error
    fn arg_sort_ascending_with_axis_tensor(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;
}

/// Implementation of sort operations for Graph
impl GraphSortOps for Graph {
    fn sort(
        &self,
        tensor: &Tensor,
        axis: i64,
        descending: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                sortWithTensor: tensor,
                axis: axis,
                descending: descending,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn sort_ascending(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                sortWithTensor: tensor,
                axis: axis,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn sort_with_axis_tensor(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        descending: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                sortWithTensor: tensor,
                axisTensor: axis_tensor,
                descending: descending,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn sort_ascending_with_axis_tensor(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                sortWithTensor: tensor,
                axisTensor: axis_tensor,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn arg_sort(
        &self,
        tensor: &Tensor,
        axis: i64,
        descending: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                argSortWithTensor: tensor,
                axis: axis,
                descending: descending,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn arg_sort_ascending(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                argSortWithTensor: tensor,
                axis: axis,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn arg_sort_with_axis_tensor(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        descending: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                argSortWithTensor: tensor,
                axisTensor: axis_tensor,
                descending: descending,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn arg_sort_ascending_with_axis_tensor(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                argSortWithTensor: tensor,
                axisTensor: axis_tensor,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }
}

/// Extension trait for easier access to sort operations
pub trait GraphSortOpsExtension {
    /// Get access to sort operations
    fn sort_ops(&self) -> &dyn GraphSortOps;
}

impl GraphSortOpsExtension for Graph {
    fn sort_ops(&self) -> &dyn GraphSortOps {
        self
    }
}