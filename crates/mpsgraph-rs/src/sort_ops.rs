use objc2::runtime::AnyObject;
// In objc2, use false as NO and true as YES
const NO: bool = false;
const YES: bool = true;
use crate::core::{AsRawObject, NSString};
use crate::graph::Graph;
use crate::tensor::Tensor;
use objc2::msg_send;

/// Sort operations for Graph
impl Graph {
    /// Sorts the elements of the input tensor along the specified axis.
    ///
    /// - Parameters:
    ///   - tensor: The input tensor
    ///   - axis: The tensor dimension over which to sort the tensor
    ///   - descending: If true, reverse the sort direction
    ///   - name: The name for the operation
    /// - Returns: A valid Tensor object
    pub fn sort(
        &self,
        tensor: &Tensor,
        axis: isize,
        descending: bool,
        name: Option<&str>,
    ) -> Tensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let descending_val = if descending { YES } else { NO };

            let result: *mut AnyObject = msg_send![self.0, sortWithTensor: tensor.0,
                axis: axis,
                descending: descending_val,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }

    /// Sorts the elements of the input tensor along the specified axis (in ascending order).
    ///
    /// - Parameters:
    ///   - tensor: The input tensor
    ///   - axis: The tensor dimension over which to sort the tensor
    ///   - name: The name for the operation
    /// - Returns: A valid Tensor object
    pub fn sort_ascending(
        &self,
        tensor: &Tensor,
        axis: isize,
        name: Option<&str>,
    ) -> Tensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let result: *mut AnyObject = msg_send![self.0, sortWithTensor: tensor.0,
                axis: axis,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }

    /// Sorts the elements of the input tensor along an axis specified by a tensor.
    ///
    /// - Parameters:
    ///   - tensor: The input tensor
    ///   - axis_tensor: A scalar tensor that specifies the dimension over which to sort
    ///   - descending: If true, reverse the sort direction
    ///   - name: The name for the operation
    /// - Returns: A valid Tensor object
    pub fn sort_with_axis_tensor(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        descending: bool,
        name: Option<&str>,
    ) -> Tensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let descending_val = if descending { YES } else { NO };

            let result: *mut AnyObject = msg_send![self.0, sortWithTensor: tensor.0,
                axisTensor: axis_tensor.0,
                descending: descending_val,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }

    /// Sorts the elements of the input tensor along an axis specified by a tensor (in ascending order).
    ///
    /// - Parameters:
    ///   - tensor: The input tensor
    ///   - axis_tensor: A scalar tensor that specifies the dimension over which to sort
    ///   - name: The name for the operation
    /// - Returns: A valid Tensor object
    pub fn sort_ascending_with_axis_tensor(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        name: Option<&str>,
    ) -> Tensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let result: *mut AnyObject = msg_send![self.0, sortWithTensor: tensor.0,
                axisTensor: axis_tensor.0,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }

    /// Computes the indices that sort the elements of the input tensor along the specified axis.
    ///
    /// - Parameters:
    ///   - tensor: The input tensor
    ///   - axis: The tensor dimension over which to sort the tensor
    ///   - descending: If true, reverse the sort direction
    ///   - name: The name for the operation
    /// - Returns: A valid Tensor object with 32-bit integer data type
    pub fn arg_sort(
        &self,
        tensor: &Tensor,
        axis: isize,
        descending: bool,
        name: Option<&str>,
    ) -> Tensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let descending_val = if descending { YES } else { NO };

            let result: *mut AnyObject = msg_send![self.0, argSortWithTensor: tensor.0,
                axis: axis,
                descending: descending_val,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }

    /// Computes the indices that sort the elements of the input tensor along the specified axis (in ascending order).
    ///
    /// - Parameters:
    ///   - tensor: The input tensor
    ///   - axis: The tensor dimension over which to sort the tensor
    ///   - name: The name for the operation
    /// - Returns: A valid Tensor object with 32-bit integer data type
    pub fn arg_sort_ascending(
        &self,
        tensor: &Tensor,
        axis: isize,
        name: Option<&str>,
    ) -> Tensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let result: *mut AnyObject = msg_send![self.0, argSortWithTensor: tensor.0,
                axis: axis,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }

    /// Computes the indices that sort the elements of the input tensor along an axis specified by a tensor.
    ///
    /// - Parameters:
    ///   - tensor: The input tensor
    ///   - axis_tensor: A scalar tensor that specifies the dimension over which to sort
    ///   - descending: If true, reverse the sort direction
    ///   - name: The name for the operation
    /// - Returns: A valid Tensor object with 32-bit integer data type
    pub fn arg_sort_with_axis_tensor(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        descending: bool,
        name: Option<&str>,
    ) -> Tensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let descending_val = if descending { YES } else { NO };

            let result: *mut AnyObject = msg_send![self.0, argSortWithTensor: tensor.0,
                axisTensor: axis_tensor.0,
                descending: descending_val,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }

    /// Computes the indices that sort the elements of the input tensor along an axis specified by a tensor (in ascending order).
    ///
    /// - Parameters:
    ///   - tensor: The input tensor
    ///   - axis_tensor: A scalar tensor that specifies the dimension over which to sort
    ///   - name: The name for the operation
    /// - Returns: A valid Tensor object with 32-bit integer data type
    pub fn arg_sort_ascending_with_axis_tensor(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        name: Option<&str>,
    ) -> Tensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let result: *mut AnyObject = msg_send![self.0, argSortWithTensor: tensor.0,
                axisTensor: axis_tensor.0,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }
}
