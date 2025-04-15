use objc2::runtime::AnyObject;
// In objc2, use false as NO and true as YES
const NO: bool = false;
const YES: bool = true;
use crate::core::{AsRawObject, NSString};
use crate::graph::Graph;
use crate::tensor::Tensor;
use objc2::msg_send;

/// Cumulative operations for Graph
impl Graph {
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
    pub fn cumulative_sum(
        &self,
        tensor: &Tensor,
        axis: i64,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Tensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        let exclusive_obj = if exclusive { YES } else { NO };
        let reverse_obj = if reverse { YES } else { NO };

        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, cumulativeSumWithTensor: tensor.0,
                axis: axis,
                exclusive: exclusive_obj,
                reverse: reverse_obj,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }

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
    pub fn cumulative_sum_with_axis_tensor(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Tensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        let exclusive_obj = if exclusive { YES } else { NO };
        let reverse_obj = if reverse { YES } else { NO };

        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, cumulativeSumWithTensor: tensor.0,
                axisTensor: axis_tensor.0,
                exclusive: exclusive_obj,
                reverse: reverse_obj,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }

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
    pub fn cumulative_sum_simple(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Tensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, cumulativeSumWithTensor: tensor.0,
                axis: axis,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }

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
    pub fn cumulative_product(
        &self,
        tensor: &Tensor,
        axis: i64,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Tensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        let exclusive_obj = if exclusive { YES } else { NO };
        let reverse_obj = if reverse { YES } else { NO };

        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, cumulativeProductWithTensor: tensor.0,
                axis: axis,
                exclusive: exclusive_obj,
                reverse: reverse_obj,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }

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
    pub fn cumulative_minimum(
        &self,
        tensor: &Tensor,
        axis: i64,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Tensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        let exclusive_obj = if exclusive { YES } else { NO };
        let reverse_obj = if reverse { YES } else { NO };

        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, cumulativeMinimumWithTensor: tensor.0,
                axis: axis,
                exclusive: exclusive_obj,
                reverse: reverse_obj,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }

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
    pub fn cumulative_maximum(
        &self,
        tensor: &Tensor,
        axis: i64,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Tensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        let exclusive_obj = if exclusive { YES } else { NO };
        let reverse_obj = if reverse { YES } else { NO };

        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, cumulativeMaximumWithTensor: tensor.0,
                axis: axis,
                exclusive: exclusive_obj,
                reverse: reverse_obj,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }

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
    pub fn cumulative_product_with_axis_tensor(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Tensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        let exclusive_obj = if exclusive { YES } else { NO };
        let reverse_obj = if reverse { YES } else { NO };

        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, cumulativeProductWithTensor: tensor.0,
                axisTensor: axis_tensor.0,
                exclusive: exclusive_obj,
                reverse: reverse_obj,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }

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
    pub fn cumulative_product_simple(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Tensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, cumulativeProductWithTensor: tensor.0,
                axis: axis,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }

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
    pub fn cumulative_minimum_with_axis_tensor(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Tensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        let exclusive_obj = if exclusive { YES } else { NO };
        let reverse_obj = if reverse { YES } else { NO };

        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, cumulativeMinimumWithTensor: tensor.0,
                axisTensor: axis_tensor.0,
                exclusive: exclusive_obj,
                reverse: reverse_obj,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }

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
    pub fn cumulative_minimum_simple(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Tensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, cumulativeMinimumWithTensor: tensor.0,
                axis: axis,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }

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
    pub fn cumulative_maximum_with_axis_tensor(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Tensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        let exclusive_obj = if exclusive { YES } else { NO };
        let reverse_obj = if reverse { YES } else { NO };

        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, cumulativeMaximumWithTensor: tensor.0,
                axisTensor: axis_tensor.0,
                exclusive: exclusive_obj,
                reverse: reverse_obj,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }

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
    pub fn cumulative_maximum_simple(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Tensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, cumulativeMaximumWithTensor: tensor.0,
                axis: axis,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }
}
