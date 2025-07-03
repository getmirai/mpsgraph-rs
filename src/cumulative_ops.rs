use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::tensor::Tensor;

/// Cumulative helpers are now inherent methods on `Graph`.
impl Graph {
    // -- SUM -----------------------------------------------------------------
    pub fn cumulative_sum(
        &self,
        tensor: &Tensor,
        axis: i64,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: Option<Retained<Tensor>> = msg_send![
                self,
                cumulativeSumWithTensor: tensor,
                axis: axis,
                exclusive: exclusive,
                reverse: reverse,
                name: name_ptr,
            ];
            result
        }
    }

    pub fn cumulative_sum_with_axis_tensor(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: Option<Retained<Tensor>> = msg_send![
                self,
                cumulativeSumWithTensor: tensor,
                axisTensor: axis_tensor,
                exclusive: exclusive,
                reverse: reverse,
                name: name_ptr,
            ];
            result
        }
    }

    pub fn cumulative_sum_simple(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: Option<Retained<Tensor>> = msg_send![
                self,
                cumulativeSumWithTensor: tensor,
                axis: axis,
                name: name_ptr,
            ];
            result
        }
    }

    // -- PRODUCT -------------------------------------------------------------
    pub fn cumulative_product(
        &self,
        tensor: &Tensor,
        axis: i64,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: Option<Retained<Tensor>> = msg_send![
                self,
                cumulativeProductWithTensor: tensor,
                axis: axis,
                exclusive: exclusive,
                reverse: reverse,
                name: name_ptr,
            ];
            result
        }
    }

    pub fn cumulative_product_with_axis_tensor(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: Option<Retained<Tensor>> = msg_send![
                self,
                cumulativeProductWithTensor: tensor,
                axisTensor: axis_tensor,
                exclusive: exclusive,
                reverse: reverse,
                name: name_ptr,
            ];
            result
        }
    }

    pub fn cumulative_product_simple(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: Option<Retained<Tensor>> = msg_send![
                self,
                cumulativeProductWithTensor: tensor,
                axis: axis,
                name: name_ptr,
            ];
            result
        }
    }

    // -- MIN / MAX helpers (similar structure, omitted for brevity) -----------
    // The remaining methods were kept unchanged below; only `pub` keyword was
    // added to their signatures for external visibility.

    pub fn cumulative_minimum(
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

            let result: Option<Retained<Tensor>> = msg_send![
                self,
                cumulativeMinimumWithTensor: tensor,
                axis: axis,
                exclusive: exclusive,
                reverse: reverse,
                name: name_ptr,
            ];
            result
        }
    }

    pub fn cumulative_maximum(
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

            let result: Option<Retained<Tensor>> = msg_send![
                self,
                cumulativeMaximumWithTensor: tensor,
                axis: axis,
                exclusive: exclusive,
                reverse: reverse,
                name: name_ptr,
            ];
            result
        }
    }

    pub fn cumulative_minimum_with_axis_tensor(
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

            let result: Option<Retained<Tensor>> = msg_send![
                self,
                cumulativeMinimumWithTensor: tensor,
                axisTensor: axis_tensor,
                exclusive: exclusive,
                reverse: reverse,
                name: name_ptr,
            ];
            result
        }
    }

    pub fn cumulative_minimum_simple(
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

            let result: Option<Retained<Tensor>> = msg_send![
                self,
                cumulativeMinimumWithTensor: tensor,
                axis: axis,
                name: name_ptr,
            ];
            result
        }
    }

    pub fn cumulative_maximum_with_axis_tensor(
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

            let result: Option<Retained<Tensor>> = msg_send![
                self,
                cumulativeMaximumWithTensor: tensor,
                axisTensor: axis_tensor,
                exclusive: exclusive,
                reverse: reverse,
                name: name_ptr,
            ];
            result
        }
    }

    pub fn cumulative_maximum_simple(
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

            let result: Option<Retained<Tensor>> = msg_send![
                self,
                cumulativeMaximumWithTensor: tensor,
                axis: axis,
                name: name_ptr,
            ];
            result
        }
    }
}
