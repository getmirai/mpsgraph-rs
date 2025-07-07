use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::tensor::Tensor;

/// Cumulative helpers are now inherent methods on `Graph`.
impl Graph {
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

pub trait CumulativeSumAxisTensorExt {
    fn cumulative_sum(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;
}

pub trait CumulativeSumAxisTensorSimpleExt {
    fn cumulative_sum(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;
}

impl CumulativeSumAxisTensorExt for Graph {
    fn cumulative_sum(
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
}

impl CumulativeSumAxisTensorSimpleExt for Graph {
    fn cumulative_sum(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
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
                name: name_ptr,
            ];
            result
        }
    }
}

pub trait CumulativeSumSimpleExt {
    fn cumulative_sum(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;
}

impl CumulativeSumSimpleExt for Graph {
    fn cumulative_sum(
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
}

pub trait CumulativeProductAxisTensorExt {
    fn cumulative_product(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;
}

pub trait CumulativeProductAxisTensorSimpleExt {
    fn cumulative_product(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;
}

pub trait CumulativeProductSimpleExt {
    fn cumulative_product(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;
}

impl CumulativeProductAxisTensorExt for Graph {
    fn cumulative_product(
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
}

impl CumulativeProductAxisTensorSimpleExt for Graph {
    fn cumulative_product(
        &self,
        tensor: &Tensor,
        axis_tensor: &Tensor,
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
                name: name_ptr,
            ];
            result
        }
    }
}

impl CumulativeProductSimpleExt for Graph {
    fn cumulative_product(
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
}
