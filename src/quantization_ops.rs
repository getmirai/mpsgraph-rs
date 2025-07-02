use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::tensor::{DataType, Tensor};

/// Trait for performing quantization operations on a graph
pub trait GraphQuantizationOps {
    fn dequantize_with_scale_tensor(
        &self,
        tensor: &Tensor,
        scale_tensor: &Tensor,
        data_type: DataType,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    fn dequantize_with_scale_tensor_and_zero_point_tensor(
        &self,
        tensor: &Tensor,
        scale_tensor: &Tensor,
        zero_point_tensor: &Tensor,
        data_type: DataType,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;
}

/// Implementation of quantization operations for Graph
impl GraphQuantizationOps for Graph {
    fn dequantize_with_scale_tensor(
        &self,
        tensor: &Tensor,
        scale_tensor: &Tensor,
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
                dequantizeTensor: tensor,
                scaleTensor: scale_tensor,
                dataType: data_type as u32,
                name: name_ptr
            ]
        }
    }

    fn dequantize_with_scale_tensor_and_zero_point_tensor(
        &self,
        tensor: &Tensor,
        scale_tensor: &Tensor,
        zero_point_tensor: &Tensor,
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
                dequantizeTensor: tensor,
                scaleTensor: scale_tensor,
                zeroPointTensor: zero_point_tensor,
                dataType: data_type as u32,
                name: name_ptr
            ]
        }
    }
}
