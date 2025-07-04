use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::tensor::{DataType, Tensor};

impl Graph {
    /// Quantize a floating-point tensor to int8/uint8 using scale and zero-point scalars.
    pub fn quantize(
        &self,
        tensor: &Tensor,
        scale: f64,
        zero_point: f64,
        data_type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let result: Retained<Tensor> = msg_send![self,
                quantizeTensor: tensor,
                scale: scale,
                zeroPoint: zero_point,
                dataType: data_type as u32,
                name: name_ptr
            ];
            result
        }
    }

    /// Dequantize an int8/uint8 tensor to float using scalar scale and zero-point.
    pub fn dequantize(
        &self,
        tensor: &Tensor,
        scale: f64,
        zero_point: f64,
        data_type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let result: Retained<Tensor> = msg_send![self,
                dequantizeTensor: tensor,
                scale: scale,
                zeroPoint: zero_point,
                dataType: data_type as u32,
                name: name_ptr
            ];
            result
        }
    }
}

//
