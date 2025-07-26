mod quantization_arguments;

pub use quantization_arguments::{DequantizationArguments, QuantizationArguments};

use objc2::{msg_send, rc::Retained};
use objc2_foundation::NSString;

use crate::{DataType, Graph, Tensor};

/// MPSGraphMatrixMultiplicationOps.
impl Graph {
    /// Creates a Quantize operation and returns the result tensor.
    ///
    /// - Parameters:
    /// - tensor: Input tensor to be quantized
    /// - arguments: The arguments for the quantization operation
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor array of datatype dataType
    pub fn quantize<'a>(
        &self,
        tensor: &Tensor,
        arguments: QuantizationArguments<'a>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match arguments {
            QuantizationArguments::ScaleZeroPointDataType {
                scale,
                zero_point,
                data_type,
            } => unsafe {
                msg_send![self,
                quantizeTensor: tensor,
                scale: scale,
                zeroPoint: zero_point,
                dataType: data_type,
                name: name.map(NSString::from_str).as_deref(),
                ]
            },
            QuantizationArguments::ScaleTensorZeroPointDataTypeAxis {
                scale_tensor,
                zero_point,
                data_type,
                axis,
            } => unsafe {
                msg_send![
                    self,
                    quantizeTensor: tensor,
                    scaleTensor: scale_tensor,
                    zeroPoint: zero_point,
                    dataType: data_type,
                    axis: axis,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            QuantizationArguments::ScaleTensorZeroPointTensorDataTypeAxis {
                scale_tensor,
                zero_point_tensor,
                data_type,
                axis,
            } => unsafe {
                msg_send![
                    self,
                    quantizeTensor: tensor,
                    scaleTensor: scale_tensor,
                    zeroPointTensor: zero_point_tensor,
                    dataType: data_type,
                    axis: axis,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }

    /// Creates Dequantize operation and returns the result tensor.
    ///
    /// - Parameters:
    /// - tensor: Input tensor to be dequantized
    /// - arguments: The arguments for the dequantization operation
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor array of datatype dataType
    pub fn dequantize<'a>(
        &self,
        tensor: &Tensor,
        arguments: DequantizationArguments<'a>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match arguments {
            DequantizationArguments::ScaleZeroPointDataType {
                scale,
                zero_point,
                data_type,
            } => unsafe {
                msg_send![
                    self,
                    dequantizeTensor: tensor,
                    scale: scale,
                    zeroPoint: zero_point,
                    dataType: data_type,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            DequantizationArguments::ScaleTensorZeroPointDataTypeAxis {
                scale_tensor,
                zero_point,
                data_type,
                axis,
            } => unsafe {
                msg_send![
                    self,
                    dequantizeTensor: tensor,
                    scaleTensor: scale_tensor,
                    zeroPoint: zero_point,
                    dataType: data_type,
                    axis: axis,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            DequantizationArguments::ScaleTensorZeroPointTensorDataTypeAxis {
                scale_tensor,
                zero_point_tensor,
                data_type,
                axis,
            } => unsafe {
                msg_send![
                    self,
                    dequantizeTensor: tensor,
                    scaleTensor: scale_tensor,
                    zeroPointTensor: zero_point_tensor,
                    dataType: data_type,
                    axis: axis,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            DequantizationArguments::ScaleTensorZeroPointTensorDataType {
                scale_tensor,
                zero_point_tensor,
                data_type,
            } => unsafe {
                msg_send![
                    self,
                    dequantizeTensor: tensor,
                    scaleTensor: scale_tensor,
                    zeroPointTensor: zero_point_tensor,
                    dataType: data_type,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            DequantizationArguments::ScaleTensorDataType {
                scale_tensor,
                data_type,
            } => unsafe {
                msg_send![
                    self,
                    dequantizeTensor: tensor,
                    scaleTensor: scale_tensor,
                    dataType: data_type,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            DequantizationArguments::LutTensor { lut_tensor } => unsafe {
                msg_send![
                    self,
                    dequantizeTensor: tensor,
                    LUTTensor: lut_tensor,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            DequantizationArguments::LutTensorAxis { lut_tensor, axis } => unsafe {
                msg_send![
                    self,
                    dequantizeTensor: tensor,
                    LUTTensor: lut_tensor,
                    axis: axis,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }
}
