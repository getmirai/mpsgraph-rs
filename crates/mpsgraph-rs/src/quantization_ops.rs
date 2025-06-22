use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::tensor::{DataType, Tensor};

/// Trait for performing quantization operations on a graph
pub trait GraphQuantizationOps {
    /// Creates a Quantize operation and returns the result tensor.
    ///
    /// Convert the float `tensor` to an i8 or u8 tensor by applying a scale + bias transform:
    /// result = (tensor / scale) + zeroPoint
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor to be quantized
    /// * `scale` - Scale scalar parameter
    /// * `zero_point` - Bias scalar parameter (converted to dataType of resultTensor)
    /// * `data_type` - Integer data type of the result tensor
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor of datatype `data_type` or None if error
    fn quantize(
        &self,
        tensor: &Tensor,
        scale: f64,
        zero_point: f64,
        data_type: DataType,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a Dequantize operation and returns the result tensor.
    ///
    /// Convert the i8 or u8 `tensor` to a float tensor by applying a scale + bias transform:
    /// result = scale(tensor - zeroPoint)
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor to be dequantized
    /// * `scale` - Scale scalar parameter
    /// * `zero_point` - Bias scalar parameter (converted to dataType of tensor)
    /// * `data_type` - Float data type of the result tensor
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor of datatype `data_type` or None if error
    fn dequantize(
        &self,
        tensor: &Tensor,
        scale: f64,
        zero_point: f64,
        data_type: DataType,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a Quantize operation with scale tensor and returns the result tensor.
    ///
    /// Convert the float `tensor` to an i8 or u8 tensor by applying a scale + bias transform:
    /// result = (tensor / scaleTensor) + zeroPoint
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor to be quantized
    /// * `scale_tensor` - Scale 1D Tensor parameter with size == tensor.shape[axis]
    /// * `zero_point` - Bias scalar parameter (converted to dataType of resultTensor)
    /// * `data_type` - Integer data type of the result tensor
    /// * `axis` - Axis on which the scale 1D value is being broadcasted
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor of datatype `data_type` or None if error
    fn quantize_with_scale_tensor(
        &self,
        tensor: &Tensor,
        scale_tensor: &Tensor,
        zero_point: f64,
        data_type: DataType,
        axis: i64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a Dequantize operation with scale tensor and returns the result tensor.
    ///
    /// Convert the i8 or u8 `tensor` to a float tensor by applying a scale + bias transform:
    /// result = scaleTensor(tensor - zeroPoint)
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor to be dequantized
    /// * `scale_tensor` - Scale 1D Tensor parameter with size == tensor.shape[axis]
    /// * `zero_point` - Bias scalar parameter (converted to dataType of tensor)
    /// * `data_type` - Float data type of the result tensor
    /// * `axis` - Axis on which the scale 1D value is being broadcasted
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor of datatype `data_type` or None if error
    fn dequantize_with_scale_tensor(
        &self,
        tensor: &Tensor,
        scale_tensor: &Tensor,
        zero_point: f64,
        data_type: DataType,
        axis: i64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a Quantize operation with scale and zero point tensors and returns the result tensor.
    ///
    /// Convert the float `tensor` to an i8 or u8 tensor by applying a scale + bias transform:
    /// result = (tensor / scaleTensor) + zeroPointTensor
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor to be quantized
    /// * `scale_tensor` - Scale 1D Tensor parameter with size == tensor.shape[axis]
    /// * `zero_point_tensor` - Bias 1D Tensor parameter with size == tensor.shape[axis]
    /// * `data_type` - Integer data type of the result tensor
    /// * `axis` - Axis on which the scale 1D value is being broadcasted
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor of datatype `data_type` or None if error
    fn quantize_with_tensors(
        &self,
        tensor: &Tensor,
        scale_tensor: &Tensor,
        zero_point_tensor: &Tensor,
        data_type: DataType,
        axis: i64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a Dequantize operation with scale and zero point tensors and returns the result tensor.
    ///
    /// Convert the i8 or u8 `tensor` to a float tensor by applying a scale + bias transform:
    /// result = scaleTensor(tensor - zeroPointTensor)
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor to be dequantized
    /// * `scale_tensor` - Scale 1D Tensor parameter with size == tensor.shape[axis]
    /// * `zero_point_tensor` - Bias 1D Tensor parameter with size == tensor.shape[axis]
    /// * `data_type` - Float data type of the result tensor
    /// * `axis` - Axis on which the scale 1D value is being broadcasted
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor of datatype `data_type` or None if error
    fn dequantize_with_tensors(
        &self,
        tensor: &Tensor,
        scale_tensor: &Tensor,
        zero_point_tensor: &Tensor,
        data_type: DataType,
        axis: i64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a Dequantize operation with scale and zero point tensors and returns the result tensor.
    /// This version does not take an axis.
    ///
    /// Convert the i8 or u8 `tensor` to a float tensor by applying a scale + bias transform:
    /// result = scaleTensor(tensor - zeroPointTensor)
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor to be dequantized
    /// * `scale_tensor` - Scale Tensor parameter
    /// * `zero_point_tensor` - Bias Tensor parameter
    /// * `data_type` - Float data type of the result tensor
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor of datatype `data_type` or None if error
    fn dequantize_with_tensors_no_axis(
        &self,
        tensor: &Tensor,
        scale_tensor: &Tensor,
        zero_point_tensor: &Tensor,
        data_type: DataType,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a lookup-table based dequantization operation and returns the result tensor.
    ///
    /// Converts a u8 or u4 `tensor` to a float tensor by applying a lookup operation:
    /// result[i1,...,in] = LUTTensor[i1',...,in',tensor[i1,...,in]].
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor to be dequantized
    /// * `lut_tensor` - The lookup table to use - for u4 the last dimension should have 16 elements, and for u8 256 elements
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object or None if error
    fn dequantize_with_lut(
        &self,
        tensor: &Tensor,
        lut_tensor: &Tensor,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a vector lookup-table based dequantization operation and returns the result tensor.
    ///
    /// Converts a u8 or u4 `tensor` to a float tensor by applying a lookup operation.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor to be dequantized
    /// * `lut_tensor` - The lookup table to use - for u4 the second to last dimension should have 16 elements, and for u8 256 elements
    /// * `axis` - Axis on which the scale 1D value is being broadcasted
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object or None if error
    fn dequantize_with_lut_axis(
        &self,
        tensor: &Tensor,
        lut_tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a quantize operation to convert a floating-point tensor to a quantized tensor.
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor to quantize
    /// * `scale` - The scale factor for quantization
    /// * `zero_point` - The zero point for quantization
    /// * `axis` - The axis along which to quantize
    /// * `data_type` - The target quantized data type
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object containing the quantized values
    fn quantize_tensor(
        &self,
        x: &Retained<Tensor>,
        scale: &Retained<Tensor>,
        zero_point: &Retained<Tensor>,
        axis: i64,
        data_type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a dequantize operation to convert a quantized tensor back to floating-point.
    ///
    /// # Arguments
    ///
    /// * `x` - The input quantized tensor to dequantize
    /// * `scale` - The scale factor for dequantization
    /// * `zero_point` - The zero point for dequantization
    /// * `axis` - The axis along which to dequantize
    /// * `data_type` - The source quantized data type
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object containing the dequantized values
    fn dequantize_tensor(
        &self,
        x: &Retained<Tensor>,
        scale: &Retained<Tensor>,
        zero_point: &Retained<Tensor>,
        axis: i64,
        data_type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor>;
}

/// Implementation of quantization operations for Graph
impl GraphQuantizationOps for Graph {
    fn quantize(
        &self,
        tensor: &Tensor,
        scale: f64,
        zero_point: f64,
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
                quantizeTensor: tensor,
                scale: scale,
                zeroPoint: zero_point,
                dataType: data_type as u32,
                name: name_ptr
            ]
        }
    }

    fn dequantize(
        &self,
        tensor: &Tensor,
        scale: f64,
        zero_point: f64,
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
                scale: scale,
                zeroPoint: zero_point,
                dataType: data_type as u32,
                name: name_ptr
            ]
        }
    }

    fn quantize_with_scale_tensor(
        &self,
        tensor: &Tensor,
        scale_tensor: &Tensor,
        zero_point: f64,
        data_type: DataType,
        axis: i64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                quantizeTensor: tensor,
                scaleTensor: scale_tensor,
                zeroPoint: zero_point,
                dataType: data_type as u32,
                axis: axis,
                name: name_ptr
            ]
        }
    }

    fn dequantize_with_scale_tensor(
        &self,
        tensor: &Tensor,
        scale_tensor: &Tensor,
        zero_point: f64,
        data_type: DataType,
        axis: i64,
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
                zeroPoint: zero_point,
                dataType: data_type as u32,
                axis: axis,
                name: name_ptr
            ]
        }
    }

    fn quantize_with_tensors(
        &self,
        tensor: &Tensor,
        scale_tensor: &Tensor,
        zero_point_tensor: &Tensor,
        data_type: DataType,
        axis: i64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                quantizeTensor: tensor,
                scaleTensor: scale_tensor,
                zeroPointTensor: zero_point_tensor,
                dataType: data_type as u32,
                axis: axis,
                name: name_ptr
            ]
        }
    }

    fn dequantize_with_tensors(
        &self,
        tensor: &Tensor,
        scale_tensor: &Tensor,
        zero_point_tensor: &Tensor,
        data_type: DataType,
        axis: i64,
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
                axis: axis,
                name: name_ptr
            ]
        }
    }

    fn dequantize_with_tensors_no_axis(
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

    fn dequantize_with_lut(
        &self,
        tensor: &Tensor,
        lut_tensor: &Tensor,
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
                LUTTensor: lut_tensor,
                name: name_ptr
            ]
        }
    }

    fn dequantize_with_lut_axis(
        &self,
        tensor: &Tensor,
        lut_tensor: &Tensor,
        axis: i64,
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
                LUTTensor: lut_tensor,
                axis: axis,
                name: name_ptr
            ]
        }
    }

    fn quantize_tensor(
        &self,
        x: &Retained<Tensor>,
        scale: &Retained<Tensor>,
        zero_point: &Retained<Tensor>,
        axis: i64,
        data_type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                quantizeTensor: &**x,
                scale: &**scale,
                zeroPoint: &**zero_point,
                axis: axis,
                type: data_type as u32,
                name: name_ptr
            ]
        }
    }

    fn dequantize_tensor(
        &self,
        x: &Retained<Tensor>,
        scale: &Retained<Tensor>,
        zero_point: &Retained<Tensor>,
        axis: i64,
        data_type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                dequantizeTensor: &**x,
                scale: &**scale,
                zeroPoint: &**zero_point,
                axis: axis,
                type: data_type as u32,
                name: name_ptr
            ]
        }
    }
}

/// Extension trait for easier access to quantization operations
pub trait GraphQuantizationOpsExtension {
    /// Get access to quantization operations
    fn quantization_ops(&self) -> &dyn GraphQuantizationOps;
}

impl GraphQuantizationOpsExtension for Graph {
    fn quantization_ops(&self) -> &dyn GraphQuantizationOps {
        self
    }
}
