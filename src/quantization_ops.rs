use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::tensor::{DataType, Tensor};

impl Graph {
    /// Creates a Quantize operation and returns the result tensor.
    ///
    /// Convert the float `tensor` to an i8 or u8 tensor by applying a scale + bias transform:
    /// result = (tensor / scale) + zeroPoint
    ///
    /// # Parameters
    /// * `tensor` - Input tensor to be quantized
    /// * `scale` - Scale scalar parameter
    /// * `zero_point` - Bias scalar parameter (converted to dataType of resultTensor)
    /// * `data_type` - Integer data type of the result tensor.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    /// A valid MPSGraphTensor array of datatype dataType
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

    /// Creates Dequantize operation and returns the result tensor.
    ///
    /// Convert the i8 or u8 `tensor` to a float tensor by applying a scale + bias transform:
    /// result = scale(tensor - zeroPoint)
    ///
    /// # Parameters
    /// * `tensor` - Input tensor to be dequantized
    /// * `scale` - Scale scalar parameter
    /// * `zero_point` - Bias scalar parameter (converted to dataType of tensor)
    /// * `data_type` - Float data type of the result tensor.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    /// A valid MPSGraphTensor array of datatype dataType
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

    /// Creates a Quantize operation and returns the result tensor.
    ///
    /// Convert the float `tensor` to an i8 or u8 tensor by applying a scale + bias transform:
    /// result = (tensor / scaleTensor) + zeroPoint
    ///
    /// # Parameters
    /// * `tensor` - Input tensor to be quantized
    /// * `scale_tensor` - Scale 1D Tensor parameter with size == tensor.shape[axis]
    /// * `zero_point` - Bias scalar parameter (converted to dataType of resultTensor)
    /// * `data_type` - Integer data type of the result tensor.
    /// * `axis` - Axis on which the scale 1D value is being broadcasted
    /// * `name` - The name for the operation.
    /// # Returns
    /// A valid MPSGraphTensor array of datatype dataType
    pub fn quantize_with_scale_tensor(
        &self,
        tensor: &Tensor,
        scale_tensor: &Tensor,
        zero_point: f64,
        data_type: DataType,
        axis: isize,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let result: Retained<Tensor> = msg_send![self,
                quantizeTensor: tensor,
                scaleTensor: scale_tensor,
                zeroPoint: zero_point,
                dataType: data_type as u32,
                axis: axis,
                name: name_ptr
            ];
            result
        }
    }

    /// Creates Dequantize operation and returns the result tensor.
    ///
    /// Convert the i8 or u8 `tensor` to a float tensor by applying a scale + bias transform:
    /// result = scaleTensor(tensor - zeroPoint)
    ///
    /// # Parameters
    /// * `tensor` - Input tensor to be dequantized
    /// * `scale_tensor` - Scale scalar or 1D Tensor parameter with size == tensor.shape[axis]
    /// * `zero_point` - Bias scalar parameter (converted to dataType of tensor)
    /// * `data_type` - Float data type of the result tensor.
    /// * `axis` - Axis on which the scale 1D value is being broadcasted
    /// * `name` - The name for the operation.
    /// # Returns
    /// A valid MPSGraphTensor array of datatype dataType
    pub fn dequantize_with_scale_tensor(
        &self,
        tensor: &Tensor,
        scale_tensor: &Tensor,
        zero_point: f64,
        data_type: DataType,
        axis: isize,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let result: Retained<Tensor> = msg_send![self,
                dequantizeTensor: tensor,
                scaleTensor: scale_tensor,
                zeroPoint: zero_point,
                dataType: data_type as u32,
                axis: axis,
                name: name_ptr
            ];
            result
        }
    }

    /// Creates a Quantize operation and returns the result tensor.
    ///
    /// Convert the float `tensor` to an i8 or u8 tensor by applying a scale + bias transform:
    /// result = (tensor / scaleTensor) + zeroPointTensor
    ///
    /// # Parameters
    /// * `tensor` - Input tensor to be quantized
    /// * `scale_tensor` - Scale scalar or 1D Tensor parameter with size == tensor.shape[axis]
    /// * `zero_point_tensor` - Bias scalar or 1D Tensor parameter with size == tensor.shape[axis]
    /// * `data_type` - Integer data type of the result tensor.
    /// * `axis` - Axis on which the scale 1D value is being broadcasted
    /// * `name` - The name for the operation.
    /// # Returns
    /// A valid MPSGraphTensor array of datatype dataType
    pub fn quantize_with_scale_and_zero_point_tensors(
        &self,
        tensor: &Tensor,
        scale_tensor: &Tensor,
        zero_point_tensor: &Tensor,
        data_type: DataType,
        axis: isize,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let result: Retained<Tensor> = msg_send![self,
                quantizeTensor: tensor,
                scaleTensor: scale_tensor,
                zeroPointTensor: zero_point_tensor,
                dataType: data_type as u32,
                axis: axis,
                name: name_ptr
            ];
            result
        }
    }

    /// Creates a dequantize operation and returns the result tensor.
    ///
    /// Convert the i8 or u8 `tensor` to a float tensor by applying a scale + bias transform:
    /// result = scaleTensor(tensor - zeroPointTensor)
    ///
    /// # Parameters
    /// * `tensor` - Input tensor to be dequantized
    /// * `scale_tensor` - Scale scalar or 1D Tensor parameter with size == tensor.shape[axis]
    /// * `zero_point_tensor` - Bias scalar or 1D Tensor parameter with size == tensor.shape[axis]
    /// * `data_type` - Float data type of the result tensor.
    /// * `axis` - Axis on which the scale 1D value is being broadcasted
    /// * `name` - The name for the operation.
    /// # Returns
    /// A valid MPSGraphTensor array of datatype dataType
    pub fn dequantize_with_scale_and_zero_point_tensors(
        &self,
        tensor: &Tensor,
        scale_tensor: &Tensor,
        zero_point_tensor: &Tensor,
        data_type: DataType,
        axis: isize,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let result: Retained<Tensor> = msg_send![self,
                dequantizeTensor: tensor,
                scaleTensor: scale_tensor,
                zeroPointTensor: zero_point_tensor,
                dataType: data_type as u32,
                axis: axis,
                name: name_ptr
            ];
            result
        }
    }

    /// Creates a dequantize operation and returns the result tensor.
    ///
    /// Convert the i8, u8, i4 or u4 `tensor` to a float tensor by applying a scale and bias transform:
    /// result = scaleTensor(tensor - zeroPointTensor).
    ///
    /// # Parameters
    /// * `tensor` - Input tensor to be dequantized.
    /// * `scale_tensor` - The scale tensor with groups support.
    /// * `zero_point_tensor` - The bias tensor with groups support.
    /// * `data_type` - Float data type of the result tensor.
    /// * `name` - The name for the operation.
    /// # Returns
    /// A valid ``MPSGraphTensor`` array of datatype `dataType`.
    pub fn dequantize_with_scale_and_zero_point_tensors_no_axis(
        &self,
        tensor: &Tensor,
        scale_tensor: &Tensor,
        zero_point_tensor: &Tensor,
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
                scaleTensor: scale_tensor,
                zeroPointTensor: zero_point_tensor,
                dataType: data_type as u32,
                name: name_ptr
            ];
            result
        }
    }

    /// Creates a dequantize operation and returns the result tensor.
    ///
    /// Converts the i8, u8, i4 or u4 `tensor` to a float tensor by applying a scale and bias transform:
    /// result = scaleTensor * tensor.
    ///
    /// # Parameters
    /// * `tensor` - Input tensor to be dequantized.
    /// * `scale_tensor` - Scale Tensor parameter with groups support.
    /// * `data_type` - Float data type of the result tensor.
    /// * `name` - The name for the operation.
    /// # Returns
    /// A valid ``MPSGraphTensor`` array of datatype `dataType`.
    pub fn dequantize_with_scale_tensor_no_axis(
        &self,
        tensor: &Tensor,
        scale_tensor: &Tensor,
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
                scaleTensor: scale_tensor,
                dataType: data_type as u32,
                name: name_ptr
            ];
            result
        }
    }

    /// Creates a lookup-table based quantization operation and returns the result tensor.
    ///
    /// Converts a u8 or u4 `tensor` to a float tensor by applying a lookup operation:
    /// result[i1,...,in] = LUTTensor[i1',...,in',tensor[i1,...,in]].
    /// Note: The operation supports LUT groups up to the last 3 dimensions for `tensor`.
    ///
    /// # Parameters
    /// * `tensor` - Input tensor to be dequantized.
    /// * `lut_tensor` - The lookup table to use - for u4 the last dimension should have 16 elements, and for u8 256 elements.
    /// * `name` - The name for the operation.
    /// # Returns
    /// A valid ``MPSGraphTensor`` object.
    pub fn dequantize_with_lut(
        &self,
        tensor: &Tensor,
        lut_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let result: Retained<Tensor> = msg_send![self,
                dequantizeTensor: tensor,
                LUTTensor: lut_tensor,
                name: name_ptr
            ];
            result
        }
    }

    /// Creates a vector lookup-table based quantization operation and returns the result tensor.
    ///
    /// Converts a u8 or u4 `tensor` to a float tensor by applying a lookup operation, where each
    /// input index defines a vector of values. The operation reads the vector values from the last dimension of the lookup table
    /// tensor and stores them into the dimension defined by `axis` on the result tensor.
    /// result[i1, ... , i_axis, ..., in] = LUTTensor[i1', ..., in', tensor[i1, ..., in], i_axis]
    /// Note: The operation supports LUT groups up to the last 2 dimensions for `tensor`.
    ///
    /// # Parameters
    /// * `tensor` - Input tensor to be dequantized.
    /// * `lut_tensor` - The lookup table to use - for u4 the second to last dimension should have 16 elements, and for u8 256 elements.
    /// * `axis` - Axis on which the scale 1D value is being broadcasted.
    /// * `name` - The name for the operation.
    /// # Returns
    /// A valid ``MPSGraphTensor`` object.
    pub fn dequantize_with_lut_and_axis(
        &self,
        tensor: &Tensor,
        lut_tensor: &Tensor,
        axis: isize,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let result: Retained<Tensor> = msg_send![self,
                dequantizeTensor: tensor,
                LUTTensor: lut_tensor,
                axis: axis,
                name: name_ptr
            ];
            result
        }
    }
}

//
