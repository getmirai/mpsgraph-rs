use crate::{DataType, Tensor};

pub enum QuantizationArguments<'a> {
    /// Convert the float `tensor` to an i8 or u8 tensor by applying a scale + bias transform:
    /// result = (tensor / scale) + zeroPoint
    ScaleZeroPointDataType {
        /// Scale scalar parameter
        scale: f64,
        /// Bias scalar parameter (converted to dataType of resultTensor)
        zero_point: i64,
        /// Integer data type of the result tensor.
        data_type: DataType,
    },
    /// Convert the float `tensor` to an i8 or u8 tensor by applying a scale + bias transform:
    /// result = (tensor / scaleTensor) + zeroPoint
    ScaleTensorZeroPointDataTypeAxis {
        /// Scale 1D Tensor parameter with size == tensor.shape[axis]
        scale_tensor: &'a Tensor,
        /// Bias scalar parameter (converted to dataType of resultTensor)
        zero_point: f64,
        /// Integer data type of the result tensor.
        data_type: DataType,
        /// Axis on which the scale 1D value is being broadcasted
        axis: i64,
    },
    /// Convert the float `tensor` to an i8 or u8 tensor by applying a scale + bias transform:
    /// result = (tensor / scaleTensor) + zeroPointTensor
    ScaleTensorZeroPointTensorDataTypeAxis {
        /// Scale scalar or 1D Tensor parameter with size == tensor.shape[axis]
        scale_tensor: &'a Tensor,
        /// Bias scalar or 1D Tensor parameter with size == tensor.shape[axis]
        zero_point_tensor: &'a Tensor,
        /// Integer data type of the result tensor.
        data_type: DataType,
        /// Axis on which the scale 1D value is being broadcasted
        axis: i64,
    },
}

pub enum DequantizationArguments<'a> {
    /// Convert the i8 or u8 `tensor` to a float tensor by applying a scale + bias transform:
    /// result = scale(tensor - zeroPoint)
    ScaleZeroPointDataType {
        /// Scale scalar parameter
        scale: f64,
        /// Bias scalar parameter (converted to dataType of resultTensor)
        zero_point: i64,
        /// Integer data type of the result tensor.
        data_type: DataType,
    },
    /// Convert the i8 or u8 `tensor` to a float tensor by applying a scale + bias transform:
    /// result = scaleTensor(tensor - zeroPoint)
    ScaleTensorZeroPointDataTypeAxis {
        /// Scale 1D Tensor parameter with size == tensor.shape[axis]
        scale_tensor: &'a Tensor,
        /// Bias scalar parameter (converted to dataType of resultTensor)
        zero_point: f64,
        /// Integer data type of the result tensor.
        data_type: DataType,
        /// Axis on which the scale 1D value is being broadcasted
        axis: i64,
    },
    /// Convert the i8 or u8 `tensor` to a float tensor by applying a scale + bias transform:
    /// result = scaleTensor(tensor - zeroPointTensor)
    ScaleTensorZeroPointTensorDataTypeAxis {
        /// Scale scalar or 1D Tensor parameter with size == tensor.shape[axis]
        scale_tensor: &'a Tensor,
        /// Bias scalar or 1D Tensor parameter with size == tensor.shape[axis]
        zero_point_tensor: &'a Tensor,
        /// Integer data type of the result tensor.
        data_type: DataType,
        /// Axis on which the scale 1D value is being broadcasted
        axis: i64,
    },
    /// Convert the i8, u8, i4 or u4 `tensor` to a float tensor by applying a scale and bias transform:
    /// ```md
    /// result = scaleTensor(tensor - zeroPointTensor).
    /// ```
    ScaleTensorZeroPointTensorDataType {
        /// The scale tensor with groups support.
        scale_tensor: &'a Tensor,
        /// The bias tensor with groups support.
        zero_point_tensor: &'a Tensor,
        /// Float data type of the result tensor.
        data_type: DataType,
    },
    /// Converts the i8, u8, i4 or u4 `tensor` to a float tensor by applying a scale and bias transform:
    /// ```md
    /// result = scaleTensor * tensor.
    /// ```
    ScaleTensorDataType {
        /// Scale Tensor parameter with groups support.
        scale_tensor: &'a Tensor,
        /// Float data type of the result tensor.
        data_type: DataType,
    },
    /// Converts a u8 or u4 `tensor` to a float tensor by applying a lookup operation:
    /// ```md
    /// result[i1,...,in] = LUTTensor[i1',...,in',tensor[i1,...,in]].
    /// ```
    /// Note: The operation supports LUT groups up to the last 3 dimensions for `tensor`.
    LutTensor {
        /// The lookup table to use - for u4 the last dimension should have 16 elements, and for u8 256 elements.
        lut_tensor: &'a Tensor,
    },
    /// Converts a u8 or u4 `tensor` to a float tensor by applying a lookup operation, where each
    /// input index defines a vector of values. The operation reads the vector values from the last dimension of the lookup table
    /// tensor and stores them into the dimension defined by `axis` on the result tensor.
    /// ```md
    /// result[i1, ... , i_axis, ..., in] = LUTTensor[i1', ..., in', tensor[i1, ..., in], i_axis]
    /// ```
    /// Note: The operation supports LUT groups up to the last 2 dimensions for `tensor`.
    LutTensorAxis {
        /// The lookup table to use - for u4 the second to last dimension should have 16 elements, and for u8 256 elements.
        lut_tensor: &'a Tensor,
        /// Axis on which the scale 1D value is being broadcasted.
        axis: i64,
    },
}
