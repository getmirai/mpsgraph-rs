use crate::core::{AsRawObject, MPSDataType, NSString};
use crate::graph::Graph;
use crate::operation::Operation;
use crate::shape::Shape;
use crate::tensor::Tensor;
use objc2::msg_send;
use objc2::runtime::AnyObject;
use objc2_foundation;

/// Memory operations for Graph
impl Graph {
    // The placeholder and constant_scalar methods are already implemented in graph.rs

    /// Creates a complex constant with the realPart and imaginaryPart values and returns the result tensor.
    ///
    /// # Arguments
    ///
    /// * `real_part` - The real part of the complex scalar to fill the entire tensor values with.
    /// * `imaginary_part` - The imaginary part of the complex scalar to fill the entire tensor values with.
    ///
    /// # Returns
    ///
    /// A valid Tensor object of type ComplexFloat32.
    pub fn complex_constant(&self, real_part: f64, imaginary_part: f64) -> Tensor {
        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, constantWithRealPart: real_part,
                imaginaryPart: imaginary_part,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }

    /// Creates a complex constant with the specified data type and returns the result tensor.
    ///
    /// # Arguments
    ///
    /// * `real_part` - The real part of the complex scalar to fill the entire tensor values with.
    /// * `imaginary_part` - The imaginary part of the complex scalar to fill the entire tensor values with.
    /// * `data_type` - The complex data type of the constant tensor.
    ///
    /// # Returns
    ///
    /// A valid Tensor object of complex type.
    pub fn complex_constant_with_type(
        &self,
        real_part: f64,
        imaginary_part: f64,
        data_type: MPSDataType,
    ) -> Tensor {
        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, constantWithRealPart: real_part,
                imaginaryPart: imaginary_part,
                dataType: data_type as u64
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }

    /// Creates a complex constant with shape and returns the result tensor.
    ///
    /// # Arguments
    ///
    /// * `real_part` - The real part of the complex scalar to fill the entire tensor values with.
    /// * `imaginary_part` - The imaginary part of the complex scalar to fill the entire tensor values with.
    /// * `shape` - The shape of the output tensor.
    /// * `data_type` - The complex data type of the constant tensor.
    ///
    /// # Returns
    ///
    /// A valid Tensor object of complex type.
    pub fn complex_constant_with_shape(
        &self,
        real_part: f64,
        imaginary_part: f64,
        shape: &Shape,
        data_type: MPSDataType,
    ) -> Tensor {
        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, constantWithRealPart: real_part,
                imaginaryPart: imaginary_part,
                shape: shape.0,
                dataType: data_type as u64
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }

    /// Creates a variable operation and returns the result tensor.
    ///
    /// # Arguments
    ///
    /// * `data` - The data for the tensor.
    /// * `shape` - The shape of the output tensor. This has to be statically shaped.
    /// * `data_type` - The dataType of the variable tensor.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A valid Tensor object.
    pub fn variable<T: Copy>(
        &self,
        data: &[T],
        shape: &Shape,
        data_type: MPSDataType,
        name: Option<&str>,
    ) -> Tensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            // Create NSData using objc2_foundation
            let bytes_len = std::mem::size_of_val(data);
            let data_slice = std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes_len);
            let ns_data = objc2_foundation::NSData::with_bytes(data_slice);
            let data_obj: *mut AnyObject =
                ns_data.as_ref() as *const objc2_foundation::NSData as *mut AnyObject;

            let result: *mut AnyObject = msg_send![
                self.0, variableWithData: data_obj,
                shape: shape.0,
                dataType: data_type as u64,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }

    /// Creates a variable from an input tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor from which to form the variable.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A valid Tensor object representing the variable.
    pub fn variable_from_tensor(
        &self,
        tensor: &Tensor,
        name: Option<&str>,
    ) -> Tensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, variableFromTensorWithTensor: tensor.0,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }

    /// Creates a read op which reads at this point of execution of the graph.
    ///
    /// # Arguments
    ///
    /// * `variable` - The variable resource tensor to read from.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A valid Tensor object.
    pub fn read_variable(&self, variable: &Tensor, name: Option<&str>) -> Tensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, readVariable: variable.0,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Tensor(result)
        }
    }

    /// Creates an assign operation which writes at this point of execution of the graph.
    ///
    /// # Arguments
    ///
    /// * `variable` - The variable resource tensor to assign to.
    /// * `tensor` - The tensor to assign to the variable.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A valid MPSGraphOperation object.
    pub fn assign_variable(
        &self,
        variable: &Tensor,
        tensor: &Tensor,
        name: Option<&str>,
    ) -> Operation {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let result: *mut AnyObject = msg_send![
                self.0, assignVariable: variable.0,
                withValueOfTensor: tensor.0,
                name: name_obj,
            ];

            let result = objc2::ffi::objc_retain(result as *mut _);
            Operation(result)
        }
    }
}
