use crate::core::DataType;
use crate::graph::Graph;
use crate::operation::Operation;
use crate::shape::Shape;
use crate::tensor::Tensor;
use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::{NSData, NSString};
use std::ptr;

/// Memory operations for Graph
pub trait GraphMemoryOps {
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
    fn complex_constant(&self, real_part: f64, imaginary_part: f64) -> Retained<Tensor>;

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
    fn complex_constant_with_type(
        &self,
        real_part: f64,
        imaginary_part: f64,
        data_type: DataType,
    ) -> Retained<Tensor>;

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
    fn complex_constant_with_shape(
        &self,
        real_part: f64,
        imaginary_part: f64,
        shape: &Shape,
        data_type: DataType,
    ) -> Retained<Tensor>;

    /// Creates a variable from raw data.
    ///
    /// # Arguments
    ///
    /// * `data` - The raw data for the tensor as a byte slice.
    /// * `shape` - The shape of the output tensor. This has to be statically shaped.
    /// * `data_type` - The dataType of the variable tensor.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A valid Tensor object.
    fn variable_with_bytes(
        &self,
        data: &[u8],
        shape: &Shape,
        data_type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor>;

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
    fn variable_from_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor>;

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
    fn read_variable(&self, variable: &Tensor, name: Option<&str>) -> Retained<Tensor>;

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
    /// A valid Operation object.
    fn assign_variable(
        &self,
        variable: &Tensor,
        tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Operation>;

    /// Creates a placeholder tensor with the specified shape and data type.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the placeholder tensor
    /// * `data_type` - The data type of the placeholder tensor
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn placeholder_tensor(
        &self,
        shape: &Shape,
        data_type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a storage tensor with the specified handle, shape, and data type.
    ///
    /// # Arguments
    ///
    /// * `handle` - The storage handle
    /// * `shape` - The shape of the storage tensor
    /// * `data_type` - The data type of the storage tensor
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn storage_tensor(&self, handle: u64, shape: &Shape, data_type: DataType) -> Retained<Tensor>;
}

/// Extension methods to provide generic variable creation
pub trait GraphVariableOps {
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
    fn variable<T: Copy>(
        &self,
        data: &[T],
        shape: &Shape,
        data_type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor>;
}

impl GraphMemoryOps for Graph {
    fn complex_constant(&self, real_part: f64, imaginary_part: f64) -> Retained<Tensor> {
        unsafe {
            let result: *mut Tensor = msg_send![
                self,
                constantWithRealPart: real_part,
                imaginaryPart: imaginary_part
            ];

            if result.is_null() {
                panic!("Failed to create complex constant tensor");
            } else {
                // This method returns an autoreleased object
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    fn complex_constant_with_type(
        &self,
        real_part: f64,
        imaginary_part: f64,
        data_type: DataType,
    ) -> Retained<Tensor> {
        unsafe {
            let result: *mut Tensor = msg_send![
                self,
                constantWithRealPart: real_part,
                imaginaryPart: imaginary_part,
                dataType: data_type as u64
            ];

            if result.is_null() {
                panic!("Failed to create complex constant tensor with type");
            } else {
                // This method returns an autoreleased object
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    fn complex_constant_with_shape(
        &self,
        real_part: f64,
        imaginary_part: f64,
        shape: &Shape,
        data_type: DataType,
    ) -> Retained<Tensor> {
        unsafe {
            let result: *mut Tensor = msg_send![
                self,
                constantWithRealPart: real_part,
                imaginaryPart: imaginary_part,
                shape: shape.as_ptr(),
                dataType: data_type as u64
            ];

            if result.is_null() {
                panic!("Failed to create complex constant tensor with shape");
            } else {
                // This method returns an autoreleased object
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    fn variable_with_bytes(
        &self,
        data: &[u8],
        shape: &Shape,
        data_type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(ptr::null(), |s| s as *const _);

            // Create NSData
            let ns_data = NSData::with_bytes(data);

            let result: *mut Tensor = msg_send![
                self,
                variableWithData: &*ns_data,
                shape: shape.as_ptr(),
                dataType: data_type as u64,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create variable tensor from bytes");
            } else {
                // This method returns an autoreleased object
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    fn variable_from_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                variableFromTensorWithTensor: tensor,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create variable from tensor");
            } else {
                // This method returns an autoreleased object
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    fn read_variable(&self, variable: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                readVariable: variable,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to read variable tensor");
            } else {
                // This method returns an autoreleased object
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    fn assign_variable(
        &self,
        variable: &Tensor,
        tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Operation> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(ptr::null(), |s| s as *const _);

            let result: *mut Operation = msg_send![
                self,
                assignVariable: variable,
                withValueOfTensor: tensor,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to assign variable tensor");
            } else {
                // This method returns an autoreleased object
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    fn placeholder_tensor(
        &self,
        shape: &Shape,
        data_type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                placeholderTensorWithShape: shape.as_ptr(),
                dataType: data_type as u64,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create placeholder tensor");
            } else {
                // This method returns an autoreleased object
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    fn storage_tensor(&self, handle: u64, shape: &Shape, data_type: DataType) -> Retained<Tensor> {
        unsafe {
            let result: *mut Tensor = msg_send![
                self,
                storageTensorWithHandle: handle,
                shape: shape.as_ptr(),
                dataType: data_type as u64
            ];

            if result.is_null() {
                panic!("Failed to create storage tensor");
            } else {
                // This method returns an autoreleased object
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }
}

impl GraphVariableOps for Graph {
    fn variable<T: Copy>(
        &self,
        data: &[T],
        shape: &Shape,
        data_type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            // Create NSData
            let bytes_len = std::mem::size_of_val(data);
            let data_slice = std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes_len);

            self.variable_with_bytes(data_slice, shape, data_type, name)
        }
    }
}

/// Extension trait providing a method for Graph to access memory operations
pub trait GraphMemoryOpsExtension {
    /// Access memory operations for this graph
    fn memory_ops(&self) -> &Self
    where
        Self: GraphMemoryOps;
}

impl GraphMemoryOpsExtension for Graph {
    fn memory_ops(&self) -> &Self
    where
        Self: GraphMemoryOps,
    {
        self
    }
}
