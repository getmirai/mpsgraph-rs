mod graph_variable_op;

pub use graph_variable_op::GraphVariableOp;

use crate::{DataType, Graph, Operation, Shape, Tensor};
use objc2::{extern_methods, msg_send, rc::Retained};
use objc2_foundation::{NSData, NSString};
use std::{mem::size_of_val, slice::from_raw_parts};

/// MemoryOps.
impl Graph {
    extern_methods!(
        /// Creates a constant op with a given shape and data, and returns the result tensor.
        ///
        /// # Arguments
        ///
        /// * `data` - Raw tensor bytes. The length must be `sizeof(data_type) * number_of_elements`.
        /// * `shape` - Statically shaped output [`Shape`].
        /// * `data_type` - [`DataType`] of the constant tensor.
        ///
        /// # Returns
        ///
        /// A valid [`Tensor`] object.
        #[unsafe(method(constantWithData:shape:dataType:))]
        #[unsafe(method_family = none)]
        pub fn constant_with_ns_data(
            &self,
            data: &NSData,
            shape: &Shape,
            data_type: DataType,
        ) -> Retained<Tensor>;

        /// Creates a complex constant op with a given shape and returns the result tensor.
        ///
        /// # Arguments
        ///
        /// * `real_part` - Real component of the complex scalar to fill the entire tensor values with  .
        /// * `imaginary_part` - Imaginary component of the complex scalar to fill the entire tensor values with.
        /// * `shape` - Statically shaped output [`Shape`].
        /// * `data_type` - [`DataType`] of the constant tensor.
        ///
        /// # Returns
        ///
        /// A valid [`Tensor`] object.
        #[unsafe(method(constantWithRealPart:imaginaryPart:shape:dataType:))]
        #[unsafe(method_family = none)]
        pub fn constant_with_real_imaginary_shape(
            &self,
            real_part: f64,
            imaginary_part: f64,
            shape: &Shape,
            data_type: DataType,
        ) -> Retained<Tensor>;
    );
}

impl Graph {
    /// Creates a placeholder operation and returns the result tensor.
    ///
    /// # Arguments
    ///
    /// * `shape` - Optional [`Shape`] of the output tensor. `None` produces an unranked tensor.
    /// * `data_type` - Optional [`DataType`] for the placeholder. `None` defaults to 32-bit float.
    /// * `name` - Name of the operation.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object.
    pub fn placeholder(
        &self,
        shape: Option<&Shape>,
        data_type: Option<DataType>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match data_type {
            Some(data_type) => unsafe {
                msg_send![
                    self,
                    placeholderWithShape: shape,
                    dataType: data_type,
                    name: name.map(NSString::from_str).as_deref()
                ]
            },
            None => unsafe {
                msg_send![
                    self,
                    placeholderWithShape: shape,
                    name: name.map(NSString::from_str).as_deref()
                ]
            },
        }
    }

    /// Creates a constant op with a given shape and data, and returns the result tensor.
    ///
    /// # Arguments
    ///
    /// * `data` - Slice containing tensor elements.
    /// * `shape` - Statically shaped output [`Shape`].
    /// * `data_type` - [`DataType`] of the constant tensor.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object.
    pub fn constant_with_data<T: Copy>(
        &self,
        data: &[T],
        shape: &Shape,
        data_type: DataType,
    ) -> Retained<Tensor> {
        let data_size = size_of_val(data);
        let ns_data =
            unsafe { NSData::with_bytes(from_raw_parts(data.as_ptr() as *const u8, data_size)) };
        Self::constant_with_ns_data(self, &ns_data, shape, data_type)
    }

    /// Creates a constant operation and returns the result tensor.
    ///
    /// # Arguments
    ///
    /// * `scalar` - Scalar value used to fill the tensor.
    /// * `shape` - Optional output [`Shape`]. If `None`, the tensor is scalar-shaped.
    /// * `data_type` - [`DataType`] of the constant tensor.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object.
    pub fn constant_with_scalar(
        &self,
        scalar: f64,
        shape: Option<&Shape>,
        data_type: DataType,
    ) -> Retained<Tensor> {
        match shape {
            Some(shape) => unsafe {
                msg_send![
                    self,
                    constantWithScalar: scalar,
                    shape: shape,
                    dataType: data_type
                ]
            },
            None => unsafe {
                msg_send![
                    self,
                    constantWithScalar: scalar,
                    dataType: data_type
                ]
            },
        }
    }

    /// Creates a complex constant operation and returns the result tensor.
    ///
    /// # Arguments
    ///
    /// * `real_part` - Real component of the complex scalar.
    /// * `imaginary_part` - Imaginary component of the complex scalar.
    /// * `data_type` - Optional [`DataType`] for the constant tensor. Defaults to [`DataType::ComplexFloat32`].
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object.
    pub fn constant_with_real_imaginary(
        &self,
        real_part: f64,
        imaginary_part: f64,
        data_type: Option<DataType>,
    ) -> Retained<Tensor> {
        match data_type {
            Some(data_type) => unsafe {
                msg_send![
                    self,
                    constantWithRealPart: real_part,
                    imaginaryPart: imaginary_part,
                    dataType: data_type
                ]
            },
            None => unsafe {
                msg_send![
                    self,
                    constantWithRealPart: real_part,
                    imaginaryPart: imaginary_part
                ]
            },
        }
    }

    /// Creates a variable operation and returns the result tensor.
    ///
    /// # Arguments
    ///
    /// * `data` - Raw tensor bytes.
    /// * `shape` - Statically shaped output [`Shape`].
    /// * `data_type` - [`DataType`] of the variable tensor.
    /// * `name` - Name of the operation.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object.
    pub fn variable_with_ns_data(
        &self,
        data: &NSData,
        shape: &Shape,
        data_type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                variableWithData: data,
                shape: shape,
                dataType: data_type,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Creates a variable operation and returns the result tensor.
    ///
    /// # Arguments
    ///
    /// * `data` - Slice containing tensor elements.
    /// * `shape` - Statically shaped output [`Shape`].
    /// * `data_type` - [`DataType`] of the variable tensor.
    /// * `name` - Name of the operation.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object.
    pub fn variable_with_data<T: Copy>(
        &self,
        data: &[T],
        shape: &Shape,
        data_type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        let data_size = size_of_val(data);
        let ns_data =
            unsafe { NSData::with_bytes(from_raw_parts(data.as_ptr() as *const u8, data_size)) };
        Self::variable_with_ns_data(self, &ns_data, shape, data_type, name)
    }

    /// Creates a variable from an input tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Source [`Tensor`].
    /// * `name` - Name of the operation.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object.
    pub fn variable_from_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                variableFromTensorWithTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Creates a read op which reads at this point of execution of the graph and returns the result tensor.
    ///
    /// # Arguments
    ///
    /// * `variable` - Variable [`Tensor`] to read.
    /// * `name` - Name of the operation.
    ///
    /// # Returns
    ///
    /// A valid [`Tensor`] object.
    pub fn read_variable(&self, variable: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                readVariable: variable,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Creates an assign operation which writes at this point of execution of the graph.
    ///
    /// # Arguments
    ///
    /// * `variable` - Variable [`Tensor`] to assign to.
    /// * `tensor` - Value [`Tensor`] that will be written into the variable.
    /// * `name` - Name of the operation.
    ///
    /// # Returns
    ///
    /// A valid [`Operation`] representing the assignment.
    pub fn assign_variable(
        &self,
        variable: &Tensor,
        tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Operation> {
        unsafe {
            msg_send![
                self,
                assignVariable: variable,
                withValueOfTensor: tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }
}
