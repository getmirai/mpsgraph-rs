use crate::core::DataType;
use crate::graph::Graph;
use crate::operation::Operation;
use crate::shape::Shape;
use crate::tensor::Tensor;
use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::{NSData, NSString};
use std::ptr;

impl Graph {
    pub fn complex_constant(&self, real_part: f64, imaginary_part: f64) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                constantWithRealPart: real_part,
                imaginaryPart: imaginary_part
            ]
        }
    }

    pub fn complex_constant_with_type(
        &self,
        real_part: f64,
        imaginary_part: f64,
        data_type: DataType,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                constantWithRealPart: real_part,
                imaginaryPart: imaginary_part,
                dataType: data_type as u64
            ]
        }
    }

    pub fn complex_constant_with_shape(
        &self,
        real_part: f64,
        imaginary_part: f64,
        shape: &Shape,
        data_type: DataType,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                constantWithRealPart: real_part,
                imaginaryPart: imaginary_part,
                shape: shape.as_ptr(),
                dataType: data_type as u64
            ]
        }
    }

    pub fn variable_with_bytes(
        &self,
        data: &[u8],
        shape: &Shape,
        data_type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(ptr::null(), |s| s as *const _);
            let ns_data = NSData::with_bytes(data);
            msg_send![
                self,
                variableWithData: &*ns_data,
                shape: shape.as_ptr(),
                dataType: data_type as u64,
                name: name_ptr
            ]
        }
    }

    pub fn variable_from_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(ptr::null(), |s| s as *const _);
            msg_send![
                self,
                variableFromTensorWithTensor: tensor,
                name: name_ptr
            ]
        }
    }

    pub fn read_variable(&self, variable: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(ptr::null(), |s| s as *const _);
            msg_send![
                self,
                readVariable: variable,
                name: name_ptr
            ]
        }
    }

    pub fn assign_variable(
        &self,
        variable: &Tensor,
        tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Operation> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(ptr::null(), |s| s as *const _);
            msg_send![
                self,
                assignVariable: variable,
                withValueOfTensor: tensor,
                name: name_ptr
            ]
        }
    }

    pub fn placeholder_tensor(
        &self,
        shape: &Shape,
        data_type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(ptr::null(), |s| s as *const _);
            msg_send![
                self,
                placeholderTensorWithShape: shape.as_ptr(),
                dataType: data_type as u64,
                name: name_ptr
            ]
        }
    }

    pub fn storage_tensor(
        &self,
        handle: u64,
        shape: &Shape,
        data_type: DataType,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                storageTensorWithHandle: handle,
                shape: shape.as_ptr(),
                dataType: data_type as u64
            ]
        }
    }
}
