use objc2::rc::Retained;
use objc2::{extern_class, msg_send};
use objc2::runtime::NSObject;
use objc2_foundation::{NSObjectProtocol, NSString};
use std::hash::{Hash, Hasher};

use crate::shape::{Shape, ShapeExtensions};

/// Data type for Metal Performance Shaders Graph tensors
/// 
/// These values match the MPSDataType definitions in the Metal Performance Shaders framework.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u32)]
pub enum DataType {
    Invalid = 0,
    
    // Floating point types
    Float32 = 0x10000000 | 32,
    Float16 = 0x10000000 | 16,
    Float64 = 0x10000000 | 64,
    
    // Signed integer types
    Int8 = 0x20000000 | 8,
    Int16 = 0x20000000 | 16,
    Int32 = 0x20000000 | 32,
    Int64 = 0x20000000 | 64,
    
    // Unsigned integer types
    Uint8 = 8,
    Uint16 = 16,
    Uint32 = 32,
    Uint64 = 64,
    
    // Boolean type
    Bool = 0x40000000 | 8,
    
    // Complex types
    Complex32 = 0x10000000 | 0x80000000 | 32,
    Complex64 = 0x10000000 | 0x80000000 | 64,
    
    // BFloat16 type
    BFloat16 = 0x80000000 | 0x10000000 | 16,
}

impl DataType {
    /// Convert from u64 to DataType
    pub fn from(value: u64) -> Self {
        match value as u32 {
            val if val == DataType::Float32 as u32 => DataType::Float32,
            val if val == DataType::Float16 as u32 => DataType::Float16,
            val if val == DataType::Float64 as u32 => DataType::Float64,
            val if val == DataType::Int8 as u32 => DataType::Int8,
            val if val == DataType::Int16 as u32 => DataType::Int16,
            val if val == DataType::Int32 as u32 => DataType::Int32,
            val if val == DataType::Int64 as u32 => DataType::Int64,
            val if val == DataType::Uint8 as u32 => DataType::Uint8,
            val if val == DataType::Uint16 as u32 => DataType::Uint16,
            val if val == DataType::Uint32 as u32 => DataType::Uint32,
            val if val == DataType::Uint64 as u32 => DataType::Uint64,
            val if val == DataType::Bool as u32 => DataType::Bool,
            val if val == DataType::Complex32 as u32 => DataType::Complex32,
            val if val == DataType::Complex64 as u32 => DataType::Complex64,
            val if val == DataType::BFloat16 as u32 => DataType::BFloat16,
            _ => DataType::Invalid,
        }
    }
}

extern_class!(
    #[derive(Debug, PartialEq, Eq)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraphTensor"]
    pub struct Tensor;
);

unsafe impl NSObjectProtocol for Tensor {}

impl Tensor {
    /// Returns the data type of this tensor
    pub fn data_type(&self) -> DataType {
        unsafe {
            let data_type_val: u32 = msg_send![self, dataType];
            std::mem::transmute(data_type_val)
        }
    }

    /// Returns the shape of this tensor
    pub fn shape(&self) -> Retained<Shape> {
        unsafe {
            let shape: Retained<Shape> = msg_send![self, shape];
            shape
        }
    }

    /// Returns the dimensions of the tensor
    pub fn dimensions(&self) -> Vec<i64> {
        self.shape().dimensions()
    }

    /// Returns the rank (number of dimensions) of this tensor
    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    /// Returns the name of this tensor
    pub fn name(&self) -> Option<String> {
        unsafe {
            let name: Option<Retained<NSString>> = msg_send![self, name];
            name.map(|s| s.to_string())
        }
    }
}

impl Hash for Tensor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self as *const Self as usize).hash(state);
    }
}