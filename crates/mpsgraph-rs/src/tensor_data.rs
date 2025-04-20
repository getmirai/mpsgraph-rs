use objc2::rc::Retained;
use objc2::{extern_class, ClassType, msg_send, sel};
use objc2::runtime::NSObject;
use objc2::ffi::class_getInstanceMethod;
use objc2_foundation::{NSArray, NSData, NSNumber, NSObjectProtocol};
use metal::{Buffer, Device as MetalDevice};
use metal::foreign_types::ForeignType;

use crate::device::Device;
use crate::shape::Shape;
use crate::tensor::DataType;

extern_class!(
    #[derive(Debug, PartialEq, Eq)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraphTensorData"]
    pub struct TensorData;
);

unsafe impl NSObjectProtocol for TensorData {}

impl TensorData {
    /// Creates a new TensorData from a slice of data and a shape dimensions
    pub fn from_bytes<T: Copy>(data: &[T], shape_dimensions: &[i64], data_type: DataType) -> Retained<Self> {
        // Create a Shape from dimensions
        let shape = Shape::from_dimensions(shape_dimensions);
        
        unsafe {
            // Calculate the total data size
            let data_size = std::mem::size_of_val(data);

            // Get the default Metal device
            let device = MetalDevice::system_default().expect("No Metal device found");
            
            // Create NSData with our data
            let ns_data = NSData::with_bytes(std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data_size,
            ));
            
            // Create MPSGraphDevice from MTLDevice
            let mps_device = Device::with_device(&device);
            
            // Create the TensorData with NSData
            let class = Self::class();
            let data_type_val = data_type as u32;
            
            let alloc: *mut Self = msg_send![class, alloc];
            let obj: *mut Self = msg_send![alloc, 
                initWithDevice:&*mps_device, 
                data:&*ns_data, 
                shape:shape.as_ptr(), 
                dataType:data_type_val
            ];
            let tensor_data = Retained::from_raw(obj).unwrap();
            
            tensor_data
        }
    }
    
    /// Creates a new TensorData with bytes, shape, and data type
    pub fn with_bytes<T: Copy>(
        data: &[T],
        shape: &Shape,
        data_type: DataType,
    ) -> Option<Retained<Self>> {
        unsafe {
            // Calculate the total data size
            let data_size = std::mem::size_of_val(data);

            // Get the default Metal device
            let device = MetalDevice::system_default().expect("No Metal device found");
            
            // Create NSData with our data
            let ns_data = NSData::with_bytes(std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data_size,
            ));
            
            // Create MPSGraphDevice from MTLDevice
            let mps_device = Device::with_device(&device);
            
            // Create the TensorData with NSData
            let class = Self::class();
            let data_type_val = data_type as u32;
            
            let alloc: *mut Self = msg_send![class, alloc];
            let obj: *mut Self = msg_send![alloc, 
                initWithDevice:&*mps_device, 
                data:&*ns_data, 
                shape:shape.as_ptr(), 
                dataType:data_type_val
            ];
            
            if obj.is_null() {
                None
            } else {
                Retained::from_raw(obj)
            }
        }
    }

    /// Creates a new TensorData from a Metal buffer
    pub fn from_buffer(buffer: &Buffer, shape_dimensions: &[i64], data_type: DataType) -> Retained<Self> {
        // Create a Shape from dimensions
        let shape = Shape::from_dimensions(shape_dimensions);
        
        unsafe {
            let class = Self::class();
            // Get the underlying MTLBuffer pointer as an Objective-C object
            let buffer_ptr = buffer.as_ptr() as *mut objc2::runtime::AnyObject;
            let data_type_val = data_type as u32;
            
            let alloc: *mut Self = msg_send![class, alloc];
            let obj: *mut Self = msg_send![alloc,
                initWithMTLBuffer:buffer_ptr,
                shape:shape.as_ptr(),
                dataType:data_type_val
            ];
            let tensor_data = Retained::from_raw(obj).unwrap();
            
            tensor_data
        }
    }

    /// Returns the shape of this tensor data
    pub fn shape(&self) -> Shape {
        unsafe {
            let array: Retained<NSArray<NSNumber>> = msg_send![self, shape];
            Shape::new(array)
        }
    }

    /// Returns the data type of this tensor data
    pub fn data_type(&self) -> DataType {
        unsafe {
            let data_type_val: u32 = msg_send![self, dataType];
            std::mem::transmute(data_type_val)
        }
    }
    
    /// Get the total number of bytes in the tensor data
    pub fn bytes_size(&self) -> usize {
        unsafe {
            let size: usize = msg_send![self, length];
            size
        }
    }
    
    /// Get the bytes as a slice of a specific type
    pub fn bytes_as<T: Copy>(&self) -> Option<Vec<T>> {
        unsafe {
            // Try to synchronize - this may not be available in all versions
            let synchronize_selector = sel!(synchronizeOnCPU);
            
            let method_exists = class_getInstanceMethod(Self::class(), synchronize_selector) != std::ptr::null_mut();
            if method_exists {
                let _: () = msg_send![self, synchronizeOnCPU];
            }
            
            // Get the data pointer
            let bytes_ptr: *const u8 = msg_send![self, bytes];
            if bytes_ptr.is_null() {
                return None;
            }
            
            // Get the size of the data
            let bytes_size: usize = msg_send![self, length];
            let element_size = std::mem::size_of::<T>();
            let count = bytes_size / element_size;
            
            if bytes_size % element_size != 0 {
                return None; // Size mismatch
            }
            
            // Convert to the desired type
            let typed_ptr = bytes_ptr as *const T;
            let slice = std::slice::from_raw_parts(typed_ptr, count);
            
            // Make a copy to ensure memory safety
            Some(slice.to_vec())
        }
    }

    /// Synchronize this tensor data to CPU
    ///
    /// This method ensures that any data on the GPU is synchronized to CPU-accessible memory.
    /// Use this method when you need to access the tensor data from the CPU after GPU operations.
    pub fn synchronize(&self) -> bool {
        unsafe {
            let synchronize_selector = sel!(synchronizeOnCPU);
            
            let method_exists = class_getInstanceMethod(Self::class(), synchronize_selector) != std::ptr::null_mut();
            if method_exists {
                let _: () = msg_send![self, synchronizeOnCPU];
                true
            } else {
                false // Method not available
            }
        }
    }
}

// Implement CustomDefault for TensorData
impl crate::CustomDefault for TensorData {
    fn custom_default() -> Retained<Self> {
        // Create a default tensor data with a scalar shape and zero value
        let scalar_shape = Shape::scalar();
        let zero: [f32; 1] = [0.0];
        Self::with_bytes(&zero, &scalar_shape, DataType::Float32).unwrap()
    }
}