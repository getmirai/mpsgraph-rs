use metal::foreign_types::ForeignType;
use metal::{Buffer, Device as MetalDevice};
use objc2::ffi::class_getInstanceMethod;
use objc2::rc::{Allocated, Retained};
use objc2::runtime::NSObject;
use objc2::{extern_class, msg_send, sel, ClassType};
use objc2_foundation::{NSArray, NSData, NSNumber, NSObjectProtocol};

use crate::device::Device;
use crate::shape::Shape;
use crate::tensor::DataType;

extern_class!(
    #[derive(Debug, PartialEq, Eq)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraphTensorData"]
    pub struct TensorData;
);

extern_class!(
    #[derive(Debug, PartialEq, Eq)]
    #[unsafe(super = NSObject)]
    #[name = "MPSMatrix"]
    pub struct Matrix;
);

extern_class!(
    #[derive(Debug, PartialEq, Eq)]
    #[unsafe(super = NSObject)]
    #[name = "MPSVector"]
    pub struct Vector;
);

extern_class!(
    #[derive(Debug, PartialEq, Eq)]
    #[unsafe(super = NSObject)]
    #[name = "MPSNDArray"]
    pub struct NDArray;
);

extern_class!(
    #[derive(Debug, PartialEq, Eq)]
    #[unsafe(super = NSObject)]
    #[name = "MPSImageBatch"]
    pub struct ImageBatch;
);

unsafe impl NSObjectProtocol for TensorData {}
unsafe impl NSObjectProtocol for Matrix {}
unsafe impl NSObjectProtocol for Vector {}
unsafe impl NSObjectProtocol for NDArray {}
unsafe impl NSObjectProtocol for ImageBatch {}

impl TensorData {
    /// Creates a new TensorData from a slice of data and a shape dimensions
    pub fn from_bytes<T: Copy>(
        data: &[T],
        shape_dimensions: &[i64],
        data_type: DataType,
    ) -> Retained<Self> {
        let shape = Shape::from_dimensions(shape_dimensions);
        unsafe {
            let data_size = std::mem::size_of_val(data);
            let device = MetalDevice::system_default().expect("No Metal device found");
            let ns_data = NSData::with_bytes(std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data_size,
            ));
            let mps_device = Device::with_device(&device);
            let class = Self::class();
            let data_type_val = data_type as u32;

            let allocated: Allocated<Self> = msg_send![class, alloc];
            let initialized: Retained<Self> = msg_send![allocated,
                initWithDevice:&*mps_device,
                data:&*ns_data,
                shape:shape.as_ptr(),
                dataType:data_type_val
            ];
            initialized
        }
    }

    /// Creates a new TensorData with bytes, shape, and data type
    pub fn with_bytes<T: Copy>(
        data: &[T],
        shape: &Shape,
        data_type: DataType,
    ) -> Option<Retained<Self>> {
        unsafe {
            let data_size = std::mem::size_of_val(data);
            let device = MetalDevice::system_default().expect("No Metal device found");
            let ns_data = NSData::with_bytes(std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data_size,
            ));
            let mps_device = Device::with_device(&device);
            let class = Self::class();
            let data_type_val = data_type as u32;

            let allocated: Allocated<Self> = msg_send![class, alloc];
            let initialized: Option<Retained<Self>> = msg_send![allocated,
                initWithDevice:&*mps_device,
                data:&*ns_data,
                shape:shape.as_ptr(),
                dataType:data_type_val
            ];
            initialized
        }
    }

    /// Creates a new TensorData from a Metal buffer
    pub fn from_buffer(buffer: &Buffer, shape: &Shape, data_type: DataType) -> Retained<Self> {
        unsafe {
            let class = Self::class();
            let buffer_ptr = buffer.as_ptr() as *mut objc2::runtime::AnyObject;

            let allocated: Allocated<Self> = msg_send![class, alloc];
            let initialized: Retained<Self> = msg_send![allocated,
                initWithMTLBuffer: buffer_ptr,
                shape: shape.as_ptr(),
                dataType: data_type as u32
            ];
            initialized
        }
    }

    /// Creates a new TensorData from a Metal buffer specifying rowBytes (stride between rows)
    pub fn from_buffer_row_bytes(
        buffer: &Buffer,
        shape: &Shape,
        data_type: DataType,
        row_bytes: u64,
    ) -> Retained<Self> {
        unsafe {
            let class = Self::class();
            let buffer_ptr = buffer.as_ptr() as *mut objc2::runtime::AnyObject;

            let allocated: Allocated<Self> = msg_send![class, alloc];
            let initialized: Retained<Self> = msg_send![allocated,
                initWithMTLBuffer: buffer_ptr,
                shape: shape.as_ptr(),
                dataType: data_type as u32,
                rowBytes: row_bytes
            ];
            initialized
        }
    }

    /// Returns the shape of this tensor data
    pub fn shape(&self) -> Shape {
        unsafe {
            let array: Retained<NSArray<NSNumber>> = msg_send![self, shape];
            Shape::new(&array)
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

            let method_exists = class_getInstanceMethod(Self::class(), synchronize_selector)
                != std::ptr::null_mut();
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

            let method_exists = class_getInstanceMethod(Self::class(), synchronize_selector)
                != std::ptr::null_mut();
            if method_exists {
                let _: () = msg_send![self, synchronizeOnCPU];
                true
            } else {
                false // Method not available
            }
        }
    }

    // -------- Additional initializers matching MPSGraphTensorData.h -----------------------------

    pub fn from_matrix(matrix: &Matrix) -> Retained<Self> {
        unsafe {
            let class = Self::class();
            let allocated: Allocated<Self> = msg_send![class, alloc];
            let initialized: Retained<Self> = msg_send![allocated, initWithMPSMatrix: matrix];
            initialized
        }
    }

    pub fn from_matrix_rank(matrix: &Matrix, rank: usize) -> Retained<Self> {
        unsafe {
            let class = Self::class();
            let allocated: Allocated<Self> = msg_send![class, alloc];
            let initialized: Retained<Self> =
                msg_send![allocated, initWithMPSMatrix: matrix, rank: rank];
            initialized
        }
    }

    pub fn from_vector(vector: &Vector) -> Retained<Self> {
        unsafe {
            let class = Self::class();
            let allocated: Allocated<Self> = msg_send![class, alloc];
            let initialized: Retained<Self> = msg_send![allocated, initWithMPSVector: vector];
            initialized
        }
    }

    pub fn from_vector_rank(vector: &Vector, rank: usize) -> Retained<Self> {
        unsafe {
            let class = Self::class();
            let allocated: Allocated<Self> = msg_send![class, alloc];
            let initialized: Retained<Self> =
                msg_send![allocated, initWithMPSVector: vector, rank: rank];
            initialized
        }
    }

    pub fn from_ndarray(ndarray: &NDArray) -> Retained<Self> {
        unsafe {
            let class = Self::class();
            let allocated: Allocated<Self> = msg_send![class, alloc];
            let initialized: Retained<Self> = msg_send![allocated, initWithMPSNDArray: ndarray];
            initialized
        }
    }

    pub fn from_image_batch(batch: &ImageBatch) -> Retained<Self> {
        unsafe {
            let class = Self::class();
            let allocated: Allocated<Self> = msg_send![class, alloc];
            let initialized: Retained<Self> = msg_send![allocated, initWithMPSImageBatch: batch];
            initialized
        }
    }

    /// Returns an `MPSNDArray` representing the tensor data, copying if necessary.
    pub fn to_ndarray(&self) -> Option<Retained<NDArray>> {
        unsafe { msg_send![self, mpsndarray] }
    }
}
