use crate::{DataType, Device, GraphObject, Shape};
use metal::foreign_types::ForeignType;
use metal::Buffer;
use objc2::rc::{Allocated, Retained};
use objc2::runtime::{AnyObject, NSObject};
use objc2::{extern_class, extern_conformance, extern_methods, msg_send, ClassType};
use objc2_foundation::{NSData, NSObjectProtocol};
use std::mem::size_of_val;
use std::slice::from_raw_parts;

extern_class!(
    /// The representation of a compute data type.
    ///
    /// Pass data to a graph using a tensor data, a reference will be taken to your data and used just in time when the graph is run.
    ///
    /// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphtensordata?language=objc)
    #[unsafe(super(GraphObject, NSObject))]
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[name = "MPSGraphTensorData"]
    pub struct TensorData;
);

extern_conformance!(
    unsafe impl NSObjectProtocol for TensorData {}
);

impl TensorData {
    extern_methods!(
        /// The shape of the tensor data.
        #[unsafe(method(shape))]
        #[unsafe(method_family = none)]
        pub unsafe fn shape(&self) -> Retained<Shape>;

        /// The data type of the tensor data.
        #[unsafe(method(dataType))]
        #[unsafe(method_family = none)]
        pub unsafe fn data_type(&self) -> DataType;

        /// The device of the tensor data.
        #[unsafe(method(device))]
        #[unsafe(method_family = none)]
        pub unsafe fn device(&self) -> Retained<Device>;

        /// Initializes the tensor data with an `NSData` on a device.
        ///
        /// - Parameters:
        /// - device: MPSDevice on which the MPSGraphTensorData exists
        /// - data: NSData from which to copy the contents
        /// - shape: shape of the output tensor
        /// - dataType: dataType of the placeholder tensor
        /// - Returns: A valid MPSGraphTensorData, or nil if allocation failure.
        #[unsafe(method(initWithDevice:data:shape:dataType:))]
        #[unsafe(method_family = init)]
        pub unsafe fn init_with_device_data_shape_data_type(
            this: Allocated<Self>,
            device: &Device,
            data: &NSData,
            shape: &Shape,
            data_type: DataType,
        ) -> Retained<Self>;
    );
}

impl TensorData {
    /// Initializes the tensor data with an `NSData` on a device.
    ///
    /// - Parameters:
    /// - device: Device on which the TensorData exists
    /// - data: NSData from which to copy the contents
    /// - shape: shape of the output tensor
    /// - dataType: dataType of the placeholder tensor
    /// - Returns: A valid TensorData, or nil if allocation failure.
    pub fn new_with_device_ns_data_shape_data_type(
        device: &Device,
        data: &NSData,
        shape: &Shape,
        data_type: DataType,
    ) -> Retained<Self> {
        unsafe {
            let class = Self::class();
            let allocated: Allocated<Self> = msg_send![class, alloc];
            Self::init_with_device_data_shape_data_type(allocated, device, data, shape, data_type)
        }
    }

    /// Initializes the tensor data with a slice on a device.
    ///
    /// - Parameters:
    /// - device: Device on which the TensorData exists
    /// - data: Slice from which to copy the contents
    /// - shape: shape of the output tensor
    /// - dataType: dataType of the placeholder tensor
    /// - Returns: A valid TensorData, or nil if allocation failure.
    pub fn new_with_device_data_shape_data_type<T: Copy>(
        device: &Device,
        data: &[T],
        shape: &Shape,
        data_type: DataType,
    ) -> Retained<Self> {
        unsafe {
            let data_size = size_of_val(data);
            let ns_data = NSData::with_bytes(from_raw_parts(data.as_ptr() as *const u8, data_size));
            Self::new_with_device_ns_data_shape_data_type(device, &ns_data, shape, data_type)
        }
    }

    /// Initializes the tensor data with a Metal buffer.
    ///
    /// - Parameters:
    /// - buffer: Metal buffer from which to copy the contents
    /// - shape: shape of the output tensor
    /// - dataType: dataType of the placeholder tensor
    /// - Returns: A valid TensorData, or nil if allocation failure.
    pub fn new_with_mtl_buffer_shape_data_type(
        buffer: &Buffer,
        shape: &Shape,
        data_type: DataType,
    ) -> Retained<Self> {
        unsafe {
            let class = Self::class();
            let buffer_ptr = buffer.as_ptr() as *mut AnyObject;
            let allocated: Allocated<Self> = msg_send![class, alloc];
            msg_send![
                allocated,
                initWithMTLBuffer: buffer_ptr,
                shape: shape,
                dataType: data_type as u32
            ]
        }
    }

    /// Initializes the tensor data with a Metal buffer specifying rowBytes (stride between rows)
    ///
    /// - Parameters:
    /// - buffer: Metal buffer from which to copy the contents
    /// - shape: shape of the output tensor
    /// - dataType: dataType of the placeholder tensor
    /// - rowBytes: rowBytes for the fastest moving dimension, must be larger than or equal to sizeOf(dataType)shape[rank - 1] and must be a multiple of sizeOf(dataType)
    /// - Returns: A valid TensorData, or nil if allocation failure.
    pub fn new_with_mtl_buffer_shape_data_type_row_bytes(
        buffer: &Buffer,
        shape: &Shape,
        data_type: DataType,
        row_bytes: u64,
    ) -> Retained<Self> {
        unsafe {
            let class = Self::class();
            let buffer_ptr = buffer.as_ptr() as *mut AnyObject;
            let allocated: Allocated<Self> = msg_send![class, alloc];
            msg_send![
                allocated,
                initWithMTLBuffer: buffer_ptr,
                shape: shape,
                dataType: data_type as u32,
                rowBytes: row_bytes
            ]
        }
    }
}
