use crate::core::DataType;
use crate::graph::Graph;
use crate::shape::Shape;
use crate::tensor::Tensor;
use objc2::extern_class;
use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::AnyClass;
use objc2_foundation::{NSArray, NSObject, NSObjectProtocol, NSString};

/// The sparse storage options for MPSGraph sparse operations.
#[repr(u64)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum SparseStorageType {
    /// COO (Coordinate) Storage format
    COO = 0,
    /// CSC (Compressed Sparse Column) Storage format
    CSC = 1,
    /// CSR (Compressed Sparse Row) Storage format
    CSR = 2,
}

extern_class!(
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraphCreateSparseOpDescriptor"]
    /// Descriptor for sparse tensor creation
    pub struct CreateSparseOpDescriptor;
);

unsafe impl NSObjectProtocol for CreateSparseOpDescriptor {}

impl CreateSparseOpDescriptor {
    /// Creates a new descriptor for a sparse tensor.
    ///
    /// # Arguments
    ///
    /// * `storage_type` - The storage format of the sparse tensor
    /// * `data_type` - The data type of the sparse tensor
    ///
    /// # Returns
    ///
    /// A new sparse tensor descriptor
    pub fn new(storage_type: SparseStorageType, data_type: DataType) -> Retained<Self> {
        unsafe {
            let cls = AnyClass::get(c"MPSGraphCreateSparseOpDescriptor").unwrap();
            msg_send![
                cls,
                descriptorWithStorageType: storage_type as u64,
                dataType: data_type as u32
            ]
        }
    }

    /// Sets the sparse storage type
    pub fn set_sparse_storage_type(&self, storage_type: SparseStorageType) -> &Self {
        unsafe {
            let _: () = msg_send![self, setSparseStorageType: storage_type as u64];
            self
        }
    }

    /// Sets the data type
    pub fn set_data_type(&self, data_type: DataType) -> &Self {
        unsafe {
            let _: () = msg_send![self, setDataType: data_type as u32];
            self
        }
    }
}

/// Sparse operations for Graph
impl Graph {
    /// Creates a sparse tensor with the specified type.
    ///
    /// # Arguments
    ///
    /// * `storage_type` - The storage format of the sparse tensor
    /// * `tensors` - Tensors representing the sparse tensor components (indices, values)
    /// * `shape` - The shape of the corresponding dense tensor
    /// * `data_type` - The data type of the sparse tensor
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new Tensor representing the sparse tensor
    pub fn sparse_tensor_with_type(
        &self,
        storage_type: SparseStorageType,
        tensors: &[&Tensor],
        shape: &Shape,
        data_type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };

            // Create NSArray of input tensors
            let tensor_refs: Vec<&Tensor> = tensors.iter().copied().collect();
            let tensors_array = NSArray::from_slice(&tensor_refs);

            msg_send![
                self,
                sparseTensorWithType: storage_type as u64,
                tensors: &*tensors_array,
                shape: shape.as_ptr(),
                dataType: data_type as u32,
                name: name_obj
            ]
        }
    }

    /// Creates a sparse tensor with the specified descriptor.
    ///
    /// # Arguments
    ///
    /// * `descriptor` - Descriptor specifying storage format and data type
    /// * `tensors` - Tensors representing the sparse tensor components (indices, values)
    /// * `shape` - The shape of the corresponding dense tensor
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new Tensor representing the sparse tensor
    pub fn sparse_tensor_with_descriptor(
        &self,
        descriptor: &CreateSparseOpDescriptor,
        tensors: &[&Tensor],
        shape: &Shape,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };

            // Create NSArray of input tensors
            let tensor_refs: Vec<&Tensor> = tensors.iter().copied().collect();
            let tensors_array = NSArray::from_slice(&tensor_refs);

            msg_send![
                self,
                sparseTensorWithDescriptor: descriptor,
                tensors: &*tensors_array,
                shape: shape.as_ptr(),
                name: name_obj
            ]
        }
    }

    /// Creates a sparse tensor with the specified indices and values.
    ///
    /// # Arguments
    ///
    /// * `indices` - Indices of non-zero values for each dimension
    /// * `values` - The non-zero values
    /// * `dense_shape` - The shape of the corresponding dense tensor
    /// * `descriptor` - Descriptor specifying storage format and data type
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new Tensor representing the sparse tensor
    pub fn sparse_tensor_with_indices_and_values(
        &self,
        indices: &[&Tensor],
        values: &Tensor,
        dense_shape: &Shape,
        descriptor: &CreateSparseOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };

            // Create NSArray of indices tensors
            let indices_refs: Vec<&Tensor> = indices.iter().copied().collect();
            let indices_array = NSArray::from_slice(&indices_refs);

            msg_send![
                self,
                sparseTensorWithIndicesTensors: &*indices_array,
                valuesTensor: values,
                denseShape: dense_shape,
                descriptor: descriptor,
                name: name_obj
            ]
        }
    }

    /// Creates a SparseToDense operation that converts a sparse tensor to a dense tensor.
    ///
    /// # Arguments
    ///
    /// * `indices` - A tensor containing the indices of nonzero elements
    /// * `values` - A tensor containing the values of nonzero elements
    /// * `dense_shape` - The desired shape of the dense tensor
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A dense tensor containing the values from the sparse format
    pub fn sparse_to_dense(
        &self,
        indices: &[&Tensor],
        values: &Tensor,
        dense_shape: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };

            // Create NSArray of indices tensors
            let indices_refs: Vec<&Tensor> = indices.iter().copied().collect();
            let indices_array = NSArray::from_slice(&indices_refs);

            msg_send![
                self,
                sparseToDenseWithIndicesTensors: &*indices_array,
                valuesTensor: values,
                denseShapeTensor: dense_shape,
                name: name_obj
            ]
        }
    }
}
