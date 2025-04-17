use crate::sparse_ops::{CreateSparseOpDescriptor, SparseStorageType};
use crate::tensor::DataType;
use crate::graph::Graph;
use crate::shape::{Shape, ShapeHelper};
use crate::tensor::Tensor;

#[test]
fn test_create_sparse_op_descriptor() {
    // Create a descriptor
    let _descriptor = CreateSparseOpDescriptor::new(SparseStorageType::CSC, DataType::Float32);
    
    // If we got here without crashing, the test passes
    assert!(true);
}

#[test]
fn test_create_sparse_op_descriptor_with_settings() {
    // Create a descriptor with CSR storage type and Float32 data type
    let descriptor = CreateSparseOpDescriptor::new(SparseStorageType::CSR, DataType::Float32);
    
    // Change the storage type to CSC
    descriptor.set_sparse_storage_type(SparseStorageType::CSC);
    
    // Change the data type to Float16
    descriptor.set_data_type(DataType::Float16);
    
    // If we got here without crashing, the test passes
    assert!(true);
}

// This test only compiles the API but doesn't run actual computations
// since it would require real data for sparse tensors
#[test]
fn test_sparse_api_compiles() {
    let graph = Graph::new();
    let shape = ShapeHelper::from_dimensions(&[2, 3, 4]);
    
    // Just test that the methods exist and the types are correct
    // We're not running actual GPU operations
    
    let _ = |tensors: &[&Tensor], data_type: DataType| {
        graph.sparse_tensor_with_type(tensors, &shape, data_type, Some("test"))
    };
    
    let _ = |descriptor: &CreateSparseOpDescriptor, tensors: &[&Tensor]| {
        graph.sparse_tensor_with_descriptor(descriptor, tensors, &shape, Some("test"))
    };
    
    let _ = |indices: &[&Tensor], values: &Tensor, descriptor: &CreateSparseOpDescriptor| {
        graph.sparse_tensor_with_indices_and_values(indices, values, &shape, descriptor, Some("test"))
    };
    
    let _ = |indices: &[&Tensor], values: &Tensor, dense_shape: &Tensor| {
        graph.sparse_to_dense(indices, values, dense_shape, Some("test"))
    };
}