use crate::tensor_ops::{abs, exp, log, relu, sigmoid, silu, sqrt, square, tanh};
use crate::tensor_ops::{GraphExtensions, GraphTensor, TensorOps};
use mpsgraph::{DataType, Graph, Shape};
use objc2::rc::Retained;
use objc2_foundation::NSNumber;

// Helper function to create Shape from usize slice
fn create_shape_from_usize(dimensions: &[usize]) -> Retained<Shape> {
    let numbers: Vec<Retained<NSNumber>> =
        dimensions.iter().map(|&d| NSNumber::new_usize(d)).collect();

    let refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();
    Shape::from_slice(&refs)
}

#[test]
fn test_placeholder_tensor_creation() {
    let graph = Graph::new();
    let shape = create_shape_from_usize(&[2usize, 3usize]);

    // Test creating a placeholder tensor
    let tensor = graph.placeholder_tensor(&shape, DataType::Float32, None);
    assert_eq!(tensor.tensor().data_type(), DataType::Float32);
}
