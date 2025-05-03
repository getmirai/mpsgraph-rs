use crate::tensor::Tensor;
use mpsgraph::{DataType, Graph, Shape};
use objc2_foundation::NSNumber;

#[test]
fn test_tensor_creation() {
    let graph = Graph::new();

    // Create a shape like in the example file
    let numbers = [NSNumber::new_usize(2), NSNumber::new_usize(3)];
    let number_refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();
    let shape = Shape::from_slice(&number_refs);

    // Test creating a placeholder tensor
    let tensor = Tensor::placeholder(&graph, DataType::Float32, &shape);
    assert_eq!(tensor.data_type(), DataType::Float32);

    // Test creating constant tensors
    let const_tensor = Tensor::constant(&graph, 5.0, DataType::Float32);
    assert_eq!(const_tensor.data_type(), DataType::Float32);

    // Test zeros and ones
    let zeros = Tensor::zeros(&graph, DataType::Float32, &shape);
    let ones = Tensor::ones(&graph, DataType::Float32, &shape);
    assert_eq!(zeros.data_type(), DataType::Float32);
    assert_eq!(ones.data_type(), DataType::Float32);
}
