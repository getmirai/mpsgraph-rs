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

#[test]
fn test_tensor_arithmetic_operators() {
    let graph = Graph::new();

    // Create shape
    let numbers = [NSNumber::new_usize(2), NSNumber::new_usize(3)];
    let number_refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();
    let shape = Shape::from_slice(&number_refs);

    let a = Tensor::placeholder(&graph, DataType::Float32, &shape);
    let b = Tensor::placeholder(&graph, DataType::Float32, &shape);

    // Test addition with references
    let sum_ref = &a + &b;
    assert_eq!(sum_ref.data_type(), DataType::Float32);

    // Test addition with values
    let sum_val = a.clone() + b.clone();
    assert_eq!(sum_val.data_type(), DataType::Float32);

    // Test mixed addition
    let sum_mixed1 = a.clone() + &b;
    let sum_mixed2 = &a + b.clone();
    assert_eq!(sum_mixed1.data_type(), DataType::Float32);
    assert_eq!(sum_mixed2.data_type(), DataType::Float32);

    // Recreate tensors for subsequent tests
    let a = Tensor::placeholder(&graph, DataType::Float32, &shape);
    let b = Tensor::placeholder(&graph, DataType::Float32, &shape);

    // Test subtraction
    let diff = &a - &b;
    assert_eq!(diff.data_type(), DataType::Float32);

    // Test multiplication
    let product = &a * &b;
    assert_eq!(product.data_type(), DataType::Float32);

    // Test division
    let quotient = &a / &b;
    assert_eq!(quotient.data_type(), DataType::Float32);

    // Test negation
    let negated = -&a;
    assert_eq!(negated.data_type(), DataType::Float32);

    // Test chained operations
    let chain = (&a + &b) * (&a - &b);
    assert_eq!(chain.data_type(), DataType::Float32);
}

#[test]
fn test_tensor_scalar_operations() {
    let graph = Graph::new();

    // Create shape
    let numbers = [NSNumber::new_usize(2), NSNumber::new_usize(3)];
    let number_refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();
    let shape = Shape::from_slice(&number_refs);

    let a = Tensor::placeholder(&graph, DataType::Float32, &shape);

    // Test scalar addition
    let scalar_sum = &a + 2.0;
    assert_eq!(scalar_sum.data_type(), DataType::Float32);

    let scalar_sum_commutative = 2.0 + &a;
    assert_eq!(scalar_sum_commutative.data_type(), DataType::Float32);

    // Test scalar subtraction
    let scalar_sub = &a - 2.0;
    assert_eq!(scalar_sub.data_type(), DataType::Float32);

    let scalar_sub_reversed = 2.0 - &a;
    assert_eq!(scalar_sub_reversed.data_type(), DataType::Float32);

    // Test scalar multiplication
    let scalar_mul = &a * 2.0;
    assert_eq!(scalar_mul.data_type(), DataType::Float32);

    let scalar_mul_commutative = 2.0 * &a;
    assert_eq!(scalar_mul_commutative.data_type(), DataType::Float32);

    // Test scalar division
    let scalar_div = &a / 2.0;
    assert_eq!(scalar_div.data_type(), DataType::Float32);

    let scalar_div_reversed = 2.0 / &a;
    assert_eq!(scalar_div_reversed.data_type(), DataType::Float32);

    // Test with zero/one special cases
    let add_zero = &a + 0.0;
    let mul_one = &a * 1.0;
    assert_eq!(add_zero.data_type(), DataType::Float32);
    assert_eq!(mul_one.data_type(), DataType::Float32);
}

#[test]
fn test_activation_functions() {
    let graph = Graph::new();

    // Create shape
    let numbers = [NSNumber::new_usize(2), NSNumber::new_usize(3)];
    let number_refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();
    let shape = Shape::from_slice(&number_refs);

    let a = Tensor::placeholder(&graph, DataType::Float32, &shape);

    // Test sigmoid
    let sigmoid_a = a.sigmoid();
    assert_eq!(sigmoid_a.data_type(), DataType::Float32);

    // Test tanh
    let tanh_a = a.tanh();
    assert_eq!(tanh_a.data_type(), DataType::Float32);

    // Test relu
    let relu_a = a.relu();
    assert_eq!(relu_a.data_type(), DataType::Float32);

    // Test silu
    let silu_a = a.silu();
    assert_eq!(silu_a.data_type(), DataType::Float32);

    // Test gelu
    let gelu_a = a.gelu();
    assert_eq!(gelu_a.data_type(), DataType::Float32);
}

#[test]
fn test_other_operations() {
    let graph = Graph::new();

    // Create shape
    let numbers = [NSNumber::new_usize(2), NSNumber::new_usize(3)];
    let number_refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();
    let shape = Shape::from_slice(&number_refs);

    let a = Tensor::placeholder(&graph, DataType::Float32, &shape);
    let b = Tensor::placeholder(&graph, DataType::Float32, &shape);

    // Test square
    let squared = a.square();
    assert_eq!(squared.data_type(), DataType::Float32);

    // Test minimum/maximum
    let min_tensor = a.minimum(&b);
    let max_tensor = a.maximum(&b);
    assert_eq!(min_tensor.data_type(), DataType::Float32);
    assert_eq!(max_tensor.data_type(), DataType::Float32);

    // Test clamp
    let clamped = a.clamp(0.0, 1.0);
    assert_eq!(clamped.data_type(), DataType::Float32);
}

#[test]
fn test_complex_expressions() {
    let graph = Graph::new();

    // Create shape
    let numbers = [NSNumber::new_usize(2), NSNumber::new_usize(3)];
    let number_refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();
    let shape = Shape::from_slice(&number_refs);

    let a = Tensor::placeholder(&graph, DataType::Float32, &shape);
    let b = Tensor::placeholder(&graph, DataType::Float32, &shape);

    // Test complex expression with mixed operations
    let expr1 = (&a + &b).square() / (2.0 - &a);
    assert_eq!(expr1.data_type(), DataType::Float32);

    // Test expression with activation functions
    let expr2 = a.sigmoid() * b.tanh();
    assert_eq!(expr2.data_type(), DataType::Float32);

    // Test expression with scalar operations
    let expr3 = (&a * 2.0 + 3.0) / &b;
    assert_eq!(expr3.data_type(), DataType::Float32);
}

#[test]
fn test_tensor_inner_access() {
    let graph = Graph::new();

    // Create shape
    let numbers = [NSNumber::new_usize(2), NSNumber::new_usize(3)];
    let number_refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();
    let shape = Shape::from_slice(&number_refs);

    let a = Tensor::placeholder(&graph, DataType::Float32, &shape);

    // Test access to inner MPSTensor
    let inner = a.inner();
    assert_eq!(inner.data_type(), DataType::Float32);

    // Test access to operation and graph
    let op = a.operation();
    let graph_from_tensor = a.graph();

    // Just verify these don't panic
    let _ = op; // Verify operation can be retrieved
    let _ = graph_from_tensor; // Verify graph can be retrieved
}
