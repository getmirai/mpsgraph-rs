use crate::tensor_ops::{abs, exp, log, relu, sigmoid, silu, sqrt, square, tanh};
use crate::tensor_ops::{GraphExtensions, GraphTensor, TensorOps};
use mpsgraph::{DataType, Graph, Shape};
use objc2::rc::Retained;
use objc2_foundation::NSNumber;

// Helper function to create Shape from usize slice
fn create_shape_from_usize(dimensions: &[usize]) -> Retained<Shape> {
    let numbers: Vec<Retained<NSNumber>> = dimensions
        .iter()
        .map(|&d| NSNumber::new_usize(d))
        .collect();
        
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

#[test]
fn test_tensor_arithmetic_operators() {
    let graph = Graph::new();
    let shape = create_shape_from_usize(&[2usize, 3usize]);

    let a = graph.placeholder_tensor(&shape, DataType::Float32, Some("a"));
    let b = graph.placeholder_tensor(&shape, DataType::Float32, Some("b"));

    // Test addition
    let sum = &a + &b;
    assert_eq!(sum.tensor().data_type(), DataType::Float32);

    // Test subtraction
    let diff = &a - &b;
    assert_eq!(diff.tensor().data_type(), DataType::Float32);

    // Test multiplication
    let product = &a * &b;
    assert_eq!(product.tensor().data_type(), DataType::Float32);

    // Test division
    let quotient = &a / &b;
    assert_eq!(quotient.tensor().data_type(), DataType::Float32);

    // Test negation
    let negated = -&a;
    assert_eq!(negated.tensor().data_type(), DataType::Float32);

    // Test chained operations
    let chain = &(&a + &b) * &(&a - &b);
    assert_eq!(chain.tensor().data_type(), DataType::Float32);
}

#[test]
fn test_tensor_unary_operations() {
    let graph = Graph::new();
    let shape = create_shape_from_usize(&[2usize, 3usize]);
    let a = graph.placeholder_tensor(&shape, DataType::Float32, Some("a"));

    // Test square
    let squared = square(&a, None);
    assert_eq!(squared.tensor().data_type(), DataType::Float32);
    let squared_method = a.square(None);
    assert_eq!(squared_method.tensor().data_type(), DataType::Float32);

    // Test sqrt
    let sqrt_a = sqrt(&a, None);
    assert_eq!(sqrt_a.tensor().data_type(), DataType::Float32);
    let sqrt_method = a.sqrt(None);
    assert_eq!(sqrt_method.tensor().data_type(), DataType::Float32);

    // Test abs
    let abs_a = abs(&a, None);
    assert_eq!(abs_a.tensor().data_type(), DataType::Float32);
    let abs_method = a.abs(None);
    assert_eq!(abs_method.tensor().data_type(), DataType::Float32);

    // Test exp
    let exp_a = exp(&a, None);
    assert_eq!(exp_a.tensor().data_type(), DataType::Float32);
    let exp_method = a.exp(None);
    assert_eq!(exp_method.tensor().data_type(), DataType::Float32);

    // Test log
    let log_a = log(&a, None);
    assert_eq!(log_a.tensor().data_type(), DataType::Float32);
    let log_method = a.log(None);
    assert_eq!(log_method.tensor().data_type(), DataType::Float32);
}

#[test]
fn test_activation_functions() {
    let graph = Graph::new();
    let shape = create_shape_from_usize(&[2usize, 3usize]);
    let a = graph.placeholder_tensor(&shape, DataType::Float32, Some("a"));

    // Test sigmoid
    let sigmoid_a = sigmoid(&a, None);
    assert_eq!(sigmoid_a.tensor().data_type(), DataType::Float32);
    let sigmoid_method = a.sigmoid(None);
    assert_eq!(sigmoid_method.tensor().data_type(), DataType::Float32);

    // Test tanh
    let tanh_a = tanh(&a, None);
    assert_eq!(tanh_a.tensor().data_type(), DataType::Float32);
    let tanh_method = a.tanh(None);
    assert_eq!(tanh_method.tensor().data_type(), DataType::Float32);

    // Test relu
    let relu_a = relu(&a, None);
    assert_eq!(relu_a.tensor().data_type(), DataType::Float32);
    let relu_method = a.relu(None);
    assert_eq!(relu_method.tensor().data_type(), DataType::Float32);

    // Test silu
    let silu_a = silu(&a, None);
    assert_eq!(silu_a.tensor().data_type(), DataType::Float32);
}

#[test]
fn test_complex_expressions() {
    let graph = Graph::new();
    let shape = create_shape_from_usize(&[2usize, 3usize]);
    let a = graph.placeholder_tensor(&shape, DataType::Float32, Some("a"));
    let b = graph.placeholder_tensor(&shape, DataType::Float32, Some("b"));

    // Test nested function calls
    let nested = abs(&sqrt(&(&a + &b), None), None);
    assert_eq!(nested.tensor().data_type(), DataType::Float32);

    // Test mixed operation styles
    let square_a = a.square(None);
    let diff_abs = abs(&(&a - &b), None);
    let mixed = &square_a * &diff_abs;
    assert_eq!(mixed.tensor().data_type(), DataType::Float32);

    // Test complex activation functions
    let custom_activation = &(relu(&a, None)) + &tanh(&b, None);
    assert_eq!(custom_activation.tensor().data_type(), DataType::Float32);
}
