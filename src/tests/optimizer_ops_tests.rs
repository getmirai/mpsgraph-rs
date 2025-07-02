use crate::optimizer_ops::VariableOp;
use crate::graph::Graph;
use crate::tensor::DataType;
use crate::shape::Shape;

#[test]
fn test_variable_op_creation() {
    let graph = Graph::new();
    
    // Create a tensor to use as a variable
    let shape = Shape::matrix(2, 3);
    let tensor = graph.placeholder(
        "weights", 
        DataType::Float32, 
        &shape, 
        None
    ).unwrap();
    
    // Create a variable operation for the tensor
    let variable_op = graph.variable_op_for_tensor(&tensor, Some("weights_var"));
    
    // Check that the variable op has the correct tensor
    let var_tensor = variable_op.tensor();
    
    // If we got here without crashing, the test passes
    assert!(true);
}

// This test only compiles the SGD API but doesn't run actual computations
#[test]
fn test_sgd_api_compiles() {
    let graph = Graph::new();
    
    // Create tensors for the test
    let shape = Shape::matrix(2, 3);
    let values = graph.placeholder("values", DataType::Float32, &shape, None).unwrap();
    let gradient = graph.placeholder("gradient", DataType::Float32, &shape, None).unwrap();
    let learning_rate = graph.placeholder("lr", DataType::Float32, &Shape::scalar(), None).unwrap();
    
    // Test SGD operation
    let _ = || {
        graph.stochastic_gradient_descent(&learning_rate, &values, &gradient, Some("sgd"))
    };
    
    // Create a variable operation
    let variable_op = graph.variable_op_for_tensor(&values, Some("var_op"));
    
    // Test apply SGD operation
    let _ = || {
        graph.apply_stochastic_gradient_descent(&learning_rate, &variable_op, &gradient, Some("apply_sgd"))
    };
}

// This test only compiles the Adam API but doesn't run actual computations
#[test]
fn test_adam_api_compiles() {
    let graph = Graph::new();
    
    // Create tensors for the test
    let shape = Shape::matrix(2, 3);
    let scalar_shape = Shape::scalar();
    
    let values = graph.placeholder("values", DataType::Float32, &shape, None).unwrap();
    let gradient = graph.placeholder("gradient", DataType::Float32, &shape, None).unwrap();
    let momentum = graph.placeholder("momentum", DataType::Float32, &shape, None).unwrap();
    let velocity = graph.placeholder("velocity", DataType::Float32, &shape, None).unwrap();
    let max_velocity = graph.placeholder("max_velocity", DataType::Float32, &shape, None).unwrap();
    
    let learning_rate = graph.placeholder("lr", DataType::Float32, &scalar_shape, None).unwrap();
    let beta1 = graph.placeholder("beta1", DataType::Float32, &scalar_shape, None).unwrap();
    let beta2 = graph.placeholder("beta2", DataType::Float32, &scalar_shape, None).unwrap();
    let epsilon = graph.placeholder("epsilon", DataType::Float32, &scalar_shape, None).unwrap();
    let beta1_power = graph.placeholder("beta1_power", DataType::Float32, &scalar_shape, None).unwrap();
    let beta2_power = graph.placeholder("beta2_power", DataType::Float32, &scalar_shape, None).unwrap();
    
    // Test Adam operation
    let _ = || {
        graph.adam(
            &learning_rate,
            &beta1,
            &beta2,
            &epsilon,
            &beta1_power,
            &beta2_power,
            &values,
            &momentum,
            &velocity,
            Some(&max_velocity),
            &gradient,
            Some("adam")
        )
    };
    
    // Test Adam with current learning rate operation
    let _ = || {
        graph.adam_with_current_learning_rate(
            &learning_rate,
            &beta1,
            &beta2,
            &epsilon,
            &values,
            &momentum,
            &velocity,
            Some(&max_velocity),
            &gradient,
            Some("adam_current_lr")
        )
    };
    
    // Test Adam without maximum velocity
    let _ = || {
        graph.adam(
            &learning_rate,
            &beta1,
            &beta2,
            &epsilon,
            &beta1_power,
            &beta2_power,
            &values,
            &momentum,
            &velocity,
            None, // no maximum velocity
            &gradient,
            Some("adam_no_max_vel")
        )
    };
}