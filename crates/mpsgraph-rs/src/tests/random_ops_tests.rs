use crate::random_ops::{RandomOpDescriptor, RandomDistribution, RandomNormalSamplingMethod};
use crate::tensor::DataType;
use crate::graph::Graph;
use crate::shape::Shape;
use crate::device::CustomDefault;

#[test]
fn test_create_random_op_descriptor() {
    // Create a random descriptor with uniform distribution
    let descriptor = RandomOpDescriptor::new(RandomDistribution::Uniform, DataType::Float32);
    
    // Configure the descriptor
    descriptor
        .set_min(0.0)
        .set_max(1.0);
    
    // If we got here without crashing, the test passes
    assert!(true);
}

#[test]
fn test_create_normal_distribution_descriptor() {
    // Create a random descriptor with normal distribution
    let descriptor = RandomOpDescriptor::new(RandomDistribution::Normal, DataType::Float32);
    
    // Configure the descriptor
    descriptor
        .set_mean(0.0)
        .set_standard_deviation(1.0)
        .set_sampling_method(RandomNormalSamplingMethod::BoxMuller);
    
    // If we got here without crashing, the test passes
    assert!(true);
}

#[test]
fn test_create_integer_distribution_descriptor() {
    // Create a random descriptor with uniform distribution for integers
    let descriptor = RandomOpDescriptor::new(RandomDistribution::Uniform, DataType::Int32);
    
    // Configure the descriptor
    descriptor
        .set_min_integer(0)
        .set_max_integer(100);
    
    // If we got here without crashing, the test passes
    assert!(true);
}

#[test]
fn test_custom_default() {
    // Get a default random descriptor
    let descriptor = RandomOpDescriptor::custom_default();
    
    // If we got here without crashing, the test passes
    assert!(true);
}

// This test only compiles the API but doesn't run actual computations
#[test]
fn test_random_ops_api_compiles() {
    let graph = Graph::new();
    
    // Create a tensor shape
    let shape = Shape::tensor3d(2, 3, 4); // 2x3x4 tensor
    
    // Test creating a random state tensor
    let _ = |seed| {
        graph.random_philox_state_tensor_with_seed(seed, Some("state_from_seed"))
    };
    
    let _ = |counter_low, counter_high, key| {
        graph.random_philox_state_tensor_with_counter(counter_low, counter_high, key, Some("state_from_counter"))
    };
    
    // Test random tensor generation
    let _ = |descriptor| {
        graph.random_tensor(&shape, &descriptor, Some("random_tensor"))
    };
    
    let _ = |descriptor, seed| {
        graph.random_tensor_with_seed(&shape, &descriptor, seed, Some("random_tensor_with_seed"))
    };
    
    let _ = |descriptor, state| {
        let (random_tensor, updated_state) = graph.random_tensor_with_state(
            &shape, 
            &descriptor, 
            state, 
            Some("random_tensor_with_state")
        );
        (random_tensor, updated_state)
    };
    
    // Test random uniform tensor generation
    let _ = || {
        graph.random_uniform_tensor(&shape, Some("uniform_tensor"))
    };
    
    let _ = |seed| {
        graph.random_uniform_tensor_with_seed(&shape, seed, Some("uniform_tensor_with_seed"))
    };
    
    let _ = |state| {
        let (uniform_tensor, updated_state) = graph.random_uniform_tensor_with_state(
            &shape, 
            state, 
            Some("uniform_tensor_with_state")
        );
        (uniform_tensor, updated_state)
    };
    
    // Test dropout operations
    let _ = |tensor, rate| {
        graph.dropout(tensor, rate, Some("dropout"))
    };
    
    let _ = |tensor, rate_tensor| {
        graph.dropout_with_rate_tensor(tensor, rate_tensor, Some("dropout_with_tensor"))
    };
}