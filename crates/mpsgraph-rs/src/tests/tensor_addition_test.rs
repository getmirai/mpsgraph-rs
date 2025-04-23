use crate::{DataType, Graph, Shape, TensorData};
use objc2_foundation::NSNumber;
use std::collections::HashMap;

#[test]
fn test_tensor_addition_metal_execution() {
    // Create a new graph
    let graph = Graph::new();

    // Create shape for our tensors (2x2 matrices)
    let numbers = [
        NSNumber::new_usize(2), // 2 rows
        NSNumber::new_usize(2), // 2 columns
    ];
    let number_refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();
    let shape = Shape::from_slice(&number_refs);

    // Create placeholder tensors for the inputs
    let a = graph.placeholder(DataType::Float32, &shape, Some("a"));
    let b = graph.placeholder(DataType::Float32, &shape, Some("b"));

    // Perform the addition operation
    let c = graph.add(&a, &b, Some("c"));

    // Create matrix A data: [[1.0, 2.0], [3.0, 4.0]]
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let shape_dimensions: Vec<i64> = vec![2, 2]; // 2x2 matrix
    let a_tensor_data = TensorData::from_bytes(&a_data, &shape_dimensions, DataType::Float32);

    // Create matrix B data: [[5.0, 6.0], [7.0, 8.0]]
    let b_data: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
    let b_tensor_data = TensorData::from_bytes(&b_data, &shape_dimensions, DataType::Float32);

    // Set up input feeds with the correct type
    let mut feeds = HashMap::new();
    feeds.insert(&*a, &*a_tensor_data);
    feeds.insert(&*b, &*b_tensor_data);

    // Execute the graph and get results
    let results = graph.run_with_feeds(&feeds, &[&*c]);

    // Verify we got a result for tensor C
    assert_eq!(results.len(), 1);

    // Get the result tensor data
    let c_tensor_data = results.iter().next().unwrap().1;

    // Ensure data is synchronized to CPU
    c_tensor_data.synchronize();

    // Get the data as f32 values
    if let Some(result_values) = c_tensor_data.bytes_as::<f32>() {
        // Verify the results are correct
        assert_eq!(result_values[0], 6.0); // 1.0 + 5.0
        assert_eq!(result_values[1], 8.0); // 2.0 + 6.0
        assert_eq!(result_values[2], 10.0); // 3.0 + 7.0
        assert_eq!(result_values[3], 12.0); // 4.0 + 8.0
    } else {
        panic!("Failed to read result data");
    }
}
