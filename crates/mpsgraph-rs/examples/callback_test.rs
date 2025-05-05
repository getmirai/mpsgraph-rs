use mpsgraph::{CommandBuffer, DataType, ExecutionDescriptor, Graph, Shape, TensorData};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

fn main() {
    // Create the graph and device
    let graph = Graph::new();

    // Create input placeholders with shape
    let shape_dimensions = [2, 2];
    let shape = Shape::matrix(2, 2);
    let a = graph.placeholder(DataType::Float32, &shape, None);
    let b = graph.placeholder(DataType::Float32, &shape, None);

    // Define operations: C = A + B
    let result = graph.add(&a, &b, None);

    // Get a Metal device
    let metal_device = metal::Device::system_default().unwrap();

    // Create input data using TensorData from buffers
    let a_data_values = [1.0f32, 2.0, 3.0, 4.0];
    let b_data_values = [5.0f32, 6.0, 7.0, 8.0];

    // Create Metal buffers for our data
    let a_buffer = metal_device.new_buffer_with_data(
        a_data_values.as_ptr() as *const _,
        (a_data_values.len() * std::mem::size_of::<f32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let b_buffer = metal_device.new_buffer_with_data(
        b_data_values.as_ptr() as *const _,
        (b_data_values.len() * std::mem::size_of::<f32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Create result buffer (empty)
    let result_buffer = metal_device.new_buffer(
        (4 * std::mem::size_of::<f32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Create TensorData from buffers
    let a_data = TensorData::from_buffer(&a_buffer, &shape, DataType::Float32);
    let b_data = TensorData::from_buffer(&b_buffer, &shape, DataType::Float32);
    let result_data = TensorData::from_buffer(&result_buffer, &shape, DataType::Float32);

    // Create input feeds
    let mut feeds = HashMap::new();
    feeds.insert(&a, &a_data);
    feeds.insert(&b, &b_data);

    // Create results map
    let mut results = HashMap::new();
    results.insert(&result, &result_data);

    // Get a Metal command queue
    let command_queue = metal_device.new_command_queue();

    // Create execution descriptor with asynchronous execution preference
    let execution_descriptor = ExecutionDescriptor::new();
    execution_descriptor.prefer_asynchronous_execution();

    // Create a command buffer
    let command_buffer = CommandBuffer::from_command_queue(&command_queue);
    command_buffer.set_label("Async Execution");

    // Create a flag to signal when the callback has completed
    let callback_completed = Arc::new(Mutex::new(false));
    let callback_completed_clone = Arc::clone(&callback_completed);

    println!("Starting asynchronous execution...");

    // Encode the graph to the command buffer
    graph.encode_to_command_buffer_with_results(
        &command_buffer,
        &feeds,
        None, // No specific target tensors
        &results,
        Some(&execution_descriptor),
    );

    // Commit the command buffer
    command_buffer.commit();

    // Simulate an asynchronous callback by waiting a bit
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(500));
        println!("Execution completed (simulated callback)");
        *callback_completed_clone.lock().unwrap() = true;
    });

    // Wait for the callback to be completed
    let mut attempts = 0;
    while !*callback_completed.lock().unwrap() && attempts < 10 {
        println!("Waiting for callback to complete...");
        thread::sleep(Duration::from_millis(100));
        attempts += 1;
    }

    // For comparison, run synchronously
    println!("\nRunning synchronously for comparison:");

    let sync_descriptor = ExecutionDescriptor::new();
    sync_descriptor.prefer_synchronous_execution();
    sync_descriptor.set_wait_until_completed(true);

    // Create a new command buffer for synchronous execution
    let sync_command_buffer = CommandBuffer::from_command_queue(&command_queue);
    sync_command_buffer.set_label("Sync Execution");

    // Create another result buffer and tensor data
    let sync_result_buffer = metal_device.new_buffer(
        (4 * std::mem::size_of::<f32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let sync_result_data = TensorData::from_buffer(&sync_result_buffer, &shape, DataType::Float32);
    let mut sync_results = HashMap::new();
    sync_results.insert(&result, &sync_result_data);

    // Encode the graph to the command buffer synchronously
    graph.encode_to_command_buffer_with_results(
        &sync_command_buffer,
        &feeds,
        None,
        &sync_results,
        Some(&sync_descriptor),
    );

    // Commit and wait for completion
    sync_command_buffer.commit();
    sync_command_buffer.wait_until_completed();

    println!("Synchronous execution completed");

    // Read and display the result
    let result_data = unsafe {
        let ptr = sync_result_buffer.contents() as *const f32;
        std::slice::from_raw_parts(ptr, 4).to_vec()
    };
    println!("Result (A + B): {:?}", result_data);

    // Verify the result
    let expected = [6.0, 8.0, 10.0, 12.0];
    let is_correct = result_data
        .iter()
        .zip(expected.iter())
        .all(|(a, b)| (a - b).abs() < 0.00001);

    if is_correct {
        println!("✅ Result is correct!");
    } else {
        println!("❌ Result doesn't match expected values.");
    }

    println!("Callback test completed!");
}
