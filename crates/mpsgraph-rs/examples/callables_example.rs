use metal::{Device, MTLResourceOptions};
use mpsgraph::{
    CompilationDescriptor, DataType, Executable, ExecutableExecutionDescriptor, Graph,
    GraphCallOps, Shape, ShapedType, TensorData,
};
use ndarray::Array2;
use objc2::rc::Retained;
use std::collections::HashMap;

/// An example demonstrating MPSGraph callable executables
///
/// This example shows:
/// 1. Creating two separate graph executables: one for addition and one for multiplication
/// 2. Using the callables API to register these executables with a main graph
/// 3. Calling these executables from the main graph and chaining their operations
fn main() {
    println!("MPSGraph Callables Example\n");

    //-- Setup --//
    // Get the default Metal device
    let device = Device::system_default().expect("No Metal device found");
    println!("Using device: {}", device.name());

    // Create a command queue for execution
    let command_queue = device.new_command_queue();

    //-- Create the first executable (Addition) --//
    println!("Creating addition executable...");
    let add_executable = create_addition_executable();

    //-- Create the second executable (Multiplication) --//
    println!("Creating multiplication executable...");
    let mul_executable = create_multiplication_executable();

    //-- Create main graph --//
    println!("Creating main graph with callables...");

    // Create a compilation descriptor with callables
    let compilation_descriptor = CompilationDescriptor::new();

    // Register our executables with the compilation descriptor
    compilation_descriptor.add_callable("addition", &add_executable);
    compilation_descriptor.add_callable("multiplication", &mul_executable);

    // Print the current callables
    let callables = compilation_descriptor.get_callables();
    println!(
        "Registered callables: {:?}",
        callables.keys().collect::<Vec<_>>()
    );

    // Create a new graph for our main computation
    let graph = Graph::new();

    //-- Define main computation graph that calls the executables --//

    // Define input tensors for the main graph
    let shape = Shape::matrix(2, 2);
    let a = graph.placeholder(DataType::Float32, &shape, None);
    let b = graph.placeholder(DataType::Float32, &shape, None);

    // Create a shaped type for our tensors (needed for the call operation)
    let float32_matrix = ShapedType::new(&shape, DataType::Float32);

    // Call the addition executable
    let addition_result = graph.call(
        "addition",
        &[a.clone(), b.clone()],
        &[float32_matrix.clone()],
        Some("AdditionCall"),
    );

    // Call the multiplication executable
    let multiplication_result = graph.call(
        "multiplication",
        &[a.clone(), b.clone()],
        &[float32_matrix.clone()],
        Some("MultiplicationCall"),
    );

    // Add the results of the two operations
    let final_result = graph.add(
        &addition_result[0],
        &multiplication_result[0],
        Some("FinalResult"),
    );

    //-- Create buffers for input and output data --//
    println!("Creating Metal buffers for inputs and outputs...");

    // Define input data
    let a_input = [1.0f32, 2.0, 3.0, 4.0];
    let b_input = [5.0f32, 6.0, 7.0, 8.0];

    // Calculate buffer size for a 2x2 matrix of f32
    let buffer_size = (4 * std::mem::size_of::<f32>()) as u64;

    // Create Metal buffers
    let a_buffer = device.new_buffer_with_data(
        a_input.as_ptr() as *const _,
        buffer_size,
        MTLResourceOptions::StorageModeShared,
    );

    let b_buffer = device.new_buffer_with_data(
        b_input.as_ptr() as *const _,
        buffer_size,
        MTLResourceOptions::StorageModeShared,
    );

    // Output buffers
    let add_buffer = device.new_buffer(buffer_size, MTLResourceOptions::StorageModeShared);
    let mul_buffer = device.new_buffer(buffer_size, MTLResourceOptions::StorageModeShared);
    let final_buffer = device.new_buffer(buffer_size, MTLResourceOptions::StorageModeShared);

    //-- Create TensorData from Metal Buffers --//
    println!("Creating TensorData from Metal buffers...");

    // Wrap Metal buffers in TensorData
    let a_data = TensorData::from_buffer(&a_buffer, &shape, DataType::Float32);
    let b_data = TensorData::from_buffer(&b_buffer, &shape, DataType::Float32);
    // Create tensor data for output buffers, prefixed with underscore to avoid warnings
    // since they're not directly used but needed for the buffers to be properly set up
    let _add_data = TensorData::from_buffer(&add_buffer, &shape, DataType::Float32);
    let _mul_data = TensorData::from_buffer(&mul_buffer, &shape, DataType::Float32);
    let final_data = TensorData::from_buffer(&final_buffer, &shape, DataType::Float32);

    //-- Compile the graph --//
    println!("Compiling the main graph with callables...");

    let device_obj = mpsgraph::Device::new();

    // Create the feeds tensor types for compilation
    let mut feeds = HashMap::new();
    feeds.insert(&a, &float32_matrix);
    feeds.insert(&b, &float32_matrix);

    // Compile the graph with our compilation descriptor that has the callables
    let executable = graph.compile(
        &device_obj,
        &feeds,
        &[&final_result],
        Some(&compilation_descriptor),
    );

    println!("Graph compiled successfully!");

    //-- Set Up Feeds and Results for Execution --//
    println!("Setting up feeds and results...");

    // Create feed dictionary (inputs)
    let mut feed_data = Vec::new();
    feed_data.push(&a_data);
    feed_data.push(&b_data);

    // Create results dictionary (outputs)
    let mut results_data = Vec::new();
    results_data.push(&final_data);

    //-- Create CommandBuffer --//
    println!("Creating CommandBuffer and encoding executable...");

    // Create a new CommandBuffer from the command queue
    let command_buffer = mpsgraph::CommandBuffer::from_command_queue(&command_queue);

    // Create execution descriptor
    let execution_descriptor = ExecutableExecutionDescriptor::new();
    execution_descriptor.prefer_synchronous_execution();

    // Convert our vectors to the right format
    let inputs: Vec<&Retained<TensorData>> = feed_data.iter().map(|x| *x).collect();
    let results: Vec<&Retained<TensorData>> = results_data.iter().map(|x| *x).collect();

    // Encode executable to command buffer
    println!("Encoding executable to command buffer...");

    executable.encode_to_command_buffer(
        &command_buffer,
        &inputs,
        Some(&results),
        Some(&execution_descriptor),
    );

    // Get the Metal command buffer to commit and wait
    let mtl_command_buffer = command_buffer.command_buffer();

    println!("Committing command buffer and waiting for completion...");

    // Commit the command buffer and wait
    mtl_command_buffer.commit();
    mtl_command_buffer.wait_until_completed();

    println!("Execution completed successfully!");

    //-- Read Results --//
    println!("\nReading final result:");

    // Read final result
    let final_result = unsafe {
        let ptr = final_buffer.contents() as *const f32;
        std::slice::from_raw_parts(ptr, 4).to_vec()
    };
    println!("Final result: {:?}", final_result);

    //-- Verify Results with ndarray --//
    println!("\nVerifying results using ndarray:");

    // Convert Metal buffer results to ndarray format
    let result_array = Array2::from_shape_vec((2, 2), final_result.clone()).unwrap();

    // Create ndarray arrays for the inputs and expected computations
    let a_array = Array2::from_shape_vec((2, 2), a_input.to_vec()).unwrap();
    let b_array = Array2::from_shape_vec((2, 2), b_input.to_vec()).unwrap();

    // Compute using ndarray:
    // 1. Addition: a + b
    let add_result = &a_array + &b_array;
    println!("ndarray addition:\n{}", add_result);

    // 2. Multiplication: a * b (element-wise)
    let mul_result = &a_array * &b_array;
    println!("ndarray multiplication:\n{}", mul_result);

    // 3. Addition + Multiplication: (a + b) + (a * b)
    let expected_result = &add_result + &mul_result;
    println!("ndarray expected final result:\n{}", expected_result);
    println!("MPSGraph final result reshaped:\n{}", result_array);

    // Check if results match with a small tolerance
    let difference = &result_array - &expected_result;
    let max_diff = difference
        .iter()
        .fold(0.0f32, |max_val, &val| max_val.max(val.abs()));

    // Also check with the old method for compatibility
    // Expected results as a flat array
    let expected_final = [11.0, 20.0, 31.0, 44.0];
    let final_correct = final_result
        .iter()
        .zip(expected_final.iter())
        .all(|(a, b)| (a - b).abs() < 0.00001);

    if max_diff < 0.00001 && final_correct {
        println!("✅ Results correct! Maximum difference: {}", max_diff);
    } else {
        println!("❌ Results don't match expected values.");
        println!("Maximum difference: {}", max_diff);
        println!("Expected: \n{}", expected_result);
        println!("Got: \n{}", result_array);
    }

    println!("\nCallables execution complete!");
}

/// Creates an executable for addition
fn create_addition_executable() -> Retained<Executable> {
    // Create a graph for addition
    let graph = Graph::new();

    // Define input tensors
    let shape = Shape::matrix(2, 2);
    let a = graph.placeholder(DataType::Float32, &shape, None);
    let b = graph.placeholder(DataType::Float32, &shape, None);

    // Define computation: C = A + B
    let c = graph.add(&a, &b, Some("Addition"));

    // Create shaped types for inputs and outputs
    let float32_matrix = ShapedType::new(&shape, DataType::Float32);

    // Set up feeds for compilation
    let mut feeds = HashMap::new();
    feeds.insert(&a, &float32_matrix);
    feeds.insert(&b, &float32_matrix);

    // Create device object
    let device = mpsgraph::Device::new();

    // Compile the graph
    let compilation_descriptor = CompilationDescriptor::new();
    compilation_descriptor.set_optimization_level(mpsgraph::Optimization::Level1);

    graph.compile(&device, &feeds, &[&c], Some(&compilation_descriptor))
}

/// Creates an executable for multiplication
fn create_multiplication_executable() -> Retained<Executable> {
    // Create a graph for multiplication
    let graph = Graph::new();

    // Define input tensors
    let shape = Shape::matrix(2, 2);
    let a = graph.placeholder(DataType::Float32, &shape, None);
    let b = graph.placeholder(DataType::Float32, &shape, None);

    // Define computation: C = A * B (element-wise)
    let c = graph.multiply(&a, &b, Some("Multiplication"));

    // Create shaped types for inputs and outputs
    let float32_matrix = ShapedType::new(&shape, DataType::Float32);

    // Set up feeds for compilation
    let mut feeds = HashMap::new();
    feeds.insert(&a, &float32_matrix);
    feeds.insert(&b, &float32_matrix);

    // Create device object
    let device = mpsgraph::Device::new();

    // Compile the graph
    let compilation_descriptor = CompilationDescriptor::new();
    compilation_descriptor.set_optimization_level(mpsgraph::Optimization::Level1);

    graph.compile(&device, &feeds, &[&c], Some(&compilation_descriptor))
}
