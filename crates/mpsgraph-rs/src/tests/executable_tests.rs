#[cfg(test)]
use crate::command_buffer::CommandBuffer;
use crate::data_types::ShapedType;
use crate::device::Device;
use crate::executable::{CompilationDescriptor, ExecutableExecutionDescriptor};
use crate::graph::Graph;
use crate::shape::Shape;
use crate::tensor::DataType;
use crate::tensor_data::TensorData;
use metal::{Device as MetalDevice, MTLResourceOptions};
use std::collections::HashMap;

/// Test for the encode_to_command_buffer method of Executable
#[test]
fn test_encode_to_command_buffer() {
    // We'll need to:
    // 1. Create a graph with a simple operation
    // 2. Compile the graph to get an executable
    // 3. Create inputs and expected outputs
    // 4. Create a command buffer
    // 5. Call encode_to_command_buffer
    // 6. Verify the results

    // Step 1: Create a simple graph with an addition operation
    let graph = Graph::new();

    // Create a shape for our tensors
    let shape = Shape::from_dimensions(&[2, 2]);
    let shape_dims = shape.dimensions().to_vec();

    // Create placeholder tensors for the inputs
    let a = graph.placeholder(DataType::Float32, &shape, Some("a"));
    let b = graph.placeholder(DataType::Float32, &shape, Some("b"));

    // Create the add operation
    let result = graph.add(&a, &b, Some("result"));

    // Step 2: Compile the graph to get an executable
    // Get a Metal device
    let metal_device = MetalDevice::system_default().expect("No Metal device found");
    let device = Device::with_device(&metal_device);

    // Create a compilation descriptor
    let compilation_descriptor = CompilationDescriptor::new();

    // Create input shapes for compilation
    let a_shape = ShapedType::new(&shape, DataType::Float32);
    let b_shape = ShapedType::new(&shape, DataType::Float32);

    // Maps of inputs for compilation
    let mut feeds = HashMap::new();
    feeds.insert(&a, &a_shape);
    feeds.insert(&b, &b_shape);

    // Compile the graph
    let targets = [&result];
    let executable = graph.compile(&device, &feeds, &targets, Some(&compilation_descriptor));

    // Step 3: Create inputs and expected outputs
    // Create a command queue from the Metal device
    let cmd_queue = metal_device.new_command_queue();

    // Create input data
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_data = vec![5.0f32, 6.0, 7.0, 8.0];

    // Create Metal buffers for inputs and outputs
    let a_buffer = metal_device.new_buffer_with_data(
        a_data.as_ptr() as *const std::ffi::c_void,
        (a_data.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let b_buffer = metal_device.new_buffer_with_data(
        b_data.as_ptr() as *const std::ffi::c_void,
        (b_data.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Expected result: a + b = [6.0, 8.0, 10.0, 12.0]
    let result_buffer = metal_device.new_buffer(
        (a_data.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Create TensorData from Metal buffers
    let shape_for_tensors = Shape::from_dimensions(&shape_dims);
    let a_tensor_data = TensorData::from_buffer(&a_buffer, &shape_for_tensors, DataType::Float32);
    let b_tensor_data = TensorData::from_buffer(&b_buffer, &shape_for_tensors, DataType::Float32);
    let result_tensor_data =
        TensorData::from_buffer(&result_buffer, &shape_for_tensors, DataType::Float32);

    // Step 4: Create a command buffer
    let cmd_buffer = CommandBuffer::from_command_buffer(&cmd_queue.new_command_buffer().to_owned());

    // Step 5: Create an execution descriptor
    let execution_descriptor = ExecutableExecutionDescriptor::new();
    execution_descriptor.set_wait_until_completed(true);

    // Step 6: Call encode_to_command_buffer
    let inputs = [&a_tensor_data, &b_tensor_data];
    let results = [&result_tensor_data];

    let _output = executable.encode_to_command_buffer(
        &cmd_buffer,
        &inputs,
        Some(&results),
        Some(&execution_descriptor),
    );

    // Commit the command buffer
    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();

    // Step 7: Verify the results
    // Read back the result data
    let result_ptr = result_buffer.contents() as *const f32;
    let result_data = unsafe { std::slice::from_raw_parts(result_ptr, a_data.len()) };

    // Check the results
    assert_eq!(result_data[0], 6.0);
    assert_eq!(result_data[1], 8.0);
    assert_eq!(result_data[2], 10.0);
    assert_eq!(result_data[3], 12.0);

    println!("Encode to command buffer test passed!");
}

/// Test for memory management in encode_to_command_buffer method
#[test]
fn test_encode_to_command_buffer_memory_management() {
    // This test specifically focuses on verifying the memory management
    // patterns in the encode_to_command_buffer method

    // Create a graph with a simple operation
    let graph = Graph::new();
    let shape = Shape::from_dimensions(&[2, 2]);
    let shape_dims = shape.dimensions().to_vec();

    // Create placeholders and operation
    let a = graph.placeholder(DataType::Float32, &shape, Some("a"));
    let b = graph.placeholder(DataType::Float32, &shape, Some("b"));
    let result = graph.add(&a, &b, Some("result"));

    // Compile to get an executable
    let metal_device = MetalDevice::system_default().expect("No Metal device found");
    let device = Device::with_device(&metal_device);
    let compilation_descriptor = CompilationDescriptor::new();

    // Set up inputs for compilation
    let a_shape = ShapedType::new(&shape, DataType::Float32);
    let b_shape = ShapedType::new(&shape, DataType::Float32);
    let mut feeds = HashMap::new();
    feeds.insert(&a, &a_shape);
    feeds.insert(&b, &b_shape);

    // Compile the graph
    let targets = [&result];
    let executable = graph.compile(&device, &feeds, &targets, Some(&compilation_descriptor));

    // Create inputs
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_data = vec![5.0f32, 6.0, 7.0, 8.0];

    // Create Metal buffers
    let a_buffer = metal_device.new_buffer_with_data(
        a_data.as_ptr() as *const std::ffi::c_void,
        (a_data.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let b_buffer = metal_device.new_buffer_with_data(
        b_data.as_ptr() as *const std::ffi::c_void,
        (b_data.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let result_buffer = metal_device.new_buffer(
        (a_data.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Create TensorData with proper memory management
    let shape_for_tensors = Shape::from_dimensions(&shape_dims);
    let a_tensor_data = TensorData::from_buffer(&a_buffer, &shape_for_tensors, DataType::Float32);
    let b_tensor_data = TensorData::from_buffer(&b_buffer, &shape_for_tensors, DataType::Float32);
    let result_tensor_data =
        TensorData::from_buffer(&result_buffer, &shape_for_tensors, DataType::Float32);

    // Create command queue and execution descriptor
    let cmd_queue = metal_device.new_command_queue();

    // Create a single command buffer that we'll reuse
    let cmd_buffer = CommandBuffer::from_command_buffer(&cmd_queue.new_command_buffer().to_owned());

    let execution_descriptor = ExecutableExecutionDescriptor::new();

    // Call encode_to_command_buffer multiple times to test memory management
    for i in 0..5 {
        let inputs = [&a_tensor_data, &b_tensor_data];
        let results = [&result_tensor_data];

        // This will test both retain_autoreleased patterns in encode_to_command_buffer:
        // 1. On the result array from the Objective-C method
        // 2. On each tensor inside the array
        let output = executable.encode_to_command_buffer(
            &cmd_buffer,
            &inputs,
            Some(&results),
            Some(&execution_descriptor),
        );

        // Verify output contains expected results (only applicable if results were returned)
        if let Some(output_tensors) = output {
            println!(
                "Iteration {}: Got {} output tensors",
                i,
                output_tensors.len()
            );
        }

        // Commit the current command buffer operations and continue
        // This allows us to reuse the same command buffer
        cmd_buffer.commit_and_continue();
    }

    // Wait for all operations to complete at the end
    cmd_buffer.wait_until_completed();

    // If we got here without crashes, memory management likely works correctly
    println!("Encode to command buffer memory management test passed!");
}
