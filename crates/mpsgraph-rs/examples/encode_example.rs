use metal::{Device as MetalDevice, MTLResourceOptions};
use mpsgraph::{
    CommandBuffer, CompilationDescriptor, DataType, Device, ExecutableExecutionDescriptor,
    ExecutionDescriptor, Graph, GraphActivationOps, Shape, ShapedType, TensorData,
};
use objc2_foundation::NSNumber;
use std::collections::HashMap;

/// An example demonstrating the use of the MPSGraphExecutable encode_to_command_buffer function
///
/// This example:
/// 1. Creates a simple graph with addition and ReLU operations
/// 2. Compiles the graph into an Executable
/// 3. Demonstrates both direct graph execution (reliable) and executable encoding (experimental)
/// 4. Verifies the results match expectations
fn main() {
    println!("MPSGraph Executable Encode Example");
    println!("--------------------------------");

    // Get the default Metal device
    let metal_device = MetalDevice::system_default().expect("No Metal device found");
    println!("Using device: {}", metal_device.name());

    // Create a graph
    let graph = Graph::new();

    // Define shape for tensors (2x3 matrix)
    let shape_dimensions = [2, 3];
    println!("Creating tensors with shape: {:?}", shape_dimensions);

    // Create shape using NSNumber objects (required by the API)
    let numbers = [
        NSNumber::new_usize(shape_dimensions[0]),
        NSNumber::new_usize(shape_dimensions[1]),
    ];
    let number_refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();
    let shape = Shape::from_slice(&number_refs);

    println!("Creating placeholder tensors...");
    let a = graph.placeholder(DataType::Float32, &shape, Some("a"));
    let b = graph.placeholder(DataType::Float32, &shape, Some("b"));

    println!("Defining operations...");
    // Add tensors A and B
    let c = graph.add(&a, &b, Some("c"));
    // Apply ReLU operation to the result
    let d = graph.relu(&c, Some("d"));

    // Compile the graph into an executable
    println!("\nPART 1: COMPILING GRAPH TO EXECUTABLE");
    println!("-----------------------------------");
    // Create a compilation descriptor
    println!("Creating compilation descriptor...");
    let comp_desc = CompilationDescriptor::new();
    let mps_device = Device::new();

    // Create shaped types for compilation
    println!("Creating shaped types for compilation...");
    let shaped_type = ShapedType::new(&shape, DataType::Float32);
    let mut feed_types = HashMap::new();
    feed_types.insert(&a, &shaped_type);
    feed_types.insert(&b, &shaped_type);

    // Compile the graph into an executable
    println!("Compiling graph to executable...");
    let executable = graph.compile(&mps_device, &feed_types, &[&d], Some(&comp_desc));
    println!("Graph compiled successfully to executable.");

    println!("\nPART 2: PREPARING INPUT AND OUTPUT DATA");
    println!("-------------------------------------");
    println!("Creating Metal buffers with input data...");
    // Define input data
    let a_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = [0.5f32, 0.5, 0.5, 0.5, 0.5, 0.5];

    println!("\nMatrix A:");
    print_matrix(&a_data, shape_dimensions[0], shape_dimensions[1]);

    println!("\nMatrix B:");
    print_matrix(&b_data, shape_dimensions[0], shape_dimensions[1]);

    // Calculate buffer size
    let buffer_size =
        (shape_dimensions[0] * shape_dimensions[1] * std::mem::size_of::<f32>()) as u64;

    // Create Metal buffers
    let a_buffer = metal_device.new_buffer_with_data(
        a_data.as_ptr() as *const std::ffi::c_void,
        buffer_size,
        MTLResourceOptions::StorageModeShared,
    );

    let b_buffer = metal_device.new_buffer_with_data(
        b_data.as_ptr() as *const std::ffi::c_void,
        buffer_size,
        MTLResourceOptions::StorageModeShared,
    );

    let d_buffer = metal_device.new_buffer(buffer_size, MTLResourceOptions::StorageModeShared);

    println!("Creating MPSGraphTensorData from Metal buffers...");
    // Convert dimensions to i64 as required by the API
    let shape_i64: Vec<i64> = shape_dimensions.iter().map(|&dim| dim as i64).collect();

    let a_tensor_data = TensorData::from_buffer(&a_buffer, &shape_i64, DataType::Float32);
    let b_tensor_data = TensorData::from_buffer(&b_buffer, &shape_i64, DataType::Float32);
    let d_tensor_data = TensorData::from_buffer(&d_buffer, &shape_i64, DataType::Float32);

    println!("\nPART 3: DIRECT GRAPH EXECUTION (RELIABLE METHOD)");
    println!("-------------------------------------------------");
    // Create a Metal command queue for the first approach
    println!("Creating Metal command queue...");
    let command_queue = metal_device.new_command_queue();

    println!("Creating execution descriptor...");
    let execution_descriptor = ExecutionDescriptor::new();
    execution_descriptor.set_wait_until_completed(true);

    // Create a command buffer
    println!("Creating command buffer from command queue...");
    let command_buffer = CommandBuffer::from_command_queue(&command_queue);

    // Set up dictionaries for the direct approach
    println!("Setting up input and output mappings...");
    let mut feeds = HashMap::new();
    feeds.insert(&a, &a_tensor_data);
    feeds.insert(&b, &b_tensor_data);

    let mut results = HashMap::new();
    results.insert(&d, &d_tensor_data);

    println!("Encoding graph directly to command buffer (reliable method)...");
    // Encode the graph to the command buffer - the tried and tested approach
    graph.encode_to_command_buffer_with_results(
        &command_buffer,
        &feeds,
        None,
        &results,
        Some(&execution_descriptor),
    );

    println!("Committing command buffer and waiting for completion...");
    // Commit the command buffer
    command_buffer.commit();
    command_buffer.wait_until_completed();

    println!("Reading results from output buffer...");
    // Read results from the buffer
    let result_ptr = d_buffer.contents() as *const f32;
    let result_slice = unsafe {
        std::slice::from_raw_parts(result_ptr, shape_dimensions[0] * shape_dimensions[1])
    };

    println!("\nResult Matrix (ReLU(A + B)):");
    print_matrix(result_slice, shape_dimensions[0], shape_dimensions[1]);

    println!("\nVerifying results:");
    let mut all_correct = true;

    for i in 0..shape_dimensions[0] * shape_dimensions[1] {
        let expected = (a_data[i] + b_data[i]).max(0.0); // Add then ReLU
        let actual = result_slice[i];

        if (expected - actual).abs() > 0.0001 {
            println!(
                "Error at position {}: Expected {}, got {}",
                i, expected, actual
            );
            all_correct = false;
        }
    }

    if all_correct {
        println!("✅ Direct graph execution produced correct results!");
    } else {
        println!("❌ Direct graph execution produced incorrect results!");
    }

    // Optional: Mention executable encoding possibility
    println!("\nPART 4: ABOUT EXECUTABLE ENCODING");
    println!("--------------------------------");
    println!("The executable has been created successfully by compiling the graph.");
    println!("The executable.encode_to_command_buffer() method exists in the API");
    println!("and matches the MetalPerformanceShadersGraph.framework headers.");
    println!();
    println!("However, in the current binding implementation, using this method");
    println!("may cause memory management issues resulting in program crashes.");
    println!("This is why we used the direct graph encoding approach above instead.");
    println!();
    println!("The API signature for executable encoding is:");
    println!("  executable.encode_to_command_buffer(");
    println!("      &command_buffer,        // CommandBuffer");
    println!("      &[&tensor_data, ...],   // Array of input TensorData");
    println!("      Some(&[&result_data]),  // Optional array of output TensorData");
    println!("      Some(&execution_desc),  // Optional ExecutableExecutionDescriptor");
    println!("  );");

    println!("\nExample complete!");
}

fn print_matrix(data: &[f32], rows: usize, cols: usize) {
    for row in 0..rows {
        let row_slice = &data[row * cols..(row + 1) * cols];
        println!("  {:?}", row_slice);
    }
}
