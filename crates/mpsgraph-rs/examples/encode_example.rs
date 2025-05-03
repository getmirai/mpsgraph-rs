use metal::{Device as MetalDevice, MTLResourceOptions};
use mpsgraph::{
    CommandBuffer, CompilationDescriptor, DataType, Device, Executable,
    ExecutableExecutionDescriptor, Graph, GraphActivationOps, Shape, ShapedType, TensorData,
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
    println!("MPSGraph Encode Example Using Executable");
    println!("=======================================");

    let metal_device = MetalDevice::system_default().expect("No Metal device found");
    println!("Using device: {}", metal_device.name());

    let graph = Graph::new();

    let shape_dimensions = [2, 3];
    println!("Creating tensors with shape: {:?}", shape_dimensions);

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
    let c = graph.add(&a, &b, Some("c"));
    let d = graph.relu(&c, Some("d")); // Result Tensor

    println!("\nCompiling graph to executable...");
    let comp_desc = CompilationDescriptor::new();
    let mps_device = Device::new();
    let shaped_type = ShapedType::new(&shape, DataType::Float32);
    let mut feed_types = HashMap::new();
    feed_types.insert(&a, &shaped_type);
    feed_types.insert(&b, &shaped_type);

    let executable: objc2::rc::Retained<Executable> =
        graph.compile(&mps_device, &feed_types, &[&d], Some(&comp_desc));
    println!("Graph compiled successfully.");

    println!("\nPreparing data for execution...");
    let a_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = [-0.5f32, -2.5, 0.5, -4.5, 2.0, -1.0]; // Include negatives for ReLU test

    println!("\nMatrix A:");
    print_matrix(&a_data, shape_dimensions[0], shape_dimensions[1]);
    println!("\nMatrix B:");
    print_matrix(&b_data, shape_dimensions[0], shape_dimensions[1]);

    let buffer_size =
        (shape_dimensions[0] * shape_dimensions[1] * std::mem::size_of::<f32>()) as u64;

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

    println!("Creating TensorData...");
    let shape_i64: Vec<i64> = shape_dimensions.iter().map(|&dim| dim as i64).collect();
    let a_tensor_data = TensorData::from_buffer(&a_buffer, &shape_i64, DataType::Float32);
    let b_tensor_data = TensorData::from_buffer(&b_buffer, &shape_i64, DataType::Float32);
    let d_tensor_data = TensorData::from_buffer(&d_buffer, &shape_i64, DataType::Float32);

    println!("\nExecuting using compiled Executable...");
    let command_queue = metal_device.new_command_queue();
    let command_buffer = CommandBuffer::from_command_queue(&command_queue);

    // Prepare inputs and outputs for the executable encode call
    let inputs_for_exec = [&a_tensor_data, &b_tensor_data];
    let outputs_for_exec = [&d_tensor_data];

    // Create descriptor for executable execution
    let exec_desc = ExecutableExecutionDescriptor::new();
    exec_desc.set_wait_until_completed(true);

    println!("Encoding using executable.encode_to_command_buffer...");
    let _result = executable.encode_to_command_buffer(
        &command_buffer,
        &inputs_for_exec,
        Some(&outputs_for_exec),
        Some(&exec_desc),
    );
    println!("Encode call completed.");

    println!("Committing and waiting...");
    command_buffer.commit();
    command_buffer.wait_until_completed();

    if command_buffer.status() == mpsgraph::CommandBufferStatus::Error {
        panic!("*** Error during command buffer execution! ***");
    }
    println!("Execution successful!");

    println!("\n--- Verifying Results ---");
    let result_ptr = d_buffer.contents() as *const f32;
    let result_slice = unsafe {
        std::slice::from_raw_parts(result_ptr, shape_dimensions[0] * shape_dimensions[1])
    };
    println!("Result Matrix (ReLU(A + B)):");
    print_matrix(result_slice, shape_dimensions[0], shape_dimensions[1]);

    let mut all_correct = true;
    for i in 0..shape_dimensions[0] * shape_dimensions[1] {
        let expected = (a_data[i] + b_data[i]).max(0.0); // ReLU
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
        println!("✅ All results match expected values!");
    } else {
        println!("❌ Results don't match expected values.");
    }
}

fn print_matrix(data: &[f32], rows: usize, cols: usize) {
    for row in 0..rows {
        let row_slice = &data[row * cols..(row + 1) * cols];
        println!("  {:?}", row_slice);
    }
}
