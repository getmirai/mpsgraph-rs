use metal::{Device as MetalDevice, MTLResourceOptions};
use mpsgraph::{CommandBuffer, DataType, ExecutionDescriptor, Graph, Shape, TensorData};
use objc2_foundation::NSNumber;
use std::collections::HashMap;

fn main() {
    println!("MPSGraph Tensor Addition Example");
    println!("--------------------------------");

    let metal_device = MetalDevice::system_default().expect("No Metal device found");
    println!("Metal device: {}", metal_device.name());

    let graph = Graph::new();

    let shape_dimensions = [2, 2];
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

    println!("Defining addition operation...");
    let c = graph.add(&a, &b, Some("c"));

    println!("Creating Metal buffers with input data...");

    let matrix_a = [1.0f32, 2.0f32, 3.0f32, 4.0f32];

    let matrix_b = [5.0f32, 6.0f32, 7.0f32, 8.0f32];

    println!("\nMatrix A:");
    print_matrix(&matrix_a, shape_dimensions[0], shape_dimensions[1]);

    println!("\nMatrix B:");
    print_matrix(&matrix_b, shape_dimensions[0], shape_dimensions[1]);

    let buffer_size =
        (shape_dimensions[0] * shape_dimensions[1] * std::mem::size_of::<f32>()) as u64;

    unsafe {
        let a_buffer = metal_device.new_buffer_with_data(
            matrix_a.as_ptr() as *const std::ffi::c_void,
            buffer_size,
            MTLResourceOptions::StorageModeShared,
        );

        let b_buffer = metal_device.new_buffer_with_data(
            matrix_b.as_ptr() as *const std::ffi::c_void,
            buffer_size,
            MTLResourceOptions::StorageModeShared,
        );

        let c_buffer = metal_device.new_buffer(buffer_size, MTLResourceOptions::StorageModeShared);

        println!("Creating MPSGraphTensorData from Metal buffers...");

        let shape_i64: Vec<i64> = shape_dimensions.iter().map(|&dim| dim as i64).collect();

        // Create a proper Shape object from dimensions
        let tensor_shape = Shape::from_dimensions(&shape_i64);

        let a_tensor_data = TensorData::from_buffer(&a_buffer, &tensor_shape, DataType::Float32);
        let b_tensor_data = TensorData::from_buffer(&b_buffer, &tensor_shape, DataType::Float32);
        let c_tensor_data = TensorData::from_buffer(&c_buffer, &tensor_shape, DataType::Float32);

        println!("Creating Metal command queue...");
        let command_queue = metal_device.new_command_queue();

        println!("Setting up input and output mappings...");

        let mut feeds = HashMap::new();
        feeds.insert(&a, &a_tensor_data);
        feeds.insert(&b, &b_tensor_data);

        let mut results = HashMap::new();
        results.insert(&c, &c_tensor_data);

        println!("Creating execution descriptor...");
        let execution_descriptor = ExecutionDescriptor::new();
        execution_descriptor.set_wait_until_completed(true);

        println!("Creating command buffer from command queue...");
        let command_buffer = CommandBuffer::from_command_queue(&command_queue);

        println!("Encoding graph to command buffer...");
        graph.encode_to_command_buffer_with_results(
            &command_buffer,
            &feeds,
            None,
            &results,
            Some(&execution_descriptor),
        );

        println!("Committing command buffer and waiting for completion...");
        command_buffer.commit();
        command_buffer.wait_until_completed();

        println!("Reading results from output buffer...");

        let result_ptr = c_buffer.contents() as *const f32;
        let result_slice =
            std::slice::from_raw_parts(result_ptr, shape_dimensions[0] * shape_dimensions[1]);

        println!("\nResult Matrix (A + B):");
        print_matrix(result_slice, shape_dimensions[0], shape_dimensions[1]);

        println!("\nVerifying results:");

        let mut all_correct = true;
        for i in 0..shape_dimensions[0] * shape_dimensions[1] {
            let expected = matrix_a[i] + matrix_b[i];
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
            println!("❌ Some results are incorrect!");
        }
    }
}

fn print_matrix(data: &[f32], rows: usize, cols: usize) {
    for row in 0..rows {
        let row_slice = &data[row * cols..(row + 1) * cols];
        println!("  {:?}", row_slice);
    }
}
