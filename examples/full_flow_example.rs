// examples/full_flow_example.rs
use metal::{Device as MetalDevice, MTLResourceOptions};
use mpsgraph::{
    CommandBuffer, CompilationDescriptor, DataType, DeploymentPlatform, Device, Executable,
    ExecutableExecutionDescriptor, Graph, SerializationDescriptor, Shape, ShapedType, Tensor,
    TensorData,
};
use objc2::rc::Retained;
use objc2_foundation::NSNumber;
use std::{collections::HashMap, env, fs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("MPSGraph Full Flow Example: Compile -> Serialize -> Deserialize -> Execute");
    println!("=======================================================================");

    // --- 1. Setup ---
    let metal_device = MetalDevice::system_default().expect("No Metal device found");
    println!("Using device: {}", metal_device.name());
    let command_queue = metal_device.new_command_queue();
    let mps_device = Device::new();

    // --- 2. Define Graph ---
    let graph = Graph::new();
    let shape_dimensions = [2, 2];
    let numbers = [
        NSNumber::new_usize(shape_dimensions[0]),
        NSNumber::new_usize(shape_dimensions[1]),
    ];
    let number_refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();
    let shape = Shape::from_slice(&number_refs);

    let a = graph.placeholder(DataType::Float32, &shape, Some("a"));
    let b = graph.placeholder(DataType::Float32, &shape, Some("b"));
    let c = graph.add(&a, &b, Some("c")); // C = A + B
    println!("Graph defined: C = A + B");

    // --- 3. Compile Graph ---
    println!("\n--- Compiling Graph ---");
    let comp_desc = CompilationDescriptor::new();
    let shaped_type = ShapedType::new(&shape, DataType::Float32);
    let mut feed_types_refs = HashMap::new();
    feed_types_refs.insert(a.as_ref(), shaped_type.as_ref());
    feed_types_refs.insert(b.as_ref(), shaped_type.as_ref());

    let executable = graph.compile(
        &mps_device,
        &feed_types_refs,
        &[c.as_ref()],
        Some(&comp_desc),
    );
    println!("Graph compiled successfully.");

    // --- 4. Serialize Executable ---
    println!("\n--- Serializing Executable ---");
    let serialization_desc = SerializationDescriptor::new();
    serialization_desc.set_deployment_platform(DeploymentPlatform::MacOS);

    // Create a path for the package in the examples directory
    let examples_dir = env::current_dir()?;
    // Create a directory for the package
    let package_path = examples_dir.join("output.mpsgraphpackage");
    let package_path_str = package_path.to_str().expect("Failed to create path string");

    // Ensure the directory doesn't already exist from a previous run
    if package_path.exists() {
        fs::remove_dir_all(&package_path).expect("Failed to remove existing package dir");
    }

    // We need to create the directory first for objc2 NSURL to work
    fs::create_dir_all(&package_path).expect("Failed to create package directory");

    println!("Serializing to: {:?}", package_path);
    executable.serialize_to_url(&package_path, &serialization_desc);
    println!("Serialization complete.");

    // Verify package contents (optional)
    if package_path.exists() {
        println!("Package contents:");
        for entry in fs::read_dir(&package_path).unwrap() {
            let entry = entry.unwrap();
            println!("  - {}", entry.file_name().to_string_lossy());
        }
    } else {
        println!("Warning: Serialized package not found at expected path.");
    }

    // --- 5. Deserialize Executable ---
    println!("\n--- Deserializing Executable ---");
    // Note: Using the same compilation descriptor used for the original compilation
    let deserialized_executable_opt =
        Executable::from_serialized_package(&package_path, Some(&comp_desc));

    let deserialized_executable: Retained<Executable>;
    match deserialized_executable_opt {
        Some(exec) => {
            println!("Deserialization successful.");
            deserialized_executable = exec;
        }
        None => {
            println!("Error: Deserialization failed. This might be due to framework limitations (e.g., missing resources.bin).");
            println!("Skipping execution phase.");
            // Clean up the created package
            if package_path.exists() {
                fs::remove_dir_all(&package_path)
                    .expect("Failed to remove package dir after failed deserialization");
            }
            return Ok(());
        }
    }

    // --- 6. Execute Deserialized Executable ---
    println!("\n--- Executing Deserialized Executable ---");

    // Prepare input data and buffers
    let a_data = [1.0f32, 2.0, 3.0, 4.0];
    let b_data = [0.5f32, 0.5, 0.5, 0.5];
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
    let c_buffer = metal_device.new_buffer(buffer_size, MTLResourceOptions::StorageModeShared);

    // Create TensorData
    let shape_i64: Vec<i64> = shape_dimensions.iter().map(|&dim| dim as i64).collect();
    let tensor_shape = Shape::from_dimensions(&shape_i64);
    let a_tensor_data = TensorData::from_buffer(&a_buffer, &tensor_shape, DataType::Float32);
    let b_tensor_data = TensorData::from_buffer(&b_buffer, &tensor_shape, DataType::Float32);
    let c_tensor_data = TensorData::from_buffer(&c_buffer, &tensor_shape, DataType::Float32);
    println!("Input/Output TensorData created.");

    // **Important:** Get tensors from the *deserialized* executable
    let feed_tensors: Vec<Retained<Tensor>> = deserialized_executable
        .feed_tensors()
        .expect("Failed to get feed tensors from deserialized executable");
    let target_tensors: Vec<Retained<Tensor>> = deserialized_executable
        .target_tensors()
        .expect("Failed to get target tensors from deserialized executable");

    // Map TensorData to the deserialized tensors (assuming order or matching by name if possible)
    // Here we assume the order [a, b] for feeds and [c] for targets matches compilation.
    if feed_tensors.len() != 2 || target_tensors.len() != 1 {
        println!(
            "Error: Mismatch in expected number of feed/target tensors after deserialization."
        );
        if package_path.exists() {
            fs::remove_dir_all(&package_path).expect("Failed to remove package dir");
        }
        return Ok(());
    }
    let inputs_for_exec_refs = [a_tensor_data.as_ref(), b_tensor_data.as_ref()];
    let outputs_for_exec_refs = [c_tensor_data.as_ref()];

    // Execute using the deserialized executable
    let command_buffer = CommandBuffer::from_command_queue(&command_queue);
    let exec_desc = ExecutableExecutionDescriptor::new();
    exec_desc.set_wait_until_completed(true);

    println!("Encoding using deserialized executable...");
    let _result = deserialized_executable.encode_to_command_buffer(
        &command_buffer,
        &inputs_for_exec_refs,
        Some(&outputs_for_exec_refs),
        Some(&exec_desc),
    );
    println!("Encode call completed.");

    println!("Committing and waiting...");
    let mtl_command_buffer = command_buffer.command_buffer();
    mtl_command_buffer.commit();
    mtl_command_buffer.wait_until_completed();

    if mtl_command_buffer.status() == metal::MTLCommandBufferStatus::Error {
        println!("*** Error during command buffer execution! ***");
        if package_path.exists() {
            fs::remove_dir_all(&package_path).expect("Failed to remove package dir");
        }
        return Ok(());
    }

    println!("Execution successful!");

    // --- 7. Verify Results ---
    println!("\n--- Verifying Results ---");
    let result_ptr = c_buffer.contents() as *const f32;
    let result_slice = unsafe {
        std::slice::from_raw_parts(result_ptr, shape_dimensions[0] * shape_dimensions[1])
    };

    println!("Result Matrix (A + B):");
    print_matrix(result_slice, shape_dimensions[0], shape_dimensions[1]);

    let mut all_correct = true;
    for i in 0..shape_dimensions[0] * shape_dimensions[1] {
        let expected = a_data[i] + b_data[i];
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

    // --- 8. Cleanup ---
    println!("\n--- Cleaning up ---");
    if package_path.exists() {
        fs::remove_dir_all(&package_path).expect("Failed to remove package dir");
        println!("Removed temporary package: {}", package_path_str);
    }

    Ok(())
}

// Helper function to print matrix
fn print_matrix(data: &[f32], rows: usize, cols: usize) {
    for row in 0..rows {
        let row_slice = &data[row * cols..(row + 1) * cols];
        println!("  {:?}", row_slice);
    }
}
