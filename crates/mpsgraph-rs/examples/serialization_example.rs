use metal::{Device as MetalDevice, MTLResourceOptions};
// use mpsgraph::device::CustomDefault as DeviceCustomDefault; // Removed
use mpsgraph::{
    // CommandBuffer, CommandBufferStatus, // Removed
    CompilationDescriptor,
    DataType,
    Device,
    Executable,
    ExecutableExecutionDescriptor,
    // ExecutionDescriptor, // Removed
    SerializationDescriptor,
    Shape,
    ShapedType,
    TensorData,
};
use objc2_foundation::NSArray; // Added import

use std::{collections::HashMap, env, fs, path::PathBuf}; // Added PathBuf back

fn main() {
    println!("MPSGraph Serialization/Deserialization Example");
    println!("=============================================");

    // --- 1. Graph Definition ---
    let graph = mpsgraph::Graph::new();
    let shape_dims = [2, 2];
    let shape = Shape::matrix(shape_dims[0] as i64, shape_dims[1] as i64);
    let a = graph.placeholder(DataType::Float32, &shape, Some("a"));
    let b = graph.placeholder(DataType::Float32, &shape, Some("b"));
    let c = graph.add(&a, &b, Some("c")); // c = a + b
    println!("Graph defined: C = A + B");

    // --- 2. Compilation ---
    println!("\n--- Compilation ---");
    let metal_device = MetalDevice::system_default().expect("No Metal device found");
    println!("Using Metal device: {}", metal_device.name());
    let mps_device = Device::with_device(&metal_device); // Use with_device

    // Feeds require ShapedType, not TensorData for compilation
    let shaped_type = ShapedType::new(&shape, DataType::Float32);
    let mut feeds = HashMap::new();
    feeds.insert(&a, &shaped_type);
    feeds.insert(&b, &shaped_type);

    let targets = [&c];
    let comp_desc = CompilationDescriptor::new();
    // Set compilation options if needed: comp_desc.set_optimization_level(...)

    println!("Calling graph.compile()..."); // Added print
    let executable = graph.compile(&mps_device, &feeds, &targets, Some(&comp_desc));
    println!("Graph compiled successfully.");

    // --- 3. Serialization ---
    println!("\n--- Serialization ---");

    // ** Explicitly specialize before serializing **
    println!("Specializing executable...");
    let input_types_vec: Vec<&mpsgraph::Type> = vec![shaped_type.as_ref(), shaped_type.as_ref()];
    let input_types_array = NSArray::from_slice(&input_types_vec);
    executable.specialize_with_device(Some(&mps_device), &input_types_array, Some(&comp_desc));
    println!("Specialization complete.");

    let mut temp_file_path = env::temp_dir();
    temp_file_path.push("example_graph.mpsgraphpkg"); // Use different name from test
    let temp_file_path_str = temp_file_path.to_str().unwrap();
    let _ = fs::remove_file(&temp_file_path); // Clean up previous run
    println!("Serializing executable to: {}", temp_file_path_str);
    let ser_desc = SerializationDescriptor::new();
    ser_desc.set_deployment_platform(mpsgraph::DeploymentPlatform::MacOS); // Set platform
    executable.serialize_to_url(temp_file_path_str, &ser_desc);
    println!("Serialization successful.");

    // --- 4. Deserialization ---
    println!("\n--- Deserialization ---");
    println!("Deserializing executable from: {}", temp_file_path_str);
    // Pass the same compilation descriptor used for the original compile
    let deserialized_executable_opt =
        Executable::from_serialized_package(temp_file_path_str, Some(&comp_desc));
    assert!(
        deserialized_executable_opt.is_some(),
        "Failed to deserialize executable"
    );
    let deserialized_executable = deserialized_executable_opt.unwrap(); // Keep variable
    println!("Deserialization successful.");

    // --- 5. Execution (Requires additions to Executable bindings) ---
    println!("\n--- Execution ---");

    // Get feed/target tensors from the deserialized executable
    let feed_tensors = deserialized_executable
        .feed_tensors()
        .expect("Failed to get feed tensors from deserialized executable");
    let target_tensors = deserialized_executable
        .target_tensors()
        .expect("Failed to get target tensors from deserialized executable");
    assert_eq!(feed_tensors.len(), 2, "Incorrect number of feed tensors");
    assert_eq!(
        target_tensors.len(),
        1,
        "Incorrect number of target tensors"
    );
    println!(
        "Retrieved {} feed tensors and {} target tensors from executable.",
        feed_tensors.len(),
        target_tensors.len()
    );
    // Note: Relying on order here. A more robust example might check tensor names.
    // let _input_tensor_a = &feed_tensors[0]; // Variables are unused currently
    // let _input_tensor_b = &feed_tensors[1];
    // let _output_tensor_c = &target_tensors[0];

    // Create input data buffers and TensorData
    let a_data_vec: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b_data_vec: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
    let result_buffer_size = (shape_dims[0] * shape_dims[1] * std::mem::size_of::<f32>()) as u64;

    let a_buffer = metal_device.new_buffer_with_data(
        a_data_vec.as_ptr() as *const _,
        result_buffer_size, // Use same size calculation
        MTLResourceOptions::StorageModeShared,
    );
    let b_buffer = metal_device.new_buffer_with_data(
        b_data_vec.as_ptr() as *const _,
        result_buffer_size, // Use same size calculation
        MTLResourceOptions::StorageModeShared,
    );
    let result_buffer =
        metal_device.new_buffer(result_buffer_size, MTLResourceOptions::StorageModeShared);

    let shape_i64: Vec<i64> = shape_dims.iter().map(|&d| d as i64).collect();
    let a_tensor_data = TensorData::from_buffer(&a_buffer, &shape_i64, DataType::Float32);
    let b_tensor_data = TensorData::from_buffer(&b_buffer, &shape_i64, DataType::Float32);
    let result_tensor_data = TensorData::from_buffer(&result_buffer, &shape_i64, DataType::Float32);

    // Map executable's tensors to data
    // IMPORTANT: The order must match the order returned by feed_tensors/target_tensors
    let inputs_vec = vec![&a_tensor_data, &b_tensor_data];
    let results_vec = vec![&result_tensor_data];

    // Prepare for execution
    let command_queue = metal_device.new_command_queue();
    let command_buffer = mpsgraph::CommandBuffer::from_command_queue(&command_queue); // Use mpsgraph::CommandBuffer
    let exec_desc = ExecutableExecutionDescriptor::new();
    exec_desc.set_wait_until_completed(true); // Run synchronously for simplicity

    println!("Executing deserialized graph...");
    // Call the execution method
    let _execution_results = deserialized_executable.encode_to_command_buffer(
        &command_buffer,
        &inputs_vec,
        Some(&results_vec), // Provide results buffer
        Some(&exec_desc),
    );
    // We provided results buffer, so ignore return value for now.

    command_buffer.commit();
    command_buffer.wait_until_completed();
    println!("Execution complete.");

    // --- 6. Verification ---
    println!("\n--- Verification ---");
    println!("Verifying results...");
    let result_ptr = result_buffer.contents() as *const f32;
    let result_slice =
        unsafe { std::slice::from_raw_parts(result_ptr, shape_dims[0] * shape_dims[1]) };
    let expected_result: Vec<f32> = vec![6.0, 8.0, 10.0, 12.0];
    assert_eq!(
        result_slice,
        &expected_result[..],
        "Result verification failed"
    );
    println!("✅ Results verified successfully!");

    // --- 7. Cleanup ---
    println!("\n--- Cleanup ---");
    println!("Cleaning up temporary file: {}", temp_file_path_str);
    let _ = fs::remove_file(temp_file_path);

    println!("\nExample finished successfully!");
}
