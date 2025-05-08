use metal::{Buffer, Device, MTLResourceOptions};
use mpsgraph::{DataType, ExecutionDescriptor, Graph, Shape, TensorData};
use objc2::rc::Retained;
use std::collections::HashMap;

// A struct that pairs an MTLBuffer with its TensorData
#[derive(Clone)]
struct TensorBuffer {
    buffer: Buffer,
    tensor_data: Retained<TensorData>,
}

impl TensorBuffer {
    // Create a new TensorBuffer from a vector of f32 data
    fn new(device: &Device, data: &[f32], shape_dims: &[usize], data_type: DataType) -> Self {
        // Calculate size in bytes
        let byte_length = data.len() * std::mem::size_of::<f32>();

        // Create MTLBuffer with storage mode shared for CPU/GPU access
        let buffer = device.new_buffer_with_data(
            data.as_ptr() as *const _,
            byte_length as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Convert dimensions to i64 and create a proper Shape
        let shape_i64: Vec<i64> = shape_dims.iter().map(|&dim| dim as i64).collect();
        let shape = Shape::from_dimensions(&shape_i64);

        // Create tensor data that references this buffer
        let tensor_data = TensorData::from_buffer(&buffer, &shape, data_type);

        Self {
            buffer,
            tensor_data,
        }
    }

    // Create an empty TensorBuffer for results
    fn new_empty(device: &Device, size: usize, shape_dims: &[usize], data_type: DataType) -> Self {
        // Calculate size in bytes
        let byte_length = size * std::mem::size_of::<f32>();

        // Create empty MTLBuffer with storage mode shared
        let buffer = device.new_buffer(byte_length as u64, MTLResourceOptions::StorageModeShared);

        // Convert dimensions to i64 and create a proper Shape
        let shape_i64: Vec<i64> = shape_dims.iter().map(|&dim| dim as i64).collect();
        let shape = Shape::from_dimensions(&shape_i64);

        // Create tensor data that references this buffer
        let tensor_data = TensorData::from_buffer(&buffer, &shape, data_type);

        Self {
            buffer,
            tensor_data,
        }
    }

    // Read data directly from the buffer
    fn get_f32_data(&self, count: usize) -> Vec<f32> {
        let ptr = self.buffer.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, count).to_vec() }
    }
}

fn main() {
    // Get Metal device
    let device = Device::system_default().expect("No Metal device found");

    // Input dimensions (use usize for array indexing)
    let m: usize = 2; // Matrix A rows
    let k: usize = 3; // Matrix A cols / Matrix B rows
    let n: usize = 2; // Matrix B cols

    // Create input data
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
    let b_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2 matrix

    // Create a graph
    let graph = Graph::new();

    // Create input placeholders
    let a_shape = [m, k]; // usize array for TensorBuffer
    let b_shape = [k, n];
    let result_shape = [m, n];

    // Create shapes for the tensors (convert usize to i64 for Shape)
    let a_tensor_shape = Shape::matrix(m as i64, k as i64);
    let b_tensor_shape = Shape::matrix(k as i64, n as i64);

    let a = graph.placeholder(DataType::Float32, &a_tensor_shape, None);
    let b = graph.placeholder(DataType::Float32, &b_tensor_shape, None);

    // Perform matrix multiplication: A * B
    // The matmul method takes transpose flags for both inputs
    let result = graph.matmul(&a, &b, false, false, None);

    // Create TensorBuffers for inputs
    let a_tensor = TensorBuffer::new(&device, &a_data, &a_shape, DataType::Float32);
    let b_tensor = TensorBuffer::new(&device, &b_data, &b_shape, DataType::Float32);

    // Create TensorBuffer for output
    let result_size = m * n;
    let result_tensor =
        TensorBuffer::new_empty(&device, result_size, &result_shape, DataType::Float32);

    // Create command queue
    let command_queue = device.new_command_queue();

    // Prepare feeds
    let mut feeds = HashMap::new();
    feeds.insert(&a, &a_tensor.tensor_data);
    feeds.insert(&b, &b_tensor.tensor_data);

    // Output tensor
    let mut results = HashMap::new();
    results.insert(&result, &result_tensor.tensor_data);

    println!("Executing matrix multiplication...");

    // Create and use a command buffer
    let command_buffer = mpsgraph::CommandBuffer::from_command_queue(&command_queue);
    let execution_descriptor = ExecutionDescriptor::new();
    execution_descriptor.set_wait_until_completed(true);

    // Encode graph to command buffer
    graph.encode_to_command_buffer_with_results(
        &command_buffer,
        &feeds,
        None, // No specific target tensors
        &results,
        Some(&execution_descriptor),
    );

    // Commit and wait for completion
    let mtl_command_buffer = command_buffer.command_buffer();
    mtl_command_buffer.commit();
    mtl_command_buffer.wait_until_completed();

    // Wait for GPU work to complete
    println!("Waiting for GPU execution to complete...");
    std::thread::sleep(std::time::Duration::from_millis(500));

    // Now read the result directly from the MTLBuffer
    println!("Reading result from buffer...");
    let result_floats = result_tensor.get_f32_data(result_size);
    println!("Result data: {:?}", result_floats);

    // Display the matrices and result
    println!("Matrix A ({m}x{k}):");
    for i in 0..m {
        let row: Vec<f32> = a_data[i * k..(i + 1) * k].to_vec();
        println!("  {:?}", row);
    }

    println!("Matrix B ({k}x{n}):");
    for i in 0..k {
        let row: Vec<f32> = b_data[i * n..(i + 1) * n].to_vec();
        println!("  {:?}", row);
    }

    println!("Result ({m}x{n}):");
    for i in 0..m {
        let row: Vec<f32> = result_floats[i * n..(i + 1) * n].to_vec();
        println!("  {:?}", row);
    }

    // Expected result for matrix multiplication
    println!("\nVerifying result:");
    let expected_result = vec![
        // First row: 1*1 + 2*3 + 3*5, 1*2 + 2*4 + 3*6
        1.0 * 1.0 + 2.0 * 3.0 + 3.0 * 5.0,
        1.0 * 2.0 + 2.0 * 4.0 + 3.0 * 6.0,
        // Second row: 4*1 + 5*3 + 6*5, 4*2 + 5*4 + 6*6
        4.0 * 1.0 + 5.0 * 3.0 + 6.0 * 5.0,
        4.0 * 2.0 + 5.0 * 4.0 + 6.0 * 6.0,
    ];
    println!("Expected: {:?}", expected_result);

    // Check if results match
    let result_correct = result_floats
        .iter()
        .zip(expected_result.iter())
        .all(|(a, b)| (a - b).abs() < 0.00001);

    if result_correct {
        println!("✅ Result is correct!");
    } else {
        println!("❌ Result doesn't match expected values.");
    }
}
