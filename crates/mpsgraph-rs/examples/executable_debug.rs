use metal::{Device as MetalDevice, MTLResourceOptions};
use mpsgraph::{
    CommandBuffer, CompilationDescriptor, DataType, Device, ExecutableExecutionDescriptor,
    ExecutionDescriptor, Graph, GraphActivationOps, Shape, ShapedType, TensorData,
};
use objc2::rc::Retained;
use objc2_foundation::NSNumber;
use std::{collections::HashMap, env, fs::File, io::Write};

/// Debug example for MPSGraphExecutable's encode_to_command_buffer function
///
/// This creates a simpler test case with logging to help pinpoint where the crash occurs
fn main() {
    // Setup debug logging
    let debug_log = File::create("executable_debug.log").unwrap();
    let mut debug_log = std::io::BufWriter::new(debug_log);

    writeln!(debug_log, "=== MPSGraphExecutable Debug Log ===\n").unwrap();
    writeln!(debug_log, "Starting debug session...").unwrap();
    debug_log.flush().unwrap();

    // Set up for debugging with LLDB if needed
    let enable_lldb_wait = env::var("DEBUG_WAIT").unwrap_or_default() == "1";
    if enable_lldb_wait {
        println!("Process ID: {}", std::process::id());
        println!("Attach debugger now and press Enter to continue...");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
    }

    writeln!(debug_log, "Creating Metal device...").unwrap();
    debug_log.flush().unwrap();

    // Get the default Metal device
    let metal_device = MetalDevice::system_default().expect("No Metal device found");
    writeln!(debug_log, "Using device: {}", metal_device.name()).unwrap();
    debug_log.flush().unwrap();

    // Create a simpler graph for testing
    writeln!(debug_log, "Creating simple graph...").unwrap();
    debug_log.flush().unwrap();

    let graph = Graph::new();

    // Simple 2x2 shape for debugging
    let shape_dimensions = [2, 2];
    writeln!(debug_log, "Creating shape: {:?}", shape_dimensions).unwrap();
    debug_log.flush().unwrap();

    // Create shape
    let numbers = [
        NSNumber::new_usize(shape_dimensions[0]),
        NSNumber::new_usize(shape_dimensions[1]),
    ];
    let number_refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();
    let shape = Shape::from_slice(&number_refs);

    // Create simple tensors and operations
    writeln!(debug_log, "Creating placeholder tensors...").unwrap();
    debug_log.flush().unwrap();

    let a = graph.placeholder(DataType::Float32, &shape, Some("a"));
    let b = graph.placeholder(DataType::Float32, &shape, Some("b"));

    writeln!(debug_log, "Creating operations...").unwrap();
    debug_log.flush().unwrap();

    // Just add the tensors - keeping it simple
    let c = graph.add(&a, &b, Some("c"));

    // Compile the graph
    writeln!(debug_log, "Compiling graph...").unwrap();
    debug_log.flush().unwrap();

    let comp_desc = CompilationDescriptor::new();
    let mps_device = Device::new();

    let shaped_type = ShapedType::new(&shape, DataType::Float32);
    let mut feed_types = HashMap::new();
    feed_types.insert(&a, &shaped_type);
    feed_types.insert(&b, &shaped_type);

    writeln!(debug_log, "Calling graph.compile()...").unwrap();
    debug_log.flush().unwrap();

    let executable = graph.compile(&mps_device, &feed_types, &[&c], Some(&comp_desc));

    writeln!(debug_log, "Graph compiled successfully.").unwrap();
    debug_log.flush().unwrap();

    // Create simple input data
    writeln!(debug_log, "Preparing data...").unwrap();
    debug_log.flush().unwrap();

    let a_data = [1.0f32, 2.0, 3.0, 4.0];
    let b_data = [0.5f32, 0.5, 0.5, 0.5];

    // Calculate buffer size
    let buffer_size =
        (shape_dimensions[0] * shape_dimensions[1] * std::mem::size_of::<f32>()) as u64;

    writeln!(debug_log, "Creating Metal buffers...").unwrap();
    debug_log.flush().unwrap();

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

    let c_buffer = metal_device.new_buffer(buffer_size, MTLResourceOptions::StorageModeShared);

    writeln!(debug_log, "Creating TensorData...").unwrap();
    debug_log.flush().unwrap();

    // Convert dimensions
    let shape_i64: Vec<i64> = shape_dimensions.iter().map(|&dim| dim as i64).collect();

    let a_tensor_data = TensorData::from_buffer(&a_buffer, &shape_i64, DataType::Float32);
    let b_tensor_data = TensorData::from_buffer(&b_buffer, &shape_i64, DataType::Float32);
    let c_tensor_data = TensorData::from_buffer(&c_buffer, &shape_i64, DataType::Float32);

    writeln!(debug_log, "Setting up command execution...").unwrap();
    debug_log.flush().unwrap();

    // Create command queue
    let command_queue = metal_device.new_command_queue();

    // Prepare inputs and outputs arrays
    let inputs = [&a_tensor_data, &b_tensor_data];
    let outputs = [&c_tensor_data];

    writeln!(debug_log, "About to call encode_to_command_buffer...").unwrap();
    debug_log.flush().unwrap();

    // --- Direct Graph Execution (Known Working Method) ---
    println!("Testing direct graph execution (known working method)...");
    writeln!(debug_log, "Testing direct graph execution first...").unwrap();
    debug_log.flush().unwrap();

    // Create a new command buffer for the direct execution test
    let command_buffer_direct = CommandBuffer::from_command_queue(&command_queue);

    // Set up dictionaries for direct approach
    let mut feeds = HashMap::new();
    feeds.insert(&a, &a_tensor_data);
    feeds.insert(&b, &b_tensor_data);

    let mut results = HashMap::new();
    results.insert(&c, &c_tensor_data);

    // Use ExecutionDescriptor for graph encoding
    let graph_exec_desc = ExecutionDescriptor::new();
    graph_exec_desc.set_wait_until_completed(true);

    // Execute directly
    graph.encode_to_command_buffer_with_results(
        &command_buffer_direct,
        &feeds,
        None,
        &results,
        Some(&graph_exec_desc),
    );

    // Commit and wait
    command_buffer_direct.commit();
    command_buffer_direct.wait_until_completed();

    writeln!(debug_log, "Direct graph execution completed successfully.").unwrap();
    debug_log.flush().unwrap();
    println!("✅ Direct graph execution completed successfully.");

    // --- Executable Encoding Attempt (Known Unstable) ---
    println!("\nSetting up for executable.encode_to_command_buffer (known unstable)... ");

    // If you want to attach LLDB right before the problematic call:
    if enable_lldb_wait {
        println!("Attach debugger before encode_to_command_buffer call and press Enter...");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
    }

    // Create a command buffer for the executable encoding attempt
    let command_buffer_exec = CommandBuffer::from_command_queue(&command_queue);
    let exec_desc_for_exec = ExecutableExecutionDescriptor::new();
    exec_desc_for_exec.set_wait_until_completed(true);

    println!("Attempting executable.encode_to_command_buffer (EXPECTED TO CRASH)... ");
    writeln!(
        debug_log,
        "Attempting executable.encode_to_command_buffer..."
    )
    .unwrap();
    debug_log.flush().unwrap();

    // This call is known to cause segmentation faults.
    let _result = executable.encode_to_command_buffer(
        &command_buffer_exec,
        &inputs,
        Some(&outputs),
        Some(&exec_desc_for_exec),
    );

    // Code below this point likely won't execute due to the crash above.
    println!("⚠️ If you see this, the encode_to_command_buffer call somehow didn't crash!");
    writeln!(
        debug_log,
        "encode_to_command_buffer call completed without immediate crash."
    )
    .unwrap();
    debug_log.flush().unwrap();

    command_buffer_exec.commit();
    command_buffer_exec.wait_until_completed();

    println!("✅ Executable encoding and execution completed (UNEXPECTED).");
    writeln!(
        debug_log,
        "Executable command buffer executed (UNEXPECTED)."
    )
    .unwrap();
    debug_log.flush().unwrap();

    println!("\nDebugging completed. See executable_debug.log for details.");
}
