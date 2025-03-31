# mpsgraph-rs

A Rust wrapper for Apple's MetalPerformanceShadersGraph (MPSGraph) API, enabling high-performance, GPU-accelerated machine learning and numerical computing on Apple platforms.

## Features

- **Complete API Coverage**: Comprehensive bindings to MetalPerformanceShadersGraph
- **Safe Memory Management**: Proper Rust ownership semantics with automatic resource cleanup
- **Efficient Graph Execution**: Synchronous and asynchronous execution options
- **Type Safety**: Strong typing with Rust's type system
- **Tensor Operations**: Full suite of tensor operations for numerical computing and machine learning

## Requirements

- macOS, iOS, tvOS or other Apple platform with Metal support
- Rust 1.58+

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
mpsgraph = "0.1.0"
```

For development with the latest version:

```toml
[dependencies]
mpsgraph = { git = "https://github.com/computer-graphics-tools/mpsgraph-rs", package = "mpsgraph" }
```

## Dependencies

This crate depends on:

- **objc2** (0.6.0): Safe Rust bindings to Objective-C
- **objc2-foundation** (0.3.0): Rust bindings for Apple's Foundation framework
- **metal** (0.32.0): Rust bindings for Apple's Metal API
- **bitflags** (2.9.0): Macro for generating bitflag structures
- **foreign-types** (0.5): FFI type handling utilities
- **block** (0.1.6): Support for Objective-C blocks
- **rand** (0.9.0): Random number generation utilities

The crate also requires linking against:

- MetalPerformanceShaders.framework
- Metal.framework
- Foundation.framework

## Example

```rust
use mpsgraph::{Graph, MPSShapeDescriptor, MPSDataType};
use metal::{Device, MTLResourceOptions};
use std::collections::HashMap;

fn main() {
    // Get the Metal device
    let device = Device::system_default().expect("No Metal device found");
    
    // Create a graph
    let graph = Graph::new().expect("Failed to create graph");
    
    // Create input tensors
    let shape = MPSShapeDescriptor::new(vec![2, 3], MPSDataType::Float32);
    let x = graph.placeholder(&shape, Some("x"));
    let y = graph.placeholder(&shape, Some("y"));
    
    // Define the computation: z = x + y
    let z = graph.add(&x, &y, Some("z"));
    
    // Create input data
    let x_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
    let y_data = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]; // 2x3 matrix
    
    // Create Metal buffers
    let buffer_size = (6 * std::mem::size_of::<f32>()) as u64;
    let x_buffer = device.new_buffer_with_data(
        x_data.as_ptr() as *const _, 
        buffer_size, 
        MTLResourceOptions::StorageModeShared
    );
    let y_buffer = device.new_buffer_with_data(
        y_data.as_ptr() as *const _, 
        buffer_size, 
        MTLResourceOptions::StorageModeShared
    );
    
    // Create feed dictionary
    let mut feed_dict = HashMap::new();
    feed_dict.insert(&x, x_buffer.deref());
    feed_dict.insert(&y, y_buffer.deref());
    
    // Run the graph
    let results = graph.run(&device, feed_dict, &[&z]);
    
    // Process results
    unsafe {
        let result_ptr = results[0].contents() as *const f32;
        let result_values = std::slice::from_raw_parts(result_ptr, 6);
        println!("Result: {:?}", result_values);
        // Outputs: [8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
    }
}
```

## Additional Features

- Matrix multiplication and other linear algebra operations
- Activation functions (ReLU, sigmoid, tanh, etc.)
- Reduction operations (sum, mean, max, min)
- Tensor reshaping and transposition
- Graph compilation for repeated execution

## License

Licensed under the MIT License.
