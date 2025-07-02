# mpsgraph-rs

Modern Rust bindings for Apple's Metal Performance Shaders Graph framework (MPSGraph).

## Overview

This crate provides Rust bindings for the Metal Performance Shaders Graph framework, a high-level graph-based API for defining neural network models that run on the GPU. It uses:

- Modern `extern_class!` pattern from objc2 for class definitions
- Automatic memory management with `Retained<T>` instead of manual retain/release
- More idiomatic Rust API with improved type safety
- Better Debug and display support via standard traits
- Cleaner FFI boundary between Rust and Objective-C

## Status

This crate is stable and ready for production use. It is the result of a completed migration effort from the original AnyObject-based implementation to a modern Objective-C integration approach. See the [Migration Progress](../../../MIGRATION-PROGRESS.md) document for details.

## Features

- Create and manipulate graph-based neural network models
- Define tensor operations with strong typing
- Execute models on Metal-compatible GPUs
- Automatic memory management for Objective-C objects

## Requirements

- macOS 10.15 or newer
- Rust 1.56 or newer
- Metal-compatible GPU

## Usage Example

```rust
use mpsgraph::{
    DataType, Device, ExecutionDescriptor, Graph, ShapeHelper, TensorData
};
use std::collections::HashMap;
use objc2::rc::Retained;

fn main() {
    // Create a Metal device and MPS graph
    let metal_device = metal::Device::system_default().expect("No Metal device found");
    let device = Device::with_device(&metal_device);
    let graph = Graph::new();
    
    // Create a 2x2 matrix shape and tensors
    let matrix_shape = ShapeHelper::matrix(2, 2);
    let input_a = graph.placeholder(DataType::Float32, &matrix_shape).unwrap();
    let input_b = graph.placeholder(DataType::Float32, &matrix_shape).unwrap();
    
    // Create a matrix multiplication operation
    let output = graph.matmul(
        &input_a, &input_b, 
        false, false, 
        None
    ).unwrap();
    
    // Create input tensor data
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_data = vec![5.0f32, 6.0, 7.0, 8.0];
    
    let a_tensor_data = TensorData::with_bytes(
        &a_data,
        &matrix_shape,
        DataType::Float32,
    ).unwrap();
    
    let b_tensor_data = TensorData::with_bytes(
        &b_data,
        &matrix_shape,
        DataType::Float32,
    ).unwrap();
    
    // Set up execution
    let exec_desc = ExecutionDescriptor::new();
    exec_desc.prefer_synchronous_execution();
    
    // Create feeds dictionary and run graph
    let mut feeds = HashMap::new();
    feeds.insert(&*input_a, &*a_tensor_data);
    feeds.insert(&*input_b, &*b_tensor_data);
    
    let results = graph.run_with_feeds_and_descriptor(
        &feeds,
        &[&*output],
        Some(&exec_desc)
    );
    
    // Process results
    if let Some(result_data) = results.get(&output) {
        if let Some(result_array) = result_data.bytes_as::<f32>() {
            println!("Result matrix:");
            for i in 0..2 {
                for j in 0..2 {
                    print!("{} ", result_array[i * 2 + j]);
                }
                println!();
            }
        }
    }
}
```

## Key Features

The main features of this crate are:

1. **Memory Management**: Uses `Retained<T>` for automatic memory management instead of raw pointers and manual retain/release
2. **Type Safety**: Strong type safety through the use of generics and proper Objective-C class wrappers
3. **API Design**: Idiomatic Rust API with methods returning `Option<T>` and `Result<T, E>` where appropriate
4. **Naming**: Uses simpler type names without the MPS prefix (e.g., `Device` instead of `MPSGraphDevice`)
5. **Error Handling**: Comprehensive error handling with better Rust integration
6. **Block Support**: Full support for Objective-C blocks with proper type safety and memory management

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## License

Licensed under MIT license.
