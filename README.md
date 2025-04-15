# mpsgraph-rs

Idiomatic Rust bindings for Apple's Metal Performance Shaders Graph API.

## Workspace Structure

This repository is organized as a Rust workspace with the following crates:

- **mpsgraph**: Core bindings for Metal Performance Shaders Graph API
- **mpsgraph-tools**: High-level utilities and ergonomic tensor operations API

## Installation

Add one or both crates to your `Cargo.toml`:

```toml
[dependencies]
# Core bindings only
mpsgraph = "0.1.0"

# Optional: High-level tensor operations
mpsgraph-tools = "0.1.0"
```

For development with the latest version:

```toml
[dependencies]
# Core bindings from git
mpsgraph = { git = "https://github.com/computer-graphics-tools/mpsgraph-rs", package = "mpsgraph" }

# Optional: High-level tensor operations from git
mpsgraph-tools = { git = "https://github.com/computer-graphics-tools/mpsgraph-rs", package = "mpsgraph-tools" }
```

## Dependencies

- **objc2** (0.6.0): Safe Rust bindings to Objective-C
- **objc2-foundation** (0.3.0): Rust bindings for Apple's Foundation framework
- **metal** (0.32.0): Rust bindings for Apple's Metal API
- **bitflags** (2.9.0): Macro for generating bitflag structures
- **foreign-types** (0.5): FFI type handling utilities
- **block** (0.1.6): Support for Objective-C blocks

## Platform requirements:

- macOS 13.0+ (or other Apple platform with Metal support)
- Metal-supporting GPU
- Rust 1.85+

## Examples

### Core MPSGraph Examples

Run an example from the core mpsgraph crate:

```bash
cargo run -p mpsgraph --example simple_compile
```

Available examples:

- matmul: Matrix multiplication using MPSGraph
- simple_compile: Simple graph compilation and execution
- type_test: Test of data type conversions
- callback_test: Testing callback functionality

### Tensor Operations Examples

Run an example from the mpsgraph-tools crate:

```bash
cargo run -p mpsgraph-tools --example tensor_ops
```

This example demonstrates:

- Operator overloading for tensor arithmetic
- Functional-style tensor operations
- Activation functions and other neural network operations
- Tensor creation utilities

## Features

### mpsgraph

- **link**: Links against MetalPerformanceShadersGraph.framework (enabled by default)

## Building

```bash
# Build all crates
cargo build

# Build and run examples
cargo run -p mpsgraph --example simple_compile
cargo run -p mpsgraph-tools --example tensor_ops

# Run tests
cargo test -p mpsgraph
cargo test -p mpsgraph-tools
```

## License

Licensed under MIT license.
