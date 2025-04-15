# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands
- Build all crates: `cargo build`
- Run all tests: `cargo test`
- Run tests for specific crate: `cargo test -p mpsgraph`
- Run a single test: `cargo test test_name`
- Run examples: `cargo run -p mpsgraph --example simple_compile`

## Workflow Rules
- Always compile and run tests after making code changes: `cargo build && cargo test`
- Check that changes work on both x86_64 and aarch64 macOS targets

## API Renaming
This codebase is undergoing a transition to more idiomatic Rust type naming:
- All types are being renamed to remove the MPS prefix (e.g., `MPSGraphDevice` → `Device`)
- When working with this codebase, update types to follow this convention

### Progress on Renaming
- Completed:
  - `MPSGraphDevice` → `Device`
  - `MPSShape` → `Shape`
  - `MPSGraph` → `Graph`
  - `MPSTensorDataScalar` → `TensorDataScalar`

- Files Updated:
  - device.rs: `MPSGraphDevice` → `Device`
  - shape.rs: `MPSShape` → `Shape`
  - graph.rs: `MPSGraph` → `Graph`, `MPSTensorDataScalar` → `TensorDataScalar`
  - operation.rs: Updated to use `Graph`
  - tensor.rs: Updated to use `Shape`
  - tensor_data.rs: Updated to use `Shape`
  - data_types.rs: Updated to use `Shape`
  - activation_ops.rs: Updated to use `Graph`
  - arithmetic_ops.rs: Updated to use `Graph`
  - convolution_transpose_ops.rs: Updated to use `Graph` and `Shape`
  - depthwise_convolution_ops.rs: Updated to use `Graph` and `Shape`
  - gradient_ops.rs: Updated to use `Graph`
  - loss_ops.rs: Updated to use `Graph` and documentation examples
  - matrix_inverse_ops.rs: Updated to use `Graph`
  - normalization_ops.rs: Updated to use `Graph`
  - optimizer_ops.rs: Updated to use `Graph` and documentation examples
  - pooling_ops.rs: Updated to use `Graph`
  - random_ops.rs: Updated to use `Graph` and `Shape`
  - reduction_ops.rs: Updated to use `Graph`
  - resize_ops.rs: Updated to use `Graph` and `Shape`
  - call_ops.rs: Updated to use `Graph`
  - control_flow_ops.rs: Updated to use `Graph` and `Device` in tests
  - cumulative_ops.rs: Updated to use `Graph`
  - fourier_transform_ops.rs: Updated to use `Graph`
  - gather_ops.rs: Updated to use `Graph`
  - im2col_ops.rs: Updated to use `Graph` and `Shape`
  - linear_algebra_ops.rs: Updated to use `Graph`
  - memory_ops.rs: Updated to use `Graph` and `Shape`
  - non_maximum_suppression_ops.rs: Updated to use `Graph`
  - non_zero_ops.rs: Updated to use `Graph`
  - one_hot_ops.rs: Updated to use `Graph`
  - quantization_ops.rs: Updated to use `Graph`
  - rnn_ops.rs: Updated to use `Graph`
  - sample_grid_ops.rs: Updated to use `Graph`
  - stencil_ops.rs: Updated to use `Graph` and `Shape`
  - tensor_shape_ops.rs: Updated to use `Graph`
  - top_k_ops.rs: Updated to use `Graph`
  - sparse_ops.rs: Updated to use `Graph` and `Shape` 
  - sort_ops.rs: Updated to use `Graph`
  - scatter_nd_ops.rs: Updated to use `Graph` and `Shape`

- All Tasks Completed:
  - Updated all type names to remove the MPS prefix
  - Updated API documentation to reflect the new type names

The renaming is complete and the codebase compiles successfully. All source files, examples, tests, and documentation have been updated to use the new type names.

The following type renaming is now complete:
  - `MPSGraphDevice` → `Device`
  - `MPSShape` → `Shape`
  - `MPSGraph` → `Graph`
  - `MPSTensorDataScalar` → `TensorDataScalar`
  - `MPSGraphTensor` → `Tensor`
  - `MPSGraphTensorData` → `TensorData`
  - `MPSGraphCompilationDescriptor` → `CompilationDescriptor`
  - `MPSGraphExecutionDescriptor` → `ExecutionDescriptor`

### Renaming Steps
For each type to be renamed:
1. Update the struct/enum/type definition
2. Update all implementations and methods 
3. Update re-exports in lib.rs (both direct exports and in prelude)
4. Update all imports of this type in other files
5. Update all uses of the type in method signatures and return types
6. Update Debug implementations (change debug output strings)
7. Build and fix any remaining issues

## Code Style Guidelines
- Use 4-space indentation
- Follow Rust 2021 edition conventions
- Types: Use simple `PascalCase` names without MPS prefix (e.g., `Device` instead of `MPSGraphDevice`)
- Methods/variables: Use `snake_case`
- Error handling: Use `Result<T, E>` with descriptive error types
- Imports: Standard library first, then external crates, then internal modules
- Memory management: Implement `Drop` trait for proper Objective-C cleanup
- Documentation: Add doc comments for all public APIs
- Testing: Name tests as `test_functionality_being_tested`
- Type safety: Use appropriate Rust types to wrap Objective-C objects
- Follow idiomatic Rust FFI patterns for Objective-C interop
- Type access is via namespace: `mpsgraph::Device` (not `mpsgraph::MPSGraphDevice`)