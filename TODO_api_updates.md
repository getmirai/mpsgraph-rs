# API Update TODO List for mpsgraph-rs

This file lists all Rust source files in `crates/mpsgraph-rs/src` that will be analyzed and potentially updated to align with the `Retained<T>` usage pattern (return `Retained<T>`, accept `&T`).

## Files to Process:

- `crates/mpsgraph-rs/src/graph.rs`
- `crates/mpsgraph-rs/src/executable.rs`
- `crates/mpsgraph-rs/src/optimizer_ops.rs`
- `crates/mpsgraph-rs/src/memory_ops.rs`
- `crates/mpsgraph-rs/src/convolution_transpose_ops.rs`
- `crates/mpsgraph-rs/src/sample_grid_ops.rs`
- `crates/mpsgraph-rs/src/top_k_ops.rs`
- `crates/mpsgraph-rs/src/non_maximum_suppression_ops.rs`
- `crates/mpsgraph-rs/src/loss_ops.rs`
- `crates/mpsgraph-rs/src/im2col_ops.rs`
- `crates/mpsgraph-rs/src/call_ops.rs`
- `crates/mpsgraph-rs/src/lib.rs`
- `crates/mpsgraph-rs/src/tensor_data.rs`
- `crates/mpsgraph-rs/src/gradient_ops.rs`
- `crates/mpsgraph-rs/src/command_buffer.rs`
- `crates/mpsgraph-rs/src/data_types.rs`
- `crates/mpsgraph-rs/src/cumulative_ops.rs`
- `crates/mpsgraph-rs/src/control_flow_ops.rs`
- `crates/mpsgraph-rs/src/tensor_shape_ops.rs`
- `crates/mpsgraph-rs/src/one_hot_ops.rs`
- `crates/mpsgraph-rs/src/non_zero_ops.rs`
- `crates/mpsgraph-rs/src/scatter_nd_ops.rs`
- `crates/mpsgraph-rs/src/fourier_transform_ops.rs`
- `crates/mpsgraph-rs/src/tensor.rs`
- `crates/mpsgraph-rs/src/device.rs`
- `crates/mpsgraph-rs/src/depthwise_convolution_ops.rs`
- `crates/mpsgraph-rs/src/convolution_ops.rs`
- `crates/mpsgraph-rs/src/core.rs`
- `crates/mpsgraph-rs/src/arithmetic_ops.rs`
- `crates/mpsgraph-rs/src/activation_ops.rs`
- `crates/mpsgraph-rs/src/gather_ops.rs`
- `crates/mpsgraph-rs/src/utils_old.rs`
- `crates/mpsgraph-rs/src/normalization_ops.rs`
- `crates/mpsgraph-rs/src/linear_algebra_ops.rs`
- `crates/mpsgraph-rs/src/matrix_ops.rs`
- `crates/mpsgraph-rs/src/shape.rs`
- `crates/mpsgraph-rs/src/reduction_ops.rs`
- `crates/mpsgraph-rs/src/pooling_ops.rs`
- `crates/mpsgraph-rs/src/sparse_ops.rs`
- `crates/mpsgraph-rs/src/random_ops.rs`
- `crates/mpsgraph-rs/src/matrix_inverse_ops.rs`
- `crates/mpsgraph-rs/src/resize_ops.rs`
- `crates/mpsgraph-rs/src/quantization_ops.rs`
- `crates/mpsgraph-rs/src/sort_ops.rs`
- `crates/mpsgraph-rs/src/operation.rs`
- `crates/mpsgraph-rs/src/stencil_ops.rs`
- `crates/mpsgraph-rs/src/rnn_ops.rs`

## Files in Subdirectories (to be investigated):

- `crates/mpsgraph-rs/src/tests/executable_tests.rs`
- `crates/mpsgraph-rs/src/tests/mod.rs`
- `crates/mpsgraph-rs/src/tests/data_types_tests.rs`
- `crates/mpsgraph-rs/src/tests/random_ops_tests.rs`
- `crates/mpsgraph-rs/src/tests/sparse_ops_tests.rs`
- `crates/mpsgraph-rs/src/tests/optimizer_ops_tests.rs`
- `crates/mpsgraph-rs/src/tests/rnn_ops_tests.rs`
- `crates/mpsgraph-rs/src/tests/stencil_ops_tests.rs`
- `crates/mpsgraph-rs/src/tests/im2col_ops_tests.rs`
- `crates/mpsgraph-rs/src/tests/device_tests.rs`
- `crates/mpsgraph-rs/src/tests/depthwise_convolution_ops_tests.rs`
- `crates/mpsgraph-rs/src/tests/convolution_ops_tests.rs`
- `crates/mpsgraph-rs/src/tests/pooling_ops_tests.rs`
- `crates/mpsgraph-rs/src/utils/mod.rs`
- `crates/mpsgraph-rs/src/utils/block_wrapper.rs`
- `crates/mpsgraph-rs/src/utils/buffer.rs` 