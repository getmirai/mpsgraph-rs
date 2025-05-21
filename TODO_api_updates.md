# API Update TODO List for mpsgraph-rs

This document tracks the refactoring of the `mpsgraph-rs` crate.

## Phase 1: Initial API Signature Refactoring (Completed for core files)
- Goal: Functions return `Retained<T>`, arguments take `&T` for ObjC types.
- Status: COMPLETED for all library files.

## Phase 2: msg_send! Simplification (Completed for core files)
- Goal: Leverage `msg_send!` macro's automatic memory management for `Retained<T>` return types.
- Status: COMPLETED for all library files.

## Phase 3: Comprehensive Refactoring for Ops Files & Nullability (Current Phase)
- Goal:
    1. Apply Phase 1 (argument types `&ObjCType`) and Phase 2 (`msg_send!` simplification) to all `*_ops.rs` files.
    2. Refine return types from `Option<Retained<T>>` to `Retained<T>` where ObjC headers indicate non-null returns.
- Status: COMPLETED.

Files Processed/Re-verified in Phase 3:
- `crates/mpsgraph-rs/src/activation_ops.rs`
- `crates/mpsgraph-rs/src/arithmetic_ops.rs`
- `crates/mpsgraph-rs/src/call_ops.rs`
- `crates/mpsgraph-rs/src/control_flow_ops.rs`
- `crates/mpsgraph-rs/src/convolution_ops.rs`
- `crates/mpsgraph-rs/src/convolution_transpose_ops.rs`
- `crates/mpsgraph-rs/src/cumulative_ops.rs`
- `crates/mpsgraph-rs/src/depthwise_convolution_ops.rs`
- `crates/mpsgraph-rs/src/fourier_transform_ops.rs`
- `crates/mpsgraph-rs/src/gather_ops.rs`
- `crates/mpsgraph-rs/src/gradient_ops.rs`
- `crates/mpsgraph-rs/src/im2col_ops.rs`
- `crates/mpsgraph-rs/src/linear_algebra_ops.rs`
- `crates/mpsgraph-rs/src/loss_ops.rs`
- `crates/mpsgraph-rs/src/matrix_inverse_ops.rs`
- `crates/mpsgraph-rs/src/matrix_ops.rs`
- `crates/mpsgraph-rs/src/memory_ops.rs`
- `crates/mpsgraph-rs/src/non_maximum_suppression_ops.rs`
- `crates/mpsgraph-rs/src/non_zero_ops.rs`
- `crates/mpsgraph-rs/src/normalization_ops.rs`
- `crates/mpsgraph-rs/src/one_hot_ops.rs`
- `crates/mpsgraph-rs/src/optimizer_ops.rs`
- `crates/mpsgraph-rs/src/pooling_ops.rs`
- `crates/mpsgraph-rs/src/quantization_ops.rs`
- `crates/mpsgraph-rs/src/random_ops.rs`
- `crates/mpsgraph-rs/src/reduction_ops.rs`
- `crates/mpsgraph-rs/src/resize_ops.rs`
- `crates/mpsgraph-rs/src/rnn_ops.rs`
- `crates/mpsgraph-rs/src/sample_grid_ops.rs`
- `crates/mpsgraph-rs/src/scatter_nd_ops.rs`
- `crates/mpsgraph-rs/src/sort_ops.rs`
- `crates/mpsgraph-rs/src/sparse_ops.rs`
- `crates/mpsgraph-rs/src/stencil_ops.rs`
- `crates/mpsgraph-rs/src/tensor_shape_ops.rs`
- `crates/mpsgraph-rs/src/top_k_ops.rs`

Other files verified as compliant or not needing these specific changes:
- `crates/mpsgraph-rs/src/lib.rs`
- `crates/mpsgraph-rs/src/core.rs`
- `crates/mpsgraph-rs/src/utils_old.rs`
- `crates/mpsgraph-rs/src/utils/mod.rs`
- `crates/mpsgraph-rs/src/utils/block_wrapper.rs`
- `crates/mpsgraph-rs/src/utils/buffer.rs`

Test and Example Files Updated & Verified:
- All files in `crates/mpsgraph-rs/src/tests/`
- All files in `crates/mpsgraph-rs/examples/`

## Phase 4: Final User Review
- Status: PENDING USER REVIEW.
- Tasks:
    - Manually review the logic within all modified method bodies, especially complex ones or those with involved FFI (e.g., block callbacks in `control_flow_ops.rs`), to ensure runtime correctness and proper memory handling beyond what compiler checks and existing tests cover.
    - Perform any additional manual testing or code review as needed.

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