# Memory Management TODO

This document outlines the completed tasks to improve Objective-C memory management in the mpsgraph-rs codebase.

## Memory Management Patterns

Two key patterns should be followed:

1. **`Retained::from_raw`** - Use for methods following Objective-C's ownership transfer conventions:
   - Methods using `alloc` followed by `init`
   - Methods with names containing `new`, `copy`, or `mutableCopy`
   - Methods starting with `create` or `make`
   - These methods return objects with +1 retain count (ownership transferred)

2. **`Retained::retain_autoreleased`** - Use for:
   - Regular methods returning temporary/autoreleased objects
   - Accessing objects from collections via methods like `objectAtIndex:` and `objectForKey:`
   - Most computation and query methods
   - These methods return objects with +0 retain count (autoreleased)

## Completed Tasks

1. **Updated operation files to use correct memory management**:
   - [x] matrix_ops.rs
   - [x] optimizer_ops.rs
   - [x] arithmetic_ops.rs
   - [x] sort_ops.rs
   - [x] quantization_ops.rs
   - [x] resize_ops.rs
   - [x] normalization_ops.rs
   - [x] matrix_inverse_ops.rs
   - [x] gather_ops.rs
   - [x] convolution_transpose_ops.rs
   - [x] sample_grid_ops.rs

2. **Checked the Graph implementation methods**:
   - [x] `stochastic_gradient_descent`
   - [x] `adam`
   - [x] `adam_with_current_learning_rate`
   - [x] `variable_op_for_tensor`
   - [x] `apply_stochastic_gradient_descent`

3. **Verified NSArray handling**:
   - [x] Check all places where NSArray elements are accessed to ensure they use `retain_autoreleased`

4. **Created documentation**:
   - [x] Add code comments in key files explaining the memory management patterns
   - [x] Create guidelines for future development (this document)

## Testing

- [x] Run the complete test suite
- [x] Test all examples
- [ ] Test the library in real-world applications if possible

## Summary of Work Completed

1. **Updated Property Accessors**:
   - [x] `tensor.rs`: Fixed `shape()` and `name()` methods
   - [x] `tensor_data.rs`: Fixed `shape()` method
   - [x] `command_buffer.rs`: Fixed `label()` method

2. **Updated Object Creation**:
   - [x] `executable.rs`: Fixed `from_serialized_package` and URL handling
   - [x] `graph.rs`: Updated constructor methods

3. **Updated Computational Methods**:
   - [x] Fixed memory management in all computational methods in multiple files 