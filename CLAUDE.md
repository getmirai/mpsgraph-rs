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

## AnyObject to NSObject Migration

This codebase is transitioning from using `objc2::runtime::AnyObject` with manual retain/release to using `extern_class!` with `objc2_foundation::NSObject` and automatic memory management. This change provides:

1. Automatic memory management (no manual retain/release)
2. Better Rust integration (Debug, PartialEq, Eq, Hash implementations)
3. Access to NSObject methods (description, hash_code, etc.)
4. More type safety and idiomatic Rust code

### Migration Guide

For each type to be migrated:

1. **Change the type definition**:

   ```rust
   // Before
   pub struct Device(pub(crate) *mut AnyObject);
   
   // After
   use objc2::rc::Retained;
   use objc2_foundation::NSObject;
   
   extern_class!(
       #[derive(Debug, PartialEq, Eq, Hash)]
       pub struct Device;
       
       unsafe impl NSObjectProtocol for Device {}
   );
   ```

2. **Update function signatures**:

   ```rust
   // Before
   pub fn new() -> Self { ... }
   
   // After
   pub fn new() -> Retained<Self> { ... }
   ```

3. **Update method implementations**:

   ```rust
   // Before
   pub fn with_device(device: &MetalDevice) -> Self {
       unsafe {
           let class_name = c"MPSGraphDevice";
           let cls = objc2::runtime::AnyClass::get(class_name)
               .unwrap_or_else(|| panic!("MPSGraphDevice class not found"));
           let device_ptr = device.as_ptr() as *mut AnyObject;
           let obj: *mut AnyObject = msg_send![cls, deviceWithMTLDevice:device_ptr];
           let obj = objc2::ffi::objc_retain(obj as *mut _);
           Device(obj)
       }
   }
   
   // After
   pub fn with_device(device: &MetalDevice) -> Retained<Self> {
       unsafe {
           let device_ptr = device.as_ptr();
           msg_send![Self::class(), deviceWithMTLDevice:device_ptr]
       }
   }
   ```

4. **Remove manual memory management**:
   - Delete `Drop` implementation with `objc_release`
   - Delete `Clone` implementation with `objc_retain`
   - These are handled automatically by `Retained<T>`

5. **Update Default implementation**:

   ```rust
   // Before
   impl Default for Device {
       fn default() -> Self {
           Self::new()
       }
   }
   
   // After
   impl Default for Device {
       fn default() -> Retained<Self> {
           Self::new()
       }
   }
   ```

6. **Update callers**:
   - When a function returns `Retained<T>` instead of `T`, callers may need adjustments
   - Use `&*retained_object` to get a reference to the object

7. **Internal field updates**:

   ```rust
   // Before
   struct SomeType {
       device: *mut AnyObject,
       // other fields
   }
   
   // After
   struct SomeType {
       device: Retained<Device>,
       // other fields
   }
   ```

### Migration Testing

After each type migration:

1. Run tests: `cargo test -p mpsgraph`
2. Build examples: `cargo build --examples`
3. Run examples: `cargo run --example simple_compile`
4. Ensure no memory leaks or crashes

### Classes to Migrate

The following classes need to be migrated from AnyObject to NSObject:

1. Core Types (migrate first):
   - `Device` in `device.rs`
   - `Graph` in `graph.rs`
   - `Shape` in `shape.rs`
   - `Tensor` in `tensor.rs`
   - `TensorData` in `tensor_data.rs`
   - `Operation` in `operation.rs`

2. Executable Types:
   - `Executable` in `executable.rs`
   - `CompilationDescriptor` in `executable.rs`
   - `ExecutionDescriptor` in `executable.rs`
   - `MPSCommandBuffer` in `command_buffer.rs`

3. Descriptor Types:
   - `PoolingDescriptor` in `pooling_ops.rs`
   - `ConvolutionDescriptor` in `convolution_ops.rs`
   - `DepthwiseConvolutionDescriptor` in `depthwise_convolution_ops.rs`
   - `StencilDescriptor` in `stencil_ops.rs`
   - `CSCFormatDescriptor` in `sparse_ops.rs`
   - `MpsIm2ColDescriptor` in `im2col_ops.rs`
   - `RNNDescriptor` in `rnn_ops.rs`
   - `RandomDescriptor` in `random_ops.rs`
   - `OptimizerDescriptor` in `optimizer_ops.rs`
   - `DataTypeAttributeValue` in `data_types.rs`

## Code Style Guidelines

- Use 4-space indentation
- Follow Rust 2021 edition conventions
- Types: Use simple `PascalCase` names without MPS prefix (e.g., `Device` instead of `MPSGraphDevice`)
- Methods/variables: Use `snake_case`
- Error handling: Use `Result<T, E>` with descriptive error types
- Imports: Standard library first, then external crates, then internal modules
- Documentation: Add doc comments for all public APIs
- Testing: Name tests as `test_functionality_being_tested`
- Type safety: Use appropriate Rust types to wrap Objective-C objects
- Follow idiomatic Rust FFI patterns for Objective-C interop
- Type access is via namespace: `mpsgraph::Device` (not `mpsgraph::MPSGraphDevice`)

## Objective-C Integration

- Use `extern_class!` macro for existing Objective-C classes
- Use `define_class!` macro for custom Objective-C classes:

```rust
define_class! {
    #[unsafe(super(NSObject))]
    #[name = "MyCustomClass"]
    #[ivars = MyIvars]  // Optional, struct with instance variables
    struct MyClass;
    
    impl MyClass {
        // Instance methods
        #[unsafe(method(myMethod:))]
        fn my_method(&self, arg: &NSString) -> bool {
            // Implementation
        }
        
        // Class methods
        #[unsafe(class_method(sharedInstance))]
        fn shared_instance() -> Id<Self> {
            // Implementation
        }
    }
    
    // Protocol implementation (optional)
    unsafe impl MyProtocol for MyClass {
        // Protocol methods
    }
}
```

- Use `Retained<T>` for owned references and `Id<T>` for memory-managed references
- Implement `NSObjectProtocol` for all Objective-C class types
- When interfacing with Metal and MPS, follow their memory ownership patterns
