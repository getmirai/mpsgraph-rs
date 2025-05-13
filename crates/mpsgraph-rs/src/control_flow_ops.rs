use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::{NSArray, NSMutableArray, NSString};
use std::ffi::c_void;
use std::ops::Fn;

use crate::graph::Graph;
use crate::operation::Operation;
use crate::tensor::Tensor;
use crate::utils::block_wrapper::{
    convert_nsarray_to_vec, create_condition_block, create_index_tensor_array_block,
    create_ns_array_from_tensors, create_tensor_array_block, create_tensor_block,
};

/// Trait for control flow operations on Graph
pub trait GraphControlFlowOps {
    /// Creates a control dependency between operations.
    ///
    /// This ensures that operations in the dependent_block are executed
    /// only after all operations in the operations list have been executed.
    ///
    /// # Parameters
    ///
    /// * `operations` - Operations that must be completed before the dependent block executes
    /// * `dependent_block` - A closure that returns tensors that depend on the operations
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A vector of tensors that are the result of the dependent_block
    fn control_dependency<F>(
        &self,
        operations: &[&Operation],
        dependent_block: F,
        name: Option<&str>,
    ) -> Vec<Retained<Tensor>>
    where
        F: Fn() -> Vec<Retained<Tensor>> + 'static;

    /// Creates an if-then-else operation.
    ///
    /// This allows for conditional execution of operations based on a predicate tensor.
    ///
    /// # Parameters
    ///
    /// * `predicate` - A scalar tensor that determines which branch to execute
    /// * `then_block` - A closure that returns tensors for the "then" branch
    /// * `else_block` - An optional closure that returns tensors for the "else" branch
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A vector of tensors that are the result of either the then_block or else_block,
    /// depending on the value of the predicate tensor
    fn if_op<F, G>(
        &self,
        predicate: &Tensor,
        then_block: F,
        else_block: Option<G>,
        name: Option<&str>,
    ) -> Vec<Retained<Tensor>>
    where
        F: Fn() -> Vec<Retained<Tensor>> + 'static,
        G: Fn() -> Vec<Retained<Tensor>> + 'static;

    /// Creates a while loop operation.
    ///
    /// This allows for iterative execution of operations until a condition is met.
    ///
    /// # Parameters
    ///
    /// * `initial_inputs` - Initial tensors passed to the loop
    /// * `before_block` - A closure that evaluates the condition and produces intermediate tensors
    /// * `after_block` - A closure that executes the loop body
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A vector of tensors that are the final result of the while loop
    fn while_loop<F, G>(
        &self,
        initial_inputs: &[&Tensor],
        before_block: F,
        after_block: G,
        name: Option<&str>,
    ) -> Vec<Retained<Tensor>>
    where
        F: Fn(&[Retained<Tensor>], &mut Vec<Retained<Tensor>>) -> Retained<Tensor> + 'static,
        G: Fn(&[Retained<Tensor>]) -> Vec<Retained<Tensor>> + 'static;

    /// Creates a for loop operation.
    ///
    /// This allows for iterative execution of operations for a specified range.
    ///
    /// # Parameters
    ///
    /// * `lower_bound` - Lower bound value of the loop (inclusive)
    /// * `upper_bound` - Upper bound value of the loop (exclusive)
    /// * `step` - Step value of the loop (must be positive)
    /// * `initial_body_arguments` - Initial tensors passed to the loop body
    /// * `body_block` - A closure that executes the loop body for each iteration
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A vector of tensors that are the final result of the for loop
    fn for_loop<F>(
        &self,
        lower_bound: &Tensor,
        upper_bound: &Tensor,
        step: &Tensor,
        initial_body_arguments: &[&Tensor],
        body_block: F,
        name: Option<&str>,
    ) -> Vec<Retained<Tensor>>
    where
        F: Fn(&Tensor, &[Retained<Tensor>]) -> Vec<Retained<Tensor>> + 'static;

    /// Creates a for loop operation with a specific number of iterations.
    ///
    /// This is a more direct version of the for loop that just specifies the total number of iterations.
    ///
    /// # Parameters
    ///
    /// * `num_iterations` - Number of iterations for the loop
    /// * `initial_body_arguments` - Initial tensors passed to the loop body
    /// * `body_block` - A closure that executes the loop body for each iteration
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A vector of tensors that are the final result of the for loop
    fn for_loop_with_iterations<F>(
        &self,
        num_iterations: &Tensor,
        initial_body_arguments: &[&Tensor],
        body_block: F,
        name: Option<&str>,
    ) -> Vec<Retained<Tensor>>
    where
        F: Fn(&Tensor, &[Retained<Tensor>]) -> Vec<Retained<Tensor>> + 'static;
}

impl GraphControlFlowOps for Graph {
    fn control_dependency<F>(
        &self,
        operations: &[&Operation],
        dependent_block: F,
        name: Option<&str>,
    ) -> Vec<Retained<Tensor>>
    where
        F: Fn() -> Vec<Retained<Tensor>> + 'static,
    {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            // Convert operations to NSArray
            let operations_array = NSArray::from_slice(operations);

            // Create the wrapped block
            let wrapped_block = create_tensor_block(move || {
                // Call the user's block
                let tensors = dependent_block();
                // Convert Vec<Retained<Tensor>> to NSArray
                let tensor_array = create_ns_array_from_tensors(tensors);
                // Convert the NSArray to a raw pointer **before** releasing the `Retained` so that ARC
                // on the Objective-C side can retain it. We purposely `forget` the Rust value to hand over
                // ownership to Objective-C and keep the object alive for the caller.
                let ptr = &*tensor_array as *const _ as *mut c_void;
                std::mem::forget(tensor_array);
                ptr
            });

            // Call the Objective-C method
            let result_array: *mut NSArray<Tensor> = msg_send![
                self,
                controlDependencyWithOperations: &*operations_array,
                dependentBlock: wrapped_block.as_block_ptr(),
                name: name_ptr,
            ];

            // Convert NSArray to Vec<Retained<Tensor>> using the helper function
            convert_nsarray_to_vec(result_array)
        }
    }

    fn if_op<F, G>(
        &self,
        predicate: &Tensor,
        then_block: F,
        else_block: Option<G>,
        name: Option<&str>,
    ) -> Vec<Retained<Tensor>>
    where
        F: Fn() -> Vec<Retained<Tensor>> + 'static,
        G: Fn() -> Vec<Retained<Tensor>> + 'static,
    {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            // Create the then block callback with our wrapper
            let then_wrapped_block = create_tensor_block(move || {
                // Call the user's block
                let tensors = then_block();
                // Convert Vec<Retained<Tensor>> to NSArray
                let tensor_array = create_ns_array_from_tensors(tensors);
                // Convert the NSArray to a raw pointer **before** releasing the `Retained` so that ARC
                // on the Objective-C side can retain it. We purposely `forget` the Rust value to hand over
                // ownership to Objective-C and keep the object alive for the caller.
                let ptr = &*tensor_array as *const _ as *mut c_void;
                std::mem::forget(tensor_array);
                ptr
            });

            // Create the else block callback if provided
            let else_wrapped_block_opt = else_block.map(|else_fn| {
                create_tensor_block(move || {
                    // Call the user's block
                    let tensors = else_fn();
                    // Convert Vec<Retained<Tensor>> to NSArray
                    let tensor_array = create_ns_array_from_tensors(tensors);
                    // Convert the NSArray to a raw pointer **before** releasing the `Retained` so that ARC
                    // on the Objective-C side can retain it. We purposely `forget` the Rust value to hand over
                    // ownership to Objective-C and keep the object alive for the caller.
                    let ptr = &*tensor_array as *const _ as *mut c_void;
                    std::mem::forget(tensor_array);
                    ptr
                })
            });

            // Get a pointer to the else block or null
            let else_block_ptr = match &else_wrapped_block_opt {
                Some(block) => block.as_block_ptr(),
                None => std::ptr::null(),
            };

            // Call the Objective-C method
            let result_array: *mut NSArray<Tensor> = msg_send![
                self,
                ifWithPredicateTensor: predicate,
                thenBlock: then_wrapped_block.as_block_ptr(),
                elseBlock: else_block_ptr,
                name: name_ptr,
            ];

            // Convert NSArray to Vec<Retained<Tensor>> using the helper function
            convert_nsarray_to_vec(result_array)
        }
    }

    fn while_loop<F, G>(
        &self,
        initial_inputs: &[&Tensor],
        before_block: F,
        after_block: G,
        name: Option<&str>,
    ) -> Vec<Retained<Tensor>>
    where
        F: Fn(&[Retained<Tensor>], &mut Vec<Retained<Tensor>>) -> Retained<Tensor> + 'static,
        G: Fn(&[Retained<Tensor>]) -> Vec<Retained<Tensor>> + 'static,
    {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            // Convert initial inputs to NSArray
            let initial_inputs_array = NSArray::from_slice(initial_inputs);

            // Create the before block callback with our wrapper
            let before_wrapped_block = create_condition_block(
                move |inputs_array_ptr: *mut c_void, results_array_ptr: *mut c_void| {
                    // Convert raw pointers to NSArray and NSMutableArray
                    let inputs_array = inputs_array_ptr as *mut NSArray<Tensor>;
                    let results_array = results_array_ptr as *mut NSMutableArray<Tensor>;

                    // Convert NSArray to Vec<Retained<Tensor>>
                    let inputs = convert_nsarray_to_vec(inputs_array);

                    // Create a mutable Vec to hold result tensors
                    let mut result_tensors = Vec::new();

                    // Call the user's block
                    let condition = before_block(&inputs, &mut result_tensors);

                    // Set results array
                    let results_ns = Retained::from_raw(results_array).unwrap();
                    for tensor in result_tensors {
                        let _: () = msg_send![&*results_ns, addObject: &*tensor];
                    }

                    // Return the condition tensor as a void pointer
                    (&*condition) as *const Tensor as *mut c_void
                },
            );

            // Create the after block callback with our wrapper
            let after_wrapped_block =
                create_tensor_array_block(move |body_args_array_ptr: *mut c_void| {
                    // Convert raw pointer to NSArray
                    let body_args_array = body_args_array_ptr as *mut NSArray<Tensor>;

                    // Convert NSArray to Vec<Retained<Tensor>>
                    let body_arguments = convert_nsarray_to_vec(body_args_array);

                    // Call the user's block
                    let result_tensors = after_block(&body_arguments);

                    // Convert Vec<Retained<Tensor>> to NSArray
                    let tensor_array = create_ns_array_from_tensors(result_tensors);

                    // Convert the NSArray to a raw pointer **before** releasing the `Retained` so that ARC
                    // on the Objective-C side can retain it. We purposely `forget` the Rust value to hand over
                    // ownership to Objective-C and keep the object alive for the caller.
                    let ptr = &*tensor_array as *const _ as *mut c_void;
                    std::mem::forget(tensor_array);
                    ptr
                });

            // Call the Objective-C method
            let result_array: *mut NSArray<Tensor> = msg_send![
                self,
                whileWithInitialInputs: &*initial_inputs_array,
                before: before_wrapped_block.as_block_ptr(),
                after: after_wrapped_block.as_block_ptr(),
                name: name_ptr,
            ];

            // Convert NSArray to Vec<Retained<Tensor>> using the helper function
            convert_nsarray_to_vec(result_array)
        }
    }

    fn for_loop<F>(
        &self,
        lower_bound: &Tensor,
        upper_bound: &Tensor,
        step: &Tensor,
        initial_body_arguments: &[&Tensor],
        body_block: F,
        name: Option<&str>,
    ) -> Vec<Retained<Tensor>>
    where
        F: Fn(&Tensor, &[Retained<Tensor>]) -> Vec<Retained<Tensor>> + 'static,
    {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            // Convert initial body arguments to NSArray
            let initial_args_array = NSArray::from_slice(initial_body_arguments);

            // Create the body block callback with our wrapper
            let body_wrapped_block = create_index_tensor_array_block(
                move |index_ptr: *mut c_void, args_array_ptr: *mut c_void| {
                    // Convert raw pointers to Tensor and NSArray
                    let index = index_ptr as *mut Tensor;
                    let args_array = args_array_ptr as *mut NSArray<Tensor>;

                    // Convert to Tensor
                    let index_tensor = Retained::from_raw(index).unwrap();

                    // Convert NSArray to Vec<Retained<Tensor>>
                    let body_arguments = convert_nsarray_to_vec(args_array);

                    // Call the user's block
                    let result_tensors = body_block(&index_tensor, &body_arguments);

                    // Convert Vec<Retained<Tensor>> to NSArray
                    let tensor_array = create_ns_array_from_tensors(result_tensors);

                    // Convert the NSArray to a raw pointer **before** releasing the `Retained` so that ARC
                    // on the Objective-C side can retain it. We purposely `forget` the Rust value to hand over
                    // ownership to Objective-C and keep the object alive for the caller.
                    let ptr = &*tensor_array as *const _ as *mut c_void;
                    std::mem::forget(tensor_array);
                    ptr
                },
            );

            // Call the Objective-C method
            let result_array: *mut NSArray<Tensor> = msg_send![
                self,
                forLoopWithLowerBound: lower_bound,
                upperBound: upper_bound,
                step: step,
                initialBodyArguments: &*initial_args_array,
                body: body_wrapped_block.as_block_ptr(),
                name: name_ptr,
            ];

            // Convert NSArray to Vec<Retained<Tensor>> using the helper function
            convert_nsarray_to_vec(result_array)
        }
    }

    fn for_loop_with_iterations<F>(
        &self,
        num_iterations: &Tensor,
        initial_body_arguments: &[&Tensor],
        body_block: F,
        name: Option<&str>,
    ) -> Vec<Retained<Tensor>>
    where
        F: Fn(&Tensor, &[Retained<Tensor>]) -> Vec<Retained<Tensor>> + 'static,
    {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            // Convert initial body arguments to NSArray
            let initial_args_array = NSArray::from_slice(initial_body_arguments);

            // Create the body block callback with our wrapper
            let body_wrapped_block = create_index_tensor_array_block(
                move |index_ptr: *mut c_void, args_array_ptr: *mut c_void| {
                    // Convert raw pointers to Tensor and NSArray
                    let index = index_ptr as *mut Tensor;
                    let args_array = args_array_ptr as *mut NSArray<Tensor>;

                    // Convert to Tensor
                    let index_tensor = Retained::from_raw(index).unwrap();

                    // Convert NSArray to Vec<Retained<Tensor>>
                    let body_arguments = convert_nsarray_to_vec(args_array);

                    // Call the user's block
                    let result_tensors = body_block(&index_tensor, &body_arguments);

                    // Convert Vec<Retained<Tensor>> to NSArray
                    let tensor_array = create_ns_array_from_tensors(result_tensors);

                    // Convert the NSArray to a raw pointer **before** releasing the `Retained` so that ARC
                    // on the Objective-C side can retain it. We purposely `forget` the Rust value to hand over
                    // ownership to Objective-C and keep the object alive for the caller.
                    let ptr = &*tensor_array as *const _ as *mut c_void;
                    std::mem::forget(tensor_array);
                    ptr
                },
            );

            // Call the Objective-C method
            let result_array: *mut NSArray<Tensor> = msg_send![
                self,
                forLoopWithNumberOfIterations: num_iterations,
                initialBodyArguments: &*initial_args_array,
                body: body_wrapped_block.as_block_ptr(),
                name: name_ptr,
            ];

            // Convert NSArray to Vec<Retained<Tensor>> using the helper function
            convert_nsarray_to_vec(result_array)
        }
    }
}
