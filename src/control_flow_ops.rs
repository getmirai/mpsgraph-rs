use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::{NSArray, NSMutableArray, NSString};
use std::ffi::c_void;
use std::ops::Fn;

use crate::graph::Graph;
use crate::operation::Operation;
use crate::tensor::Tensor;
use crate::utils::block_wrapper::{
    convert_nsarray_to_vec, convert_retained_nsarray_to_vec, create_condition_block,
    create_index_tensor_array_block, create_ns_array_from_tensors, create_tensor_array_block,
    create_tensor_block,
};

/// Control-flow helpers are now inherent methods on `Graph`.
impl Graph {
    // Public wrappers ---------------------------------------------------------
    pub fn control_dependency<F>(
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

            let operations_array = NSArray::from_slice(operations);

            let wrapped_block = create_tensor_block(move || {
                let tensors = dependent_block();
                let tensor_array = create_ns_array_from_tensors(tensors);
                let ptr = &*tensor_array as *const _ as *mut c_void;
                std::mem::forget(tensor_array);
                ptr
            });

            let result_array_opt: Option<Retained<NSArray<Tensor>>> = msg_send![
                self,
                controlDependencyWithOperations: &*operations_array,
                dependentBlock: wrapped_block.as_block_ptr(),
                name: name_ptr,
            ];

            result_array_opt.map_or(Vec::new(), convert_retained_nsarray_to_vec)
        }
    }

    pub fn if_op<F, G>(
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

            let then_wrapped_block = create_tensor_block(move || {
                let tensors = then_block();
                let tensor_array = create_ns_array_from_tensors(tensors);
                let ptr = &*tensor_array as *const _ as *mut c_void;
                std::mem::forget(tensor_array);
                ptr
            });

            let else_wrapped_block_opt = else_block.map(|else_fn| {
                create_tensor_block(move || {
                    let tensors = else_fn();
                    let tensor_array = create_ns_array_from_tensors(tensors);
                    let ptr = &*tensor_array as *const _ as *mut c_void;
                    std::mem::forget(tensor_array);
                    ptr
                })
            });

            let else_block_ptr = match &else_wrapped_block_opt {
                Some(block) => block.as_block_ptr(),
                None => std::ptr::null(),
            };

            let result_array_opt: Option<Retained<NSArray<Tensor>>> = msg_send![
                self,
                ifWithPredicateTensor: predicate,
                thenBlock: then_wrapped_block.as_block_ptr(),
                elseBlock: else_block_ptr,
                name: name_ptr,
            ];

            result_array_opt.map_or(Vec::new(), convert_retained_nsarray_to_vec)
        }
    }

    pub fn while_loop<F, G>(
        &self,
        initial_inputs: &[&Tensor],
        before_block: F,
        after_block: G,
        name: Option<&str>,
    ) -> Vec<Retained<Tensor>>
    where
        F: Fn(&[&Tensor], &mut Vec<Retained<Tensor>>) -> Retained<Tensor> + 'static,
        G: Fn(&[&Tensor]) -> Vec<Retained<Tensor>> + 'static,
    {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let initial_inputs_array = NSArray::from_slice(initial_inputs);

            let before_wrapped_block = create_condition_block(
                move |inputs_array_ptr: *mut c_void, results_array_ptr: *mut c_void| {
                    let inputs_array_raw = inputs_array_ptr as *mut NSArray<Tensor>;
                    let results_array = results_array_ptr as *mut NSMutableArray<Tensor>;

                    let inputs_retained = convert_nsarray_to_vec(inputs_array_raw);
                    let inputs_refs: Vec<&Tensor> =
                        inputs_retained.iter().map(|t| t.as_ref()).collect();

                    let mut result_tensors_for_block = Vec::new();
                    let condition = before_block(&inputs_refs, &mut result_tensors_for_block);

                    let results_ns = Retained::retain_autoreleased(results_array).unwrap();
                    for tensor in result_tensors_for_block {
                        let _: () = msg_send![&*results_ns, addObject: &*tensor];
                    }
                    (&*condition) as *const Tensor as *mut c_void
                },
            );

            let after_wrapped_block =
                create_tensor_array_block(move |body_args_array_ptr: *mut c_void| {
                    let body_args_array_raw = body_args_array_ptr as *mut NSArray<Tensor>;
                    let body_arguments_retained = convert_nsarray_to_vec(body_args_array_raw);
                    let body_arguments_refs: Vec<&Tensor> =
                        body_arguments_retained.iter().map(|t| t.as_ref()).collect();

                    let result_tensors = after_block(&body_arguments_refs);

                    let tensor_array = create_ns_array_from_tensors(result_tensors);
                    let ptr = &*tensor_array as *const _ as *mut c_void;
                    std::mem::forget(tensor_array);
                    ptr
                });

            let result_array_opt: Option<Retained<NSArray<Tensor>>> = msg_send![
                self,
                whileWithInitialInputs: &*initial_inputs_array,
                before: before_wrapped_block.as_block_ptr(),
                after: after_wrapped_block.as_block_ptr(),
                name: name_ptr,
            ];
            result_array_opt.map_or(Vec::new(), convert_retained_nsarray_to_vec)
        }
    }

    pub fn for_loop<F>(
        &self,
        lower_bound: &Tensor,
        upper_bound: &Tensor,
        step: &Tensor,
        initial_body_arguments: &[&Tensor],
        body_block: F,
        name: Option<&str>,
    ) -> Vec<Retained<Tensor>>
    where
        F: Fn(&Tensor, &[&Tensor]) -> Vec<Retained<Tensor>> + 'static,
    {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let initial_args_array = NSArray::from_slice(initial_body_arguments);

            let body_wrapped_block = create_index_tensor_array_block(
                move |index_ptr: *mut c_void, args_array_ptr: *mut c_void| {
                    let index = index_ptr as *mut Tensor;
                    let args_array_raw = args_array_ptr as *mut NSArray<Tensor>;
                    let index_tensor = Retained::retain_autoreleased(index).unwrap();
                    let body_arguments_retained = convert_nsarray_to_vec(args_array_raw);
                    let body_arguments_refs: Vec<&Tensor> =
                        body_arguments_retained.iter().map(|t| t.as_ref()).collect();

                    let result_tensors = body_block(&index_tensor, &body_arguments_refs);

                    let tensor_array = create_ns_array_from_tensors(result_tensors);
                    let ptr = &*tensor_array as *const _ as *mut c_void;
                    std::mem::forget(tensor_array);
                    ptr
                },
            );

            let result_array_opt: Option<Retained<NSArray<Tensor>>> = msg_send![
                self,
                forLoopWithLowerBound: lower_bound,
                upperBound: upper_bound,
                step: step,
                initialBodyArguments: &*initial_args_array,
                body: body_wrapped_block.as_block_ptr(),
                name: name_ptr,
            ];

            result_array_opt.map_or(Vec::new(), convert_retained_nsarray_to_vec)
        }
    }

    // Private helper implementing the number-of-iterations variant.
    fn for_loop_impl_with_iterations<F>(
        &self,
        num_iterations: &Tensor,
        initial_body_arguments: &[&Tensor],
        body_block: F,
        name: Option<&str>,
    ) -> Vec<Retained<Tensor>>
    where
        F: Fn(&Tensor, &[&Tensor]) -> Vec<Retained<Tensor>> + 'static,
    {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let initial_args_array = NSArray::from_slice(initial_body_arguments);

            let body_wrapped_block = create_index_tensor_array_block(
                move |index_ptr: *mut c_void, args_array_ptr: *mut c_void| {
                    let index = index_ptr as *mut Tensor;
                    let args_array_raw = args_array_ptr as *mut NSArray<Tensor>;
                    let index_tensor = Retained::retain_autoreleased(index).unwrap();
                    let body_arguments_retained = convert_nsarray_to_vec(args_array_raw);
                    let body_arguments_refs: Vec<&Tensor> =
                        body_arguments_retained.iter().map(|t| t.as_ref()).collect();

                    let result_tensors = body_block(&index_tensor, &body_arguments_refs);

                    let tensor_array = create_ns_array_from_tensors(result_tensors);
                    let ptr = &*tensor_array as *const _ as *mut c_void;
                    std::mem::forget(tensor_array);
                    ptr
                },
            );

            let result_array_opt: Option<Retained<NSArray<Tensor>>> = msg_send![
                self,
                forLoopWithNumberOfIterations: num_iterations,
                initialBodyArguments: &*initial_args_array,
                body: body_wrapped_block.as_block_ptr(),
                name: name_ptr,
            ];

            result_array_opt.map_or(Vec::new(), convert_retained_nsarray_to_vec)
        }
    }
}

// -------------------------------------------------------------------------
// Extension trait providing the overloaded `for_loop` variant (iterations)
// -------------------------------------------------------------------------

pub trait ForLoopIterationsExt {
    fn for_loop<F>(
        &self,
        num_iterations: &Tensor,
        initial_body_arguments: &[&Tensor],
        body_block: F,
        name: Option<&str>,
    ) -> Vec<Retained<Tensor>>
    where
        F: Fn(&Tensor, &[&Tensor]) -> Vec<Retained<Tensor>> + 'static;
}

impl ForLoopIterationsExt for Graph {
    fn for_loop<F>(
        &self,
        num_iterations: &Tensor,
        initial_body_arguments: &[&Tensor],
        body_block: F,
        name: Option<&str>,
    ) -> Vec<Retained<Tensor>>
    where
        F: Fn(&Tensor, &[&Tensor]) -> Vec<Retained<Tensor>> + 'static,
    {
        self.for_loop_impl_with_iterations(num_iterations, initial_body_arguments, body_block, name)
    }
}
