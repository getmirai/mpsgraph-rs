mod control_flow_dependency_block;
mod for_loop_body_block;
mod if_then_else_block;
mod while_after_block;
mod while_before_block;

use control_flow_dependency_block::ControlFlowDependencyBlock;
use for_loop_body_block::ForLoopBodyBlock;
use if_then_else_block::IfThenElseBlock;
use while_after_block::WhileAfterBlock;
use while_before_block::WhileBeforeBlock;

use crate::{Graph, Operation, Tensor};
use objc2::{extern_methods, msg_send, rc::Retained};
use objc2_foundation::{NSArray, NSString};
use std::ptr::NonNull;

/// MPSGraphControlFlowOps.
impl Graph {
    extern_methods!(
        /// Adds a while loop operation.
        ///
        /// - Parameters:
        /// - initialInputs: inputTensors to the `beforeBlock`, for the 1st iteration will be same as initialInputs passed to the while loop.
        /// - before: `beforeBlock`, this will be run first and then call the `afterBlock` with results or return results from the loop.
        /// - after: `afterBlock`, this will execute after the condition evaluation.
        /// - name: name of operation.
        /// - Returns: A valid MPSGraphTensor array with results returned from the conditionBlock depending on the predicate tensor.
        #[unsafe(method(whileWithInitialInputs:before:after:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn while_with_initial_inputs(
            &self,
            initial_inputs: &NSArray<Tensor>,
            before: WhileBeforeBlock,
            after: WhileAfterBlock,
            name: Option<&NSString>,
        ) -> Retained<NSArray<Tensor>>;

        /// Adds a for loop operation, The lower and upper bounds specify a half-open range: the range includes the lower bound but does not include the upper bound.
        ///
        /// - Parameters:
        /// - lowerBound: Lower bound value of the loop, this is a scalar tensor, this is the index the loop will start with.
        /// - upperBound: Upper bound value of the loop, this is a scalar tensor.
        /// - step: Step value of the loop, this is a scalar tensor and must be positive.
        /// - initialBodyArguments: initial set of iteration arguments passed to the bodyBlock of the for loop.
        /// - body: This block will execute the body of the for loop.
        /// - name: name of operation.
        /// - Returns: A valid `MPSGraphTensor` array with same count and corresponding element types as `initialIterationArguments` and return types of the for loop.
        #[unsafe(method(forLoopWithLowerBound:upperBound:step:initialBodyArguments:body:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn for_loop(
            &self,
            lower_bound: &Tensor,
            upper_bound: &Tensor,
            step: &Tensor,
            initial_body_arguments: &NSArray<Tensor>,
            body: ForLoopBodyBlock,
            name: Option<&NSString>,
        ) -> Retained<NSArray<Tensor>>;

        /// Adds a for loop operation, with a specific number of iterations.
        ///
        /// - Parameters:
        /// - numberOfIterations: tensor with number of iterations the loop will execute
        /// - initialBodyArguments: initial set of iteration arguments passed to the bodyBlock of the for loop
        /// - body: bodyBlock, this will execute the body of the for loop, index will go from 0 to numberOfIterations-1
        /// - name: name of operation
        /// - Returns: A valid MPSGraphTensor array with same count and corresponding elementTypes as initialIterationArguments and return types of the for loop
        #[unsafe(method(forLoopWithNumberOfIterations:initialBodyArguments:body:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn for_loop_with_number_of_iterations(
            &self,
            number_of_iterations: &Tensor,
            initial_body_arguments: &NSArray<Tensor>,
            body: ForLoopBodyBlock,
            name: Option<&NSString>,
        ) -> Retained<NSArray<Tensor>>;
    );
}

impl Graph {
    /// Runs the graph for the given feeds and returns the target tensor values, ensuring all target operations also executed.
    ///
    /// This call blocks until execution has completed.
    ///
    /// - Parameters:
    /// - operations: Operations marked as control dependency for all ops created inside the dependent block
    /// - dependent_ops: closure which is provided by caller to create dependent ops
    /// - name: name of scope
    /// - Returns: A valid MPSGraphTensor array with results returned from dependent_block forwarded
    fn control_dependency<F>(
        &self,
        operations: &[&Operation],
        dependent_ops: F,
        name: Option<&str>,
    ) -> Box<[Retained<Tensor>]>
    where
        F: Fn() -> Box<[Retained<Tensor>]> + 'static,
    {
        let operations_array = NSArray::from_slice(operations);
        let dependent_block = ControlFlowDependencyBlock::new(dependent_ops);
        let result: Retained<NSArray<Tensor>> = unsafe {
            msg_send![
                self,
                controlDependencyWithOperations: &*operations_array,
                dependentBlock: dependent_block.as_deref(),
                name: name.map(NSString::from_str).as_deref(),
            ]
        };
        result.to_vec().into_boxed_slice()
    }

    /// Adds an if-then-else operation to the graph.
    ///
    /// - Parameters:
    /// - predicate_tensor: Tensor must have a single scalar value, used to decide between then/else branches
    /// - then_block: If predicate is true operations in this block are executed
    /// - else_block: If predicate is false operations in this block are executed
    /// - name: name of operation
    /// - Returns: results If no error, the tensors returned by user. If not empty, user must define both then/else block,
    /// both should have same number of arguments and each corresponding argument should have same elementTypes.
    pub fn if_then_else<T, E>(
        &self,
        predicate_tensor: &Tensor,
        then_block: T,
        else_block: E,
        name: Option<&str>,
    ) -> Box<[Retained<Tensor>]>
    where
        T: Fn() -> Box<[Retained<Tensor>]> + 'static,
        E: Fn() -> Box<[Retained<Tensor>]> + 'static,
    {
        let then_block = IfThenElseBlock::new(then_block);
        let else_block = IfThenElseBlock::new(else_block);
        let result: Retained<NSArray<Tensor>> = unsafe {
            msg_send![
                self,
                ifWithPredicateTensor: predicate_tensor,
                thenBlock: then_block.as_deref(),
                elseBlock: else_block.as_deref(),
                name: name.map(NSString::from_str).as_deref(),
            ]
        };
        result.to_vec().into_boxed_slice()
    }
}
