use std::ptr::NonNull;

use objc2_foundation::{NSArray, NSMutableArray};

use crate::Tensor;

/// The block that executes before the condition evaluates for each iteration.
///
/// - Parameters:
/// - inputTensors: Input tensors to the `whileConditionBlock`, for the first iteration will be same as initialInputs passed to the while loop.
/// - resultTensors: A valid `MPSGraphTensor` array with results forwarded to after block or returned from the while loop depending on the predicate tensor. It will be empty and the caller block should fill it up before returning.
/// - Returns: Tensor MUST be set and have a single scalar value, used to decide between executing the body block or returning from the while loop.
///
/// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphwhilebeforeblock?language=objc)
pub type WhileBeforeBlock = *mut block2::DynBlock<
    dyn Fn(NonNull<NSArray<Tensor>>, NonNull<NSMutableArray<Tensor>>) -> NonNull<Tensor>,
>;
