use crate::Tensor;
use block2::{Block, IntoBlock, RcBlock};
use objc2::rc::Retained;
use objc2_foundation::{NSArray, NSMutableArray};
use std::ptr::NonNull;

/// The block that executes before the condition evaluates for each iteration.
///
/// - Parameters:
///     - input_tensors: Input tensors to the `whileConditionBlock`, for the first iteration will be same as initialInputs passed to the while loop.
///     - result_tensors: A valid `MPSGraphTensor` array with results forwarded to after block or returned from the while loop depending on the predicate tensor. It will be empty and the caller block should fill it up before returning.
/// - Returns: Tensor MUST be set and have a single scalar value, used to decide between executing the body block or returning from the while loop.
///
/// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphwhilebeforeblock?language=objc)
#[repr(transparent)]
pub struct WhileBeforeBlock {
    block: RcBlock<
        dyn Fn(NonNull<NSArray<Tensor>>, NonNull<NSMutableArray<Tensor>>) -> NonNull<Tensor>,
    >,
}

impl WhileBeforeBlock {
    pub fn new<F>(while_before_ops: F) -> Self
    where
        F: Fn(&[&Tensor], &mut [&Tensor]) -> Retained<Tensor> + 'static,
    {
        Self {
            block: RcBlock::new(
                move |input_tensors: NonNull<NSArray<Tensor>>,
                      result_tensors: NonNull<NSMutableArray<Tensor>>| {
                    let inputs = unsafe { input_tensors.as_ref().to_vec_unchecked() };
                    let mut results = unsafe { result_tensors.as_ref().to_vec_unchecked() };
                    let tensor = while_before_ops(&inputs, &mut results);
                    let raw = Retained::autorelease_return(tensor);
                    unsafe { NonNull::new_unchecked(raw) }
                },
            ),
        }
    }

    pub fn as_deref(
        &self,
    ) -> &Block<dyn Fn(NonNull<NSArray<Tensor>>, NonNull<NSMutableArray<Tensor>>) -> NonNull<Tensor>>
    {
        &*self.block
    }
}
