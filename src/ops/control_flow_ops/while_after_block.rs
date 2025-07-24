use crate::Tensor;
use block2::{Block, IntoBlock, RcBlock};
use objc2::rc::Retained;
use objc2_foundation::{NSArray, NSMutableArray};
use std::ptr::NonNull;

/// The block that executes after the condition evaluates for each iteration.
///
/// - Parameters:
///   - body_block_arguments: Inputs to the body of the while loop passed by the condition block return, and should be the same element types as the return of the while loop.
/// - Returns: A valid `MPSGraphTensor` array with results forwarded to the condition block.
///
/// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphwhileafterblock?language=objc)
#[repr(transparent)]
pub struct WhileAfterBlock {
    block: RcBlock<dyn Fn(NonNull<NSArray<Tensor>>) -> NonNull<NSArray<Tensor>>>,
}

impl WhileAfterBlock {
    pub fn new<F>(while_after_ops: F) -> Self
    where
        F: Fn(&[&Tensor]) -> Box<[Retained<Tensor>]> + 'static,
    {
        Self {
            block: RcBlock::new(move |body_block_arguments: NonNull<NSArray<Tensor>>| {
                let body_block_arguments =
                    unsafe { body_block_arguments.as_ref().to_vec_unchecked() };
                let results = while_after_ops(&body_block_arguments);
                let results_array = NSArray::from_retained_slice(&results);
                let raw = Retained::autorelease_return(results_array);
                unsafe { NonNull::new_unchecked(raw) }
            }),
        }
    }

    pub fn as_deref(&self) -> &Block<dyn Fn(NonNull<NSArray<Tensor>>) -> NonNull<NSArray<Tensor>>> {
        &*self.block
    }
}
