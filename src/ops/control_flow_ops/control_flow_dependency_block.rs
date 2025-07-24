use crate::Tensor;
use block2::{Block, RcBlock};
use objc2::rc::Retained;
use objc2_foundation::NSArray;
use std::ptr::NonNull;

/// The scope where all the operations defined in this block get control-dependency operations.
///
/// - Returns: A valid tensor with the results forwarded to the return of `controlDependency` call.
///
/// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphcontrolflowdependencyblock?language=objc)
#[repr(transparent)]
pub struct ControlFlowDependencyBlock {
    block: RcBlock<dyn Fn() -> NonNull<NSArray<Tensor>>>,
}

impl ControlFlowDependencyBlock {
    pub fn new<F>(dependent_ops: F) -> Self
    where
        F: Fn() -> Box<[Retained<Tensor>]> + 'static,
    {
        Self {
            block: RcBlock::new(move || {
                let tensors = dependent_ops();
                let arr = NSArray::from_retained_slice(&tensors);
                let raw = Retained::autorelease_return(arr);
                unsafe { NonNull::new_unchecked(raw) }
            }),
        }
    }

    pub fn as_deref(&self) -> &Block<dyn Fn() -> NonNull<NSArray<Tensor>>> {
        &*self.block
    }
}
