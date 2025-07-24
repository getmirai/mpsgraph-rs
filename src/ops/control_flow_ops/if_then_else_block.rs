use std::ptr::NonNull;

use objc2_foundation::NSArray;

use crate::Tensor;

/// A block of operations executed under either the if or else condition.
///
/// - Returns: Tensors returned by user. If not empty, the user must define both the then and else blocks,
/// both should have the same number of arguments, and each corresponding argument should have the same element types.
///
/// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphifthenelseblock?language=objc)

pub type IfThenElseBlock = *mut block2::DynBlock<dyn Fn() -> NonNull<NSArray<Tensor>>>;
