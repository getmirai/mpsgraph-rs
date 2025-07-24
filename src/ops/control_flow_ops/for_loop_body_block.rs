use std::ptr::NonNull;

use objc2_foundation::NSArray;

use crate::Tensor;

/// A block for the body in the for loop.
///
/// - Parameters:
/// - index: The for loop index per iteration, it is a scalar tensor.
/// - iterationArguments: Arguments for this iteration, with the same count and corresponding element types as `initialIterationArguments` and return types of the `for` loop.
/// - Returns: A valid MPSGraphTensor array with same count and corresponding element types as `initialIterationArguments` and return types of the `for` loop.
///
/// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphforloopbodyblock?language=objc)
pub type ForLoopBodyBlock = *mut block2::DynBlock<
    dyn Fn(NonNull<Tensor>, NonNull<NSArray<Tensor>>) -> NonNull<NSArray<Tensor>>,
>;
