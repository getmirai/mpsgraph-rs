use crate::TensorData;
use block2::DynBlock;
use objc2_foundation::{NSArray, NSError};
use std::ptr::NonNull;

/// A notification when graph executable execution finishes.
///
/// - Parameters:
/// - results: If no error, the results produced by the graph operation. If Graph hasn't yet allocated the results, this will be `NSNull`.
/// - error: If an error occurs, more information might be found here.
///
/// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphexecutablecompletionhandler?language=objc)
pub type ExecutableCompletionHandler =
    *mut DynBlock<dyn Fn(NonNull<NSArray<TensorData>>, *mut NSError)>;
