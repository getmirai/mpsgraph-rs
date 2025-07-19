use crate::TensorData;
use block2::DynBlock;
use objc2_foundation::{NSArray, NSError};
use std::ptr::NonNull;

/// A notification when graph executable execution schedules.
///
/// - Parameters:
/// - results: If no error, the results produced by the graph operation.
/// - error: If an error occurs, more information might be found here.
///
/// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphexecutablescheduledhandler?language=objc)
pub type ExecutableScheduledHandler =
    *mut DynBlock<dyn Fn(NonNull<NSArray<TensorData>>, *mut NSError)>;
