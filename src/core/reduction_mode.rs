use objc2::Encoding;
use objc2::{Encode, RefEncode};
use objc2_foundation::NSUInteger;

/// Reduction mode (MPSGraphReductionMode)
#[allow(dead_code)]
#[repr(usize)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum MPSGraphReductionMode {
    /// Min
    Min = 0,
    /// Max
    Max = 1,
    /// Sum
    Sum = 2,
    /// Product
    Product = 3,
    /// Argument Min
    ArgumentMin = 4,
    /// Argument Max
    ArgumentMax = 5,
}

impl From<MPSGraphReductionMode> for NSUInteger {
    fn from(mode: MPSGraphReductionMode) -> Self {
        mode as NSUInteger
    }
}

unsafe impl Encode for MPSGraphReductionMode {
    const ENCODING: Encoding = NSUInteger::ENCODING;
}

unsafe impl RefEncode for MPSGraphReductionMode {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}
