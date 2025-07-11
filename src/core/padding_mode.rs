use objc2::Encoding;
use objc2::{Encode, RefEncode};
use objc2_foundation::NSInteger;

/// Tensor padding mode
#[allow(dead_code)]
#[repr(isize)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum MPSGraphPaddingMode {
    /// Constant
    Constant = 0,
    /// Reflect
    Reflect = 1,
    /// Symmetric
    Symmetric = 2,
    /// ClampToEdge (PyTorch ReplicationPad)
    ClampToEdge = 3,
    /// Zero
    Zero = 4,
    /// Periodic `x[-2] -> x[L-3]`
    Periodic = 5,
    /// Anti Periodic `x[-2] -> -x[L-3]`
    AntiPeriodic = 6,
}

impl From<MPSGraphPaddingMode> for NSInteger {
    fn from(mode: MPSGraphPaddingMode) -> Self {
        mode as NSInteger
    }
}

unsafe impl Encode for MPSGraphPaddingMode {
    const ENCODING: Encoding = NSInteger::ENCODING;
}

unsafe impl RefEncode for MPSGraphPaddingMode {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}
