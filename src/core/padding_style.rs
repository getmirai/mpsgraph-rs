use objc2::Encoding;
use objc2::{Encode, RefEncode};
use objc2_foundation::NSUInteger;

/// Tensor padding style
#[allow(dead_code)]
#[repr(usize)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum PaddingStyle {
    /// Explicit
    Explicit = 0,
    /// ONNX_SAME_LOWER (old TF_VALID naming)
    TFValid = 1,
    /// TF_SAME
    TFSame = 2,
    /// Explicit offsets
    ExplicitOffset = 3,
    /// ONNX_SAME_LOWER
    ONNXSAMELOWER = 4,
}

impl From<PaddingStyle> for NSUInteger {
    fn from(style: PaddingStyle) -> Self {
        style as NSUInteger
    }
}

unsafe impl Encode for PaddingStyle {
    const ENCODING: Encoding = NSUInteger::ENCODING;
}

unsafe impl RefEncode for PaddingStyle {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}
