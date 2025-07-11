use objc2::Encoding;
use objc2::{Encode, RefEncode};
use objc2_foundation::NSUInteger;

bitflags::bitflags! {
    #[allow(dead_code)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub struct AliasingStrategy: NSUInteger {
        const DEFAULT = 0;
        const DONT_CARE = Self::DEFAULT.bits();
        const SHALL_ALIAS = 1 << 0;
        const SHALL_NOT_ALIAS = 1 << 1;
        const ALIASING_RESERVED = Self::SHALL_ALIAS.bits() | Self::SHALL_NOT_ALIAS.bits();
        const PREFER_TEMPORARY_MEMORY = 1 << 2;
        const PREFER_NON_TEMPORARY_MEMORY = 1 << 3;
    }
}

impl From<AliasingStrategy> for NSUInteger {
    fn from(flags: AliasingStrategy) -> Self {
        flags.bits()
    }
}

unsafe impl Encode for AliasingStrategy {
    const ENCODING: Encoding = NSUInteger::ENCODING;
}

unsafe impl RefEncode for AliasingStrategy {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}
