/// Convolution padding mode
#[repr(u64)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingMode {
    /// Valid padding - no padding
    Valid = 0,
    /// Same padding - pad to maintain same size
    Same = 1,
    /// Explicit padding - user-specified padding values
    Explicit = 2,
}
