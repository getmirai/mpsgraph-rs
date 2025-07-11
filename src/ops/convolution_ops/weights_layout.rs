/// Weight layout for convolution
#[repr(u64)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightsLayout {
    /// Weights arranged as HWIO (height, width, input channels, output channels)
    HWIO = 0,
    /// Weights arranged as OHWI (output channels, height, width, input channels)
    OHWI = 1,
    /// Weights arranged as IHWO (input channels, height, width, output channels)
    IHWO = 2,
}
