/// Dataflow direction for convolution
#[repr(u64)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvolutionDataLayout {
    /// Data is arranged as NHWC (batch, height, width, channels)
    NHWC = 0,
    /// Data is arranged as NCHW (batch, channels, height, width)
    NCHW = 1,
}
