use objc2::extern_class;
use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::AnyClass;
use objc2_foundation::{NSObject, NSObjectProtocol};

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

/// Dataflow direction for convolution
#[repr(u64)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvolutionDataLayout {
    /// Data is arranged as NHWC (batch, height, width, channels)
    NHWC = 0,
    /// Data is arranged as NCHW (batch, channels, height, width)
    NCHW = 1,
}

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

extern_class!(
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraphConvolution2DOpDescriptor"]
    /// Descriptor for 2D convolution operations
    pub struct Convolution2DOpDescriptor;
);

unsafe impl NSObjectProtocol for Convolution2DOpDescriptor {}

extern_class!(
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraphConvolution3DOpDescriptor"]
    /// Descriptor for 3D convolution operations
    pub struct Convolution3DOpDescriptor;
);

unsafe impl NSObjectProtocol for Convolution3DOpDescriptor {}

impl Convolution2DOpDescriptor {
    /// Creates a new descriptor with default parameters
    pub fn new() -> Retained<Self> {
        unsafe {
            let cls = AnyClass::get(c"MPSGraphConvolution2DOpDescriptor").unwrap();
            msg_send![cls, descriptor]
        }
    }

    /// Sets the stride width (x direction)
    pub fn set_stride_x(&self, stride_x: usize) {
        unsafe {
            let _: () = msg_send![self, setStrideInX: stride_x as u64];
        }
    }

    /// Sets the stride height (y direction)
    pub fn set_stride_y(&self, stride_y: usize) {
        unsafe {
            let _: () = msg_send![self, setStrideInY: stride_y as u64];
        }
    }

    /// Sets the dilation rate in width (x direction)
    pub fn set_dilation_x(&self, dilation_x: usize) {
        unsafe {
            let _: () = msg_send![self, setDilationRateInX: dilation_x as u64];
        }
    }

    /// Sets the dilation rate in height (y direction)
    pub fn set_dilation_y(&self, dilation_y: usize) {
        unsafe {
            let _: () = msg_send![self, setDilationRateInY: dilation_y as u64];
        }
    }

    /// Sets the padding mode
    pub fn set_padding_mode(&self, mode: PaddingMode) {
        unsafe {
            let _: () = msg_send![self, setPaddingMode: mode as u64];
        }
    }

    /// Sets the padding values for explicit padding mode
    pub fn set_padding_values(&self, left: usize, right: usize, top: usize, bottom: usize) {
        unsafe {
            let _: () = msg_send![self,
                setPaddingLeft: left as u64,
                paddingRight: right as u64,
                paddingTop: top as u64,
                paddingBottom: bottom as u64
            ];
        }
    }

    /// Sets the data layout (NHWC or NCHW)
    pub fn set_data_layout(&self, layout: ConvolutionDataLayout) {
        unsafe {
            let _: () = msg_send![self, setDataLayout: layout as u64];
        }
    }

    /// Sets the weights layout (HWIO, OHWI, or IHWO)
    pub fn set_weights_layout(&self, layout: WeightsLayout) {
        unsafe {
            let _: () = msg_send![self, setWeightsLayout: layout as u64];
        }
    }

    /// Sets the number of groups for grouped convolution
    pub fn set_groups(&self, groups: usize) {
        unsafe {
            let _: () = msg_send![self, setGroups: groups as u64];
        }
    }
}

impl Convolution3DOpDescriptor {
    /// Creates a new descriptor with default parameters
    pub fn new() -> Retained<Self> {
        unsafe {
            let cls = AnyClass::get(c"MPSGraphConvolution3DOpDescriptor").unwrap();
            msg_send![cls, descriptor]
        }
    }

    /// Sets the stride in each dimension
    pub fn set_strides(&self, stride_x: usize, stride_y: usize, stride_z: usize) {
        unsafe {
            let _: () = msg_send![self,
                setStrideInX: stride_x as u64,
                strideInY: stride_y as u64,
                strideInZ: stride_z as u64
            ];
        }
    }

    /// Sets the dilation rate in each dimension
    pub fn set_dilation_rates(&self, dilation_x: usize, dilation_y: usize, dilation_z: usize) {
        unsafe {
            let _: () = msg_send![self,
                setDilationRateInX: dilation_x as u64,
                dilationRateInY: dilation_y as u64,
                dilationRateInZ: dilation_z as u64
            ];
        }
    }

    /// Sets the padding mode
    pub fn set_padding_mode(&self, mode: PaddingMode) {
        unsafe {
            let _: () = msg_send![self, setPaddingMode: mode as u64];
        }
    }

    /// Sets the padding values for explicit padding mode in each dimension
    pub fn set_padding_values(
        &self,
        left: usize,
        right: usize,
        top: usize,
        bottom: usize,
        front: usize,
        back: usize,
    ) {
        unsafe {
            let _: () = msg_send![self,
                setPaddingLeft: left as u64,
                paddingRight: right as u64,
                paddingTop: top as u64,
                paddingBottom: bottom as u64,
                paddingFront: front as u64,
                paddingBack: back as u64
            ];
        }
    }

    /// Sets the data layout
    pub fn set_data_layout(&self, layout: ConvolutionDataLayout) {
        unsafe {
            let _: () = msg_send![self, setDataLayout: layout as u64];
        }
    }

    /// Sets the weights layout
    pub fn set_weights_layout(&self, layout: WeightsLayout) {
        unsafe {
            let _: () = msg_send![self, setWeightsLayout: layout as u64];
        }
    }

    /// Sets the number of groups for grouped convolution
    pub fn set_groups(&self, groups: usize) {
        unsafe {
            let _: () = msg_send![self, setGroups: groups as u64];
        }
    }
}
