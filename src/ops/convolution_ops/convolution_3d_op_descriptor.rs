use super::{ConvolutionDataLayout, PaddingMode, WeightsLayout};
use objc2::extern_class;
use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::AnyClass;
use objc2_foundation::{NSObject, NSObjectProtocol};

extern_class!(
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraphConvolution3DOpDescriptor"]
    /// Descriptor for 3D convolution operations
    pub struct Convolution3DOpDescriptor;
);

unsafe impl NSObjectProtocol for Convolution3DOpDescriptor {}

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
