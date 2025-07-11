use super::{ConvolutionDataLayout, PaddingMode, WeightsLayout};
use objc2::extern_class;
use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::AnyClass;
use objc2_foundation::{NSObject, NSObjectProtocol};

extern_class!(
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraphConvolution2DOpDescriptor"]
    /// A struct that describes the properties of a 2D-convolution operator.
    ///
    /// Use it to add a 2D-convolution operator with the desired properties to the graph.
    pub struct Convolution2DOpDescriptor;
);

unsafe impl NSObjectProtocol for Convolution2DOpDescriptor {}

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
