use crate::graph::Graph;
use crate::shape::Shape;
use crate::tensor::Tensor;
use objc2::extern_class;
use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::AnyClass;
use objc2_foundation::{NSObject, NSObjectProtocol, NSString};

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

impl Graph {
    /// Creates a 2-D (forward) convolution operation and returns the result tensor.
    ///
    /// Corresponds to Objective-C selector `convolution2DWithSourceTensor:weightsTensor:descriptor:name:`.
    ///
    /// Parameters
    /// * `source_tensor` – Rank-4 source tensor. The layout is defined by `descriptor.data_layout`.
    /// * `weights_tensor` – Rank-4 weights tensor. The layout is defined by `descriptor.weights_layout`.
    /// * `descriptor` – Strides, dilation rates, padding and layout information.
    /// * `name` – Optional debug name.
    ///
    /// Returns a valid `Retained<Tensor>` with the convolution result.
    pub fn convolution_2d(
        &self,
        source_tensor: &Tensor,
        weights_tensor: &Tensor,
        descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                convolution2DWithSourceTensor: source_tensor,
                weightsTensor: weights_tensor,
                descriptor: descriptor,
                name: name_ptr
            ]
        }
    }

    /// Creates a 2-D convolution gradient operation with respect to the *source* tensor.
    ///
    /// Computes `dL/dS = dL/dR · dR/dS`, where `R` is the forward convolution result and
    /// `incoming_gradient_tensor` provides `dL/dR`.
    ///
    /// Parameters
    /// * `incoming_gradient_tensor` – Loss gradient wrt. forward result.
    /// * `weights_tensor` – Forward-pass weights tensor.
    /// * `output_shape` – Shape of the forward-pass source tensor.
    /// * `forward_convolution_descriptor` – Descriptor used in the forward op.
    /// * `name` – Optional debug name.
    pub fn convolution_2d_data_gradient(
        &self,
        incoming_gradient_tensor: &Tensor,
        weights_tensor: &Tensor,
        output_shape: &Shape,
        forward_convolution_descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                convolution2DDataGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                weightsTensor: weights_tensor,
                outputShape: output_shape.as_ptr(),
                forwardConvolutionDescriptor: forward_convolution_descriptor,
                name: name_ptr
            ]
        }
    }

    /// Same as [`convolution_2d_data_gradient_with_incoming_gradient_tensor_weights_tensor_output_shape_forward_convolution_descriptor`]
    /// but accepts the source *shape tensor* instead of a `Shape` object.
    pub fn convolution_2d_data_gradient_with_incoming_gradient_tensor_weights_tensor_output_shape_tensor_forward_convolution_descriptor(
        &self,
        incoming_gradient_tensor: &Tensor,
        weights_tensor: &Tensor,
        output_shape_tensor: &Tensor,
        forward_convolution_descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                convolution2DDataGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                weightsTensor: weights_tensor,
                outputShapeTensor: output_shape_tensor,
                forwardConvolutionDescriptor: forward_convolution_descriptor,
                name: name_ptr
            ]
        }
    }

    /// Creates a 2-D convolution gradient operation with respect to the *weights* tensor.
    ///
    /// Computes `dL/dW = dL/dR · dR/dW`, where `R` is the forward convolution result.
    ///
    /// Parameters are analogous to the data-gradient variant, replacing `weights_tensor` with
    /// `source_tensor`.
    pub fn convolution_2d_weights_gradient(
        &self,
        incoming_gradient_tensor: &Tensor,
        source_tensor: &Tensor,
        output_shape: &Shape,
        forward_convolution_descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                convolution2DWeightsGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                sourceTensor: source_tensor,
                outputShape: output_shape.as_ptr(),
                forwardConvolutionDescriptor: forward_convolution_descriptor,
                name: name_ptr
            ]
        }
    }

    /// As above, but receives the output shape as a tensor.
    pub fn convolution_2d_weights_gradient_with_incoming_gradient_tensor_source_tensor_output_shape_tensor_forward_convolution_descriptor(
        &self,
        incoming_gradient_tensor: &Tensor,
        source_tensor: &Tensor,
        output_shape_tensor: &Tensor,
        forward_convolution_descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                convolution2DWeightsGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                sourceTensor: source_tensor,
                outputShapeTensor: output_shape_tensor,
                forwardConvolutionDescriptor: forward_convolution_descriptor,
                name: name_ptr
            ]
        }
    }

    /// Creates a 3-D (forward) convolution operation. See the 2-D variant for parameter meaning.
    pub fn convolution_3d(
        &self,
        source_tensor: &Tensor,
        weights_tensor: &Tensor,
        descriptor: &Convolution3DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                convolution3DWithSourceTensor: source_tensor,
                weightsTensor: weights_tensor,
                descriptor: descriptor,
                name: name_ptr
            ]
        }
    }

    /// 3-D convolution data-gradient (output shape object).
    pub fn convolution_3d_data_gradient(
        &self,
        incoming_gradient_tensor: &Tensor,
        weights_tensor: &Tensor,
        output_shape: &Shape,
        forward_convolution_descriptor: &Convolution3DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                convolution3DDataGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                weightsTensor: weights_tensor,
                outputShape: output_shape.as_ptr(),
                forwardConvolutionDescriptor: forward_convolution_descriptor,
                name: name_ptr
            ]
        }
    }

    /// 3-D convolution data-gradient (output shape tensor).
    pub fn convolution_3d_data_gradient_with_incoming_gradient_tensor_weights_tensor_output_shape_tensor_forward_convolution_descriptor(
        &self,
        incoming_gradient_tensor: &Tensor,
        weights_tensor: &Tensor,
        output_shape_tensor: &Tensor,
        forward_convolution_descriptor: &Convolution3DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                convolution3DDataGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                weightsTensor: weights_tensor,
                outputShapeTensor: output_shape_tensor,
                forwardConvolutionDescriptor: forward_convolution_descriptor,
                name: name_ptr
            ]
        }
    }

    /// 3-D convolution weights-gradient (output shape object).
    pub fn convolution_3d_weights_gradient(
        &self,
        incoming_gradient_tensor: &Tensor,
        source_tensor: &Tensor,
        output_shape: &Shape,
        forward_convolution_descriptor: &Convolution3DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                convolution3DWeightsGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                sourceTensor: source_tensor,
                outputShape: output_shape.as_ptr(),
                forwardConvolutionDescriptor: forward_convolution_descriptor,
                name: name_ptr
            ]
        }
    }

    /// 3-D convolution weights-gradient (output shape tensor).
    pub fn convolution_3d_weights_gradient_with_incoming_gradient_tensor_source_tensor_output_shape_tensor_forward_convolution_descriptor(
        &self,
        incoming_gradient_tensor: &Tensor,
        source_tensor: &Tensor,
        output_shape_tensor: &Tensor,
        forward_convolution_descriptor: &Convolution3DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                convolution3DWeightsGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                sourceTensor: source_tensor,
                outputShapeTensor: output_shape_tensor,
                forwardConvolutionDescriptor: forward_convolution_descriptor,
                name: name_ptr
            ]
        }
    }
}

// -------------------------------------------------------------------------
// Extension traits providing tensor-shape overloads
// -------------------------------------------------------------------------

pub trait Convolution2DDataGradientTensorShapeExt {
    fn convolution_2d_data_gradient(
        &self,
        incoming_gradient_tensor: &Tensor,
        weights_tensor: &Tensor,
        output_shape_tensor: &Tensor,
        forward_convolution_descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;
}

impl Convolution2DDataGradientTensorShapeExt for Graph {
    fn convolution_2d_data_gradient(
        &self,
        incoming_gradient_tensor: &Tensor,
        weights_tensor: &Tensor,
        output_shape_tensor: &Tensor,
        forward_convolution_descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                convolution2DDataGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                weightsTensor: weights_tensor,
                outputShapeTensor: output_shape_tensor,
                forwardConvolutionDescriptor: forward_convolution_descriptor,
                name: name_ptr
            ]
        }
    }
}

pub trait Convolution2DWeightsGradientTensorShapeExt {
    fn convolution_2d_weights_gradient(
        &self,
        incoming_gradient_tensor: &Tensor,
        source_tensor: &Tensor,
        output_shape_tensor: &Tensor,
        forward_convolution_descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;
}

impl Convolution2DWeightsGradientTensorShapeExt for Graph {
    fn convolution_2d_weights_gradient(
        &self,
        incoming_gradient_tensor: &Tensor,
        source_tensor: &Tensor,
        output_shape_tensor: &Tensor,
        forward_convolution_descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                convolution2DWeightsGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                sourceTensor: source_tensor,
                outputShapeTensor: output_shape_tensor,
                forwardConvolutionDescriptor: forward_convolution_descriptor,
                name: name_ptr
            ]
        }
    }
}

pub trait Convolution3DDataGradientTensorShapeExt {
    fn convolution_3d_data_gradient(
        &self,
        incoming_gradient_tensor: &Tensor,
        weights_tensor: &Tensor,
        output_shape_tensor: &Tensor,
        forward_convolution_descriptor: &Convolution3DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;
}

impl Convolution3DDataGradientTensorShapeExt for Graph {
    fn convolution_3d_data_gradient(
        &self,
        incoming_gradient_tensor: &Tensor,
        weights_tensor: &Tensor,
        output_shape_tensor: &Tensor,
        forward_convolution_descriptor: &Convolution3DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                convolution3DDataGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                weightsTensor: weights_tensor,
                outputShapeTensor: output_shape_tensor,
                forwardConvolutionDescriptor: forward_convolution_descriptor,
                name: name_ptr
            ]
        }
    }
}

pub trait Convolution3DWeightsGradientTensorShapeExt {
    fn convolution_3d_weights_gradient(
        &self,
        incoming_gradient_tensor: &Tensor,
        source_tensor: &Tensor,
        output_shape_tensor: &Tensor,
        forward_convolution_descriptor: &Convolution3DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;
}

impl Convolution3DWeightsGradientTensorShapeExt for Graph {
    fn convolution_3d_weights_gradient(
        &self,
        incoming_gradient_tensor: &Tensor,
        source_tensor: &Tensor,
        output_shape_tensor: &Tensor,
        forward_convolution_descriptor: &Convolution3DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                convolution3DWeightsGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                sourceTensor: source_tensor,
                outputShapeTensor: output_shape_tensor,
                forwardConvolutionDescriptor: forward_convolution_descriptor,
                name: name_ptr
            ]
        }
    }
}
