use crate::graph::Graph;
use crate::shape::Shape;
use crate::tensor::Tensor;
use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

mod convolution_2d_op_descriptor;
mod convolution_3d_op_descriptor;
mod convolution_data_layout;
mod padding_mode;
mod weights_layout;

pub use convolution_2d_op_descriptor::Convolution2DOpDescriptor;
pub use convolution_3d_op_descriptor::Convolution3DOpDescriptor;
pub use convolution_data_layout::ConvolutionDataLayout;
pub use padding_mode::PaddingMode;
pub use weights_layout::WeightsLayout;

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
