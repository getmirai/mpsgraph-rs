use crate::{Graph, Shape, ShapeOrTensor, Tensor};
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
            msg_send![
                self,
                convolution2DWithSourceTensor: source_tensor,
                weightsTensor: weights_tensor,
                descriptor: descriptor,
                name: name.map(NSString::from_str).as_deref()
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
        output_shape: &ShapeOrTensor,
        forward_convolution_descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            match output_shape {
                ShapeOrTensor::Shape(output_shape) => {
                    msg_send![
                        self,
                        convolution2DDataGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                        weightsTensor: weights_tensor,
                        outputShape: output_shape,
                        forwardConvolutionDescriptor: forward_convolution_descriptor,
                        name: name.map(NSString::from_str).as_deref()
                    ]
                }
                ShapeOrTensor::Tensor(output_shape_tensor) => {
                    msg_send![
                        self,
                        convolution2DDataGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                        weightsTensor: weights_tensor,
                        outputShapeTensor: output_shape_tensor,
                        forwardConvolutionDescriptor: forward_convolution_descriptor,
                        name: name.map(NSString::from_str).as_deref()
                    ]
                }
            }
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
        output_shape: &ShapeOrTensor,
        forward_convolution_descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            match output_shape {
                ShapeOrTensor::Shape(output_shape) => {
                    msg_send![
                        self,
                        convolution2DWeightsGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                        sourceTensor: source_tensor,
                        outputShape: output_shape,
                        forwardConvolutionDescriptor: forward_convolution_descriptor,
                        name: name.map(NSString::from_str).as_deref()
                    ]
                }
                ShapeOrTensor::Tensor(output_shape_tensor) => {
                    msg_send![
                        self,
                        convolution2DWeightsGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                        sourceTensor: source_tensor,
                        outputShapeTensor: output_shape_tensor,
                        forwardConvolutionDescriptor: forward_convolution_descriptor,
                        name: name.map(NSString::from_str).as_deref()
                    ]
                }
            }
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
            msg_send![
                self,
                convolution3DWithSourceTensor: source_tensor,
                weightsTensor: weights_tensor,
                descriptor: descriptor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Creates a 3-D convolution gradient operation with respect to the *source* tensor.
    ///
    /// Computes `dL/dS = dL/dR · dR/dS`, where `R` is the forward convolution result and
    /// `incoming_gradient_tensor` provides `dL/dR`.
    ///
    /// The `output_shape` can be supplied either as a static `Shape` or dynamically as a `Tensor`
    /// via `ShapeOrTensor`, matching the interface of the 2-D variant.
    pub fn convolution_3d_data_gradient(
        &self,
        incoming_gradient_tensor: &Tensor,
        weights_tensor: &Tensor,
        output_shape: &ShapeOrTensor,
        forward_convolution_descriptor: &Convolution3DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            match output_shape {
                ShapeOrTensor::Shape(output_shape) => {
                    msg_send![
                        self,
                        convolution3DDataGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                        weightsTensor: weights_tensor,
                        outputShape: output_shape,
                        forwardConvolutionDescriptor: forward_convolution_descriptor,
                        name: name.map(NSString::from_str).as_deref()
                    ]
                }
                ShapeOrTensor::Tensor(output_shape_tensor) => {
                    msg_send![
                        self,
                        convolution3DDataGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                        weightsTensor: weights_tensor,
                        outputShapeTensor: output_shape_tensor,
                        forwardConvolutionDescriptor: forward_convolution_descriptor,
                        name: name.map(NSString::from_str).as_deref()
                    ]
                }
            }
        }
    }

    /// Creates a 3-D convolution gradient operation with respect to the *weights* tensor.
    ///
    /// Computes `dL/dW = dL/dR · dR/dW`, where `R` is the forward convolution result.
    /// `output_shape` can be passed either as a static `Shape` or a dynamic `Tensor` via
    /// `ShapeOrTensor`, analogous to the 2-D variant.
    pub fn convolution_3d_weights_gradient(
        &self,
        incoming_gradient_tensor: &Tensor,
        source_tensor: &Tensor,
        output_shape: &ShapeOrTensor,
        forward_convolution_descriptor: &Convolution3DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            match output_shape {
                ShapeOrTensor::Shape(output_shape) => {
                    msg_send![
                        self,
                        convolution3DWeightsGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                        sourceTensor: source_tensor,
                        outputShape: output_shape,
                        forwardConvolutionDescriptor: forward_convolution_descriptor,
                        name: name.map(NSString::from_str).as_deref()
                    ]
                }
                ShapeOrTensor::Tensor(output_shape_tensor) => {
                    msg_send![
                        self,
                        convolution3DWeightsGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                        sourceTensor: source_tensor,
                        outputShapeTensor: output_shape_tensor,
                        forwardConvolutionDescriptor: forward_convolution_descriptor,
                        name: name.map(NSString::from_str).as_deref()
                    ]
                }
            }
        }
    }
}
