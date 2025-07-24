//! Convolution-transpose helpers implemented directly on `Graph`.

use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use super::Convolution2DOpDescriptor;
use crate::{Graph, ShapeOrTensor, Tensor};

impl Graph {
    // ----- Forward ----------------------------------------------------------
    /// Creates a 2-D convolution-transpose operation and returns the result tensor.
    ///
    /// Convolution-transpose is identical to the convolution *data-gradient* operation
    /// (`convolution_2d_data_gradient_with_incoming_gradient_tensor_weights_tensor_output_shape_forward_convolution_descriptor`).
    /// A stride `s` upsamples the spatial dimensions by a factor of `s`.
    /// The relationship between the width of the *source* and the width of the *destination* is:
    ///
    /// `(sourceWidth - 1) * stride + 1 + (kernelWidth - 1) * dilationRate`
    /// `    <= destinationWidth + paddingLeft + paddingRight`.
    ///
    /// Because this inequality can hold for `stride - 1` different destination widths,
    /// the `output_shape` parameter is used to disambiguate.
    ///
    /// Parameters
    /// * `source` — Input tensor.
    /// * `weights` — Weights tensor.
    /// * `output_shape` — Desired shape of the result tensor.
    /// * `descriptor` — Descriptor of the corresponding forward 2-D convolution.
    /// * `name` — Optional debug name.
    pub fn convolution_transpose_2d(
        &self,
        source: &Tensor,
        weights: &Tensor,
        output_shape: &ShapeOrTensor,
        descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            match output_shape {
                ShapeOrTensor::Shape(shape) => {
                    msg_send![self,
                        convolutionTranspose2DWithSourceTensor: source,
                        weightsTensor: weights,
                        outputShape: shape,
                        descriptor: descriptor,
                        name: name.map(NSString::from_str).as_deref(),
                    ]
                }
                ShapeOrTensor::Tensor(shape_tensor) => {
                    msg_send![self,
                        convolutionTranspose2DWithSourceTensor: source,
                        weightsTensor: weights,
                        outputShapeTensor: shape_tensor,
                        descriptor: descriptor,
                        name: name.map(NSString::from_str).as_deref(),
                    ]
                }
            }
        }
    }

    // The dedicated tensor-output-shape variant is now redundant and has been removed.

    // ----- Gradients --------------------------------------------------------
    /// Creates a convolution-transpose *data-gradient* operation and returns the gradient
    /// with respect to the **source** tensor of the forward convolution-transpose.
    ///
    /// Parameters
    /// * `incoming_gradient_tensor` — Incoming gradient.
    /// * `weights_tensor` — Forward-pass weights tensor.
    /// * `output_shape` — Shape of the forward-pass *source* tensor.
    /// * `forward_convolution_descriptor` — Descriptor used in the forward op.
    /// * `name` — Optional debug name.
    pub fn convolution_transpose_2d_data_gradient(
        &self,
        incoming_gradient_tensor: &Tensor,
        weights_tensor: &Tensor,
        output_shape: &ShapeOrTensor,
        forward_convolution_descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            match output_shape {
                ShapeOrTensor::Shape(shape) => {
                    msg_send![self,
                        convolutionTranspose2DDataGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                        weightsTensor: weights_tensor,
                        outputShape: shape,
                        forwardConvolutionDescriptor: forward_convolution_descriptor,
                        name: name.map(NSString::from_str).as_deref()
                    ]
                }
                ShapeOrTensor::Tensor(shape_tensor) => {
                    msg_send![self,
                        convolutionTranspose2DDataGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                        weightsTensor: weights_tensor,
                        outputShapeTensor: shape_tensor,
                        forwardConvolutionDescriptor: forward_convolution_descriptor,
                        name: name.map(NSString::from_str).as_deref()
                    ]
                }
            }
        }
    }

    // Tensor-only overload removed.

    /// Creates a convolution-transpose *weights-gradient* operation and returns the gradient
    /// with respect to the **weights** tensor of the forward convolution-transpose.
    ///
    /// Parameters are analogous to the data-gradient variant, replacing `weights_tensor`
    /// with `source_tensor` and `output_shape` with the *weights* shape.
    pub fn convolution_transpose_2d_weights_gradient(
        &self,
        incoming_gradient_tensor: &Tensor,
        source_tensor: &Tensor,
        output_shape: &ShapeOrTensor,
        forward_convolution_descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            match output_shape {
                ShapeOrTensor::Shape(shape) => {
                    msg_send![self,
                        convolutionTranspose2DWeightsGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                        sourceTensor: source_tensor,
                        outputShape: shape,
                        forwardConvolutionDescriptor: forward_convolution_descriptor,
                        name: name.map(NSString::from_str).as_deref()
                    ]
                }
                ShapeOrTensor::Tensor(shape_tensor) => {
                    msg_send![self,
                        convolutionTranspose2DWeightsGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                        sourceTensor: source_tensor,
                        outputShapeTensor: shape_tensor,
                        forwardConvolutionDescriptor: forward_convolution_descriptor,
                        name: name.map(NSString::from_str).as_deref()
                    ]
                }
            }
        }
    }
}
