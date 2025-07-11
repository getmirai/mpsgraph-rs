//! Convolution-transpose helpers implemented directly on `Graph`.

use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::convolution_ops::Convolution2DOpDescriptor;
use crate::graph::Graph;
use crate::tensor::Tensor;
use crate::Shape;
// Re-export types for external convenience
pub use crate::pooling_ops::{PaddingStyle, TensorNamedDataLayout};

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
    pub fn convolution_transpose_2d_with_source_tensor_weights_tensor_output_shape_descriptor(
        &self,
        source: &Tensor,
        weights: &Tensor,
        output_shape: &Shape,
        descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self,
                convolutionTranspose2DWithSourceTensor: source,
                weightsTensor: weights,
                outputShape: output_shape.as_ptr(),
                descriptor: descriptor,
                name: name_ptr]
        }
    }

    /// Same as [`convolution_transpose_2d_with_source_tensor_weights_tensor_output_shape_descriptor`]
    /// but receives the *output shape* as a rank-1 Int32/Int64 tensor.
    pub fn convolution_transpose_2d_with_source_tensor_weights_tensor_output_shape_tensor_descriptor(
        &self,
        source: &Tensor,
        weights: &Tensor,
        output_shape_tensor: &Tensor,
        descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self,
                convolutionTranspose2DWithSourceTensor: source,
                weightsTensor: weights,
                outputShapeTensor: output_shape_tensor,
                descriptor: descriptor,
                name: name_ptr]
        }
    }

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
    pub fn convolution_transpose_2d_data_gradient_with_incoming_gradient_tensor_weights_tensor_output_shape_forward_convolution_descriptor(
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
            msg_send![self,
                convolutionTranspose2DDataGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                weightsTensor: weights_tensor,
                outputShape: output_shape.as_ptr(),
                forwardConvolutionDescriptor: forward_convolution_descriptor,
                name: name_ptr]
        }
    }

    /// Same as [`convolution_transpose_2d_data_gradient_with_incoming_gradient_tensor_weights_tensor_output_shape_forward_convolution_descriptor`]
    /// but takes `output_shape_tensor` instead of a [`Shape`] object.
    pub fn convolution_transpose_2d_data_gradient_with_incoming_gradient_tensor_weights_tensor_output_shape_tensor_forward_convolution_descriptor(
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
            msg_send![self,
                convolutionTranspose2DDataGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                weightsTensor: weights_tensor,
                outputShapeTensor: output_shape_tensor,
                forwardConvolutionDescriptor: forward_convolution_descriptor,
                name: name_ptr]
        }
    }

    /// Creates a convolution-transpose *weights-gradient* operation and returns the gradient
    /// with respect to the **weights** tensor of the forward convolution-transpose.
    ///
    /// Parameters are analogous to the data-gradient variant, replacing `weights_tensor`
    /// with `source_tensor` and `output_shape` with the *weights* shape.
    pub fn convolution_transpose_2d_weights_gradient_with_incoming_gradient_tensor_source_tensor_output_shape_forward_convolution_descriptor(
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
            msg_send![self,
                convolutionTranspose2DWeightsGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                sourceTensor: source_tensor,
                outputShape: output_shape.as_ptr(),
                forwardConvolutionDescriptor: forward_convolution_descriptor,
                name: name_ptr]
        }
    }

    /// Same as [`convolution_transpose_2d_weights_gradient_with_incoming_gradient_tensor_source_tensor_output_shape_forward_convolution_descriptor`]
    /// but takes `output_shape_tensor` instead of a [`Shape`] object.
    pub fn convolution_transpose_2d_weights_gradient_with_incoming_gradient_tensor_source_tensor_output_shape_tensor_forward_convolution_descriptor(
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
            msg_send![self,
                convolutionTranspose2DWeightsGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                sourceTensor: source_tensor,
                outputShapeTensor: output_shape_tensor,
                forwardConvolutionDescriptor: forward_convolution_descriptor,
                name: name_ptr]
        }
    }
}
