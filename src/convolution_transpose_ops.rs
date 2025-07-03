//! Convolution-transpose helpers implemented directly on `Graph`.

use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::convolution_ops::{Convolution2DOpDescriptor, PaddingMode};
use crate::graph::Graph;
use crate::shape::Shape;
use crate::tensor::Tensor;
// Re-export types for external convenience
pub use crate::pooling_ops::{PaddingStyle, TensorNamedDataLayout};

impl Graph {
    // ----- Forward ----------------------------------------------------------
    pub fn convolution_transpose_2d(
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
                outputShape: output_shape,
                descriptor: descriptor,
                name: name_ptr]
        }
    }

    pub fn convolution_transpose_2d_with_tensor_shape(
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
    pub fn convolution_transpose_2d_data_gradient(
        &self,
        incoming_gradient: &Tensor,
        weights: &Tensor,
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
                convolutionTranspose2DDataGradientWithIncomingGradientTensor: incoming_gradient,
                weightsTensor: weights,
                outputShape: output_shape,
                forwardConvolutionDescriptor: forward_convolution_descriptor,
                name: name_ptr]
        }
    }

    pub fn convolution_transpose_2d_data_gradient_with_tensor_shape(
        &self,
        incoming_gradient: &Tensor,
        weights: &Tensor,
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
                convolutionTranspose2DDataGradientWithIncomingGradientTensor: incoming_gradient,
                weightsTensor: weights,
                outputShapeTensor: output_shape_tensor,
                forwardConvolutionDescriptor: forward_convolution_descriptor,
                name: name_ptr]
        }
    }

    pub fn convolution_transpose_2d_weights_gradient(
        &self,
        incoming_gradient: &Tensor,
        source: &Tensor,
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
                convolutionTranspose2DWeightsGradientWithIncomingGradientTensor: incoming_gradient,
                sourceTensor: source,
                outputShape: output_shape,
                forwardConvolutionDescriptor: forward_convolution_descriptor,
                name: name_ptr]
        }
    }

    pub fn convolution_transpose_2d_weights_gradient_with_tensor_shape(
        &self,
        incoming_gradient: &Tensor,
        source: &Tensor,
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
                convolutionTranspose2DWeightsGradientWithIncomingGradientTensor: incoming_gradient,
                sourceTensor: source,
                outputShapeTensor: output_shape_tensor,
                forwardConvolutionDescriptor: forward_convolution_descriptor,
                name: name_ptr]
        }
    }
}
