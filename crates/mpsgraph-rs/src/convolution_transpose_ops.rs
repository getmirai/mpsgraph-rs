use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::convolution_ops::Convolution2DOpDescriptor;
use crate::graph::Graph;
use crate::shape::Shape;
use crate::tensor::Tensor;

// Import PaddingMode from convolution_ops and PaddingStyle from pooling_ops
pub use crate::convolution_ops::PaddingMode;
pub use crate::pooling_ops::{PaddingStyle, TensorNamedDataLayout};

/// Trait for convolution transpose operations on a Graph
pub trait GraphConvolutionTransposeOps {
    /// Creates a 2D convolution transpose operation and returns the result tensor.
    ///
    /// Convolution Tranpose operation is exactly the same as convolution gradient with respect to input image.
    /// Weights tensor and source tensors are interpreted as they are in convolution data gradient.
    /// Convolution with stride `s` downsamples source tensor by factor `s` in spatial dimensions whereas
    /// convolution tranpose with stride `s` upsamples source tensor by factor `s`.
    ///
    /// # Arguments
    ///
    /// * `source` - Source tensor
    /// * `weights` - Weights tensor
    /// * `output_shape` - Shape of the result tensor
    /// * `descriptor` - Descriptor for the corresponding forward 2D-convolution operation
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new Tensor containing the result
    fn convolution_transpose_2d(
        &self,
        source: &Tensor,
        weights: &Tensor,
        output_shape: &Shape,
        descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a 2D convolution transpose operation with tensor output shape
    ///
    /// # Arguments
    ///
    /// * `source` - Source tensor
    /// * `weights` - Weights tensor
    /// * `output_shape_tensor` - 1D Int32 or Int64 tensor. Shape of the result tensor
    /// * `descriptor` - Descriptor for the corresponding forward 2D-convolution operation
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new Tensor containing the result
    fn convolution_transpose_2d_with_tensor_shape(
        &self,
        source: &Tensor,
        weights: &Tensor,
        output_shape_tensor: &Tensor,
        descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a convolution transpose gradient operation with respect to the source tensor
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient` - Incoming gradient tensor
    /// * `weights` - Forward pass weights tensor
    /// * `output_shape` - Shape of the forward pass source tensor
    /// * `forward_convolution_descriptor` - Forward pass op descriptor
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new Tensor containing the gradient with respect to source
    fn convolution_transpose_2d_data_gradient(
        &self,
        incoming_gradient: &Tensor,
        weights: &Tensor,
        output_shape: &Shape,
        forward_convolution_descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a convolution transpose gradient operation with respect to the source tensor (with tensor output shape)
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient` - Incoming gradient tensor
    /// * `weights` - Forward pass weights tensor
    /// * `output_shape_tensor` - 1D Int32 or Int64 tensor. Shape of the forward pass source tensor
    /// * `forward_convolution_descriptor` - Forward pass op descriptor
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new Tensor containing the gradient with respect to source
    fn convolution_transpose_2d_data_gradient_with_tensor_shape(
        &self,
        incoming_gradient: &Tensor,
        weights: &Tensor,
        output_shape_tensor: &Tensor,
        forward_convolution_descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a convolution transpose weights gradient operation
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient` - Incoming gradient tensor
    /// * `source` - Forward pass source tensor
    /// * `output_shape` - Shape of the forward pass weights tensor
    /// * `forward_convolution_descriptor` - Forward pass op descriptor
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new Tensor containing the gradient with respect to weights
    fn convolution_transpose_2d_weights_gradient(
        &self,
        incoming_gradient: &Tensor,
        source: &Tensor,
        output_shape: &Shape,
        forward_convolution_descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a convolution transpose weights gradient operation (with tensor output shape)
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient` - Incoming gradient tensor
    /// * `source` - Forward pass source tensor
    /// * `output_shape_tensor` - 1D Int32 or Int64 tensor. Shape of the forward pass weights tensor
    /// * `forward_convolution_descriptor` - Forward pass op descriptor
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new Tensor containing the gradient with respect to weights
    fn convolution_transpose_2d_weights_gradient_with_tensor_shape(
        &self,
        incoming_gradient: &Tensor,
        source: &Tensor,
        output_shape_tensor: &Tensor,
        forward_convolution_descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;
}

/// Implementation of convolution transpose operations for Graph
impl GraphConvolutionTransposeOps for Graph {
    fn convolution_transpose_2d(
        &self,
        source: &Tensor,
        weights: &Tensor,
        output_shape: &Shape,
        descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: Retained<Tensor> = msg_send![
                self,
                convolutionTranspose2DWithSourceTensor: source,
                weightsTensor: weights,
                outputShape: output_shape,
                descriptor: descriptor,
                name: name_ptr,
            ];
            result
        }
    }

    fn convolution_transpose_2d_with_tensor_shape(
        &self,
        source: &Tensor,
        weights: &Tensor,
        output_shape_tensor: &Tensor,
        descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: Retained<Tensor> = msg_send![
                self,
                convolutionTranspose2DWithSourceTensor: source,
                weightsTensor: weights,
                outputShapeTensor: output_shape_tensor,
                descriptor: descriptor,
                name: name_ptr,
            ];
            result
        }
    }

    fn convolution_transpose_2d_data_gradient(
        &self,
        incoming_gradient: &Tensor,
        weights: &Tensor,
        output_shape: &Shape,
        forward_convolution_descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: Retained<Tensor> = msg_send![
                self,
                convolutionTranspose2DDataGradientWithIncomingGradientTensor: incoming_gradient,
                weightsTensor: weights,
                outputShape: output_shape,
                forwardConvolutionDescriptor: forward_convolution_descriptor,
                name: name_ptr,
            ];
            result
        }
    }

    fn convolution_transpose_2d_data_gradient_with_tensor_shape(
        &self,
        incoming_gradient: &Tensor,
        weights: &Tensor,
        output_shape_tensor: &Tensor,
        forward_convolution_descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: Retained<Tensor> = msg_send![
                self,
                convolutionTranspose2DDataGradientWithIncomingGradientTensor: incoming_gradient,
                weightsTensor: weights,
                outputShapeTensor: output_shape_tensor,
                forwardConvolutionDescriptor: forward_convolution_descriptor,
                name: name_ptr,
            ];
            result
        }
    }

    fn convolution_transpose_2d_weights_gradient(
        &self,
        incoming_gradient: &Tensor,
        source: &Tensor,
        output_shape: &Shape,
        forward_convolution_descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: Retained<Tensor> = msg_send![
                self,
                convolutionTranspose2DWeightsGradientWithIncomingGradientTensor: incoming_gradient,
                sourceTensor: source,
                outputShape: output_shape,
                forwardConvolutionDescriptor: forward_convolution_descriptor,
                name: name_ptr,
            ];
            result
        }
    }

    fn convolution_transpose_2d_weights_gradient_with_tensor_shape(
        &self,
        incoming_gradient: &Tensor,
        source: &Tensor,
        output_shape_tensor: &Tensor,
        forward_convolution_descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: Retained<Tensor> = msg_send![
                self,
                convolutionTranspose2DWeightsGradientWithIncomingGradientTensor: incoming_gradient,
                sourceTensor: source,
                outputShapeTensor: output_shape_tensor,
                forwardConvolutionDescriptor: forward_convolution_descriptor,
                name: name_ptr,
            ];
            result
        }
    }
}
