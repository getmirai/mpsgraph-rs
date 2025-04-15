use crate::core::{AsRawObject, NSString};
use crate::graph::Graph;
use crate::shape::Shape;
use crate::tensor::Tensor;
use objc2::msg_send;
use objc2::runtime::AnyObject;

/// Defines the data layout for tensors
#[repr(u64)]
#[derive(Debug, Copy, Clone)]
pub enum TensorNamedDataLayout {
    NCHW = 0, // Batch, Channels, Height, Width
    NHWC = 1, // Batch, Height, Width, Channels
    CHWN = 2, // Channels, Height, Width, Batch
    HWC = 3,  // Height, Width, Channels
    CHW = 4,  // Channels, Height, Width
}

/// Defines the padding style for convolution operations
#[repr(u64)]
#[derive(Debug, Copy, Clone)]
pub enum PaddingStyle {
    Explicit = 0,
    TfSame = 1,
    TfValid = 2,
}

// Re-export Convolution2DOpDescriptor from convolution_ops
pub use crate::convolution_ops::Convolution2DOpDescriptor;

/// Transposed convolution operations for Graph
impl Graph {
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
    pub fn convolution_transpose_2d(
        &self,
        source: &Tensor,
        weights: &Tensor,
        output_shape: &Shape,
        descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Tensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0, convolutionTranspose2DWithSourceTensor: source.0,
                weightsTensor: weights.0,
                outputShape: output_shape.0,
                descriptor: descriptor.0,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            Tensor(tensor)
        }
    }

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
    pub fn convolution_transpose_2d_with_tensor_shape(
        &self,
        source: &Tensor,
        weights: &Tensor,
        output_shape_tensor: &Tensor,
        descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Tensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0, convolutionTranspose2DWithSourceTensor: source.0,
                weightsTensor: weights.0,
                outputShapeTensor: output_shape_tensor.0,
                descriptor: descriptor.0,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            Tensor(tensor)
        }
    }

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
    pub fn convolution_transpose_2d_data_gradient(
        &self,
        incoming_gradient: &Tensor,
        weights: &Tensor,
        output_shape: &Shape,
        forward_convolution_descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Tensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0, convolutionTranspose2DDataGradientWithIncomingGradientTensor: incoming_gradient.0,
                weightsTensor: weights.0,
                outputShape: output_shape.0,
                forwardConvolutionDescriptor: forward_convolution_descriptor.0,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            Tensor(tensor)
        }
    }

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
    pub fn convolution_transpose_2d_data_gradient_with_tensor_shape(
        &self,
        incoming_gradient: &Tensor,
        weights: &Tensor,
        output_shape_tensor: &Tensor,
        forward_convolution_descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Tensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0, convolutionTranspose2DDataGradientWithIncomingGradientTensor: incoming_gradient.0,
                weightsTensor: weights.0,
                outputShapeTensor: output_shape_tensor.0,
                forwardConvolutionDescriptor: forward_convolution_descriptor.0,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            Tensor(tensor)
        }
    }

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
    pub fn convolution_transpose_2d_weights_gradient(
        &self,
        incoming_gradient: &Tensor,
        source: &Tensor,
        output_shape: &Shape,
        forward_convolution_descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Tensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0, convolutionTranspose2DWeightsGradientWithIncomingGradientTensor: incoming_gradient.0,
                sourceTensor: source.0,
                outputShape: output_shape.0,
                forwardConvolutionDescriptor: forward_convolution_descriptor.0,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            Tensor(tensor)
        }
    }

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
    pub fn convolution_transpose_2d_weights_gradient_with_tensor_shape(
        &self,
        incoming_gradient: &Tensor,
        source: &Tensor,
        output_shape_tensor: &Tensor,
        forward_convolution_descriptor: &Convolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Tensor {
        let name_obj = match name {
            Some(s) => NSString::from_str(s).as_raw_object(),
            None => std::ptr::null_mut(),
        };

        unsafe {
            let tensor: *mut AnyObject = msg_send![
                self.0, convolutionTranspose2DWeightsGradientWithIncomingGradientTensor: incoming_gradient.0,
                sourceTensor: source.0,
                outputShapeTensor: output_shape_tensor.0,
                forwardConvolutionDescriptor: forward_convolution_descriptor.0,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            Tensor(tensor)
        }
    }
}
