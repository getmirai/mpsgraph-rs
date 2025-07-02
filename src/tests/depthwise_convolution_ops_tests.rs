use crate::convolution_ops::PaddingMode;
use crate::depthwise_convolution_ops::{
    DepthwiseConvolution2DOpDescriptor, DepthwiseConvolution3DOpDescriptor,
};
use crate::pooling_ops::TensorNamedDataLayout;

#[test]
fn test_depthwise_convolution_2d_descriptor() {
    // Create a new descriptor
    let descriptor = DepthwiseConvolution2DOpDescriptor::new();
    
    // Set strides
    descriptor.set_stride_in_x(2);
    descriptor.set_stride_in_y(2);
    
    // Set dilation rates
    descriptor.set_dilation_rate_in_x(1);
    descriptor.set_dilation_rate_in_y(1);
    
    // Set padding values
    descriptor.set_padding_left(1);
    descriptor.set_padding_right(1);
    descriptor.set_padding_top(1);
    descriptor.set_padding_bottom(1);
    
    // Set padding style
    descriptor.set_padding_style(PaddingMode::Explicit);
    
    // Set data and weights layouts
    descriptor.set_data_layout(TensorNamedDataLayout::NCHW);
    descriptor.set_weights_layout(TensorNamedDataLayout::NCHW);
    
    // Set explicit padding
    descriptor.set_explicit_padding(1, 1, 1, 1);
}

#[test]
fn test_depthwise_convolution_2d_descriptor_with_layouts() {
    // Create a new descriptor with specified layouts
    let descriptor = DepthwiseConvolution2DOpDescriptor::new_with_layouts(
        TensorNamedDataLayout::NCHW,
        TensorNamedDataLayout::NHWC,
    );
    
    // Set strides
    descriptor.set_stride_in_x(1);
    descriptor.set_stride_in_y(1);
    
    // Set padding style
    descriptor.set_padding_style(PaddingMode::Same);
}

#[test]
fn test_depthwise_convolution_3d_descriptor() {
    // Create a new descriptor
    let descriptor = DepthwiseConvolution3DOpDescriptor::new(PaddingMode::Valid);
    
    // Set strides
    descriptor.set_strides(&[1, 1, 1]);
    
    // Set dilation rates
    descriptor.set_dilation_rates(&[1, 1, 1]);
    
    // Set padding values (left, right, top, bottom, front, back)
    descriptor.set_padding_values(&[0, 0, 0, 0, 0, 0]);
    
    // Set the channel dimension index
    descriptor.set_channel_dimension_index(-4);
}

#[test]
fn test_depthwise_convolution_3d_descriptor_with_values() {
    // Create a new descriptor with specified values
    let descriptor = DepthwiseConvolution3DOpDescriptor::new_with_values(
        &[2, 2, 2],             // strides
        &[1, 1, 1],             // dilation rates
        &[1, 1, 1, 1, 1, 1],    // padding values
        PaddingMode::Explicit,
    );
    
    // Set the channel dimension index
    descriptor.set_channel_dimension_index(-4);
    
    // Set padding style
    descriptor.set_padding_style(PaddingMode::Same);
}