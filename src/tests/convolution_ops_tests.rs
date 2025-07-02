use crate::convolution_ops::{
    Convolution2DOpDescriptor, Convolution3DOpDescriptor,
    PaddingMode, ConvolutionDataLayout, WeightsLayout,
};

#[test]
fn test_convolution_2d_descriptor() {
    // Create a new 2D convolution descriptor
    let descriptor = Convolution2DOpDescriptor::new();
    
    // Set stride values
    descriptor.set_stride_x(2);
    descriptor.set_stride_y(2);
    
    // Set dilation rates
    descriptor.set_dilation_x(1);
    descriptor.set_dilation_y(1);
    
    // Set padding mode to explicit
    descriptor.set_padding_mode(PaddingMode::Explicit);
    
    // Set explicit padding values
    descriptor.set_padding_values(1, 1, 1, 1);
    
    // Set data layout
    descriptor.set_data_layout(ConvolutionDataLayout::NCHW);
    
    // Set weights layout
    descriptor.set_weights_layout(WeightsLayout::HWIO);
    
    // Set groups for grouped convolution
    descriptor.set_groups(1);
}

#[test]
fn test_convolution_3d_descriptor() {
    // Create a new 3D convolution descriptor
    let descriptor = Convolution3DOpDescriptor::new();
    
    // Set stride values
    descriptor.set_strides(1, 1, 1);
    
    // Set dilation rates
    descriptor.set_dilation_rates(1, 1, 1);
    
    // Set padding mode to same
    descriptor.set_padding_mode(PaddingMode::Same);
    
    // Set explicit padding values
    descriptor.set_padding_values(0, 0, 0, 0, 0, 0);
    
    // Set data layout
    descriptor.set_data_layout(ConvolutionDataLayout::NCHW);
    
    // Set weights layout
    descriptor.set_weights_layout(WeightsLayout::OHWI);
    
    // Set groups for grouped convolution
    descriptor.set_groups(1);
}