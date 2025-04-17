use crate::im2col_ops::ImToColOpDescriptor;
use crate::pooling_ops::TensorNamedDataLayout;
use crate::graph::Graph;
use crate::shape::ShapeHelper;
use crate::tensor::{Tensor, DataType};

#[test]
fn test_create_im2col_op_descriptor() {
    // Create a descriptor with full parameters
    let _descriptor = ImToColOpDescriptor::descriptor_with_kernel_dimensions(
        3, // kernel_width
        3, // kernel_height
        1, // stride_in_x
        1, // stride_in_y
        1, // dilation_rate_in_x
        1, // dilation_rate_in_y
        1, // padding_left
        1, // padding_right
        1, // padding_top
        1, // padding_bottom
        TensorNamedDataLayout::NCHW, // data_layout
    );
    
    // If we got here without crashing, the test passes
    assert!(true);
}

#[test]
fn test_create_im2col_op_descriptor_simple() {
    // Create a descriptor with simple parameters
    let _descriptor = ImToColOpDescriptor::descriptor_with_kernel_dimensions_simple(
        3, // kernel_width
        3, // kernel_height
        1, // stride_in_x
        1, // stride_in_y
        1, // dilation_rate_in_x
        1, // dilation_rate_in_y
        TensorNamedDataLayout::NCHW, // data_layout
    );
    
    // If we got here without crashing, the test passes
    assert!(true);
}

#[test]
fn test_im2col_op_descriptor_set_explicit_padding() {
    // Create a descriptor
    let descriptor = ImToColOpDescriptor::descriptor_with_kernel_dimensions_simple(
        3, // kernel_width
        3, // kernel_height
        1, // stride_in_x
        1, // stride_in_y
        1, // dilation_rate_in_x
        1, // dilation_rate_in_y
        TensorNamedDataLayout::NCHW, // data_layout
    );
    
    // Set explicit padding
    descriptor.set_explicit_padding(2, 2, 2, 2);
    
    // If we got here without crashing, the test passes
    assert!(true);
}

// This test only compiles the API but doesn't run actual computations
// since it would require real data
#[test]
fn test_im2col_api_compiles() {
    let graph = Graph::new();
    
    // Placeholder source tensor
    let source_shape = ShapeHelper::tensor4d(1, 3, 32, 32); // batch, channels, height, width
    let source = graph.placeholder("source", DataType::Float32, &source_shape, None).unwrap();
    
    // Create descriptor
    let descriptor = ImToColOpDescriptor::descriptor_with_kernel_dimensions(
        3, 3, 1, 1, 1, 1, 1, 1, 1, 1, TensorNamedDataLayout::NCHW
    );
    
    // Test im2col operation
    let _im2col_result = graph.im_to_col(&source, &descriptor, Some("im2col"));
    
    // Test col2im operation
    let output_shape = ShapeHelper::tensor4d(1, 3, 32, 32);
    let _col2im_result = graph.col_to_im(&source, &output_shape, &descriptor, Some("col2im"));
    
    // Test alias methods for backward compatibility
    let _image_to_column_result = graph.image_to_column(&source, &descriptor, Some("image_to_column"));
    let _column_to_image_result = graph.column_to_image(&source, &output_shape, &descriptor, Some("column_to_image"));
    
    // If we got here without crashing, the test passes
    assert!(true);
}