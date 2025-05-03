use crate::convolution_ops::PaddingMode;
use crate::pooling_ops::TensorNamedDataLayout;
use crate::{Graph, Tensor};
use objc2::extern_class;
use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::AnyClass;
use objc2_foundation::{NSArray, NSNumber, NSObject, NSObjectProtocol, NSString};

extern_class!(
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraphDepthwiseConvolution2DOpDescriptor"]
    /// Descriptor for 2D depthwise convolution operations
    pub struct DepthwiseConvolution2DOpDescriptor;
);

unsafe impl NSObjectProtocol for DepthwiseConvolution2DOpDescriptor {}

extern_class!(
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraphDepthwiseConvolution3DOpDescriptor"]
    /// Descriptor for 3D depthwise convolution operations
    pub struct DepthwiseConvolution3DOpDescriptor;
);

unsafe impl NSObjectProtocol for DepthwiseConvolution3DOpDescriptor {}

impl DepthwiseConvolution2DOpDescriptor {
    /// Creates a new depthwise convolution 2D operation descriptor
    pub fn new() -> Retained<Self> {
        unsafe {
            let cls = AnyClass::get(c"MPSGraphDepthwiseConvolution2DOpDescriptor").unwrap();
            msg_send![cls, descriptor]
        }
    }

    /// Creates a new depthwise convolution 2D operation descriptor with specified data and weights layouts
    pub fn new_with_layouts(
        data_layout: TensorNamedDataLayout,
        weights_layout: TensorNamedDataLayout,
    ) -> Retained<Self> {
        unsafe {
            let cls = AnyClass::get(c"MPSGraphDepthwiseConvolution2DOpDescriptor").unwrap();
            msg_send![
                cls, descriptorWithDataLayout: data_layout as u64,
                weightsLayout: weights_layout as u64
            ]
        }
    }

    /// Sets the stride in X dimension
    pub fn set_stride_in_x(&self, stride: usize) {
        unsafe {
            let _: () = msg_send![self, setStrideInX: stride as u64];
        }
    }

    /// Sets the stride in Y dimension
    pub fn set_stride_in_y(&self, stride: usize) {
        unsafe {
            let _: () = msg_send![self, setStrideInY: stride as u64];
        }
    }

    /// Sets the dilation rate in X dimension
    pub fn set_dilation_rate_in_x(&self, rate: usize) {
        unsafe {
            let _: () = msg_send![self, setDilationRateInX: rate as u64];
        }
    }

    /// Sets the dilation rate in Y dimension
    pub fn set_dilation_rate_in_y(&self, rate: usize) {
        unsafe {
            let _: () = msg_send![self, setDilationRateInY: rate as u64];
        }
    }

    /// Sets the padding on the left
    pub fn set_padding_left(&self, padding: usize) {
        unsafe {
            let _: () = msg_send![self, setPaddingLeft: padding as u64];
        }
    }

    /// Sets the padding on the right
    pub fn set_padding_right(&self, padding: usize) {
        unsafe {
            let _: () = msg_send![self, setPaddingRight: padding as u64];
        }
    }

    /// Sets the padding on the top
    pub fn set_padding_top(&self, padding: usize) {
        unsafe {
            let _: () = msg_send![self, setPaddingTop: padding as u64];
        }
    }

    /// Sets the padding on the bottom
    pub fn set_padding_bottom(&self, padding: usize) {
        unsafe {
            let _: () = msg_send![self, setPaddingBottom: padding as u64];
        }
    }

    /// Sets the padding style
    pub fn set_padding_style(&self, style: PaddingMode) {
        unsafe {
            let _: () = msg_send![self, setPaddingStyle: style as u64];
        }
    }

    /// Sets the data layout
    pub fn set_data_layout(&self, layout: TensorNamedDataLayout) {
        unsafe {
            let _: () = msg_send![self, setDataLayout: layout as u64];
        }
    }

    /// Sets the weights layout
    pub fn set_weights_layout(&self, layout: TensorNamedDataLayout) {
        unsafe {
            let _: () = msg_send![self, setWeightsLayout: layout as u64];
        }
    }

    /// Sets the explicit padding values
    pub fn set_explicit_padding(&self, left: usize, right: usize, top: usize, bottom: usize) {
        unsafe {
            let _: () = msg_send![
                self, setExplicitPaddingWithPaddingLeft: left as u64,
                paddingRight: right as u64,
                paddingTop: top as u64,
                paddingBottom: bottom as u64
            ];
        }
    }
}

impl DepthwiseConvolution3DOpDescriptor {
    /// Creates a new depthwise convolution 3D operation descriptor with default values
    pub fn new(padding_style: PaddingMode) -> Retained<Self> {
        unsafe {
            let cls = AnyClass::get(c"MPSGraphDepthwiseConvolution3DOpDescriptor").unwrap();
            msg_send![cls, descriptorWithPaddingStyle: padding_style as u64]
        }
    }

    /// Creates a new depthwise convolution 3D operation descriptor with given values
    pub fn new_with_values(
        strides: &[usize],
        dilation_rates: &[usize],
        padding_values: &[usize],
        padding_style: PaddingMode,
    ) -> Retained<Self> {
        unsafe {
            // Convert Rust arrays to NSArrays
            let strides_array = create_number_array(strides);
            let dilation_rates_array = create_number_array(dilation_rates);
            let padding_values_array = create_number_array(padding_values);

            let cls = AnyClass::get(c"MPSGraphDepthwiseConvolution3DOpDescriptor").unwrap();
            let descriptor: Retained<Self> = msg_send![
                cls, descriptorWithStrides: &*strides_array,
                dilationRates: &*dilation_rates_array,
                paddingValues: &*padding_values_array,
                paddingStyle: padding_style as u64
            ];

            descriptor
        }
    }

    /// Sets the strides for spatial dimensions (3 values)
    pub fn set_strides(&self, strides: &[usize]) {
        unsafe {
            let strides_array = create_number_array(strides);
            let _: () = msg_send![self, setStrides: &*strides_array];
        }
    }

    /// Sets the dilation rates for spatial dimensions (3 values)
    pub fn set_dilation_rates(&self, dilation_rates: &[usize]) {
        unsafe {
            let dilation_rates_array = create_number_array(dilation_rates);
            let _: () = msg_send![self, setDilationRates: &*dilation_rates_array];
        }
    }

    /// Sets the padding values for spatial dimensions (6 values)
    pub fn set_padding_values(&self, padding_values: &[usize]) {
        unsafe {
            let padding_values_array = create_number_array(padding_values);
            let _: () = msg_send![self, setPaddingValues: &*padding_values_array];
        }
    }

    /// Sets the padding style
    pub fn set_padding_style(&self, style: PaddingMode) {
        unsafe {
            let _: () = msg_send![self, setPaddingStyle: style as u64];
        }
    }

    /// Sets the channel dimension index
    pub fn set_channel_dimension_index(&self, index: isize) {
        unsafe {
            let _: () = msg_send![self, setChannelDimensionIndex: index];
        }
    }
}

/// Helper function to create an NSArray of NSNumbers from a slice of usizes
fn create_number_array(values: &[usize]) -> Retained<NSArray<NSNumber>> {
    // Create NSNumber objects
    let numbers: Vec<Retained<NSNumber>> = values
        .iter()
        .map(|&val| NSNumber::new_u64(val as u64))
        .collect();

    // Get references to NSNumber objects
    let number_refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();

    // Create NSArray from the NSNumber objects
    NSArray::from_slice(&number_refs)
}

/// Depthwise convolution operations for Graph
impl Graph {
    /// Creates a 2D depthwise convolution operation and returns the result tensor.
    ///
    /// # Arguments
    ///
    /// * `source` - A 2D image source tensor (rank=4, layout defined by descriptor.dataLayout)
    /// * `weights` - Weights tensor (rank=4, layout defined by descriptor.weightsLayout)
    /// * `descriptor` - Descriptor that specifies strides, dilation rates, paddings and layouts
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new Tensor containing the result
    pub fn depthwise_convolution_2d(
        &self,
        source: &Tensor,
        weights: &Tensor,
        descriptor: &DepthwiseConvolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };

            msg_send![
                self,
                depthwiseConvolution2DWithSourceTensor: source,
                weightsTensor: weights,
                descriptor: descriptor,
                name: name_obj
            ]
        }
    }

    /// Creates a 2D depthwise convolution gradient for data operation.
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient` - A 2D input gradient tensor (rank=4, layout defined by descriptor.dataLayout)
    /// * `weights` - Weights tensor (rank=4, layout defined by descriptor.weightsLayout)
    /// * `output_shape` - Shape of the output tensor (and input tensor of forward pass)
    /// * `descriptor` - Descriptor that specifies strides, dilation rates, paddings and layouts
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new Tensor containing the gradient with respect to data
    pub fn depthwise_convolution_2d_data_gradient(
        &self,
        incoming_gradient: &Tensor,
        weights: &Tensor,
        output_shape: &crate::Shape,
        descriptor: &DepthwiseConvolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };

            msg_send![
                self,
                depthwiseConvolution2DDataGradientWithIncomingGradientTensor: incoming_gradient,
                weightsTensor: weights,
                outputShape: output_shape,
                descriptor: descriptor,
                name: name_obj
            ]
        }
    }

    /// Creates a 2D depthwise convolution gradient for weights operation.
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient` - A 2D input gradient tensor (rank=4, layout defined by descriptor.dataLayout)
    /// * `source` - A 2D image source tensor (rank=4, layout defined by descriptor.dataLayout)
    /// * `output_shape` - Shape of the output tensor (and weight tensor of forward pass)
    /// * `descriptor` - Descriptor that specifies strides, dilation rates, paddings and layouts
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new Tensor containing the gradient with respect to weights
    pub fn depthwise_convolution_2d_weights_gradient(
        &self,
        incoming_gradient: &Tensor,
        source: &Tensor,
        output_shape: &crate::Shape,
        descriptor: &DepthwiseConvolution2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };

            msg_send![
                self,
                depthwiseConvolution2DWeightsGradientWithIncomingGradientTensor: incoming_gradient,
                sourceTensor: source,
                outputShape: output_shape,
                descriptor: descriptor,
                name: name_obj
            ]
        }
    }

    /// Creates a 3D depthwise convolution operation.
    ///
    /// # Arguments
    ///
    /// * `source` - A 3D image source tensor (at least rank=4, CDHW when channelDimensionIndex = -4)
    /// * `weights` - Weights tensor (rank=4, axes interpreted as CDHW when channelDimensionIndex = -4)
    /// * `descriptor` - Descriptor that specifies strides, dilation rates and paddings
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new Tensor containing the result
    pub fn depthwise_convolution_3d(
        &self,
        source: &Tensor,
        weights: &Tensor,
        descriptor: &DepthwiseConvolution3DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };

            msg_send![
                self,
                depthwiseConvolution3DWithSourceTensor: source,
                weightsTensor: weights,
                descriptor: descriptor,
                name: name_obj
            ]
        }
    }

    /// Creates a 3D depthwise convolution gradient for data operation.
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient` - A 3D input gradient tensor (at least rank=4, CDHW)
    /// * `weights` - Weights tensor (rank=4, axes interpreted as CDHW)
    /// * `output_shape` - Shape of the output tensor (and input tensor of forward pass)
    /// * `descriptor` - Descriptor that specifies strides, dilation rates and paddings
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new Tensor containing the gradient with respect to data
    pub fn depthwise_convolution_3d_data_gradient(
        &self,
        incoming_gradient: &Tensor,
        weights: &Tensor,
        output_shape: &crate::Shape,
        descriptor: &DepthwiseConvolution3DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };

            msg_send![
                self,
                depthwiseConvolution3DDataGradientWithIncomingGradientTensor: incoming_gradient,
                weightsTensor: weights,
                outputShape: output_shape,
                descriptor: descriptor,
                name: name_obj
            ]
        }
    }

    /// Creates a 3D depthwise convolution gradient for weights operation.
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient` - A 3D input gradient tensor (at least rank=4, NCDHW)
    /// * `source` - A 3D image source tensor (at least rank=4, NCDHW)
    /// * `output_shape` - Shape of the output tensor (and weight tensor of forward pass)
    /// * `descriptor` - Descriptor that specifies strides, dilation rates and paddings
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A new Tensor containing the gradient with respect to weights
    pub fn depthwise_convolution_3d_weights_gradient(
        &self,
        incoming_gradient: &Tensor,
        source: &Tensor,
        output_shape: &crate::Shape,
        descriptor: &DepthwiseConvolution3DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };

            msg_send![
                self,
                depthwiseConvolution3DWeightsGradientWithIncomingGradientTensor: incoming_gradient,
                sourceTensor: source,
                outputShape: output_shape,
                descriptor: descriptor,
                name: name_obj
            ]
        }
    }
}
