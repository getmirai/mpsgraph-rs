use crate::{GraphObject, PaddingStyle, TensorNamedDataLayout};
use objc2::{
    extern_class, extern_conformance, extern_methods,
    rc::{Allocated, Retained},
    runtime::AnyClass,
};
use objc2_foundation::{CopyingHelper, NSCopying, NSObject, NSObjectProtocol};

extern_class!(
    /// A class that describes the properties of a 2D-convolution operator.
    ///
    /// Use an instance of this class is to add a 2D-convolution operator with the desired properties to the graph.
    ///
    /// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphconvolution2dopdescriptor?language=objc)
    #[unsafe(super(GraphObject, NSObject))]
    #[derive(Debug, PartialEq, Eq, Hash)]
    pub struct Convolution2DOpDescriptor;
);

extern_conformance!(
    unsafe impl NSCopying for Convolution2DOpDescriptor {}
);

unsafe impl CopyingHelper for Convolution2DOpDescriptor {
    type Result = Self;
}

extern_conformance!(
    unsafe impl NSObjectProtocol for Convolution2DOpDescriptor {}
);

impl Convolution2DOpDescriptor {
    extern_methods!(
        /// The scale that maps `x`-coordinate of the destination to `x`-coordinate of the source.
        ///
        /// Source `x`-coordinate, `sx` is computed from destination `x`-coordinate, `dx` as `sx = strideInX*dx`.
        /// Default value is 1.
        #[unsafe(method(strideInX))]
        #[unsafe(method_family = none)]
        pub fn stride_in_x(&self) -> u64;

        /// Setter for [`strideInX`][Self::strideInX].
        #[unsafe(method(setStrideInX:))]
        #[unsafe(method_family = none)]
        pub fn set_stride_in_x(&self, stride_in_x: u64);

        /// The scale that maps `y`-coordinate of the destination to `y`-coordinate of the source.
        ///
        /// Source `y`-coordinate, `sy` is computed from destination `y`-coordinate, `dy` as `sy = strideInY*dy`.
        /// Default value is 1.
        #[unsafe(method(strideInY))]
        #[unsafe(method_family = none)]
        pub fn stride_in_y(&self) -> u64;

        /// Setter for [`strideInY`][Self::strideInY].
        #[unsafe(method(setStrideInY:))]
        #[unsafe(method_family = none)]
        pub fn set_stride_in_y(&self, stride_in_y: u64);

        /// The amount by which the weights tensor expands in the `x`-direction.
        ///
        /// The weights tensor is dilated by inserting `dilationRateInX-1` zeros between consecutive values in `x`-dimension.
        /// Dilated weights tensor width is `(dilationRateInX-1)*kernelWidth+1`.
        /// Default value is 1.
        #[unsafe(method(dilationRateInX))]
        #[unsafe(method_family = none)]
        pub fn dilation_rate_in_x(&self) -> u64;

        /// Setter for [`dilationRateInX`][Self::dilationRateInX].
        #[unsafe(method(setDilationRateInX:))]
        #[unsafe(method_family = none)]
        pub fn set_dilation_rate_in_x(&self, dilation_rate_in_x: u64);

        /// The amount by which the weights tensor expands in the `y`-direction.
        ///
        /// The weights tensor is dilated by inserting `dilationRateInY-1` zeros between consecutive values in `y`-dimension.
        /// Dilated weights tensor width is `(dilationRateInY-1)*kernelHeight+1`.
        /// Default value is 1.
        #[unsafe(method(dilationRateInY))]
        #[unsafe(method_family = none)]
        pub fn dilation_rate_in_y(&self) -> u64;

        /// Setter for [`dilationRateInY`][Self::dilationRateInY].
        #[unsafe(method(setDilationRateInY:))]
        #[unsafe(method_family = none)]
        pub fn set_dilation_rate_in_y(&self, dilation_rate_in_y: u64);

        /// The number of zeros added on the left side of the source tensor.
        #[unsafe(method(paddingLeft))]
        #[unsafe(method_family = none)]
        pub fn padding_left(&self) -> u64;

        /// Setter for [`paddingLeft`][Self::paddingLeft].
        #[unsafe(method(setPaddingLeft:))]
        #[unsafe(method_family = none)]
        pub fn set_padding_left(&self, padding_left: u64);

        /// The number of zeros added on the right side of the source tensor.
        #[unsafe(method(paddingRight))]
        #[unsafe(method_family = none)]
        pub fn padding_right(&self) -> u64;

        /// Setter for [`paddingRight`][Self::paddingRight].
        #[unsafe(method(setPaddingRight:))]
        #[unsafe(method_family = none)]
        pub fn set_padding_right(&self, padding_right: u64);

        /// The number of zeros added at the top of the source tensor.
        #[unsafe(method(paddingTop))]
        #[unsafe(method_family = none)]
        pub fn padding_top(&self) -> u64;

        /// Setter for [`paddingTop`][Self::paddingTop].
        #[unsafe(method(setPaddingTop:))]
        #[unsafe(method_family = none)]
        pub fn set_padding_top(&self, padding_top: u64);

        /// The number of zeros added at the bottom of the source tensor.
        #[unsafe(method(paddingBottom))]
        #[unsafe(method_family = none)]
        pub fn padding_bottom(&self) -> u64;

        /// Setter for [`paddingBottom`][Self::paddingBottom].
        #[unsafe(method(setPaddingBottom:))]
        #[unsafe(method_family = none)]
        pub fn set_padding_bottom(&self, padding_bottom: u64);

        /// The type of padding applied to the source tensor.
        ///
        /// If paddingStyle is `MPSGraphPaddingStyleExplicit`, `paddingLeft`, `laddingRight`, `paddingTop`,
        /// and `paddingBottom` must to be specified. For all other padding styles, framework compute these values so you dont need to provide these values.
        #[unsafe(method(paddingStyle))]
        #[unsafe(method_family = none)]
        pub fn padding_style(&self) -> PaddingStyle;

        /// Setter for [`paddingStyle`][Self::paddingStyle].
        #[unsafe(method(setPaddingStyle:))]
        #[unsafe(method_family = none)]
        pub fn set_padding_style(&self, padding_style: PaddingStyle);

        /// The named layout of data in the source tensor.
        ///
        /// It defines the order of named dimensions (Batch, Channel, Height, Width). The convolution operation uses this to interpret data in the source tensor.
        /// For example, if `dataLayout` is `MPSGraphTensorNamedDataLayoutNCHW`, frameork interprets data in source tensor as `batch x channels x height x width`
        /// with `width` as fastest moving dimension.
        #[unsafe(method(dataLayout))]
        #[unsafe(method_family = none)]
        pub fn data_layout(&self) -> TensorNamedDataLayout;

        /// Setter for [`dataLayout`][Self::dataLayout].
        #[unsafe(method(setDataLayout:))]
        #[unsafe(method_family = none)]
        pub fn set_data_layout(&self, data_layout: TensorNamedDataLayout);

        /// The named layout of data in the weights tensor.
        ///
        /// It defines the order of named dimensions (Output channels, Input channels, Kernel height, Kernel width). The convolution operation uses this to interpret data in the weights tensor.
        /// For example, if `weightsLayout` is `MPSGraphTensorNamedDataLayoutOIHW`, frameork interprets data in weights tensor as `outputChannels x inputChannels x kernelHeight x kernelWidth`
        /// with `kernelWidth` as fastest moving dimension.
        #[unsafe(method(weightsLayout))]
        #[unsafe(method_family = none)]
        pub fn weights_layout(&self) -> TensorNamedDataLayout;

        /// Setter for [`weightsLayout`][Self::weightsLayout].
        #[unsafe(method(setWeightsLayout:))]
        #[unsafe(method_family = none)]
        pub fn set_weights_layout(&self, weights_layout: TensorNamedDataLayout);

        /// The number of partitions of the input and output channels.
        ///
        /// The convolution operation divides input and output channels in `groups` partitions.
        /// input channels in a group or partition are only connected to output channels in corresponding group.
        /// Number of weights the convolution needs is `outputFeatureChannels x inputFeatureChannels/groups x kernelWidth x kernelHeight`
        #[unsafe(method(groups))]
        #[unsafe(method_family = none)]
        pub fn groups(&self) -> u64;

        /// Setter for [`groups`][Self::groups].
        #[unsafe(method(setGroups:))]
        #[unsafe(method_family = none)]
        pub fn set_groups(&self, groups: u64);

        /// Creates a convolution descriptor with given values for parameters.
        /// - Parameters:
        /// - strideInX: See ``strideInX`` property.
        /// - strideInY: See ``strideInY`` property.
        /// - dilationRateInX: See ``dilationRateInX`` property.
        /// - dilationRateInY: See ``dilationRateInY`` property.
        /// - groups: See ``groups`` property.
        /// - paddingLeft: See ``paddingLeft`` property.
        /// - paddingRight: See ``paddingRight`` property.
        /// - paddingTop: See ``paddingTop`` property.
        /// - paddingBottom: See ``paddingBottom`` property.
        /// - paddingStyle: See ``paddingStyle`` property.
        /// - dataLayout: See ``dataLayout`` property.
        /// - weightsLayout: See ``weightsLayout`` property.
        /// - Returns: The `MPSGraphConvolution2DOpDescriptor` on autoreleasepool.
        #[unsafe(method(descriptorWithStrideInX:strideInY:dilationRateInX:dilationRateInY:groups:paddingLeft:paddingRight:paddingTop:paddingBottom:paddingStyle:dataLayout:weightsLayout:))]
        #[unsafe(method_family = none)]
        pub fn descriptor_with_stride_in_x_stride_in_y_dilation_rate_in_x_dilation_rate_in_y_groups_padding_left_padding_right_padding_top_padding_bottom_padding_style_data_layout_weights_layout(
            stride_in_x: u64,
            stride_in_y: u64,
            dilation_rate_in_x: u64,
            dilation_rate_in_y: u64,
            groups: u64,
            padding_left: u64,
            padding_right: u64,
            padding_top: u64,
            padding_bottom: u64,
            padding_style: PaddingStyle,
            data_layout: TensorNamedDataLayout,
            weights_layout: TensorNamedDataLayout,
        ) -> Option<Retained<Self>>;

        /// Creates a convolution descriptor with given values for parameters.
        /// - Parameters:
        /// - strideInX: See ``strideInX`` property.
        /// - strideInY: See ``strideInY`` property.
        /// - dilationRateInX: See ``dilationRateInX`` property.
        /// - dilationRateInY: See ``dilationRateInY`` property.
        /// - groups: See ``groups`` property.
        /// - paddingStyle: See ``paddingStyle`` property.
        /// - dataLayout: See ``dataLayout`` property.
        /// - weightsLayout: See ``weightsLayout`` property.
        /// - Returns: The `MPSGraphConvolution2DOpDescriptor` on autoreleasepool.
        #[unsafe(method(descriptorWithStrideInX:strideInY:dilationRateInX:dilationRateInY:groups:paddingStyle:dataLayout:weightsLayout:))]
        #[unsafe(method_family = none)]
        pub fn descriptor_with_stride_in_x_stride_in_y_dilation_rate_in_x_dilation_rate_in_y_groups_padding_style_data_layout_weights_layout(
            stride_in_x: u64,
            stride_in_y: u64,
            dilation_rate_in_x: u64,
            dilation_rate_in_y: u64,
            groups: u64,
            padding_style: PaddingStyle,
            data_layout: TensorNamedDataLayout,
            weights_layout: TensorNamedDataLayout,
        ) -> Option<Retained<Self>>;

        /// Sets the left, right, top, and bottom padding values.
        /// - Parameters:
        /// - paddingLeft: See ``paddingLeft`` property.
        /// - paddingRight: See ``paddingRight`` property.
        /// - paddingTop: See ``paddingTop`` property.
        /// - paddingBottom: See ``paddingBottom`` property.
        #[unsafe(method(setExplicitPaddingWithPaddingLeft:paddingRight:paddingTop:paddingBottom:))]
        #[unsafe(method_family = none)]
        pub fn set_explicit_padding_with_padding_left_padding_right_padding_top_padding_bottom(
            &self,
            padding_left: u64,
            padding_right: u64,
            padding_top: u64,
            padding_bottom: u64,
        );
    );
}

/// Methods declared on superclass `NSObject`.
impl Convolution2DOpDescriptor {
    extern_methods!(
        #[unsafe(method(init))]
        #[unsafe(method_family = init)]
        pub unsafe fn init(this: Allocated<Self>) -> Retained<Self>;

        #[unsafe(method(new))]
        #[unsafe(method_family = new)]
        pub unsafe fn new() -> Retained<Self>;
    );
}
