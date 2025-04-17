use crate::graph::Graph;
use crate::shape::Shape;
use crate::tensor::Tensor;
use crate::pooling_ops::TensorNamedDataLayout;
use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::AnyClass;
use objc2::extern_class;
use objc2_foundation::{NSObject, NSObjectProtocol, NSString};

extern_class!(
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraphImToColOpDescriptor"]
    /// Descriptor for Image to Column operations
    pub struct ImToColOpDescriptor;
);

unsafe impl NSObjectProtocol for ImToColOpDescriptor {}

impl ImToColOpDescriptor {
    /// Creates a new descriptor with full parameters for im2col operations
    ///
    /// # Arguments
    ///
    /// * `kernel_width` - The kernel size in width dimension
    /// * `kernel_height` - The kernel size in height dimension
    /// * `stride_in_x` - The stride in width dimension
    /// * `stride_in_y` - The stride in height dimension
    /// * `dilation_rate_in_x` - The dilation in width dimension
    /// * `dilation_rate_in_y` - The dilation in height dimension
    /// * `padding_left` - The padding in width dimension on the left side
    /// * `padding_right` - The padding in width dimension on the right side
    /// * `padding_top` - The padding in height dimension at the top
    /// * `padding_bottom` - The padding in height dimension at the bottom
    /// * `data_layout` - The layout of source or output tensor
    pub fn descriptor_with_kernel_dimensions(
        kernel_width: usize,
        kernel_height: usize,
        stride_in_x: usize,
        stride_in_y: usize,
        dilation_rate_in_x: usize,
        dilation_rate_in_y: usize,
        padding_left: usize,
        padding_right: usize,
        padding_top: usize,
        padding_bottom: usize,
        data_layout: TensorNamedDataLayout,
    ) -> Retained<Self> {
        unsafe {
            let cls = AnyClass::get(c"MPSGraphImToColOpDescriptor").unwrap();
            msg_send![
                cls, 
                descriptorWithKernelWidth: kernel_width as u64,
                kernelHeight: kernel_height as u64,
                strideInX: stride_in_x as u64,
                strideInY: stride_in_y as u64,
                dilationRateInX: dilation_rate_in_x as u64,
                dilationRateInY: dilation_rate_in_y as u64,
                paddingLeft: padding_left as u64,
                paddingRight: padding_right as u64,
                paddingTop: padding_top as u64,
                paddingBottom: padding_bottom as u64,
                dataLayout: data_layout as u64
            ]
        }
    }

    /// Creates a new descriptor with a simpler set of parameters for im2col operations
    ///
    /// # Arguments
    ///
    /// * `kernel_width` - The kernel size in width dimension
    /// * `kernel_height` - The kernel size in height dimension
    /// * `stride_in_x` - The stride in width dimension
    /// * `stride_in_y` - The stride in height dimension
    /// * `dilation_rate_in_x` - The dilation in width dimension
    /// * `dilation_rate_in_y` - The dilation in height dimension
    /// * `data_layout` - The layout of source or output tensor
    pub fn descriptor_with_kernel_dimensions_simple(
        kernel_width: usize,
        kernel_height: usize,
        stride_in_x: usize,
        stride_in_y: usize,
        dilation_rate_in_x: usize,
        dilation_rate_in_y: usize,
        data_layout: TensorNamedDataLayout,
    ) -> Retained<Self> {
        unsafe {
            let cls = AnyClass::get(c"MPSGraphImToColOpDescriptor").unwrap();
            msg_send![
                cls, 
                descriptorWithKernelWidth: kernel_width as u64,
                kernelHeight: kernel_height as u64,
                strideInX: stride_in_x as u64,
                strideInY: stride_in_y as u64,
                dilationRateInX: dilation_rate_in_x as u64,
                dilationRateInY: dilation_rate_in_y as u64,
                dataLayout: data_layout as u64
            ]
        }
    }

    /// For backward compatibility
    pub fn new(
        kernel_width: usize,
        kernel_height: usize,
        stride_in_x: usize,
        stride_in_y: usize,
        dilation_rate_in_x: usize,
        dilation_rate_in_y: usize,
        padding_left: usize,
        padding_right: usize,
        padding_top: usize,
        padding_bottom: usize,
        data_layout: TensorNamedDataLayout,
    ) -> Retained<Self> {
        Self::descriptor_with_kernel_dimensions(
            kernel_width,
            kernel_height,
            stride_in_x,
            stride_in_y,
            dilation_rate_in_x,
            dilation_rate_in_y,
            padding_left,
            padding_right,
            padding_top,
            padding_bottom,
            data_layout,
        )
    }

    /// For backward compatibility
    pub fn new_simple(
        kernel_width: usize,
        kernel_height: usize,
        stride_in_x: usize,
        stride_in_y: usize,
        dilation_rate_in_x: usize,
        dilation_rate_in_y: usize,
        data_layout: TensorNamedDataLayout,
    ) -> Retained<Self> {
        Self::descriptor_with_kernel_dimensions_simple(
            kernel_width,
            kernel_height,
            stride_in_x,
            stride_in_y,
            dilation_rate_in_x,
            dilation_rate_in_y,
            data_layout,
        )
    }

    /// Sets the explicit padding values for the descriptor
    ///
    /// # Arguments
    ///
    /// * `padding_left` - Padding on the left side
    /// * `padding_right` - Padding on the right side
    /// * `padding_top` - Padding on the top side
    /// * `padding_bottom` - Padding on the bottom side
    ///
    /// # Returns
    ///
    /// Self reference for chaining
    pub fn set_explicit_padding(
        &self,
        padding_left: usize,
        padding_right: usize,
        padding_top: usize,
        padding_bottom: usize,
    ) -> &Self {
        unsafe {
            let _: () = msg_send![
                self,
                setExplicitPaddingWithPaddingLeft: padding_left as u64,
                paddingRight: padding_right as u64,
                paddingTop: padding_top as u64,
                paddingBottom: padding_bottom as u64
            ];
        }
        self
    }
}

/// ImToCol operations for Graph
impl Graph {
    /// Creates an image to column operation that converts a 3D or 4D input tensor to a matrix.
    ///
    /// This operation performs an im2col transformation, which is used in convolution operations.
    /// It extracts patches from the input tensor and arranges them as columns of a matrix.
    ///
    /// # Arguments
    ///
    /// * `source` - The source data tensor. The layout is specified by the descriptor.
    /// * `descriptor` - The descriptor containing parameters for the operation.
    /// * `name` - Optional name for the operation.
    ///
    /// # Returns
    ///
    /// A new Tensor containing the matrix result.
    pub fn im_to_col(
        &self,
        source: &Tensor,
        descriptor: &ImToColOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };

            msg_send![
                self,
                imToColWithSourceTensor: source,
                descriptor: descriptor,
                name: name_obj
            ]
        }
    }

    /// Creates a column to image operation that is the reverse of image to column.
    ///
    /// # Arguments
    ///
    /// * `source` - The source data tensor containing the matrix.
    /// * `output_shape` - The shape of the output image tensor.
    /// * `descriptor` - The descriptor containing parameters for the operation.
    /// * `name` - Optional name for the operation.
    ///
    /// # Returns
    ///
    /// A new Tensor containing the image result.
    pub fn col_to_im(
        &self,
        source: &Tensor,
        output_shape: &Shape,
        descriptor: &ImToColOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };

            msg_send![
                self,
                colToImWithSourceTensor: source,
                outputShape: output_shape,
                descriptor: descriptor,
                name: name_obj
            ]
        }
    }

    /// Alias for im_to_col for backward compatibility
    pub fn image_to_column(
        &self,
        source: &Tensor,
        descriptor: &ImToColOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        self.im_to_col(source, descriptor, name)
    }

    /// Alias for col_to_im for backward compatibility
    pub fn column_to_image(
        &self,
        source: &Tensor,
        output_shape: &Shape,
        descriptor: &ImToColOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        self.col_to_im(source, output_shape, descriptor, name)
    }
}