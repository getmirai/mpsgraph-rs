use crate::core::{create_ns_array_from_i64_slice, DataType};
use crate::graph::Graph;
use crate::tensor::Tensor;
use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::AnyClass;
use objc2::extern_class;
use objc2_foundation::{NSArray, NSNumber, NSObject, NSObjectProtocol, NSString};

/// Return indices mode for max pooling operations
#[repr(u64)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PoolingReturnIndicesMode {
    /// No indices returned
    None = 0,
    /// Returns indices flattened in inner most (last) dimension
    GlobalFlatten1D = 1,
    /// Returns indices flattened in 2 innermost dimensions. eg: HW in NCHW
    GlobalFlatten2D = 2,
    /// Returns indices flattened in 3 innermost dimensions. eg: HWC in NHWC
    GlobalFlatten3D = 3,
    /// Returns indices flattened in 4 innermost dimensions
    GlobalFlatten4D = 4,
    /// Returns indices within pooling window, flattened in inner most dimension
    LocalFlatten1D = 5,
    /// Returns indices within pooling window, flattened in 2 innermost dimensions. eg: HW in NCHW
    LocalFlatten2D = 6,
    /// Returns indices within pooling window, flattened in 3 innermost dimensions. eg: HWC in NHWC
    LocalFlatten3D = 7,
    /// Returns indices within pooling window, flattened in 4 innermost dimensions
    LocalFlatten4D = 8,
}

/// Data layout for 2D tensor operations
#[repr(u64)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TensorNamedDataLayout {
    /// NCHW layout (batch, channels, height, width)
    NCHW = 0,
    /// NHWC layout (batch, height, width, channels)
    NHWC = 1,
}

/// Padding style for tensor operations
#[repr(u64)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PaddingStyle {
    /// Explicit padding with specified values
    Explicit = 0,
    /// Same padding (input and output have same dimensions)
    TfSame = 1,
    /// Valid padding (no padding)
    TfValid = 2,
}

extern_class!(
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraphPooling2DOpDescriptor"]
    /// The descriptor for 2D pooling operations
    pub struct Pooling2DOpDescriptor;
);

unsafe impl NSObjectProtocol for Pooling2DOpDescriptor {}

extern_class!(
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraphPooling4DOpDescriptor"]
    /// The descriptor for 4D pooling operations
    pub struct Pooling4DOpDescriptor;
);

unsafe impl NSObjectProtocol for Pooling4DOpDescriptor {}

impl Pooling2DOpDescriptor {
    /// Creates a new 2D pooling descriptor with the given parameters
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
        padding_style: PaddingStyle,
        data_layout: TensorNamedDataLayout,
    ) -> Retained<Self> {
        unsafe {
            let cls = AnyClass::get(c"MPSGraphPooling2DOpDescriptor").unwrap();
            msg_send![cls,
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
                paddingStyle: padding_style as u64,
                dataLayout: data_layout as u64
            ]
        }
    }

    /// Creates a simplified 2D pooling descriptor
    pub fn new_simple(
        kernel_width: usize,
        kernel_height: usize,
        stride_in_x: usize,
        stride_in_y: usize,
        padding_style: PaddingStyle,
        data_layout: TensorNamedDataLayout,
    ) -> Retained<Self> {
        unsafe {
            let cls = AnyClass::get(c"MPSGraphPooling2DOpDescriptor").unwrap();
            msg_send![cls,
                descriptorWithKernelWidth: kernel_width as u64,
                kernelHeight: kernel_height as u64,
                strideInX: stride_in_x as u64,
                strideInY: stride_in_y as u64,
                paddingStyle: padding_style as u64,
                dataLayout: data_layout as u64
            ]
        }
    }

    /// Sets explicit padding values and changes padding style to explicit
    pub fn set_explicit_padding(
        &self,
        padding_left: usize,
        padding_right: usize,
        padding_top: usize,
        padding_bottom: usize,
    ) {
        unsafe {
            let _: () = msg_send![self,
                setExplicitPaddingWithPaddingLeft: padding_left as u64,
                paddingRight: padding_right as u64,
                paddingTop: padding_top as u64,
                paddingBottom: padding_bottom as u64
            ];
        }
    }

    /// Sets the return indices mode for max pooling operations
    pub fn set_return_indices_mode(&self, mode: PoolingReturnIndicesMode) {
        unsafe {
            let _: () = msg_send![self, setReturnIndicesMode: mode as u64];
        }
    }

    /// Sets the data type for returned indices
    pub fn set_return_indices_data_type(&self, data_type: DataType) {
        unsafe {
            let _: () = msg_send![self, setReturnIndicesDataType: data_type as u32];
        }
    }

    /// Sets ceil mode for computing output size
    pub fn set_ceil_mode(&self, ceil_mode: bool) {
        unsafe {
            let _: () = msg_send![self, setCeilMode: ceil_mode];
        }
    }

    /// Sets whether to include zero padding in average computation
    pub fn set_include_zero_pad_to_average(&self, include: bool) {
        unsafe {
            let _: () = msg_send![self, setIncludeZeroPadToAverage: include];
        }
    }
}

impl Pooling4DOpDescriptor {
    /// Creates a new 4D pooling descriptor with the given parameters
    pub fn new(
        kernel_sizes: &[usize],
        strides: &[usize],
        dilation_rates: &[usize],
        padding_values: &[usize],
        padding_style: PaddingStyle,
    ) -> Retained<Self> {
        unsafe {
            // Create NSArrays for parameters
            let kernel_sizes_array = Self::create_number_array(kernel_sizes);
            let strides_array = Self::create_number_array(strides);
            let dilation_rates_array = Self::create_number_array(dilation_rates);
            let padding_values_array = Self::create_number_array(padding_values);

            let cls = AnyClass::get(c"MPSGraphPooling4DOpDescriptor").unwrap();
            let descriptor: Retained<Self> = msg_send![cls,
                descriptorWithKernelSizes: &*kernel_sizes_array,
                strides: &*strides_array,
                dilationRates: &*dilation_rates_array,
                paddingValues: &*padding_values_array,
                paddingStyle: padding_style as u64
            ];

            descriptor
        }
    }

    /// Creates a simplified 4D pooling descriptor
    pub fn new_simple(kernel_sizes: &[usize], padding_style: PaddingStyle) -> Retained<Self> {
        unsafe {
            // Create NSArray for kernel sizes
            let kernel_sizes_array = Self::create_number_array(kernel_sizes);

            let cls = AnyClass::get(c"MPSGraphPooling4DOpDescriptor").unwrap();
            let descriptor: Retained<Self> = msg_send![cls,
                descriptorWithKernelSizes: &*kernel_sizes_array,
                paddingStyle: padding_style as u64
            ];

            descriptor
        }
    }

    /// Sets the return indices mode for max pooling operations
    pub fn set_return_indices_mode(&self, mode: PoolingReturnIndicesMode) {
        unsafe {
            let _: () = msg_send![self, setReturnIndicesMode: mode as u64];
        }
    }

    /// Sets the data type for returned indices
    pub fn set_return_indices_data_type(&self, data_type: DataType) {
        unsafe {
            let _: () = msg_send![self, setReturnIndicesDataType: data_type as u32];
        }
    }

    /// Sets ceil mode for computing output size
    pub fn set_ceil_mode(&self, ceil_mode: bool) {
        unsafe {
            let _: () = msg_send![self, setCeilMode: ceil_mode];
        }
    }

    /// Sets whether to include zero padding in average computation
    pub fn set_include_zero_pad_to_average(&self, include: bool) {
        unsafe {
            let _: () = msg_send![self, setIncludeZeroPadToAverage: include];
        }
    }

    /// Helper function to create NSArray of NSNumbers from a slice of usize values
    fn create_number_array(values: &[usize]) -> Retained<NSArray<NSNumber>> {
        // Create NSNumber objects
        let numbers: Vec<Retained<NSNumber>> = values
            .iter()
            .map(|&value| NSNumber::new_u64(value as u64))
            .collect();

        // Convert to slice of references
        let number_refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();

        // Create NSArray from the NSNumber objects
        NSArray::from_slice(&number_refs)
    }
}

/// Graph trait extension for pooling operations
pub trait GraphPoolingOps {
    /// Creates a 2D max pooling operation
    fn max_pooling_2d(
        &self,
        source: &Tensor,
        descriptor: &Pooling2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a 2D max pooling operation that returns indices
    fn max_pooling_2d_return_indices(
        &self,
        source: &Tensor,
        descriptor: &Pooling2DOpDescriptor,
        name: Option<&str>,
    ) -> (Retained<Tensor>, Retained<Tensor>);

    /// Creates a max pooling gradient operation
    fn max_pooling_2d_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        descriptor: &Pooling2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a max pooling gradient operation using indices
    fn max_pooling_2d_gradient_with_indices(
        &self,
        gradient: &Tensor,
        indices: &Tensor,
        output_shape: &[i64],
        descriptor: &Pooling2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a max pooling gradient operation using indices tensor
    fn max_pooling_2d_gradient_with_indices_tensor(
        &self,
        gradient: &Tensor,
        indices: &Tensor,
        output_shape_tensor: &Tensor,
        descriptor: &Pooling2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a 2D average pooling operation
    fn avg_pooling_2d(
        &self,
        source: &Tensor,
        descriptor: &Pooling2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates an average pooling gradient operation
    fn avg_pooling_2d_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        descriptor: &Pooling2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a 2D L2 norm pooling operation
    fn l2_norm_pooling_2d(
        &self,
        source: &Tensor,
        descriptor: &Pooling2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates an L2 norm pooling gradient operation
    fn l2_norm_pooling_2d_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        descriptor: &Pooling2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a 4D max pooling operation
    fn max_pooling_4d(
        &self,
        source: &Tensor,
        descriptor: &Pooling4DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a 4D max pooling operation that returns indices
    fn max_pooling_4d_return_indices(
        &self,
        source: &Tensor,
        descriptor: &Pooling4DOpDescriptor,
        name: Option<&str>,
    ) -> (Retained<Tensor>, Retained<Tensor>);

    /// Creates a max pooling gradient operation
    fn max_pooling_4d_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        descriptor: &Pooling4DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a max pooling gradient operation using indices
    fn max_pooling_4d_gradient_with_indices(
        &self,
        gradient: &Tensor,
        indices: &Tensor,
        output_shape: &[i64],
        descriptor: &Pooling4DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a max pooling gradient operation using indices tensor
    fn max_pooling_4d_gradient_with_indices_tensor(
        &self,
        gradient: &Tensor,
        indices: &Tensor,
        output_shape_tensor: &Tensor,
        descriptor: &Pooling4DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a 4D average pooling operation
    fn avg_pooling_4d(
        &self,
        source: &Tensor,
        descriptor: &Pooling4DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates an average pooling gradient operation
    fn avg_pooling_4d_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        descriptor: &Pooling4DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a 4D L2 norm pooling operation
    fn l2_norm_pooling_4d(
        &self,
        source: &Tensor,
        descriptor: &Pooling4DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates an L2 norm pooling gradient operation
    fn l2_norm_pooling_4d_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        descriptor: &Pooling4DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;
}

/// Implementation of pooling operations for Graph
impl GraphPoolingOps for Graph {
    fn max_pooling_2d(
        &self,
        source: &Tensor,
        descriptor: &Pooling2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };

            msg_send![
                self,
                maxPooling2DWithSourceTensor: source,
                descriptor: descriptor,
                name: name_obj
            ]
        }
    }

    fn max_pooling_2d_return_indices(
        &self,
        source: &Tensor,
        descriptor: &Pooling2DOpDescriptor,
        name: Option<&str>,
    ) -> (Retained<Tensor>, Retained<Tensor>) {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };
            
            let result: Retained<NSArray<Tensor>> = msg_send![
                self,
                maxPooling2DReturnIndicesWithSourceTensor: source,
                descriptor: descriptor,
                name: name_obj
            ];

            // Get the two result tensors
            let result_count: usize = msg_send![&*result, count];

            if result_count != 2 {
                panic!("maxPooling2DReturnIndices should return exactly 2 tensors");
            }

            let pooling_tensor: *mut Tensor = msg_send![&*result, objectAtIndex: 0u64];
            let indices_tensor: *mut Tensor = msg_send![&*result, objectAtIndex: 1u64];

            // Convert to retained references
            let pooling_tensor = Retained::retain(pooling_tensor).unwrap();
            let indices_tensor = Retained::retain(indices_tensor).unwrap();

            (pooling_tensor, indices_tensor)
        }
    }

    fn max_pooling_2d_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        descriptor: &Pooling2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                maxPooling2DGradientWithGradientTensor: gradient,
                sourceTensor: source,
                descriptor: descriptor,
                name: match name { Some(s) => &*NSString::from_str(s), None => std::ptr::null() }
            ]
        }
    }

    fn max_pooling_2d_gradient_with_indices(
        &self,
        gradient: &Tensor,
        indices: &Tensor,
        output_shape: &[i64],
        descriptor: &Pooling2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            // Create MPSShape from the output_shape array
            let shape_array = create_ns_array_from_i64_slice(output_shape);

            let tensor: Retained<Tensor> = msg_send![
                self,
                maxPooling2DGradientWithGradientTensor: gradient,
                indicesTensor: indices,
                outputShape: &*shape_array,
                descriptor: descriptor,
                name: match name { Some(s) => &*NSString::from_str(s), None => std::ptr::null() }
            ];

            tensor
        }
    }

    fn max_pooling_2d_gradient_with_indices_tensor(
        &self,
        gradient: &Tensor,
        indices: &Tensor,
        output_shape_tensor: &Tensor,
        descriptor: &Pooling2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                maxPooling2DGradientWithGradientTensor: gradient,
                indicesTensor: indices,
                outputShapeTensor: output_shape_tensor,
                descriptor: descriptor,
                name: match name { Some(s) => &*NSString::from_str(s), None => std::ptr::null() }
            ]
        }
    }

    fn avg_pooling_2d(
        &self,
        source: &Tensor,
        descriptor: &Pooling2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                avgPooling2DWithSourceTensor: source,
                descriptor: descriptor,
                name: match name { Some(s) => &*NSString::from_str(s), None => std::ptr::null() }
            ]
        }
    }

    fn avg_pooling_2d_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        descriptor: &Pooling2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                avgPooling2DGradientWithGradientTensor: gradient,
                sourceTensor: source,
                descriptor: descriptor,
                name: match name { Some(s) => &*NSString::from_str(s), None => std::ptr::null() }
            ]
        }
    }

    fn l2_norm_pooling_2d(
        &self,
        source: &Tensor,
        descriptor: &Pooling2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                L2NormPooling2DWithSourceTensor: source,
                descriptor: descriptor,
                name: match name { Some(s) => &*NSString::from_str(s), None => std::ptr::null() }
            ]
        }
    }

    fn l2_norm_pooling_2d_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        descriptor: &Pooling2DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                L2NormPooling2DGradientWithGradientTensor: gradient,
                sourceTensor: source,
                descriptor: descriptor,
                name: match name { Some(s) => &*NSString::from_str(s), None => std::ptr::null() }
            ]
        }
    }

    fn max_pooling_4d(
        &self,
        source: &Tensor,
        descriptor: &Pooling4DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                maxPooling4DWithSourceTensor: source,
                descriptor: descriptor,
                name: match name { Some(s) => &*NSString::from_str(s), None => std::ptr::null() }
            ]
        }
    }

    fn max_pooling_4d_return_indices(
        &self,
        source: &Tensor,
        descriptor: &Pooling4DOpDescriptor,
        name: Option<&str>,
    ) -> (Retained<Tensor>, Retained<Tensor>) {
        unsafe {
            let result: Retained<NSArray<Tensor>> = msg_send![
                self,
                maxPooling4DReturnIndicesWithSourceTensor: source,
                descriptor: descriptor,
                name: match name { Some(s) => &*NSString::from_str(s), None => std::ptr::null() }
            ];

            // Get the two result tensors
            let result_count: usize = msg_send![&*result, count];

            if result_count != 2 {
                panic!("maxPooling4DReturnIndices should return exactly 2 tensors");
            }

            let pooling_tensor: *mut Tensor = msg_send![&*result, objectAtIndex: 0u64];
            let indices_tensor: *mut Tensor = msg_send![&*result, objectAtIndex: 1u64];

            // Convert to retained references
            let pooling_tensor = Retained::retain(pooling_tensor).unwrap();
            let indices_tensor = Retained::retain(indices_tensor).unwrap();

            (pooling_tensor, indices_tensor)
        }
    }

    fn max_pooling_4d_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        descriptor: &Pooling4DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                maxPooling4DGradientWithGradientTensor: gradient,
                sourceTensor: source,
                descriptor: descriptor,
                name: match name { Some(s) => &*NSString::from_str(s), None => std::ptr::null() }
            ]
        }
    }

    fn max_pooling_4d_gradient_with_indices(
        &self,
        gradient: &Tensor,
        indices: &Tensor,
        output_shape: &[i64],
        descriptor: &Pooling4DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            // Create MPSShape from the output_shape array
            let shape_array = create_ns_array_from_i64_slice(output_shape);

            let tensor: Retained<Tensor> = msg_send![
                self,
                maxPooling4DGradientWithGradientTensor: gradient,
                indicesTensor: indices,
                outputShape: &*shape_array,
                descriptor: descriptor,
                name: match name { Some(s) => &*NSString::from_str(s), None => std::ptr::null() }
            ];

            tensor
        }
    }

    fn max_pooling_4d_gradient_with_indices_tensor(
        &self,
        gradient: &Tensor,
        indices: &Tensor,
        output_shape_tensor: &Tensor,
        descriptor: &Pooling4DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                maxPooling4DGradientWithGradientTensor: gradient,
                indicesTensor: indices,
                outputShapeTensor: output_shape_tensor,
                descriptor: descriptor,
                name: match name { Some(s) => &*NSString::from_str(s), None => std::ptr::null() }
            ]
        }
    }

    fn avg_pooling_4d(
        &self,
        source: &Tensor,
        descriptor: &Pooling4DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                avgPooling4DWithSourceTensor: source,
                descriptor: descriptor,
                name: match name { Some(s) => &*NSString::from_str(s), None => std::ptr::null() }
            ]
        }
    }

    fn avg_pooling_4d_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        descriptor: &Pooling4DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                avgPooling4DGradientWithGradientTensor: gradient,
                sourceTensor: source,
                descriptor: descriptor,
                name: match name { Some(s) => &*NSString::from_str(s), None => std::ptr::null() }
            ]
        }
    }

    fn l2_norm_pooling_4d(
        &self,
        source: &Tensor,
        descriptor: &Pooling4DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                L2NormPooling4DWithSourceTensor: source,
                descriptor: descriptor,
                name: match name { Some(s) => &*NSString::from_str(s), None => std::ptr::null() }
            ]
        }
    }

    fn l2_norm_pooling_4d_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        descriptor: &Pooling4DOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                L2NormPooling4DGradientWithGradientTensor: gradient,
                sourceTensor: source,
                descriptor: descriptor,
                name: match name { Some(s) => &*NSString::from_str(s), None => std::ptr::null() }
            ]
        }
    }
}