use crate::graph::Graph;
use crate::tensor::Tensor;
use objc2::msg_send;
use objc2::rc::Retained;
use objc2::ClassType;
use objc2_foundation::{NSArray, NSNumber, NSObject, NSObjectProtocol, NSString};

/// Scaling mode for FFT operations
///
/// Available since macOS 14.0, iOS 17.0, tvOS 17.0
#[repr(u64)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FFTScalingMode {
    /// No scaling
    None = 0,
    /// Scale by reciprocal of total FFT size
    Size = 1,
    /// Scale by reciprocal square root of total FFT size
    Unitary = 2,
}

objc2::extern_class!(
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraphFFTDescriptor"]
    /// Descriptor for FFT operations
    ///
    /// Available since macOS 14.0, iOS 17.0, tvOS 17.0
    pub struct FFTDescriptor;
);

unsafe impl NSObjectProtocol for FFTDescriptor {}

impl FFTDescriptor {
    /// Creates a new FFT descriptor with default settings
    pub fn new() -> Retained<Self> {
        unsafe { msg_send![Self::class(), descriptor] }
    }

    /// Sets whether to use inverse FFT (positive phase factor)
    pub fn set_inverse(&self, inverse: bool) {
        unsafe {
            let _: () = msg_send![self, setInverse: inverse];
        }
    }

    /// Gets whether inverse FFT is used
    pub fn inverse(&self) -> bool {
        unsafe { msg_send![self, inverse] }
    }

    /// Sets the scaling mode
    pub fn set_scaling_mode(&self, mode: FFTScalingMode) {
        unsafe {
            let _: () = msg_send![self, setScalingMode: mode as u64];
        }
    }

    /// Gets the scaling mode
    pub fn scaling_mode(&self) -> FFTScalingMode {
        unsafe {
            let mode: u64 = msg_send![self, scalingMode];
            match mode {
                0 => FFTScalingMode::None,
                1 => FFTScalingMode::Size,
                2 => FFTScalingMode::Unitary,
                _ => FFTScalingMode::None,
            }
        }
    }

    /// Sets the normalization factor for the FFT
    pub fn set_normalization_factor(&self, factor: f64) {
        unsafe {
            let _: () = msg_send![self, setNormalizationFactor:factor];
        }
    }

    /// Sets whether to round to odd Hermitean output dimensions
    pub fn set_round_to_odd_hermitean(&self, round: bool) {
        unsafe {
            let _: () = msg_send![self, setRoundToOddHermitean: round];
        }
    }

    /// Gets whether to round to odd Hermitean output dimensions
    pub fn round_to_odd_hermitean(&self) -> bool {
        unsafe { msg_send![self, roundToOddHermitean] }
    }
}

impl FFTDescriptor {
    // Helper method that serves as a replacement for Default
    pub fn default_descriptor() -> Retained<Self> {
        Self::new()
    }
}

/// Trait for performing Fourier transform operations on a graph
pub trait GraphFourierTransformOps {
    /// Creates a fast Fourier transform operation
    ///
    /// # Arguments
    ///
    /// * `tensor` - Tensor to transform (complex tensor)
    /// * `axes` - Axes along which to apply the transform
    /// * `descriptor` - FFT descriptor specifying parameters
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A transformed tensor
    ///
    /// # Availability
    ///
    /// Available since macOS 14.0, iOS 17.0, tvOS 17.0
    fn fast_fourier_transform(
        &self,
        tensor: &Tensor,
        axes: &[i64],
        descriptor: &FFTDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a fast Fourier transform operation with axes specified by a tensor
    ///
    /// # Arguments
    ///
    /// * `tensor` - Tensor to transform (complex tensor)
    /// * `axes_tensor` - Tensor containing the axes along which to apply the transform
    /// * `descriptor` - FFT descriptor specifying parameters
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A transformed tensor
    ///
    /// # Availability
    ///
    /// Available since macOS 14.0, iOS 17.0, tvOS 17.0
    fn fast_fourier_transform_with_tensor_axes(
        &self,
        tensor: &Tensor,
        axes_tensor: &Tensor,
        descriptor: &FFTDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a real-to-complex Fast Fourier Transform.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor (real tensor)
    /// * `axes` - Axes along which to perform the FFT
    /// * `descriptor` - Descriptor for the FFT operation
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object or None if error
    fn real_to_complex_fft(
        &self,
        tensor: &Tensor,
        axes: &[i64],
        descriptor: &FFTDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a real-to-complex Fast Fourier Transform using tensor axes.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor (real tensor)
    /// * `axes_tensor` - Tensor containing the axes along which to perform the FFT
    /// * `descriptor` - Descriptor for the FFT operation
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object or None if error
    fn real_to_complex_fft_with_tensor_axes(
        &self,
        tensor: &Tensor,
        axes_tensor: &Tensor,
        descriptor: &FFTDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a complex-to-real Fast Fourier Transform.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor (complex tensor with alternating real and imaginary components)
    /// * `axes` - Axes along which to perform the FFT
    /// * `descriptor` - Descriptor for the FFT operation
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object or None if error
    fn complex_to_real_fft(
        &self,
        tensor: &Tensor,
        axes: &[i64],
        descriptor: &FFTDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a complex-to-real Fast Fourier Transform using tensor axes.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor (complex tensor with alternating real and imaginary components)
    /// * `axes_tensor` - Tensor containing the axes along which to perform the FFT
    /// * `descriptor` - Descriptor for the FFT operation
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object or None if error
    fn complex_to_real_fft_with_tensor_axes(
        &self,
        tensor: &Tensor,
        axes_tensor: &Tensor,
        descriptor: &FFTDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a real-to-Hermitean fast Fourier transform operation
    ///
    /// # Arguments
    ///
    /// * `tensor` - Real tensor to transform
    /// * `axes` - Axes along which to apply the transform
    /// * `descriptor` - FFT descriptor specifying parameters
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A complex tensor in Hermitean format
    ///
    /// # Availability
    ///
    /// Available since macOS 14.0, iOS 17.0, tvOS 17.0
    fn real_to_hermitean_fft(
        &self,
        tensor: &Tensor,
        axes: &[i64],
        descriptor: &FFTDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a real-to-Hermitean fast Fourier transform operation with axes specified by a tensor
    ///
    /// # Arguments
    ///
    /// * `tensor` - Real tensor to transform
    /// * `axes_tensor` - Tensor containing the axes along which to apply the transform
    /// * `descriptor` - FFT descriptor specifying parameters
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A complex tensor in Hermitean format
    ///
    /// # Availability
    ///
    /// Available since macOS 14.0, iOS 17.0, tvOS 17.0
    fn real_to_hermitean_fft_with_tensor_axes(
        &self,
        tensor: &Tensor,
        axes_tensor: &Tensor,
        descriptor: &FFTDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a Hermitean-to-real fast Fourier transform operation
    ///
    /// # Arguments
    ///
    /// * `tensor` - Complex tensor in Hermitean format to transform
    /// * `axes` - Axes along which to apply the transform
    /// * `descriptor` - FFT descriptor specifying parameters
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A real tensor
    ///
    /// # Availability
    ///
    /// Available since macOS 14.0, iOS 17.0, tvOS 17.0
    fn hermitean_to_real_fft(
        &self,
        tensor: &Tensor,
        axes: &[i64],
        descriptor: &FFTDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a Hermitean-to-real fast Fourier transform operation with axes specified by a tensor
    ///
    /// # Arguments
    ///
    /// * `tensor` - Complex tensor in Hermitean format to transform
    /// * `axes_tensor` - Tensor containing the axes along which to apply the transform
    /// * `descriptor` - FFT descriptor specifying parameters
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A real tensor
    ///
    /// # Availability
    ///
    /// Available since macOS 14.0, iOS 17.0, tvOS 17.0
    fn hermitean_to_real_fft_with_tensor_axes(
        &self,
        tensor: &Tensor,
        axes_tensor: &Tensor,
        descriptor: &FFTDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a forward FFT operation using complex-valued input.
    ///
    /// # Arguments
    ///
    /// * `real` - Tensor with the real part of the input
    /// * `imaginary` - Tensor with the imaginary part of the input
    /// * `descriptor` - FFT descriptor specifying the transform parameters
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A tuple of Tensor objects (real_output, imaginary_output).
    ///
    /// # Deprecated
    ///
    /// This method uses the older API. Consider using `fast_fourier_transform` instead.
    fn forward_fft(
        &self,
        real: &Tensor,
        imaginary: &Tensor,
        descriptor: &FFTDescriptor,
        name: Option<&str>,
    ) -> Option<(Retained<Tensor>, Retained<Tensor>)>;

    /// Creates an inverse FFT operation using complex-valued input.
    ///
    /// # Arguments
    ///
    /// * `real` - Tensor with the real part of the input
    /// * `imaginary` - Tensor with the imaginary part of the input
    /// * `descriptor` - FFT descriptor specifying the transform parameters
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A tuple of Tensor objects (real_output, imaginary_output).
    ///
    /// # Deprecated
    ///
    /// This method uses the older API. Consider using `fast_fourier_transform` instead
    /// with `descriptor.set_inverse(true)`.
    fn inverse_fft(
        &self,
        real: &Tensor,
        imaginary: &Tensor,
        descriptor: &FFTDescriptor,
        name: Option<&str>,
    ) -> Option<(Retained<Tensor>, Retained<Tensor>)>;

    /// Creates a forward FFT operation using real-valued input.
    ///
    /// # Arguments
    ///
    /// * `real` - Tensor with the real input values
    /// * `descriptor` - FFT descriptor specifying the transform parameters
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A tuple of Tensor objects (real_output, imaginary_output).
    ///
    /// # Deprecated
    ///
    /// This method uses the older API. Consider using `real_to_hermitean_fft` instead.
    fn forward_real_fft(
        &self,
        real: &Tensor,
        descriptor: &FFTDescriptor,
        name: Option<&str>,
    ) -> Option<(Retained<Tensor>, Retained<Tensor>)>;

    /// Creates an inverse FFT operation that produces real-valued output.
    ///
    /// # Arguments
    ///
    /// * `real` - Tensor with the real part of the input
    /// * `imaginary` - Tensor with the imaginary part of the input
    /// * `descriptor` - FFT descriptor specifying the transform parameters
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// The real-valued output as an Tensor object.
    ///
    /// # Deprecated
    ///
    /// This method uses the older API. Consider using `hermitean_to_real_fft` instead.
    fn inverse_real_fft(
        &self,
        real: &Tensor,
        imaginary: &Tensor,
        descriptor: &FFTDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor>;
}

/// Implementation of Fourier transform operations for Graph
impl GraphFourierTransformOps for Graph {
    /// Creates a fast Fourier transform operation
    fn fast_fourier_transform(
        &self,
        tensor: &Tensor,
        axes: &[i64],
        descriptor: &FFTDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        let name_obj = name.map(NSString::from_str);
        let axes_numbers: Vec<Retained<NSNumber>> =
            axes.iter().map(|&x| NSNumber::new_i64(x)).collect();
        let refs: Vec<&NSNumber> = axes_numbers.iter().map(|n| n.as_ref()).collect();
        let ns_array = NSArray::from_slice(&refs);
        unsafe {
            msg_send![
                self, fastFourierTransformWithTensor: tensor,
                axes: &*ns_array,
                descriptor: descriptor,
                name: name_obj.as_deref().map_or(std::ptr::null(), |s| s as *const _),
            ]
        }
    }

    /// Creates a fast Fourier transform operation with axes specified by a tensor
    fn fast_fourier_transform_with_tensor_axes(
        &self,
        tensor: &Tensor,
        axes_tensor: &Tensor,
        descriptor: &FFTDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        let name_obj = name.map(NSString::from_str);
        unsafe {
            msg_send![
                self, fastFourierTransformWithTensor: tensor,
                axesTensor: axes_tensor,
                descriptor: descriptor,
                name: name_obj.as_deref().map_or(std::ptr::null(), |s| s as *const _),
            ]
        }
    }

    /// Creates a real-to-complex Fast Fourier Transform.
    fn real_to_complex_fft(
        &self,
        tensor: &Tensor,
        axes: &[i64],
        descriptor: &FFTDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        let name_obj = name.map(NSString::from_str);
        let axes_numbers: Vec<Retained<NSNumber>> =
            axes.iter().map(|&x| NSNumber::new_i64(x)).collect();
        let refs: Vec<&NSNumber> = axes_numbers.iter().map(|n| n.as_ref()).collect();
        let ns_array = NSArray::from_slice(&refs);
        unsafe {
            msg_send![
                self, realToComplexFFTWithTensor: tensor,
                axes: &*ns_array,
                descriptor: descriptor,
                name: name_obj.as_deref().map_or(std::ptr::null(), |s| s as *const _),
            ]
        }
    }

    /// Creates a real-to-complex Fast Fourier Transform using tensor axes.
    fn real_to_complex_fft_with_tensor_axes(
        &self,
        tensor: &Tensor,
        axes_tensor: &Tensor,
        descriptor: &FFTDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        let name_obj = name.map(NSString::from_str);
        unsafe {
            msg_send![
                self, realToComplexFFTWithTensor: tensor,
                axesTensor: axes_tensor,
                descriptor: descriptor,
                name: name_obj.as_deref().map_or(std::ptr::null(), |s| s as *const _),
            ]
        }
    }

    /// Creates a complex-to-real Fast Fourier Transform.
    fn complex_to_real_fft(
        &self,
        tensor: &Tensor,
        axes: &[i64],
        descriptor: &FFTDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        let name_obj = name.map(NSString::from_str);
        let axes_numbers: Vec<Retained<NSNumber>> =
            axes.iter().map(|&x| NSNumber::new_i64(x)).collect();
        let refs: Vec<&NSNumber> = axes_numbers.iter().map(|n| n.as_ref()).collect();
        let ns_array = NSArray::from_slice(&refs);
        unsafe {
            msg_send![
                self, complexToRealFFTWithTensor: tensor,
                axes: &*ns_array,
                descriptor: descriptor,
                name: name_obj.as_deref().map_or(std::ptr::null(), |s| s as *const _),
            ]
        }
    }

    /// Creates a complex-to-real Fast Fourier Transform using tensor axes.
    fn complex_to_real_fft_with_tensor_axes(
        &self,
        tensor: &Tensor,
        axes_tensor: &Tensor,
        descriptor: &FFTDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        let name_obj = name.map(NSString::from_str);
        unsafe {
            msg_send![
                self, complexToRealFFTWithTensor: tensor,
                axesTensor: axes_tensor,
                descriptor: descriptor,
                name: name_obj.as_deref().map_or(std::ptr::null(), |s| s as *const _),
            ]
        }
    }

    /// Creates a real-to-Hermitean fast Fourier transform operation
    fn real_to_hermitean_fft(
        &self,
        tensor: &Tensor,
        axes: &[i64],
        descriptor: &FFTDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        let name_obj = name.map(NSString::from_str);
        let axes_numbers: Vec<Retained<NSNumber>> =
            axes.iter().map(|&x| NSNumber::new_i64(x)).collect();
        let refs: Vec<&NSNumber> = axes_numbers.iter().map(|n| n.as_ref()).collect();
        let ns_array = NSArray::from_slice(&refs);
        unsafe {
            msg_send![
                self, realToHermiteanFFTWithTensor: tensor,
                axes: &*ns_array,
                descriptor: descriptor,
                name: name_obj.as_deref().map_or(std::ptr::null(), |s| s as *const _),
            ]
        }
    }

    /// Creates a real-to-Hermitean fast Fourier transform operation with axes specified by a tensor
    fn real_to_hermitean_fft_with_tensor_axes(
        &self,
        tensor: &Tensor,
        axes_tensor: &Tensor,
        descriptor: &FFTDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        let name_obj = name.map(NSString::from_str);
        unsafe {
            msg_send![
                self, realToHermiteanFFTWithTensor: tensor,
                axesTensor: axes_tensor,
                descriptor: descriptor,
                name: name_obj.as_deref().map_or(std::ptr::null(), |s| s as *const _),
            ]
        }
    }

    /// Creates a Hermitean-to-real fast Fourier transform operation
    fn hermitean_to_real_fft(
        &self,
        tensor: &Tensor,
        axes: &[i64],
        descriptor: &FFTDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        let name_obj = name.map(NSString::from_str);
        let axes_numbers: Vec<Retained<NSNumber>> =
            axes.iter().map(|&x| NSNumber::new_i64(x)).collect();
        let refs: Vec<&NSNumber> = axes_numbers.iter().map(|n| n.as_ref()).collect();
        let ns_array = NSArray::from_slice(&refs);
        unsafe {
            msg_send![
                self, hermiteanToRealFFTWithTensor: tensor,
                axes: &*ns_array,
                descriptor: descriptor,
                name: name_obj.as_deref().map_or(std::ptr::null(), |s| s as *const _),
            ]
        }
    }

    /// Creates a Hermitean-to-real fast Fourier transform operation with axes specified by a tensor
    fn hermitean_to_real_fft_with_tensor_axes(
        &self,
        tensor: &Tensor,
        axes_tensor: &Tensor,
        descriptor: &FFTDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        let name_obj = name.map(NSString::from_str);
        unsafe {
            msg_send![
                self, hermiteanToRealFFTWithTensor: tensor,
                axesTensor: axes_tensor,
                descriptor: descriptor,
                name: name_obj.as_deref().map_or(std::ptr::null(), |s| s as *const _),
            ]
        }
    }

    // Legacy API methods for backward compatibility

    /// Creates a forward FFT operation using complex-valued input.
    fn forward_fft(
        &self,
        real: &Tensor,
        imaginary: &Tensor,
        descriptor: &FFTDescriptor,
        name: Option<&str>,
    ) -> Option<(Retained<Tensor>, Retained<Tensor>)> {
        let name_obj = name.map(NSString::from_str);
        unsafe {
            let result_array_opt: Option<Retained<NSArray<Tensor>>> = msg_send![
                self, forwardFFTWithRealTensor: real,
                imaginaryTensor: imaginary,
                descriptor: descriptor,
                name: name_obj.as_deref().map_or(std::ptr::null(), |s| s as *const _),
            ];

            result_array_opt.and_then(|result_array| {
                if result_array.count() == 2 {
                    let real_output: Option<Retained<Tensor>> =
                        msg_send![&*result_array, objectAtIndex: 0u64];
                    let imag_output: Option<Retained<Tensor>> =
                        msg_send![&*result_array, objectAtIndex: 1u64];
                    match (real_output, imag_output) {
                        (Some(r), Some(i)) => Some((r, i)),
                        _ => None,
                    }
                } else {
                    None
                }
            })
        }
    }

    /// Creates an inverse FFT operation using complex-valued input.
    fn inverse_fft(
        &self,
        real: &Tensor,
        imaginary: &Tensor,
        descriptor: &FFTDescriptor,
        name: Option<&str>,
    ) -> Option<(Retained<Tensor>, Retained<Tensor>)> {
        let name_obj = name.map(NSString::from_str);
        unsafe {
            let result_array_opt: Option<Retained<NSArray<Tensor>>> = msg_send![
                self, inverseFFTWithRealTensor: real,
                imaginaryTensor: imaginary,
                descriptor: descriptor,
                name: name_obj.as_deref().map_or(std::ptr::null(), |s| s as *const _),
            ];
            result_array_opt.and_then(|result_array| {
                if result_array.count() == 2 {
                    let real_output: Option<Retained<Tensor>> =
                        msg_send![&*result_array, objectAtIndex: 0u64];
                    let imag_output: Option<Retained<Tensor>> =
                        msg_send![&*result_array, objectAtIndex: 1u64];
                    match (real_output, imag_output) {
                        (Some(r), Some(i)) => Some((r, i)),
                        _ => None,
                    }
                } else {
                    None
                }
            })
        }
    }

    /// Creates a forward FFT operation using real-valued input.
    fn forward_real_fft(
        &self,
        real: &Tensor,
        descriptor: &FFTDescriptor,
        name: Option<&str>,
    ) -> Option<(Retained<Tensor>, Retained<Tensor>)> {
        let name_obj = name.map(NSString::from_str);
        unsafe {
            let result_array_opt: Option<Retained<NSArray<Tensor>>> = msg_send![
                self, forwardRealFFTWithRealTensor: real,
                descriptor: descriptor,
                name: name_obj.as_deref().map_or(std::ptr::null(), |s| s as *const _),
            ];
            result_array_opt.and_then(|result_array| {
                if result_array.count() == 2 {
                    let real_output: Option<Retained<Tensor>> =
                        msg_send![&*result_array, objectAtIndex: 0u64];
                    let imag_output: Option<Retained<Tensor>> =
                        msg_send![&*result_array, objectAtIndex: 1u64];
                    match (real_output, imag_output) {
                        (Some(r), Some(i)) => Some((r, i)),
                        _ => None,
                    }
                } else {
                    None
                }
            })
        }
    }

    /// Creates an inverse FFT operation that produces real-valued output.
    fn inverse_real_fft(
        &self,
        real: &Tensor,
        imaginary: &Tensor,
        descriptor: &FFTDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        let name_obj = name.map(NSString::from_str);
        unsafe {
            let tensor: Retained<Tensor> = msg_send![
                self, inverseRealFFTWithRealTensor: real,
                imaginaryTensor: imaginary,
                descriptor: descriptor,
                name: name_obj.as_deref().map_or(std::ptr::null(), |s| s as *const _),
            ];
            tensor
        }
    }
}

/// Extension trait for easier access to Fourier transform operations
pub trait GraphFourierTransformOpsExtension {
    /// Get access to Fourier transform operations
    fn fourier_transform_ops(&self) -> &dyn GraphFourierTransformOps;
}

impl GraphFourierTransformOpsExtension for Graph {
    fn fourier_transform_ops(&self) -> &dyn GraphFourierTransformOps {
        self
    }
}
