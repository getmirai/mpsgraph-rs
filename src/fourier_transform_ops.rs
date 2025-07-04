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


/// Implementation of Fourier transform operations for Graph
impl Graph {
    /// Creates a fast Fourier transform operation
    pub fn fast_fourier_transform(
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
    pub fn fast_fourier_transform_with_tensor_axes(
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
    pub fn real_to_complex_fft(
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
    pub fn real_to_complex_fft_with_tensor_axes(
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
    pub fn complex_to_real_fft(
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
    pub fn complex_to_real_fft_with_tensor_axes(
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
    pub fn real_to_hermitean_fft(
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
    pub fn real_to_hermitean_fft_with_tensor_axes(
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
    pub fn hermitean_to_real_fft(
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
    pub fn hermitean_to_real_fft_with_tensor_axes(
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
    pub fn forward_fft(
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
    pub fn inverse_fft(
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
    pub fn forward_real_fft(
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
    pub fn inverse_real_fft(
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

