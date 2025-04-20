use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::AnyClass;
use objc2::extern_class;
use objc2_foundation::{NSObject, NSObjectProtocol, NSString};

use crate::graph::Graph;
use crate::shape::Shape;
use crate::tensor::Tensor;
use crate::CustomDefault;

/// Re-export padding styles from convolution_ops
pub use crate::convolution_ops::PaddingMode;

/// The reduction mode for stencil operations.
#[repr(u64)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ReductionMode {
    /// Min reduction
    Min = 0,
    /// Max reduction
    Max = 1,
    /// Sum reduction
    Sum = 2,
    /// Product reduction
    Product = 3,
    /// Argument Min reduction
    ArgumentMin = 4,
    /// Argument Max reduction
    ArgumentMax = 5,
}

/// Padding modes for stencil operations (from sample_grid_ops)
#[repr(i64)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum BoundaryMode {
    /// Constant padding
    Constant = 0,
    /// Reflect padding
    Reflect = 1,
    /// Symmetric padding
    Symmetric = 2,
    /// Clamp to edge padding (PyTorch ReplicationPad)
    ClampToEdge = 3,
    /// Zero padding
    Zero = 4,
    /// Periodic padding (x[-2] -> x[L-3], where L is size of x)
    Periodic = 5,
    /// Anti-periodic padding (x[-2] -> -x[L-3])
    AntiPeriodic = 6,
}

extern_class!(
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraphStencilOpDescriptor"]
    /// Descriptor for stencil operations
    pub struct StencilOpDescriptor;
);

unsafe impl NSObjectProtocol for StencilOpDescriptor {}

impl StencilOpDescriptor {
    /// Creates a new stencil operation descriptor with default values
    pub fn new() -> Retained<Self> {
        unsafe {
            let cls = AnyClass::get(c"MPSGraphStencilOpDescriptor").unwrap();
            msg_send![cls, descriptor]
        }
    }

    /// Creates a new stencil operation descriptor with the specified padding style
    pub fn with_padding_style(padding_style: PaddingMode) -> Retained<Self> {
        unsafe {
            let cls = AnyClass::get(c"MPSGraphStencilOpDescriptor").unwrap();
            msg_send![cls, descriptorWithPaddingStyle: padding_style as u64]
        }
    }

    /// Creates a new stencil operation descriptor with the specified explicit padding
    pub fn with_explicit_padding(explicit_padding: &Shape) -> Retained<Self> {
        unsafe {
            let cls = AnyClass::get(c"MPSGraphStencilOpDescriptor").unwrap();
            msg_send![cls, descriptorWithExplicitPadding: explicit_padding.as_ptr()]
        }
    }

    /// Creates a new stencil operation descriptor with the specified offsets and explicit padding
    pub fn with_offsets_and_explicit_padding(
        offsets: &Shape,
        explicit_padding: &Shape,
    ) -> Retained<Self> {
        unsafe {
            let cls = AnyClass::get(c"MPSGraphStencilOpDescriptor").unwrap();
            msg_send![cls, 
                descriptorWithOffsets: offsets.as_ptr(), 
                explicitPadding: explicit_padding.as_ptr()
            ]
        }
    }

    /// Creates a new stencil operation descriptor with all parameters specified
    pub fn with_all_params(
        reduction_mode: ReductionMode,
        offsets: &Shape,
        strides: &Shape,
        dilation_rates: &Shape,
        explicit_padding: &Shape,
        boundary_mode: BoundaryMode,
        padding_style: PaddingMode,
        padding_constant: f32,
    ) -> Retained<Self> {
        unsafe {
            let cls = AnyClass::get(c"MPSGraphStencilOpDescriptor").unwrap();
            msg_send![cls, 
                descriptorWithReductionMode: reduction_mode as u64,
                offsets: offsets.as_ptr(),
                strides: strides.as_ptr(),
                dilationRates: dilation_rates.as_ptr(),
                explicitPadding: explicit_padding.as_ptr(),
                boundaryMode: boundary_mode as i64,
                paddingStyle: padding_style as u64,
                paddingConstant: padding_constant
            ]
        }
    }

    /// Sets the reduction mode
    pub fn set_reduction_mode(&self, mode: ReductionMode) {
        unsafe {
            let _: () = msg_send![self, setReductionMode: mode as u64];
        }
    }

    /// Sets the offsets
    pub fn set_offsets(&self, offsets: &Shape) {
        unsafe {
            let _: () = msg_send![self, setOffsets: offsets.as_ptr()];
        }
    }

    /// Sets the strides
    pub fn set_strides(&self, strides: &Shape) {
        unsafe {
            let _: () = msg_send![self, setStrides: strides.as_ptr()];
        }
    }

    /// Sets the dilation rates
    pub fn set_dilation_rates(&self, dilation_rates: &Shape) {
        unsafe {
            let _: () = msg_send![self, setDilationRates: dilation_rates.as_ptr()];
        }
    }

    /// Sets the explicit padding
    pub fn set_explicit_padding(&self, explicit_padding: &Shape) {
        unsafe {
            let _: () = msg_send![self, setExplicitPadding: explicit_padding.as_ptr()];
        }
    }

    /// Sets the boundary mode
    pub fn set_boundary_mode(&self, mode: BoundaryMode) {
        unsafe {
            let _: () = msg_send![self, setBoundaryMode: mode as i64];
        }
    }

    /// Sets the padding style
    pub fn set_padding_style(&self, style: PaddingMode) {
        unsafe {
            let _: () = msg_send![self, setPaddingStyle: style as u64];
        }
    }

    /// Sets the padding constant
    pub fn set_padding_constant(&self, value: f32) {
        unsafe {
            let _: () = msg_send![self, setPaddingConstant: value];
        }
    }
}

impl CustomDefault for StencilOpDescriptor {
    fn custom_default() -> Retained<Self> {
        Self::new()
    }
}

/// Stencil operations for Graph
impl Graph {
    /// Creates a stencil operation and returns the result tensor.
    ///
    /// Performs a weighted reduction operation (See `ReductionMode`) on the last 4 dimensions of the `source`
    /// over the window determined by `weights`, according to the value defined in `descriptor`.
    /// The operation can be represented as:
    ///
    /// `y[i] = reduction{j in w} ( x[i + j] * w[j] )`
    ///
    /// # Arguments
    ///
    /// * `source` - The tensor containing the source data. Must be of rank 4 or greater.
    /// * `weights` - A 4-D tensor containing the weights data.
    /// * `descriptor` - The descriptor object that specifies the parameters for the stencil operation.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A valid Tensor object.
    pub fn stencil(
        &self,
        source: &Tensor,
        weights: &Tensor,
        descriptor: &StencilOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };
            
            msg_send![
                self,
                stencilWithSourceTensor: source,
                weightsTensor: weights,
                descriptor: descriptor,
                name: name_obj
            ]
        }
    }
}