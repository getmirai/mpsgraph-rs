use objc2::msg_send;
use objc2::runtime::AnyObject;
use std::ptr;

/// Convolution padding mode
#[repr(u64)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingMode {
    /// Valid padding - no padding
    Valid = 0,
    /// Same padding - pad to maintain same size
    Same = 1,
    /// Explicit padding - user-specified padding values
    Explicit = 2,
}

/// Dataflow direction for convolution
#[repr(u64)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvolutionDataLayout {
    /// Data is arranged as NHWC (batch, height, width, channels)
    NHWC = 0,
    /// Data is arranged as NCHW (batch, channels, height, width)
    NCHW = 1,
}

/// Weight layout for convolution
#[repr(u64)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightsLayout {
    /// Weights arranged as HWIO (height, width, input channels, output channels)
    HWIO = 0,
    /// Weights arranged as OHWI (output channels, height, width, input channels)
    OHWI = 1,
    /// Weights arranged as IHWO (input channels, height, width, output channels)
    IHWO = 2,
}

/// Descriptor for 2D convolution operations
pub struct Convolution2DOpDescriptor(pub(crate) *mut AnyObject);

/// Descriptor for 3D convolution operations
pub struct Convolution3DOpDescriptor(pub(crate) *mut AnyObject);

impl Drop for Convolution2DOpDescriptor {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                objc2::ffi::objc_release(self.0 as *mut _);
            }
        }
    }
}

impl Clone for Convolution2DOpDescriptor {
    fn clone(&self) -> Self {
        unsafe {
            if !self.0.is_null() {
                let obj = objc2::ffi::objc_retain(self.0 as *mut _);
                Convolution2DOpDescriptor(obj)
            } else {
                Convolution2DOpDescriptor(ptr::null_mut())
            }
        }
    }
}

impl Drop for Convolution3DOpDescriptor {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                objc2::ffi::objc_release(self.0 as *mut _);
            }
        }
    }
}

impl Clone for Convolution3DOpDescriptor {
    fn clone(&self) -> Self {
        unsafe {
            if !self.0.is_null() {
                let obj = objc2::ffi::objc_retain(self.0 as *mut _);
                Convolution3DOpDescriptor(obj)
            } else {
                Convolution3DOpDescriptor(ptr::null_mut())
            }
        }
    }
}

impl Default for Convolution2DOpDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

impl Convolution2DOpDescriptor {
    /// Creates a new descriptor with default parameters
    pub fn new() -> Self {
        unsafe {
            let cls = objc2::runtime::AnyClass::get(c"MPSGraphConvolution2DOpDescriptor").unwrap();
            let desc: *mut AnyObject = msg_send![cls, descriptor];
            Convolution2DOpDescriptor(objc2::ffi::objc_retain(desc as *mut _))
        }
    }
}

impl Default for Convolution3DOpDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

impl Convolution3DOpDescriptor {
    /// Creates a new descriptor with default parameters
    pub fn new() -> Self {
        unsafe {
            let cls = objc2::runtime::AnyClass::get(c"MPSGraphConvolution3DOpDescriptor").unwrap();
            let desc: *mut AnyObject = msg_send![cls, descriptor];
            Convolution3DOpDescriptor(objc2::ffi::objc_retain(desc as *mut _))
        }
    }
}
