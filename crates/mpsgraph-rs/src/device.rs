use metal::foreign_types::ForeignType;
use metal::Device as MetalDevice;
use objc2::msg_send;
use objc2::runtime::AnyObject;
use std::fmt;
use std::ptr;

/// A wrapper for a Metal Performance Shaders Graph device object
pub struct Device(pub(crate) *mut AnyObject);

impl Default for Device {
    fn default() -> Self {
        Self::new()
    }
}

impl Device {
    /// Creates a new Device using the system default Metal device
    pub fn new() -> Self {
        let device = metal::Device::system_default().expect("No Metal device found");
        Self::with_device(&device)
    }

    /// Creates a new Device from a Metal device
    pub fn with_device(device: &MetalDevice) -> Self {
        unsafe {
            let class_name = c"MPSGraphDevice";
            let cls = objc2::runtime::AnyClass::get(class_name)
                .unwrap_or_else(|| panic!("MPSGraphDevice class not found"));
            let device_ptr = device.as_ptr() as *mut AnyObject;
            let obj: *mut AnyObject = msg_send![cls, deviceWithMTLDevice:device_ptr];
            let obj = objc2::ffi::objc_retain(obj as *mut _);
            Device(obj)
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                objc2::ffi::objc_release(self.0 as *mut _);
            }
        }
    }
}

impl Clone for Device {
    fn clone(&self) -> Self {
        unsafe {
            if !self.0.is_null() {
                let obj = objc2::ffi::objc_retain(self.0 as *mut _);
                Device(obj)
            } else {
                Device(ptr::null_mut())
            }
        }
    }
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Device")
    }
}
