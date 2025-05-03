use metal::foreign_types::ForeignType;
use metal::Device as MetalDevice;
use objc2::rc::Retained;
use objc2::runtime::NSObject;
use objc2::{extern_class, msg_send, ClassType};
use objc2_foundation::NSObjectProtocol;

// The extern_class macro will generate the struct definition
extern_class!(
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraphDevice"]
    pub struct Device;
);

unsafe impl NSObjectProtocol for Device {}

impl Device {
    /// Creates a new Device using the system default Metal device
    pub fn new() -> Retained<Self> {
        let device = metal::Device::system_default().expect("No Metal device found");
        Self::with_device(&device)
    }

    /// Creates a new Device from a Metal device
    pub fn with_device(device: &MetalDevice) -> Retained<Self> {
        unsafe {
            let class = Self::class();
            let device_ptr = device.as_ptr();

            // Cast the raw device pointer to id type (NSObject) as expected by MPS
            let device_id = device_ptr as *mut objc2::runtime::AnyObject;

            msg_send![class, deviceWithMTLDevice: device_id]
        }
    }
}
