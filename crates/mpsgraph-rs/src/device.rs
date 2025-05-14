use metal::foreign_types::ForeignType;
use metal::Device as MetalDevice;
use objc2::rc::Retained;
use objc2::runtime::NSObject;
use objc2::{extern_class, msg_send, ClassType};
use objc2_foundation::NSObjectProtocol;
use std::fmt;
use std::str::FromStr;

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct MPSGraphComputeDevice: u64 {
        const GPU = 1 << 0;
        const ANE = 1 << 1;
        const CPU = 1 << 2;
    }
}

impl fmt::Display for MPSGraphComputeDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return write!(f, "unknown");
        }
        let mut parts = Vec::new();
        if self.contains(MPSGraphComputeDevice::GPU) {
            parts.push("gpu");
        }
        if self.contains(MPSGraphComputeDevice::ANE) {
            parts.push("ane");
        }
        if self.contains(MPSGraphComputeDevice::CPU) {
            parts.push("cpu");
        }
        write!(f, "{}", parts.join("+"))
    }
}

impl FromStr for MPSGraphComputeDevice {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "unknown" {
            // Or handle as an error, depending on desired behavior for "unknown"
            return Ok(MPSGraphComputeDevice::empty());
        }
        let mut device = MPSGraphComputeDevice::empty();
        for part in s.split('+') {
            match part {
                "gpu" => device |= MPSGraphComputeDevice::GPU,
                "ane" => device |= MPSGraphComputeDevice::ANE,
                "cpu" => device |= MPSGraphComputeDevice::CPU,
                _ => return Err(format!("Invalid device string: {}", part)),
            }
        }
        if device.is_empty() && !s.is_empty() && s != "unknown" {
            Err(format!(
                "Invalid or empty MPSGraphComputeDevice string: {}",
                s
            ))
        } else {
            Ok(device)
        }
    }
}

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
