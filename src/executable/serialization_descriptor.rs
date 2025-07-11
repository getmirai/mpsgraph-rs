use super::DeploymentPlatform;
use objc2::rc::{Allocated, Retained};
use objc2::runtime::NSObject;
use objc2::{extern_class, msg_send, ClassType};
use objc2_foundation::{NSObjectProtocol, NSString};

extern_class!(
    #[derive(Debug, PartialEq, Eq)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraphExecutableSerializationDescriptor"]
    pub struct SerializationDescriptor;
);

unsafe impl NSObjectProtocol for SerializationDescriptor {}

impl SerializationDescriptor {
    /// Create a new serialization descriptor
    pub fn new() -> Retained<Self> {
        unsafe {
            let class = Self::class();
            let allocated: Allocated<Self> = msg_send![class, alloc];
            let initialized: Retained<Self> = msg_send![allocated, init];
            initialized
        }
    }

    /// Set append flag - if true, appends to existing file instead of overwriting
    pub fn set_append(&self, append: bool) {
        unsafe {
            let _: () = msg_send![self, setAppend: append];
        }
    }

    /// Set deployment platform
    pub fn set_deployment_platform(&self, platform: DeploymentPlatform) {
        unsafe {
            let _: () = msg_send![self, setDeploymentPlatform: platform as u64];
        }
    }

    /// Set minimum deployment target as a string (e.g., "13.0")
    pub fn set_minimum_deployment_target(&self, target: &str) {
        unsafe {
            let target_str = NSString::from_str(target);
            let _: () = msg_send![self, setMinimumDeploymentTarget: &*target_str];
        }
    }
}
