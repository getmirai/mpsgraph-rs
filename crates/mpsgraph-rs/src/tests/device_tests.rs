//! Tests for the Device implementation

use crate::{Device, CustomDefault};

#[test]
fn test_device_creation() {
    // Create a new device
    let device = Device::new();
    
    // Just check that the device was created - no easy way to check validity
    // with the Retained wrapper
    assert!(format!("{:?}", device).contains("MPSGraphDevice"));
    
    // Create a device using CustomDefault
    let default_device = Device::custom_default();
    assert!(format!("{:?}", default_device).contains("MPSGraphDevice"));
}