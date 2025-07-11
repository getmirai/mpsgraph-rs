use super::Executable;
use super::Optimization;
use super::OptimizationProfile;
use crate::ComputeDevice;
use objc2::rc::{Allocated, Retained};
use objc2::runtime::NSObject;
use objc2::{extern_class, msg_send, ClassType, Message};
use objc2_foundation::{NSArray, NSDictionary, NSMutableDictionary, NSObjectProtocol, NSString};
use std::collections::HashMap;

extern_class!(
    #[derive(Debug, PartialEq, Eq)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraphCompilationDescriptor"]
    pub struct CompilationDescriptor;
);

unsafe impl NSObjectProtocol for CompilationDescriptor {}

impl CompilationDescriptor {
    /// Create a new compilation descriptor
    pub fn new() -> Retained<Self> {
        unsafe {
            let class = Self::class();
            let allocated: Allocated<Self> = msg_send![class, alloc];
            let initialized: Retained<Self> = msg_send![allocated, init];
            initialized
        }
    }

    /// Set the optimization level
    pub fn set_optimization_level(&self, level: Optimization) {
        unsafe {
            let _: () = msg_send![self, setOptimizationLevel: level as u64];
        }
    }

    /// Set the optimization profile
    pub fn set_optimization_profile(&self, profile: OptimizationProfile) {
        unsafe {
            let _: () = msg_send![self, setOptimizationProfile: profile as u64];
        }
    }

    /// Set whether to debug compile
    pub fn set_debug_compile(&self, debug_compile: bool) {
        unsafe {
            let _: () = msg_send![self, setDebugCompile: debug_compile];
        }
    }

    /// Get the callables map as a Rust HashMap
    pub fn get_callables(&self) -> HashMap<String, Retained<Executable>> {
        unsafe {
            let ns_dict_opt: Option<Retained<NSDictionary<NSString, Executable>>> =
                msg_send![self, callables];
            let ns_dict = match ns_dict_opt {
                Some(dict) => dict,
                None => return HashMap::new(),
            };

            let keys_opt: Option<Retained<NSArray<NSString>>> = msg_send![&*ns_dict, allKeys];
            let keys = match keys_opt {
                Some(arr) => arr,
                None => return HashMap::new(),
            };

            let mut result = HashMap::with_capacity(keys.len());

            for i in 0..keys.len() {
                let ns_key_opt: Option<Retained<NSString>> = msg_send![&*keys, objectAtIndex: i];
                let ns_key = match ns_key_opt {
                    Some(key) => key,
                    None => continue,
                };

                let key_str = ns_key.to_string();

                let executable_opt: Option<Retained<Executable>> =
                    msg_send![&*ns_dict, objectForKey: &*ns_key];
                let executable = match executable_opt {
                    Some(exec) => exec,
                    None => continue,
                };

                result.insert(key_str, executable);
            }
            result
        }
    }

    /// Set the callables map using a Rust HashMap
    pub fn set_callables(&self, callables: &HashMap<String, &Executable>) {
        if callables.is_empty() {
            // If the HashMap is empty, set the property to nil
            unsafe {
                let _: () = msg_send![self, setCallables: std::ptr::null::<NSDictionary<NSString, Executable>>()];
            }
            return;
        }

        // Create a mutable dictionary
        let mutable_dict = NSMutableDictionary::<NSString, Executable>::new();

        // Add each entry to the dictionary
        for (key, &exec_ref) in callables {
            let ns_key = NSString::from_str(key);
            unsafe {
                let _: () = msg_send![&*mutable_dict, setObject: exec_ref, forKey: &*ns_key];
            }
        }

        // Convert to immutable dictionary
        let immutable_dict: Retained<NSDictionary<NSString, Executable>> =
            unsafe { msg_send![&*mutable_dict, copy] };

        // Set the property
        unsafe {
            let _: () = msg_send![self, setCallables: &*immutable_dict];
        }
    }

    /// Add a callable executable for a specific symbol name
    pub fn add_callable(&self, symbol_name: &str, executable: &Executable) {
        // Get the current callables
        let mut callables_retained_map = self.get_callables();

        // Add the new callable, retaining it for the map
        callables_retained_map.insert(symbol_name.to_string(), executable.retain());

        // Convert map values from Retained<Executable> to &Executable for set_callables
        let callables_refs_map: HashMap<String, &Executable> = callables_retained_map
            .iter()
            .map(|(k, v)| (k.clone(), v.as_ref()))
            .collect();

        // Update the callables property
        self.set_callables(&callables_refs_map);
    }

    /// Remove a callable executable for a specific symbol name
    pub fn remove_callable(&self, symbol_name: &str) {
        let mut callables_retained_map = self.get_callables();
        callables_retained_map.remove(symbol_name);

        let callables_refs_map: HashMap<String, &Executable> = callables_retained_map
            .iter()
            .map(|(k, v)| (k.clone(), v.as_ref()))
            .collect();
        self.set_callables(&callables_refs_map);
    }

    pub fn set_allowed_device(&self, devices: u64) {
        unsafe {
            let _: () = msg_send![self, setAllowedComputeDevices: devices as u64];
        }
    }

    pub fn set_compiler_options(&self, options: u64) {
        unsafe {
            let _: () = msg_send![self, setCompilerOptions: options];
        }
    }

    pub fn set_ane_compiler_spatial_splitting(&self, value: u64) {
        unsafe {
            let _: () = msg_send![self, setAneCompilerSpatialSplitting: value];
        }
    }

    pub fn set_enable_ane_fw_to_fw_signal(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setEnableANEFWToFWSignal: enable];
        }
    }

    pub fn set_enable_ane_late_latch(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setEnableANELateLatch: enable];
        }
    }

    pub fn set_print_ane_placement_analysis(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setPrintANEPlacementAnalysis: enable];
        }
    }

    pub fn set_preferred_device(&self, device: ComputeDevice) {
        unsafe {
            let _: () = msg_send![self, setPreferredDevice: device.bits()];
        }
    }

    pub fn set_allowed_compute_devices(&self, devices: ComputeDevice) {
        unsafe {
            let _: () = msg_send![self, setAllowedComputeDevices: devices.bits()];
        }
    }

    pub fn set_enable_parallel_encode(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setEnableParallelEncode: enable];
        }
    }

    pub fn set_maximum_number_of_parallel_encoding_regions(&self, value: u64) {
        unsafe {
            let _: () = msg_send![self, setMaximumNumberOfParallelEncodingRegions: value];
        }
    }

    pub fn set_minimum_number_of_ops_in_parallel_region(&self, value: u64) {
        unsafe {
            let _: () = msg_send![self, setMinimumNumberOfOpsInParallelRegion: value];
        }
    }

    pub fn set_enable_mlir_diagnostics(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setEnableMLIRDiagnostics: enable];
        }
    }

    pub fn set_enable_shape_equivalence(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setEnableShapeEquivalence: enable];
        }
    }
}
