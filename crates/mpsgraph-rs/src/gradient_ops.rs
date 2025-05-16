use crate::graph::Graph;
use crate::tensor::Tensor;
use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::{NSArray, NSObject, NSString};
use std::collections::HashMap;

/// Gradient (Automatic Differentiation) operations for Graph
pub trait GraphGradientOps {
    fn gradient_for_primary_tensor(
        &self,
        primary_tensor: &Tensor,
        tensors: &[&Tensor],
        name: Option<&str>,
    ) -> Option<HashMap<Retained<Tensor>, Retained<Tensor>>>;
}

impl GraphGradientOps for Graph {
    fn gradient_for_primary_tensor(
        &self,
        primary_tensor: &Tensor,
        tensors: &[&Tensor],
        name: Option<&str>,
    ) -> Option<HashMap<Retained<Tensor>, Retained<Tensor>>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            // Create NSArray from the Tensor references
            let tensor_refs: Vec<&Tensor> = tensors.to_vec();
            let tensors_array = NSArray::from_slice(&tensor_refs);

            // Call the Objective-C method
            let dict: *mut NSObject = msg_send![
                self,
                gradientForPrimaryTensor: primary_tensor,
                withTensors: &*tensors_array,
                name: name_ptr
            ];

            if dict.is_null() {
                return None;
            }

            // Convert to Retained to get proper memory management
            let dict_retained: Retained<NSObject> = Retained::retain_autoreleased(dict).unwrap();

            // Get the keys array
            let keys: *mut NSArray<Tensor> = msg_send![&*dict_retained, allKeys];

            if keys.is_null() {
                return None;
            }

            let keys_retained: Retained<NSArray<Tensor>> =
                Retained::retain_autoreleased(keys).unwrap();
            let keys_count: usize = msg_send![&*keys_retained, count];

            // Convert NSDictionary to HashMap
            let mut result = HashMap::with_capacity(keys_count);

            for i in 0..keys_count {
                let key: *mut Tensor = msg_send![&*keys_retained, objectAtIndex: i];
                let key_retained = Retained::retain_autoreleased(key).unwrap();

                let value: *mut Tensor = msg_send![&*dict_retained, objectForKey: &*key_retained];
                let value_retained = Retained::retain_autoreleased(value).unwrap();

                result.insert(key_retained, value_retained);
            }

            Some(result)
        }
    }
}

/// Extension trait providing a method for Graph to access gradient operations
pub trait GraphGradientOpsExtension {
    /// Access gradient operations for this graph
    fn gradient_ops(&self) -> &dyn GraphGradientOps;
}

impl GraphGradientOpsExtension for Graph {
    fn gradient_ops(&self) -> &dyn GraphGradientOps {
        self
    }
}
