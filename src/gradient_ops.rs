//! Gradient (automatic differentiation) helper now inherent on `Graph`.

use crate::graph::Graph;
use crate::tensor::Tensor;
use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::{NSArray, NSDictionary, NSString};
use std::collections::HashMap;

impl Graph {
    /// Compute gradients of `tensors` with respect to `primary_tensor`.
    pub fn gradient_for_primary_tensor(
        &self,
        primary_tensor: &Tensor,
        tensors: &[&Tensor],
        name: Option<&str>,
    ) -> Option<HashMap<Retained<Tensor>, Retained<Tensor>>> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let tensors_array = NSArray::from_slice(tensors);

            let dict_opt: Option<Retained<NSDictionary<Tensor, Tensor>>> = msg_send![
                self,
                gradientForPrimaryTensor: primary_tensor,
                withTensors: &*tensors_array,
                name: name_ptr
            ];

            dict_opt.map(|dict_retained| {
                let keys_opt: Option<Retained<NSArray<Tensor>>> =
                    msg_send![&*dict_retained, allKeys];
                let mut result_map = HashMap::new();
                if let Some(keys_retained) = keys_opt {
                    let count = keys_retained.len();
                    for i in 0..count {
                        let key: Option<Retained<Tensor>> =
                            msg_send![&*keys_retained, objectAtIndex: i as u64];
                        if let Some(k) = key {
                            let value: Option<Retained<Tensor>> =
                                msg_send![&*dict_retained, objectForKey: &*k];
                            if let Some(v) = value {
                                result_map.insert(k, v);
                            }
                        }
                    }
                }
                result_map
            })
        }
    }
}
