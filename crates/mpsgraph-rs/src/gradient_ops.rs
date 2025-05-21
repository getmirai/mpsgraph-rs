use crate::graph::Graph;
use crate::tensor::Tensor;
use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::{NSArray, NSDictionary, NSString};
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

            let tensors_array = NSArray::from_slice(tensors);

            // Objective-C method returns NSDictionary*
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
                    let keys_count: usize = keys_retained.len();
                    for i in 0..keys_count {
                        let key: Option<Retained<Tensor>> =
                            msg_send![&*keys_retained, objectAtIndex: i];
                        if let Some(k_retained) = key {
                            let value: Option<Retained<Tensor>> =
                                msg_send![&*dict_retained, objectForKey: &*k_retained];
                            if let Some(v_retained) = value {
                                result_map.insert(k_retained, v_retained);
                            }
                        }
                    }
                }
                result_map
            })
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
