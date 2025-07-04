use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::tensor::Tensor;

impl Graph {
    /// Computes the inverse of a square matrix tensor. All dimensions after the
    /// first two are treated as batch dimensions; an inverse is calculated per
    /// batch.
    ///
    /// This is a thin wrapper around `-[MPSGraph inverseOfTensor:name:]`.
    pub fn inverse(&self, input: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let tensor: Retained<Tensor> = msg_send![self,
                inverseOfTensor: input,
                name: name_ptr
            ];
            tensor
        }
    }
}
