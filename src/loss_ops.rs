//! Loss helpers (softmax cross-entropy) are now inherent on `Graph`.

use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::tensor::Tensor;

/// Reduction modes for loss functions.
#[repr(u64)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum LossReductionType {
    /// Computes the loss without reduction
    None = 0,
    /// Reduces the loss down to a scalar with a sum operation
    Sum = 1,
    /// Reduces the loss down to a scalar with a mean operation
    Mean = 2,
}

// Alias for backward compatibility
#[allow(non_upper_case_globals)]
pub const Axis: LossReductionType = LossReductionType::None;

impl Graph {
    /// Creates a softmax cross-entropy loss operation and returns the result tensor.
    ///
    /// The softmax cross-entropy operation computes:
    /// ```text
    /// where softmax(source) = exp(source) / sum(exp(source))
    /// loss = reduction(-labels * ln(softmax(source)))
    /// ```
    ///
    /// # Parameters
    ///
    /// * `source_tensor` - The source tensor (logits)
    /// * `labels_tensor` - The labels tensor (ground truth)
    /// * `axis` - The axis over which the operation computes softmax
    /// * `reduction_type` - The type of reduction to apply
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A tensor containing the computed loss
    pub fn soft_max_cross_entropy(
        &self,
        source_tensor: &Tensor,
        labels_tensor: &Tensor,
        axis: i64,
        reduction_type: LossReductionType,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self,
                softMaxCrossEntropyWithSourceTensor: source_tensor,
                labelsTensor: labels_tensor,
                axis: axis,
                reductionType: reduction_type as u64,
                name: name_ptr]
        }
    }

    /// Creates the gradient of a softmax cross-entropy loss operation and returns the result tensor.
    ///
    /// # Parameters
    ///
    /// * `gradient_tensor` - The incoming gradient tensor (typically a constant tensor with value 1)
    /// * `source_tensor` - The original source tensor (logits)
    /// * `labels_tensor` - The original labels tensor (ground truth)
    /// * `axis` - The axis over which the operation computes softmax
    /// * `reduction_type` - The type of reduction that was applied
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A tensor containing the gradient with respect to the source tensor
    pub fn soft_max_cross_entropy_gradient(
        &self,
        gradient_tensor: &Tensor,
        source_tensor: &Tensor,
        labels_tensor: &Tensor,
        axis: i64,
        reduction_type: LossReductionType,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self,
                softMaxCrossEntropyGradientWithIncomingGradientTensor: gradient_tensor,
                sourceTensor: source_tensor,
                labelsTensor: labels_tensor,
                axis: axis,
                reductionType: reduction_type as u64,
                name: name_ptr]
        }
    }
}
