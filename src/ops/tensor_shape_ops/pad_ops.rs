use crate::{ns_number_array_from_slice, Graph, PaddingMode, Shape, Tensor};
use objc2::{msg_send, rc::Retained};
use objc2_foundation::NSString;

impl Graph {
    /// Creates a padding operation and returns the result tensor.
    ///
    /// - Parameters:
    /// - tensor: The input tensor.
    /// - paddingMode: The parameter that defines the padding mode.
    /// - leftPadding: The parameter that defines how much padding the operation applies to the input tensor before each dimension - must be of size `rank(tensor)`.
    /// - rightPadding: The parameter that defines how much padding the operation applies to the input tensor after each dimension - must be of size `rank(tensor)`.
    /// - constantValue: The constant value the operation uses when `paddingMode = MPSGraphPaddingModeConstant`.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object.
    pub fn pad_tensor(
        &self,
        tensor: &Tensor,
        padding_mode: PaddingMode,
        left_padding: &[i64],
        right_padding: &[i64],
        constant_value: f64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                padTensor: tensor,
                withPaddingMode: padding_mode,
                leftPadding: &*ns_number_array_from_slice(left_padding),
                rightPadding: &*ns_number_array_from_slice(right_padding),
                constantValue: constant_value,
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }

    /// Creates a padding gradient operation and returns the result tensor.
    ///
    /// - Parameters:
    /// - incomingGradientTensor: The input gradient tensor.
    /// - sourceTensor: The input tensor of the forward pass.
    /// - paddingMode: The parameter that defines the padding mode.
    /// - leftPadding: The parameter that defines how much padding the operation applies to the input tensor before each dimension - must be of size `rank(tensor)`.
    /// - rightPadding: The parameter that defines how much padding the operation applies to the input tensor after each dimension - must be of size `rank(tensor)`.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object.
    pub fn pad_gradient(
        &self,
        incoming_gradient_tensor: &Tensor,
        source_tensor: &Tensor,
        padding_mode: PaddingMode,
        left_padding: &[i64],
        right_padding: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                padGradientWithIncomingGradientTensor: incoming_gradient_tensor,
                sourceTensor: source_tensor,
                paddingMode: padding_mode,
                leftPadding: &*ns_number_array_from_slice(left_padding),
                rightPadding: &*ns_number_array_from_slice(right_padding),
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }
}
