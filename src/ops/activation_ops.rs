//! Activation operations exposed as inherent methods on `Graph`.

use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::ScalarOrTensor;
use crate::Tensor;

impl Graph {
    /// Computes the ReLU (rectified linear activation unit) function with the input tensor.
    ///
    /// The operation is:  f(x) = max(x, 0).
    ///
    /// - Parameters:
    ///   - tensor: The input tensor.
    ///   - name: The name for the operation.
    /// - Returns: A valid `Tensor` object.
    pub fn relu(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![self, reLUWithTensor: tensor, name: name.map(NSString::from_str).as_deref()]
        }
    }

    /// Computes the gradient of the ReLU  (rectified linear activation unit) function using the incoming gradient.
    ///
    /// - Parameters:
    ///   - gradient: The incoming gradient tensor.
    ///   - source: The input tensor from forward pass.
    ///   - name: The name for the operation.
    /// - Returns: A valid `Tensor` object.
    pub fn relu_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                reLUGradientWithIncomingGradient: gradient,
                sourceTensor: source,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Computes the sigmoid operation on an input tensor.
    ///
    /// - Parameters:
    ///   - tensor: The input tensor.
    ///   - name: The name for the operation.
    /// - Returns: A valid `Tensor` object.
    pub fn sigmoid(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![self, sigmoidWithTensor: tensor, name: name.map(NSString::from_str).as_deref()]
        }
    }

    /// Computes the gradient of the sigmoid function using the incoming gradient tensor.
    ///
    /// - Parameters:
    ///   - gradient: The incoming gradient tensor.
    ///   - source: The input tensor.
    ///   - name: The name for the operation.
    /// - Returns: A valid `Tensor` object.
    pub fn sigmoid_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                sigmoidGradientWithIncomingGradient: gradient,
                sourceTensor: source,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Computes the softmax function on the input tensor along the specified axis.
    ///
    /// - Parameters:
    ///   - tensor: The input tensor.
    ///   - axis: The axis along which softmax is computed.
    ///   - name: The name for the operation.
    /// - Returns: A valid `Tensor` object.
    pub fn soft_max(&self, tensor: &Tensor, axis: i64, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![self, softMaxWithTensor: tensor, axis: axis, name: name.map(NSString::from_str).as_deref()]
        }
    }

    /// Computes the gradient of the softmax function along the specified axis using the incoming gradient tensor.
    ///
    /// - Parameters:
    ///   - gradient: The incoming gradient tensor.
    ///   - source: The input tensor.
    ///   - axis: The axis along which softmax is computed.
    ///   - name: The name for the operation.
    /// - Returns: A valid `Tensor` object.
    pub fn soft_max_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                softMaxGradientWithIncomingGradient: gradient,
                sourceTensor: source,
                axis: axis,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }

    /// Computes the leaky rectified linear unit (ReLU) activation function on the input tensor.
    ///
    /// The operation is: f(x) = max(x, alpha).
    ///
    /// - Parameters:
    ///   - tensor: An input tensor.
    ///   - alpha: The scalar value alpha used by all elements in the input tensor.
    ///   - name: The name for the operation.
    /// - Returns: A valid `Tensor` object.
    pub fn leaky_relu(
        &self,
        tensor: &Tensor,
        alpha: ScalarOrTensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            match alpha {
                ScalarOrTensor::Scalar(alpha) => {
                    msg_send![self, leakyReLUWithTensor: tensor, alpha: alpha, name: name.map(NSString::from_str).as_deref()]
                }
                ScalarOrTensor::Tensor(alpha_tensor) => {
                    msg_send![self, leakyReLUWithTensor: tensor, alphaTensor: alpha_tensor, name: name.map(NSString::from_str).as_deref()]
                }
            }
        }
    }

    /// Computes the gradient of the leaky rectified linear unit (ReLU) activation.
    ///
    /// This operation supports broadcasting with the alpha tensor.
    ///
    /// - Parameters:
    ///   - gradient: The incoming gradient tensor.
    ///   - source: The input tensor in forward pass.
    ///   - alpha: The alpha tensor
    ///   - name: The name for the operation.
    /// - Returns: A valid `Tensor` object.
    pub fn leaky_relu_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        alpha_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![
                self,
                leakyReLUGradientWithIncomingGradient: gradient,
                sourceTensor: source,
                alphaTensor: alpha_tensor,
                name: name.map(NSString::from_str).as_deref()
            ]
        }
    }
}
