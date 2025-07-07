//! Activation operations exposed as inherent methods on `Graph`.

use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::tensor::Tensor;

impl Graph {
    /// Computes the ReLU (rectified linear activation unit) function with the input tensor.
    ///
    /// The operation is:  f(x) = max(x, 0).
    ///
    /// - Parameters:
    ///   - tensor: The input tensor.
    ///   - name: The name for the operation.
    /// - Returns: A valid ``MPSGraphTensor`` object.
    pub fn relu_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, reLUWithTensor: tensor, name: name_ptr]
        }
    }

    /// Computes the gradient of the ReLU  (rectified linear activation unit) function using the incoming gradient.
    ///
    /// - Parameters:
    ///   - gradient: The incoming gradient tensor.
    ///   - source: The input tensor from forward pass.
    ///   - name: The name for the operation.
    /// - Returns: A valid ``MPSGraphTensor`` object.
    pub fn relu_gradient_with_incoming_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                reLUGradientWithIncomingGradient: gradient,
                sourceTensor: source,
                name: name_ptr
            ]
        }
    }

    /// Computes the sigmoid operation on an input tensor.
    ///
    /// - Parameters:
    ///   - tensor: The input tensor.
    ///   - name: The name for the operation.
    /// - Returns: A valid ``MPSGraphTensor`` object.
    pub fn sigmoid_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, sigmoidWithTensor: tensor, name: name_ptr]
        }
    }

    /// Computes the gradient of the sigmoid function using the incoming gradient tensor.
    ///
    /// - Parameters:
    ///   - gradient: The incoming gradient tensor.
    ///   - source: The input tensor.
    ///   - name: The name for the operation.
    /// - Returns: A valid ``MPSGraphTensor`` object
    pub fn sigmoid_gradient_with_incoming_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                sigmoidGradientWithIncomingGradient: gradient,
                sourceTensor: source,
                name: name_ptr
            ]
        }
    }

    /// Computes the softmax function on the input tensor along the specified axis.
    ///
    /// - Parameters:
    ///   - tensor: The input tensor.
    ///   - axis: The axis along which softmax is computed.
    ///   - name: The name for the operation.
    /// - Returns: A valid ``MPSGraphTensor`` object
    pub fn soft_max_with_tensor(
        &self,
        tensor: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, softMaxWithTensor: tensor, axis: axis, name: name_ptr]
        }
    }

    /// Computes the gradient of the softmax function along the specified axis using the incoming gradient tensor.
    ///
    /// - Parameters:
    ///   - gradient: The incoming gradient tensor.
    ///   - source: The input tensor.
    ///   - axis: The axis along which softmax is computed.
    ///   - name: The name for the operation.
    /// - Returns: A valid ``MPSGraphTensor`` object
    pub fn soft_max_gradient_with_incoming_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                softMaxGradientWithIncomingGradient: gradient,
                sourceTensor: source,
                axis: axis,
                name: name_ptr
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
    /// - Returns: A valid ``MPSGraphTensor`` object
    pub fn leaky_relu_with_tensor_alpha(
        &self,
        tensor: &Tensor,
        alpha: f64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, leakyReLUWithTensor: tensor, alpha: alpha, name: name_ptr]
        }
    }

    /// Computes the leaky rectified linear unit (ReLU) activation function on the input tensor.
    ///
    /// The operation is: f(x) = max(x, alpha).
    /// This operation supports broadcasting with the alpha tensor.
    ///
    /// - Parameters:
    ///   - tensor: The input tensor.
    ///   - alpha: The alpha tensor.
    ///   - name: The name for the operation.
    /// - Returns: A valid ``MPSGraphTensor`` object
    pub fn leaky_relu_with_tensor_alpha_tensor(
        &self,
        tensor: &Tensor,
        alpha_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                leakyReLUWithTensor: tensor,
                alphaTensor: alpha_tensor,
                name: name_ptr
            ]
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
    /// - Returns: A valid ``MPSGraphTensor`` object
    pub fn leaky_relu_gradient_with_incoming_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        alpha_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                leakyReLUGradientWithIncomingGradient: gradient,
                sourceTensor: source,
                alphaTensor: alpha_tensor,
                name: name_ptr
            ]
        }
    }
}
