use objc2::rc::Retained;
use objc2::msg_send;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::tensor::Tensor;

/// Trait for activation operations on Graph
pub trait GraphActivationOps {
    /// Creates a ReLU operation
    fn relu(&self, x: &Tensor, name: Option<&str>) -> Option<Retained<Tensor>>;
    
    /// Creates a ReLU gradient operation
    fn relu_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;
    
    /// Creates a Sigmoid operation
    fn sigmoid(&self, x: &Tensor, name: Option<&str>) -> Option<Retained<Tensor>>;
    
    /// Creates a Sigmoid gradient operation
    fn sigmoid_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;
    
    /// Creates a Tanh operation
    fn tanh(&self, x: &Tensor, name: Option<&str>) -> Option<Retained<Tensor>>;
    
    /// Creates a SoftMax operation
    fn softmax(&self, x: &Tensor, axis: i64, name: Option<&str>) -> Option<Retained<Tensor>>;
    
    /// Creates a SoftMax gradient operation
    fn softmax_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;
    
    /// Creates a Leaky ReLU operation
    fn leaky_relu(&self, x: &Tensor, alpha: f32, name: Option<&str>) -> Option<Retained<Tensor>>;
    
    /// Creates a Leaky ReLU with alpha tensor
    fn leaky_relu_with_alpha_tensor(
        &self,
        x: &Tensor,
        alpha_tensor: &Tensor,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;
    
    /// Creates a Leaky ReLU gradient operation
    fn leaky_relu_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        alpha: f32,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates an ELU (Exponential Linear Unit) operation
    fn elu(&self, x: &Tensor, alpha: f32, name: Option<&str>) -> Option<Retained<Tensor>>;

    /// Creates a GELU (Gaussian Error Linear Unit) operation
    fn gelu(&self, x: &Tensor, name: Option<&str>) -> Option<Retained<Tensor>>;
}

impl GraphActivationOps for Graph {
    fn relu(&self, x: &Tensor, name: Option<&str>) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                reLUWithTensor: x,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }
    
    fn relu_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                reLUGradientWithIncomingGradient: gradient,
                sourceTensor: source,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }
    
    fn sigmoid(&self, x: &Tensor, name: Option<&str>) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                sigmoidWithTensor: x,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }
    
    fn sigmoid_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                sigmoidGradientWithIncomingGradient: gradient,
                sourceTensor: source,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }
    
    fn tanh(&self, x: &Tensor, name: Option<&str>) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                tanhWithTensor: x,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }
    
    fn softmax(&self, x: &Tensor, axis: i64, name: Option<&str>) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                softMaxWithTensor: x,
                axis: axis,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }
    
    fn softmax_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                softMaxGradientWithIncomingGradient: gradient,
                sourceTensor: source,
                axis: axis,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }
    
    fn leaky_relu(&self, x: &Tensor, alpha: f32, name: Option<&str>) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                leakyReLUWithTensor: x,
                alpha: alpha as f64,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }
    
    fn leaky_relu_with_alpha_tensor(
        &self,
        x: &Tensor,
        alpha_tensor: &Tensor,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                leakyReLUWithTensor: x,
                alphaTensor: alpha_tensor,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }
    
    fn leaky_relu_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        alpha: f32,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                leakyReLUGradientWithIncomingGradient: gradient,
                sourceTensor: source,
                alpha: alpha as f64,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn elu(&self, x: &Tensor, alpha: f32, name: Option<&str>) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                eluWithTensor: x,
                alpha: alpha as f64,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn gelu(&self, x: &Tensor, name: Option<&str>) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                geluWithTensor: x,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }
}

/// Extension trait providing a method for Graph to access activation operations
pub trait GraphActivationOpsExtension {
    /// Access activation operations for this graph
    fn activation_ops(&self) -> &dyn GraphActivationOps;
}

impl GraphActivationOpsExtension for Graph {
    fn activation_ops(&self) -> &dyn GraphActivationOps {
        self
    }
}