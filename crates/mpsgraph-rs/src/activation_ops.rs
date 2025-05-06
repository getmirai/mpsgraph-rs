use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::tensor::Tensor;

/// Trait for activation operations on Graph
pub trait GraphActivationOps {
    /// Creates a ReLU operation
    fn relu(&self, x: &Retained<Tensor>, name: Option<&str>) -> Retained<Tensor>;

    /// Creates a ReLU gradient operation
    fn relu_gradient(
        &self,
        gradient: &Retained<Tensor>,
        source: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a Sigmoid operation
    fn sigmoid(&self, x: &Retained<Tensor>, name: Option<&str>) -> Retained<Tensor>;

    /// Creates a Sigmoid gradient operation
    fn sigmoid_gradient(
        &self,
        gradient: &Retained<Tensor>,
        source: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a Tanh operation
    fn tanh(&self, x: &Retained<Tensor>, name: Option<&str>) -> Retained<Tensor>;

    /// Creates a SoftMax operation
    fn softmax(&self, x: &Retained<Tensor>, axis: i64, name: Option<&str>) -> Retained<Tensor>;

    /// Creates a SoftMax gradient operation
    fn softmax_gradient(
        &self,
        gradient: &Retained<Tensor>,
        source: &Retained<Tensor>,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a Leaky ReLU operation
    fn leaky_relu(&self, x: &Retained<Tensor>, alpha: f32, name: Option<&str>) -> Retained<Tensor>;

    /// Creates a Leaky ReLU with alpha tensor
    fn leaky_relu_with_alpha_tensor(
        &self,
        x: &Retained<Tensor>,
        alpha_tensor: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a Leaky ReLU gradient operation
    fn leaky_relu_gradient(
        &self,
        gradient: &Retained<Tensor>,
        source: &Retained<Tensor>,
        alpha: f32,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates an ELU (Exponential Linear Unit) operation
    fn elu(&self, x: &Retained<Tensor>, alpha: f32, name: Option<&str>) -> Retained<Tensor>;

    /// Creates a GELU (Gaussian Error Linear Unit) operation
    fn gelu(&self, x: &Retained<Tensor>, name: Option<&str>) -> Retained<Tensor>;
}

impl GraphActivationOps for Graph {
    fn relu(&self, x: &Retained<Tensor>, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                reLUWithTensor: &**x,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create ReLU tensor");
            } else {
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    fn relu_gradient(
        &self,
        gradient: &Retained<Tensor>,
        source: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                reLUGradientWithIncomingGradient: &**gradient,
                sourceTensor: &**source,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create ReLU gradient tensor");
            } else {
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    fn sigmoid(&self, x: &Retained<Tensor>, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                sigmoidWithTensor: &**x,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create Sigmoid tensor");
            } else {
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    fn sigmoid_gradient(
        &self,
        gradient: &Retained<Tensor>,
        source: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                sigmoidGradientWithIncomingGradient: &**gradient,
                sourceTensor: &**source,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create Sigmoid gradient tensor");
            } else {
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    fn tanh(&self, x: &Retained<Tensor>, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                tanhWithTensor: &**x,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create Tanh tensor");
            } else {
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    fn softmax(&self, x: &Retained<Tensor>, axis: i64, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                softMaxWithTensor: &**x,
                axis: axis,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create SoftMax tensor");
            } else {
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    fn softmax_gradient(
        &self,
        gradient: &Retained<Tensor>,
        source: &Retained<Tensor>,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                softMaxGradientWithIncomingGradient: &**gradient,
                sourceTensor: &**source,
                axis: axis,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create SoftMax gradient tensor");
            } else {
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    fn leaky_relu(&self, x: &Retained<Tensor>, alpha: f32, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                leakyReLUWithTensor: &**x,
                alpha: alpha,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create Leaky ReLU tensor");
            } else {
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    fn leaky_relu_with_alpha_tensor(
        &self,
        x: &Retained<Tensor>,
        alpha_tensor: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                leakyReLUWithTensor: &**x,
                alphaTensor: &**alpha_tensor,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create Leaky ReLU with alpha tensor");
            } else {
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    fn leaky_relu_gradient(
        &self,
        gradient: &Retained<Tensor>,
        source: &Retained<Tensor>,
        alpha: f32,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                leakyReLUGradientWithIncomingGradient: &**gradient,
                sourceTensor: &**source,
                alpha: alpha,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create Leaky ReLU gradient tensor");
            } else {
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    fn elu(&self, x: &Retained<Tensor>, alpha: f32, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                eluWithTensor: &**x,
                alpha: alpha,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create ELU tensor");
            } else {
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    fn gelu(&self, x: &Retained<Tensor>, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                geluWithTensor: &**x,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create GELU tensor");
            } else {
                Retained::retain_autoreleased(result).unwrap()
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
