use crate::core::AsRawObject;
use crate::graph::Graph;
use crate::tensor::Tensor;
use objc2::msg_send;
use objc2::runtime::AnyObject;
use objc2_foundation::NSString;

/// Activation operations for Graph
impl Graph {
    /// Creates a ReLU operation
    pub fn relu(&self, x: &Tensor, name: Option<&str>) -> Tensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let tensor: *mut AnyObject = msg_send![
                self.0,
                reLUWithTensor: x.0,
                name: name_obj
            ];

            if !tensor.is_null() {
                Tensor(tensor)
            } else {
                Tensor(std::ptr::null_mut())
            }
        }
    }

    /// Creates a ReLU gradient operation
    pub fn relu_gradient_with_incoming_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        name: Option<&str>,
    ) -> Tensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let tensor: *mut AnyObject = msg_send![
                self.0,
                reLUGradientWithIncomingGradient: gradient.0,
                sourceTensor: source.0,
                name: name_obj
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            Tensor(tensor)
        }
    }

    /// Creates a Sigmoid operation
    pub fn sigmoid(&self, x: &Tensor, name: Option<&str>) -> Tensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            // Call the Sigmoid operation with null name
            let tensor: *mut AnyObject = msg_send![self.0,
                sigmoidWithTensor: x.0,
                name: name_obj
            ];

            if !tensor.is_null() {
                Tensor(tensor)
            } else {
                // Return null tensor if the operation failed
                Tensor(std::ptr::null_mut())
            }
        }
    }

    /// Creates a Sigmoid gradient operation
    pub fn sigmoid_gradient_with_incoming_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        name: Option<&str>,
    ) -> Tensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let tensor: *mut AnyObject = msg_send![self.0, sigmoidGradientWithIncomingGradient: gradient.0,
                sourceTensor: source.0,
                name: name_obj
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            Tensor(tensor)
        }
    }

    /// Creates a Tanh operation
    pub fn tanh(&self, x: &Tensor, name: Option<&str>) -> Tensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let tensor: *mut AnyObject = msg_send![self.0, tanhWithTensor: x.0, name: name_obj];

            if !tensor.is_null() {
                Tensor(tensor)
            } else {
                Tensor(std::ptr::null_mut())
            }
        }
    }

    /// Creates a SoftMax operation
    pub fn softmax(&self, x: &Tensor, axis: i64, name: Option<&str>) -> Tensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            // Call the SoftMax operation with null name
            let tensor: *mut AnyObject = msg_send![self.0, softMaxWithTensor: x.0,
                axis: axis,
                name: name_obj
            ];

            if !tensor.is_null() {
                Tensor(tensor)
            } else {
                Tensor(std::ptr::null_mut())
            }
        }
    }

    /// Creates a SoftMax gradient operation
    pub fn softmax_gradient_with_incoming_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        axis: i64,
        name: Option<&str>,
    ) -> Tensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let tensor: *mut AnyObject = msg_send![self.0, softMaxGradientWithIncomingGradient: gradient.0,
                sourceTensor: source.0,
                axis: axis,
                name: name_obj
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            Tensor(tensor)
        }
    }

    /// Creates a Leaky ReLU operation
    pub fn leaky_relu(&self, x: &Tensor, alpha: f32, name: Option<&str>) -> Tensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let tensor: *mut AnyObject = msg_send![self.0, leakyReLUWithTensor: x.0,
                alpha: alpha as f64,
                name: name_obj
            ];

            if !tensor.is_null() {
                Tensor(tensor)
            } else {
                Tensor(std::ptr::null_mut())
            }
        }
    }

    /// Creates a Leaky ReLU with alpha tensor
    pub fn leaky_relu_with_alpha_tensor(
        &self,
        x: &Tensor,
        alpha_tensor: &Tensor,
        name: Option<&str>,
    ) -> Tensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let tensor: *mut AnyObject = msg_send![self.0, leakyReLUWithTensor: x.0,
                alphaTensor: alpha_tensor.0,
                name: name_obj,
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            Tensor(tensor)
        }
    }

    /// Creates a Leaky ReLU gradient operation
    pub fn leaky_relu_gradient_with_incoming_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        alpha: f32,
        name: Option<&str>,
    ) -> Tensor {
        unsafe {
            let name_obj = match name {
                Some(s) => NSString::from_str(s).as_raw_object(),
                None => std::ptr::null_mut(),
            };

            let tensor: *mut AnyObject = msg_send![
                self.0,
                leakyReLUGradientWithIncomingGradient: gradient.0,
                sourceTensor: source.0,
                alpha: alpha as f64,
                name: name_obj
            ];

            let tensor = objc2::ffi::objc_retain(tensor as *mut _);
            Tensor(tensor)
        }
    }
}
