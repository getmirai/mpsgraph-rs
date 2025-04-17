use crate::graph::Graph;
use crate::operation::Operation;
use crate::tensor::Tensor;
use objc2::msg_send;
use objc2::rc::Retained;
use objc2::{extern_class};
use objc2_foundation::{NSArray, NSObject, NSObjectProtocol, NSString};
use std::ptr;

extern_class!(
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraphVariableOp"]
    /// Represents a variable operation in MPSGraph.
    ///
    /// In MPSGraph, a variable operation is a special operation that creates a tensor
    /// which can be updated during the graph execution. This is particularly useful for
    /// trainable parameters in machine learning models.
    pub struct VariableOp;
);

unsafe impl NSObjectProtocol for VariableOp {}

impl VariableOp {
    /// Returns the operation associated with this variable.
    pub fn operation(&self) -> Retained<Operation> {
        unsafe {
            msg_send![self, operation]
        }
    }

    /// Returns the tensor associated with this variable.
    pub fn tensor(&self) -> Retained<Tensor> {
        unsafe {
            msg_send![self, tensor]
        }
    }
}

impl crate::device::CustomDefault for VariableOp {
    fn custom_default() -> Retained<Self> {
        // This is a placeholder. In practice, variables should be
        // created with Graph::variable_op_for_tensor
        panic!("VariableOp cannot be created with custom_default. Use Graph::variable_op_for_tensor instead");
    }
}

/// Optimizer operations for Graph
impl Graph {
    /// Stochastic gradient descent optimization.
    ///
    /// `variable = variable - (learningRate * gradient)`
    ///
    /// # Parameters
    ///
    /// * `learning_rate` - Scalar tensor which indicates the learning rate to use
    /// * `values` - Values tensor, usually representing the trainable parameters
    /// * `gradient` - Partial gradient of the trainable parameters with respect to loss
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A tensor containing the updated values
    pub fn stochastic_gradient_descent(
        &self,
        learning_rate: &Tensor,
        values: &Tensor,
        gradient: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };

            msg_send![
                self, 
                stochasticGradientDescentWithLearningRateTensor: learning_rate,
                valuesTensor: values,
                gradientTensor: gradient,
                name: name_obj
            ]
        }
    }

    /// Adam optimization.
    ///
    /// The Adam optimizer combines ideas from momentum and RMSProp, maintaining per-parameter
    /// momentum and velocity (squared gradients) to adaptively adjust learning rates.
    ///
    /// ```text
    /// m[t] = beta1 * m[t-1] + (1 - beta1) * g
    /// v[t] = beta2 * v[t-1] + (1 - beta2) * (g ^ 2)
    /// maxVel[t] = max(maxVel[t-1], v[t])
    /// variable = variable - (learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)) * m[t] / (sqrt(maxVel) + epsilon)
    /// ```
    ///
    /// # Parameters
    ///
    /// * `learning_rate` - Scalar tensor with the learning rate
    /// * `beta1` - Exponential decay rate for first moment estimates
    /// * `beta2` - Exponential decay rate for second moment estimates
    /// * `epsilon` - Small constant for numerical stability
    /// * `beta1_power` - Current beta1^t value
    /// * `beta2_power` - Current beta2^t value
    /// * `values` - Current parameter values to be updated
    /// * `momentum` - First moment estimates (momentum)
    /// * `velocity` - Second moment estimates (velocity)
    /// * `maximum_velocity` - Optional maximum velocity tensor
    /// * `gradient` - Gradient of parameters with respect to loss
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A vector containing tensors for:
    /// - Updated values
    /// - New momentum
    /// - New velocity
    /// - New maximum velocity (if maximum_velocity is provided)
    pub fn adam(
        &self,
        learning_rate: &Tensor,
        beta1: &Tensor,
        beta2: &Tensor,
        epsilon: &Tensor,
        beta1_power: &Tensor,
        beta2_power: &Tensor,
        values: &Tensor,
        momentum: &Tensor,
        velocity: &Tensor,
        maximum_velocity: Option<&Tensor>,
        gradient: &Tensor,
        name: Option<&str>,
    ) -> Vec<Retained<Tensor>> {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };

            let max_velocity_obj = match maximum_velocity {
                Some(m) => m as *const _,
                None => std::ptr::null(),
            };

            let result_array: Retained<NSArray<Tensor>> = msg_send![
                self, 
                adamWithLearningRateTensor: learning_rate,
                beta1Tensor: beta1,
                beta2Tensor: beta2,
                epsilonTensor: epsilon,
                beta1PowerTensor: beta1_power,
                beta2PowerTensor: beta2_power,
                valuesTensor: values,
                momentumTensor: momentum,
                velocityTensor: velocity,
                maximumVelocityTensor: max_velocity_obj,
                gradientTensor: gradient,
                name: name_obj
            ];

            // Get the count of result tensors
            let count = result_array.count();

            // Convert NSArray to Vec<Tensor>
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let tensor_ptr: *mut Tensor = msg_send![&*result_array, objectAtIndex: i];
                let tensor = Retained::retain(tensor_ptr).unwrap();
                result.push(tensor);
            }

            result
        }
    }

    /// Adam optimization with current learning rate.
    ///
    /// This is a variant of Adam where the learning rate adaptation is already applied
    /// to the learning rate tensor.
    ///
    /// ```text
    /// m[t] = beta1 * m[t-1] + (1 - beta1) * g
    /// v[t] = beta2 * v[t-1] + (1 - beta2) * (g ^ 2)
    /// maxVel[t] = max(maxVel[t-1], v[t])
    /// variable = variable - current_learning_rate * m[t] / (sqrt(maxVel) + epsilon)
    /// ```
    ///
    /// # Parameters
    ///
    /// * `current_learning_rate` - Scalar tensor with the already adjusted learning rate
    /// * `beta1` - Exponential decay rate for first moment estimates
    /// * `beta2` - Exponential decay rate for second moment estimates
    /// * `epsilon` - Small constant for numerical stability
    /// * `values` - Current parameter values to be updated
    /// * `momentum` - First moment estimates (momentum)
    /// * `velocity` - Second moment estimates (velocity)
    /// * `maximum_velocity` - Optional maximum velocity tensor
    /// * `gradient` - Gradient of parameters with respect to loss
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A vector containing tensors for:
    /// - Updated values
    /// - New momentum
    /// - New velocity
    /// - New maximum velocity (if maximum_velocity is provided)
    pub fn adam_with_current_learning_rate(
        &self,
        current_learning_rate: &Tensor,
        beta1: &Tensor,
        beta2: &Tensor,
        epsilon: &Tensor,
        values: &Tensor,
        momentum: &Tensor,
        velocity: &Tensor,
        maximum_velocity: Option<&Tensor>,
        gradient: &Tensor,
        name: Option<&str>,
    ) -> Vec<Retained<Tensor>> {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };

            let max_velocity_obj = match maximum_velocity {
                Some(m) => m as *const _,
                None => std::ptr::null(),
            };

            let result_array: Retained<NSArray<Tensor>> = msg_send![
                self, 
                adamWithCurrentLearningRateTensor: current_learning_rate,
                beta1Tensor: beta1,
                beta2Tensor: beta2,
                epsilonTensor: epsilon,
                valuesTensor: values,
                momentumTensor: momentum,
                velocityTensor: velocity,
                maximumVelocityTensor: max_velocity_obj,
                gradientTensor: gradient,
                name: name_obj
            ];

            // Get the count of result tensors
            let count = result_array.count();

            // Convert NSArray to Vec<Tensor>
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let tensor_ptr: *mut Tensor = msg_send![&*result_array, objectAtIndex: i];
                let tensor = Retained::retain(tensor_ptr).unwrap();
                result.push(tensor);
            }

            result
        }
    }

    /// Creates a variable operation for a tensor.
    ///
    /// This method creates a variable operation from a tensor, which can then be used
    /// with optimization operations that require direct manipulation of the variable.
    ///
    /// # Parameters
    ///
    /// * `tensor` - The tensor to convert to a variable
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A new VariableOp wrapper
    pub fn variable_op_for_tensor(
        &self,
        tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<VariableOp> {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => ptr::null(),
            };

            msg_send![
                self, 
                variableOpWithTensor: tensor,
                name: name_obj
            ]
        }
    }

    /// Applies stochastic gradient descent directly to a variable operation.
    ///
    /// This method updates the variable operation in-place using stochastic gradient descent.
    ///
    /// # Parameters
    ///
    /// * `learning_rate` - Scalar tensor which indicates the learning rate to use
    /// * `variable_op` - The variable operation to update
    /// * `gradient` - Partial gradient of the trainable parameters with respect to loss
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A tensor representing the updated value of the variable
    pub fn apply_stochastic_gradient_descent(
        &self,
        learning_rate: &Tensor,
        variable_op: &VariableOp,
        gradient: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => ptr::null(),
            };

            msg_send![
                self, 
                applyStochasticGradientDescentWithLearningRateTensor: learning_rate,
                variableOp: variable_op,
                gradientTensor: gradient,
                name: name_obj
            ]
        }
    }
}