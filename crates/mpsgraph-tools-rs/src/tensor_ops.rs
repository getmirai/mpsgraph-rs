//! Tensor Operations API for Graph
//!
//! This module provides ergonomic tensor operations with method-based and functional-style
//! programming for Graph tensors.
//!
//! # Features
//!
//! - **Method-Based API**: Use direct methods on Retained<Tensor> (e.g., `a.add(&b, None)`)
//! - **Functional API**: Apply operations using functional style (e.g., `abs(&sqrt(&a, None), None)`)
//! - **Scalar Operations**: Easily work with scalar values (e.g., `a.add_scalar(3.0, None)`)
//! - **Utility Methods**: Convenience functions for common operations
//! - **Tensor Creation**: Helper methods for creating tensors filled with zeros, ones, etc.

use mpsgraph::tensor::Tensor as MPSTensor;
use mpsgraph::{DataType, Graph, GraphActivationOps, GraphArithmeticOps, Operation, Shape};

/// Helper function to extract the graph from a tensor
fn get_graph_from_tensor(tensor: &Retained<MPSTensor>) -> Retained<Graph> {
    unsafe {
        // Get the operation that created this tensor
        let tensor_ptr = &**tensor;
        let operation_ptr: *mut Operation = msg_send![tensor_ptr, operation];
        let operation = Retained::retain_autoreleased(operation_ptr).unwrap();

        // Get the graph from the operation
        let graph_ptr: *mut Graph = msg_send![&*operation, graph];
        Retained::retain_autoreleased(graph_ptr).unwrap()
    }
}

/// Helper function to get the data type from a tensor
fn get_data_type_from_tensor(tensor: &Retained<MPSTensor>) -> DataType {
    unsafe {
        let tensor_ptr = &**tensor;
        let data_type: u32 = msg_send![tensor_ptr, dataType];
        std::mem::transmute(data_type)
    }
}
use objc2::msg_send;
use objc2::rc::Retained;

/// Extension trait for tensor arithmetic operations
///
/// This trait adds arithmetic operations to the Tensor type.
pub trait TensorOps {
    /// Add this tensor to another tensor
    fn add(&self, rhs: &Retained<MPSTensor>, name: Option<&str>) -> Retained<MPSTensor>;

    /// Subtract another tensor from this tensor
    fn sub(&self, rhs: &Retained<MPSTensor>, name: Option<&str>) -> Retained<MPSTensor>;

    /// Multiply this tensor by another tensor
    fn mul(&self, rhs: &Retained<MPSTensor>, name: Option<&str>) -> Retained<MPSTensor>;

    /// Divide this tensor by another tensor
    fn div(&self, rhs: &Retained<MPSTensor>, name: Option<&str>) -> Retained<MPSTensor>;

    /// Negate this tensor
    fn neg(&self, name: Option<&str>) -> Retained<MPSTensor>;

    /// Square each element of this tensor
    fn square(&self, name: Option<&str>) -> Retained<MPSTensor>;

    /// Take the square root of each element of this tensor
    fn sqrt(&self, name: Option<&str>) -> Retained<MPSTensor>;

    /// Take the absolute value of each element of this tensor
    fn abs(&self, name: Option<&str>) -> Retained<MPSTensor>;

    /// Take the exponential of each element of this tensor
    fn exp(&self, name: Option<&str>) -> Retained<MPSTensor>;

    /// Take the natural logarithm of each element of this tensor
    fn log(&self, name: Option<&str>) -> Retained<MPSTensor>;

    /// Apply the sigmoid function to each element of this tensor
    fn sigmoid(&self, name: Option<&str>) -> Retained<MPSTensor>;

    /// Apply the tanh function to each element of this tensor
    fn tanh(&self, name: Option<&str>) -> Retained<MPSTensor>;

    /// Apply the ReLU function to each element of this tensor
    fn relu(&self, name: Option<&str>) -> Retained<MPSTensor>;
}

/// Implementation of TensorOps for Retained<MPSTensor>
impl TensorOps for Retained<MPSTensor> {
    fn add(&self, rhs: &Retained<MPSTensor>, name: Option<&str>) -> Retained<MPSTensor> {
        let graph = get_graph_from_tensor(self);
        graph.add(self, rhs, name)
    }

    fn sub(&self, rhs: &Retained<MPSTensor>, name: Option<&str>) -> Retained<MPSTensor> {
        let graph = get_graph_from_tensor(self);
        graph.subtract(self, rhs, name)
    }

    fn mul(&self, rhs: &Retained<MPSTensor>, name: Option<&str>) -> Retained<MPSTensor> {
        let graph = get_graph_from_tensor(self);
        graph.multiply(self, rhs, name)
    }

    fn div(&self, rhs: &Retained<MPSTensor>, name: Option<&str>) -> Retained<MPSTensor> {
        let graph = get_graph_from_tensor(self);
        graph.divide(self, rhs, name)
    }

    fn neg(&self, name: Option<&str>) -> Retained<MPSTensor> {
        let graph = get_graph_from_tensor(self);
        graph.negative(self, name)
    }

    fn square(&self, name: Option<&str>) -> Retained<MPSTensor> {
        let graph = get_graph_from_tensor(self);
        graph.square(self, name)
    }

    fn sqrt(&self, name: Option<&str>) -> Retained<MPSTensor> {
        let graph = get_graph_from_tensor(self);
        graph.sqrt(self, name)
    }

    fn abs(&self, name: Option<&str>) -> Retained<MPSTensor> {
        let graph = get_graph_from_tensor(self);
        graph.abs(self, name)
    }

    fn exp(&self, name: Option<&str>) -> Retained<MPSTensor> {
        let graph = get_graph_from_tensor(self);
        graph.exp(self, name)
    }

    fn log(&self, name: Option<&str>) -> Retained<MPSTensor> {
        let graph = get_graph_from_tensor(self);
        graph.log(self, name)
    }

    fn sigmoid(&self, name: Option<&str>) -> Retained<MPSTensor> {
        let graph = get_graph_from_tensor(self);
        graph.sigmoid(self, name)
    }

    fn tanh(&self, name: Option<&str>) -> Retained<MPSTensor> {
        let graph = get_graph_from_tensor(self);
        GraphActivationOps::tanh(&*graph, self, name)
    }

    fn relu(&self, name: Option<&str>) -> Retained<MPSTensor> {
        let graph = get_graph_from_tensor(self);
        graph.relu(self, name)
    }
}

/// Extensions for Graph to create tensors with common values
pub trait GraphExtensions {
    /// Create a placeholder tensor
    fn placeholder_tensor(
        &self,
        shape: &Shape,
        data_type: DataType,
        name: Option<&str>,
    ) -> Retained<MPSTensor>;

    /// Create a constant scalar tensor (without shape)
    fn constant_scalar_tensor<T: Into<f64> + Copy>(
        &self,
        value: T,
        data_type: DataType,
        name: Option<&str>,
    ) -> Retained<MPSTensor>;

    /// Create a constant scalar tensor with a specific shape
    fn constant_scalar_shaped_tensor<T: Into<f64> + Copy>(
        &self,
        value: T,
        shape: &Shape,
        data_type: DataType,
        name: Option<&str>,
    ) -> Retained<MPSTensor>;

    /// Create a tensor filled with zeros
    fn zeros(&self, shape: &Shape, data_type: DataType, name: Option<&str>) -> Retained<MPSTensor>;

    /// Create a tensor filled with ones
    fn ones(&self, shape: &Shape, data_type: DataType, name: Option<&str>) -> Retained<MPSTensor>;

    /// Create a tensor filled with a specific value
    fn fill<T: Into<f64> + Copy>(
        &self,
        value: T,
        shape: &Shape,
        data_type: DataType,
        name: Option<&str>,
    ) -> Retained<MPSTensor>;
}

impl GraphExtensions for Graph {
    fn placeholder_tensor(
        &self,
        shape: &Shape,
        data_type: DataType,
        name: Option<&str>,
    ) -> Retained<MPSTensor> {
        self.placeholder(data_type, shape, name)
    }

    /// Create a constant scalar tensor (without shape)
    fn constant_scalar_tensor<T: Into<f64> + Copy>(
        &self,
        value: T,
        data_type: DataType,
        _name: Option<&str>,
    ) -> Retained<MPSTensor> {
        // Use unsafe direct message sending to create the scalar constant
        unsafe {
            let value_f64 = value.into();

            // Use the correct method without the name parameter
            // The constantWithScalar:dataType: method expects a 32-bit value for the data type
            let tensor_ptr: *mut MPSTensor = msg_send![
                self,
                constantWithScalar: value_f64,
                dataType: data_type as u32
            ];

            Retained::retain_autoreleased(tensor_ptr)
                .expect("Failed to create constant scalar tensor")
        }
    }

    /// Create a constant scalar tensor with a specific shape
    fn constant_scalar_shaped_tensor<T: Into<f64> + Copy>(
        &self,
        value: T,
        shape: &Shape,
        data_type: DataType,
        _name: Option<&str>,
    ) -> Retained<MPSTensor> {
        // Use unsafe direct message sending to create the scalar constant with shape
        unsafe {
            let value_f64 = value.into();

            // Use the constantWithScalar:shape:dataType: method
            let tensor_ptr: *mut MPSTensor = msg_send![
                self,
                constantWithScalar: value_f64,
                shape: shape.as_ptr(),
                dataType: data_type as u32
            ];

            Retained::retain_autoreleased(tensor_ptr)
                .expect("Failed to create constant scalar tensor with shape")
        }
    }

    /// Create a tensor filled with zeros
    fn zeros(&self, shape: &Shape, data_type: DataType, name: Option<&str>) -> Retained<MPSTensor> {
        self.constant_scalar_shaped_tensor(0.0, shape, data_type, name)
    }

    /// Create a tensor filled with ones
    fn ones(&self, shape: &Shape, data_type: DataType, name: Option<&str>) -> Retained<MPSTensor> {
        self.constant_scalar_shaped_tensor(1.0, shape, data_type, name)
    }

    /// Create a tensor filled with a specific value
    fn fill<T: Into<f64> + Copy>(
        &self,
        value: T,
        shape: &Shape,
        data_type: DataType,
        name: Option<&str>,
    ) -> Retained<MPSTensor> {
        self.constant_scalar_shaped_tensor(value, shape, data_type, name)
    }
}

/// Applies square operation to the tensor elements
///
/// Computes the square of each element in the tensor: f(x) = x²
///
/// # Parameters
///
/// * `tensor` - The input tensor
/// * `name` - Optional name for the operation
///
/// # Returns
///
/// A new tensor with each element squared
pub fn square(tensor: &Retained<MPSTensor>, name: Option<&str>) -> Retained<MPSTensor> {
    tensor.square(name)
}

/// Applies square root operation to the tensor elements
///
/// Computes the square root of each element in the tensor: f(x) = √x
///
/// # Parameters
///
/// * `tensor` - The input tensor
/// * `name` - Optional name for the operation
///
/// # Returns
///
/// A new tensor with the square root of each element
pub fn sqrt(tensor: &Retained<MPSTensor>, name: Option<&str>) -> Retained<MPSTensor> {
    tensor.sqrt(name)
}

/// Applies absolute value operation to the tensor elements
///
/// Computes the absolute value of each element in the tensor: f(x) = |x|
///
/// # Parameters
///
/// * `tensor` - The input tensor
/// * `name` - Optional name for the operation
///
/// # Returns
///
/// A new tensor with the absolute value of each element
pub fn abs(tensor: &Retained<MPSTensor>, name: Option<&str>) -> Retained<MPSTensor> {
    tensor.abs(name)
}

/// Applies exponential function to the tensor elements
///
/// Computes e^x for each element in the tensor
///
/// # Parameters
///
/// * `tensor` - The input tensor
/// * `name` - Optional name for the operation
///
/// # Returns
///
/// A new tensor with the exponential of each element
pub fn exp(tensor: &Retained<MPSTensor>, name: Option<&str>) -> Retained<MPSTensor> {
    tensor.exp(name)
}

/// Applies natural logarithm to the tensor elements
///
/// Computes ln(x) for each element in the tensor
///
/// # Parameters
///
/// * `tensor` - The input tensor
/// * `name` - Optional name for the operation
///
/// # Returns
///
/// A new tensor with the natural logarithm of each element
pub fn log(tensor: &Retained<MPSTensor>, name: Option<&str>) -> Retained<MPSTensor> {
    tensor.log(name)
}

/// Applies sigmoid activation function to the tensor elements
///
/// Computes σ(x) = 1/(1+e^(-x)) for each element in the tensor
///
/// # Parameters
///
/// * `tensor` - The input tensor
/// * `name` - Optional name for the operation
///
/// # Returns
///
/// A new tensor with the sigmoid of each element
pub fn sigmoid(tensor: &Retained<MPSTensor>, name: Option<&str>) -> Retained<MPSTensor> {
    tensor.sigmoid(name)
}

/// Applies tanh activation function to the tensor elements
///
/// Computes tanh(x) for each element in the tensor
///
/// # Parameters
///
/// * `tensor` - The input tensor
/// * `name` - Optional name for the operation
///
/// # Returns
///
/// A new tensor with the tanh of each element
pub fn tanh(tensor: &Retained<MPSTensor>, name: Option<&str>) -> Retained<MPSTensor> {
    tensor.tanh(name)
}

/// Applies ReLU activation function to the tensor elements
///
/// Computes max(0, x) for each element in the tensor
///
/// # Parameters
///
/// * `tensor` - The input tensor
/// * `name` - Optional name for the operation
///
/// # Returns
///
/// A new tensor with the ReLU activation applied
pub fn relu(tensor: &Retained<MPSTensor>, name: Option<&str>) -> Retained<MPSTensor> {
    tensor.relu(name)
}

/// Applies SiLU activation function (x * sigmoid(x))
///
/// SiLU (Sigmoid Linear Unit) is also known as the Swish activation function.
/// It computes x * sigmoid(x) for each element in the tensor.
///
/// # Parameters
///
/// * `tensor` - The input tensor
/// * `name_prefix` - Optional prefix for the operation names
///
/// # Returns
///
/// A new tensor with the SiLU activation applied
pub fn silu(tensor: &Retained<MPSTensor>, name_prefix: Option<&str>) -> Retained<MPSTensor> {
    // Implementation using the TensorOps trait methods
    let sigmoid_name = name_prefix.map(|p| format!("{}_sigmoid", p));
    let sigmoid_tensor = tensor.sigmoid(sigmoid_name.as_deref());

    tensor.mul(&sigmoid_tensor, name_prefix)
}

/// Applies GELU activation function
///
/// GELU (Gaussian Error Linear Unit) is defined as x * Φ(x) where Φ is the cumulative
/// distribution function of the standard normal distribution.
/// This implementation uses the approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
///
/// # Parameters
///
/// * `tensor` - The input tensor
/// * `name_prefix` - Optional prefix for the operation names
///
/// # Returns
///
/// A new tensor with the GELU activation applied
pub fn gelu(tensor: &Retained<MPSTensor>, name_prefix: Option<&str>) -> Retained<MPSTensor> {
    // Constants for the GELU approximation
    let sqrt_2_over_pi = 0.7978845608028654; // sqrt(2/π)
    let coeff = 0.044715;
    let graph = get_graph_from_tensor(tensor);
    let data_type = get_data_type_from_tensor(tensor);

    // Create constant tensors
    let const_0_5 = graph.constant_scalar_tensor(0.5, data_type, None);
    let const_1 = graph.constant_scalar_tensor(1.0, data_type, None);
    let const_sqrt_2_pi = graph.constant_scalar_tensor(sqrt_2_over_pi, data_type, None);
    let const_coeff = graph.constant_scalar_tensor(coeff, data_type, None);

    // Compute x^3
    let square_name = name_prefix.map(|p| format!("{}_square", p));
    let x_squared = tensor.square(square_name.as_deref());

    let cube_name = name_prefix.map(|p| format!("{}_cube", p));
    let x_cubed = tensor.mul(&x_squared, cube_name.as_deref());

    // Compute coeff * x^3
    let scaled_cube_name = name_prefix.map(|p| format!("{}_scaled_cube", p));
    let scaled_x_cubed = graph.multiply(&const_coeff, &x_cubed, scaled_cube_name.as_deref());

    // Compute x + coeff * x^3
    let inner_name = name_prefix.map(|p| format!("{}_inner", p));
    let inner = tensor.add(&scaled_x_cubed, inner_name.as_deref());

    // Compute sqrt(2/π) * (x + coeff * x^3)
    let scaled_inner_name = name_prefix.map(|p| format!("{}_scaled_inner", p));
    let scaled_inner = graph.multiply(&const_sqrt_2_pi, &inner, scaled_inner_name.as_deref());

    // Compute tanh(sqrt(2/π) * (x + coeff * x^3))
    let tanh_name = name_prefix.map(|p| format!("{}_tanh", p));
    let tanh_tensor = scaled_inner.tanh(tanh_name.as_deref());

    // Compute 1 + tanh(...)
    let one_plus_tanh_name = name_prefix.map(|p| format!("{}_one_plus_tanh", p));
    let one_plus_tanh = graph.add(&const_1, &tanh_tensor, one_plus_tanh_name.as_deref());

    // Compute 0.5 * (1 + tanh(...))
    let half_term_name = name_prefix.map(|p| format!("{}_half_term", p));
    let half_term = graph.multiply(&const_0_5, &one_plus_tanh, half_term_name.as_deref());

    // Compute x * 0.5 * (1 + tanh(...))
    tensor.mul(&half_term, name_prefix)
}

/// Element-wise power operation
///
/// Raises each element in the tensor to the specified power
///
/// # Parameters
///
/// * `tensor` - The input tensor
/// * `exponent` - The exponent tensor
/// * `name` - Optional name for the operation
///
/// # Returns
///
/// A new tensor with each element raised to the specified power
pub fn pow(
    tensor: &Retained<MPSTensor>,
    exponent: &Retained<MPSTensor>,
    name: Option<&str>,
) -> Retained<MPSTensor> {
    let graph = get_graph_from_tensor(tensor);
    graph.power(tensor, exponent, name)
}

/// Clip tensor values to a specified range
///
/// # Parameters
///
/// * `tensor` - The input tensor
/// * `min_val` - The minimum value tensor (elements smaller than this are clipped)
/// * `max_val` - The maximum value tensor (elements larger than this are clipped)
/// * `name` - Optional name for the operation
///
/// # Returns
///
/// A new tensor with values clipped to the specified range
pub fn clip(
    tensor: &Retained<MPSTensor>,
    min_val: &Retained<MPSTensor>,
    max_val: &Retained<MPSTensor>,
    name: Option<&str>,
) -> Retained<MPSTensor> {
    // First clip to minimum (max of tensor and min_val)
    let graph = get_graph_from_tensor(tensor);
    let name_min = name.map(|n| format!("{}_min", n));
    let clipped_min = graph.maximum(tensor, min_val, name_min.as_deref());

    // Then clip to maximum (min of clipped_min and max_val)
    let name_max = name.map(|n| format!("{}_max", n));
    graph.minimum(&clipped_min, max_val, name_max.as_deref())
}

// Add scalar operations similar to the Swift extension
pub trait TensorScalarOps {
    /// Add a scalar value to this tensor
    fn add_scalar<T: Into<f64> + Copy>(&self, value: T, name: Option<&str>) -> Retained<MPSTensor>;

    /// Subtract a scalar value from this tensor
    fn sub_scalar<T: Into<f64> + Copy>(&self, value: T, name: Option<&str>) -> Retained<MPSTensor>;

    /// Multiply this tensor by a scalar value
    fn mul_scalar<T: Into<f64> + Copy>(&self, value: T, name: Option<&str>) -> Retained<MPSTensor>;

    /// Divide this tensor by a scalar value
    fn div_scalar<T: Into<f64> + Copy>(&self, value: T, name: Option<&str>) -> Retained<MPSTensor>;

    /// Create a constant tensor with the same data type as this tensor
    fn constant<T: Into<f64> + Copy>(&self, value: T) -> Retained<MPSTensor>;

    /// Raise each element of the tensor to a power
    fn power_scalar<T: Into<f64> + Copy>(
        &self,
        exponent: T,
        name: Option<&str>,
    ) -> Retained<MPSTensor>;

    /// Clip tensor values to a specified range
    fn clamp<T: Into<f64> + Copy>(&self, min: T, max: T, name: Option<&str>)
        -> Retained<MPSTensor>;

    /// Get the minimum of this tensor and a scalar value
    fn minimum_scalar<T: Into<f64> + Copy>(
        &self,
        value: T,
        name: Option<&str>,
    ) -> Retained<MPSTensor>;

    /// Get the maximum of this tensor and a scalar value
    fn maximum_scalar<T: Into<f64> + Copy>(
        &self,
        value: T,
        name: Option<&str>,
    ) -> Retained<MPSTensor>;
}

impl TensorScalarOps for Retained<MPSTensor> {
    fn constant<T: Into<f64> + Copy>(&self, value: T) -> Retained<MPSTensor> {
        let graph = get_graph_from_tensor(self);
        let data_type = get_data_type_from_tensor(self);
        graph.constant_scalar_tensor(value, data_type, None)
    }

    fn add_scalar<T: Into<f64> + Copy>(&self, value: T, name: Option<&str>) -> Retained<MPSTensor> {
        // Skip the operation if value is 0
        if value.into() == 0.0 {
            return self.clone();
        }
        let const_tensor = self.constant(value);
        self.add(&const_tensor, name)
    }

    fn sub_scalar<T: Into<f64> + Copy>(&self, value: T, name: Option<&str>) -> Retained<MPSTensor> {
        // Skip the operation if value is 0
        if value.into() == 0.0 {
            return self.clone();
        }
        let const_tensor = self.constant(value);
        self.sub(&const_tensor, name)
    }

    fn mul_scalar<T: Into<f64> + Copy>(&self, value: T, name: Option<&str>) -> Retained<MPSTensor> {
        // Skip the operation if value is 1
        if value.into() == 1.0 {
            return self.clone();
        }
        let const_tensor = self.constant(value);
        self.mul(&const_tensor, name)
    }

    fn div_scalar<T: Into<f64> + Copy>(&self, value: T, name: Option<&str>) -> Retained<MPSTensor> {
        // Skip the operation if value is 1
        if value.into() == 1.0 {
            return self.clone();
        }
        let const_tensor = self.constant(value);
        self.div(&const_tensor, name)
    }

    fn power_scalar<T: Into<f64> + Copy>(
        &self,
        exponent: T,
        name: Option<&str>,
    ) -> Retained<MPSTensor> {
        let const_tensor = self.constant(exponent);
        let graph = get_graph_from_tensor(self);
        graph.power(self, &const_tensor, name)
    }

    fn clamp<T: Into<f64> + Copy>(
        &self,
        min: T,
        max: T,
        name: Option<&str>,
    ) -> Retained<MPSTensor> {
        let min_tensor = self.constant(min);
        let max_tensor = self.constant(max);
        clip(self, &min_tensor, &max_tensor, name)
    }

    fn minimum_scalar<T: Into<f64> + Copy>(
        &self,
        value: T,
        name: Option<&str>,
    ) -> Retained<MPSTensor> {
        let const_tensor = self.constant(value);
        let graph = get_graph_from_tensor(self);
        graph.minimum(self, &const_tensor, name)
    }

    fn maximum_scalar<T: Into<f64> + Copy>(
        &self,
        value: T,
        name: Option<&str>,
    ) -> Retained<MPSTensor> {
        let const_tensor = self.constant(value);
        let graph = get_graph_from_tensor(self);
        graph.maximum(self, &const_tensor, name)
    }
}
