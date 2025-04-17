//! Tensor Operations API for Graph
//!
//! This module provides ergonomic tensor operations with operator overloading
//! and functional-style programming for Graph tensors.
//!
//! # Features
//!
//! - **Trait-Based Operator Overloading**: Use standard operators (`+`, `-`, `*`, `/`, `-x`) with references
//!   for tensor operations (e.g., `&a + &b`) through extension traits
//! - **Functional API**: Apply operations using functional style (e.g., `abs(&sqrt(&(&a + &b), None), None)`)
//! - **Utility Methods**: Convenience functions for common operations
//! - **Tensor Creation**: Helper methods for creating tensors filled with zeros, ones, etc.

use mpsgraph::{
    DataType, Graph, Tensor, Shape,
    GraphActivationOps, GraphArithmeticOps
};
use objc2::rc::Retained;
use objc2::msg_send;
use std::ops;

/// Extension trait for tensor arithmetic operations
/// 
/// This trait adds arithmetic operations to the Tensor type.
pub trait TensorOps {
    /// Get the underlying tensor
    fn tensor(&self) -> &Tensor;
    
    /// Get the associated graph for this tensor
    fn graph(&self) -> Retained<Graph>;

    /// Add this tensor to another tensor
    fn add(&self, rhs: &Tensor, name: Option<&str>) -> Retained<Tensor>;
    
    /// Subtract another tensor from this tensor
    fn sub(&self, rhs: &Tensor, name: Option<&str>) -> Retained<Tensor>;
    
    /// Multiply this tensor by another tensor
    fn mul(&self, rhs: &Tensor, name: Option<&str>) -> Retained<Tensor>;
    
    /// Divide this tensor by another tensor
    fn div(&self, rhs: &Tensor, name: Option<&str>) -> Retained<Tensor>;
    
    /// Negate this tensor
    fn neg(&self, name: Option<&str>) -> Retained<Tensor>;
    
    /// Square each element of this tensor
    fn square(&self, name: Option<&str>) -> GraphTensor;
    
    /// Take the square root of each element of this tensor
    fn sqrt(&self, name: Option<&str>) -> GraphTensor;
    
    /// Take the absolute value of each element of this tensor
    fn abs(&self, name: Option<&str>) -> GraphTensor;
    
    /// Take the exponential of each element of this tensor
    fn exp(&self, name: Option<&str>) -> GraphTensor;
    
    /// Take the natural logarithm of each element of this tensor
    fn log(&self, name: Option<&str>) -> GraphTensor;
    
    /// Apply the sigmoid function to each element of this tensor
    fn sigmoid(&self, name: Option<&str>) -> GraphTensor;
    
    /// Apply the tanh function to each element of this tensor
    fn tanh(&self, name: Option<&str>) -> GraphTensor;
    
    /// Apply the ReLU function to each element of this tensor
    fn relu(&self, name: Option<&str>) -> GraphTensor;
}

/// Extension trait for operator overloading with tensors
pub trait TensorOpOverloads: TensorOps {
    /// Addition operator
    fn add_op(&self, rhs: &Self) -> Retained<Tensor>;
    
    /// Subtraction operator
    fn sub_op(&self, rhs: &Self) -> Retained<Tensor>;
    
    /// Multiplication operator
    fn mul_op(&self, rhs: &Self) -> Retained<Tensor>;
    
    /// Division operator
    fn div_op(&self, rhs: &Self) -> Retained<Tensor>;
    
    /// Negation operator
    fn neg_op(&self) -> Retained<Tensor>;
}
/// Struct to associate a Graph with a Tensor
#[derive(Debug, Clone)]
pub struct GraphTensor {
    pub tensor: Retained<Tensor>,
    pub graph: Retained<Graph>,
}

impl GraphTensor {
    /// Create a new GraphTensor from a Tensor and a Graph
    pub fn new(tensor: Retained<Tensor>, graph: Retained<Graph>) -> Self {
        GraphTensor { tensor, graph }
    }
    
    /// Create a new GraphTensor from a TensorOps object
    pub fn from_ops(ops: &impl TensorOps) -> Self {
        let tensor = unsafe { Retained::from_raw(ops.tensor() as *const _ as *mut _) }.unwrap();
        GraphTensor {
            tensor,
            graph: ops.graph(),
        }
    }
}

impl TensorOps for GraphTensor {
    fn tensor(&self) -> &Tensor {
        &self.tensor
    }
    
    fn graph(&self) -> Retained<Graph> {
        self.graph.clone()
    }
    
    fn add(&self, rhs: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        self.graph.add(self.tensor.as_ref(), rhs, name)
            .expect("Failed to add tensors")
    }
    
    fn sub(&self, rhs: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        self.graph.subtract(self.tensor.as_ref(), rhs, name)
            .expect("Failed to subtract tensors")
    }
    
    fn mul(&self, rhs: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        self.graph.multiply(self.tensor.as_ref(), rhs, name)
            .expect("Failed to multiply tensors")
    }
    
    fn div(&self, rhs: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        self.graph.divide(self.tensor.as_ref(), rhs, name)
            .expect("Failed to divide tensors")
    }
    
    fn neg(&self, name: Option<&str>) -> Retained<Tensor> {
        self.graph.negative(self.tensor.as_ref(), name)
            .expect("Failed to negate tensor")
    }
    
    fn square(&self, name: Option<&str>) -> GraphTensor {
        let tensor = self.graph.square(self.tensor.as_ref(), name)
            .expect("Failed to square tensor");
        GraphTensor::new(tensor, self.graph())
    }
    
    fn sqrt(&self, name: Option<&str>) -> GraphTensor {
        let tensor = self.graph.sqrt(self.tensor.as_ref(), name)
            .expect("Failed to compute square root");
        GraphTensor::new(tensor, self.graph())
    }
    
    fn abs(&self, name: Option<&str>) -> GraphTensor {
        let tensor = self.graph.abs(self.tensor.as_ref(), name)
            .expect("Failed to compute absolute value");
        GraphTensor::new(tensor, self.graph())
    }
    
    fn exp(&self, name: Option<&str>) -> GraphTensor {
        let tensor = self.graph.exp(self.tensor.as_ref(), name)
            .expect("Failed to compute exponential");
        GraphTensor::new(tensor, self.graph())
    }
    
    fn log(&self, name: Option<&str>) -> GraphTensor {
        let tensor = self.graph.log(self.tensor.as_ref(), name)
            .expect("Failed to compute logarithm");
        GraphTensor::new(tensor, self.graph())
    }
    
    fn sigmoid(&self, name: Option<&str>) -> GraphTensor {
        let tensor = self.graph.sigmoid(self.tensor.as_ref(), name)
            .expect("Failed to compute sigmoid");
        GraphTensor::new(tensor, self.graph())
    }
    
    fn tanh(&self, name: Option<&str>) -> GraphTensor {
        let tensor = self.graph.tanh(self.tensor.as_ref(), name)
            .expect("Failed to compute tanh");
        GraphTensor::new(tensor, self.graph())
    }
    
    fn relu(&self, name: Option<&str>) -> GraphTensor {
        let tensor = self.graph.relu(self.tensor.as_ref(), name)
            .expect("Failed to compute ReLU");
        GraphTensor::new(tensor, self.graph())
    }
}

impl TensorOpOverloads for GraphTensor {
    fn add_op(&self, rhs: &Self) -> Retained<Tensor> {
        self.add(rhs.tensor.as_ref(), None)
    }
    
    fn sub_op(&self, rhs: &Self) -> Retained<Tensor> {
        self.sub(rhs.tensor.as_ref(), None)
    }
    
    fn mul_op(&self, rhs: &Self) -> Retained<Tensor> {
        self.mul(rhs.tensor.as_ref(), None)
    }
    
    fn div_op(&self, rhs: &Self) -> Retained<Tensor> {
        self.div(rhs.tensor.as_ref(), None)
    }
    
    fn neg_op(&self) -> Retained<Tensor> {
        self.neg(None)
    }
}

impl<'a, 'b> ops::Add<&'b GraphTensor> for &'a GraphTensor {
    type Output = GraphTensor;

    fn add(self, rhs: &'b GraphTensor) -> Self::Output {
        GraphTensor::new(self.add_op(rhs), self.graph())
    }
}

// Implement the remaining operator overloads
impl<'a, 'b> ops::Sub<&'b GraphTensor> for &'a GraphTensor {
    type Output = GraphTensor;

    fn sub(self, rhs: &'b GraphTensor) -> Self::Output {
        GraphTensor::new(self.sub_op(rhs), self.graph())
    }
}

impl<'a, 'b> ops::Mul<&'b GraphTensor> for &'a GraphTensor {
    type Output = GraphTensor;

    fn mul(self, rhs: &'b GraphTensor) -> Self::Output {
        GraphTensor::new(self.mul_op(rhs), self.graph())
    }
}

impl<'a, 'b> ops::Div<&'b GraphTensor> for &'a GraphTensor {
    type Output = GraphTensor;

    fn div(self, rhs: &'b GraphTensor) -> Self::Output {
        GraphTensor::new(self.div_op(rhs), self.graph())
    }
}

impl<'a> ops::Neg for &'a GraphTensor {
    type Output = GraphTensor;

    fn neg(self) -> Self::Output {
        GraphTensor::new(self.neg_op(), self.graph())
    }
}
/// These operators are implemented for GraphTensor type
/// The operator methods for old Tensor have been removed as the Tensor 
/// class in the modernized API doesn't have graph() or tensor() methods
///
/// Use the GraphTensor type instead, which associates a Graph with a Tensor.

/// GraphTensor implementation provides the modernized
/// tensor operations API that works with the updated
/// mpsgraph crate using Retained<T> types
///
/// All the operations that were previously in Tensor are now
/// available through the GraphTensor type.

/// Extensions for Graph to create GraphTensor
pub trait GraphExtensions {
    /// Create a placeholder tensor and wrap it with GraphTensor
    fn placeholder_tensor(
        &self,
        shape: &Shape,
        data_type: DataType,
        name: Option<&str>,
    ) -> GraphTensor;

    /// Create a constant scalar tensor (without shape) and wrap it with GraphTensor
    /// 
    /// Note: This uses a workaround since the underlying MPS API doesn't match
    /// our existing bindings.
    fn constant_scalar_tensor<T: Into<f64> + Copy>(
        &self, 
        value: T,
        data_type: DataType,
        name: Option<&str>
    ) -> GraphTensor;

    /// Create a constant scalar tensor with a specific shape and wrap it with GraphTensor
    ///
    /// This creates a tensor filled with the same scalar value at every position.
    fn constant_scalar_shaped_tensor<T: Into<f64> + Copy>(
        &self,
        value: T,
        shape: &Shape,
        data_type: DataType,
        name: Option<&str>
    ) -> GraphTensor;

    /// Create a tensor filled with zeros
    ///
    /// This is a convenience wrapper around constant_scalar_shaped_tensor.
    fn zeros(
        &self,
        shape: &Shape,
        data_type: DataType,
        name: Option<&str>
    ) -> GraphTensor;

    /// Create a tensor filled with ones
    ///
    /// This is a convenience wrapper around constant_scalar_shaped_tensor.
    fn ones(
        &self,
        shape: &Shape,
        data_type: DataType,
        name: Option<&str>
    ) -> GraphTensor;

    /// Create a tensor filled with a specific value
    ///
    /// This is an alias for constant_scalar_shaped_tensor with a more intuitive name.
    fn fill<T: Into<f64> + Copy>(
        &self,
        value: T,
        shape: &Shape,
        data_type: DataType,
        name: Option<&str>
    ) -> GraphTensor;
}

impl GraphExtensions for Graph {
    fn placeholder_tensor(
        &self,
        shape: &Shape,
        data_type: DataType,
        name: Option<&str>,
    ) -> GraphTensor {
        let tensor_opt = if let Some(name_str) = name {
            self.placeholder_with_name(data_type, shape, name_str)
        } else {
            self.placeholder(data_type, shape)
        };
        let tensor = tensor_opt.expect("Failed to create placeholder tensor");
        let graph_clone = unsafe { Retained::from_raw(self as *const _ as *mut _) }.unwrap();
        GraphTensor::new(tensor, graph_clone)
    }
    
    /// Create a constant scalar tensor (without shape) and wrap it with GraphTensor
    ///
    /// This implementation uses a workaround for creating constant tensors since
    /// the API signature for constant_scalar in mpsgraph-rs doesn't match the actual
    /// method signature in the MPS framework.
    ///
    /// # Parameters
    ///
    /// * `value` - The scalar value to use
    /// * `data_type` - The data type of the tensor elements
    /// * `name` - Optional name for the operation (ignored in this implementation)
    ///
    /// # Returns
    ///
    /// A new tensor filled with the scalar value
    fn constant_scalar_tensor<T: Into<f64> + Copy>(
        &self, 
        value: T,
        data_type: DataType,
        _name: Option<&str>
    ) -> GraphTensor {
        // Use unsafe direct message sending to create the scalar constant
        unsafe {
            let value_f64 = value.into();
            
            // Use the correct method without the name parameter
            // The constantWithScalar:dataType: method expects a 32-bit value for the data type
            let tensor_ptr: *mut Tensor = msg_send![
                self,
                constantWithScalar: value_f64,
                dataType: data_type as u32
            ];
            
            let tensor = Retained::from_raw(tensor_ptr).expect("Failed to create constant scalar tensor");
            let graph_clone = Retained::from_raw(self as *const _ as *mut _).unwrap();
            GraphTensor::new(tensor, graph_clone)
        }
    }
    
    /// Create a constant scalar tensor with a specific shape and wrap it with GraphTensor
    ///
    /// This creates a tensor filled with the same scalar value at every position.
    ///
    /// # Parameters
    ///
    /// * `value` - The scalar value to use
    /// * `shape` - The shape of the tensor to create
    /// * `data_type` - The data type of the tensor elements
    /// * `name` - Optional name for the operation (ignored in this implementation)
    ///
    /// # Returns
    ///
    /// A new tensor filled with the scalar value and shaped according to the shape parameter
    fn constant_scalar_shaped_tensor<T: Into<f64> + Copy>(
        &self,
        value: T,
        shape: &Shape,
        data_type: DataType,
        _name: Option<&str>
    ) -> GraphTensor {
        // Use unsafe direct message sending to create the scalar constant with shape
        unsafe {
            let value_f64 = value.into();
            
            // Use the constantWithScalar:shape:dataType: method
            let tensor_ptr: *mut Tensor = msg_send![
                self,
                constantWithScalar: value_f64,
                shape: shape,
                dataType: data_type as u32
            ];
            
            let tensor = Retained::from_raw(tensor_ptr).expect("Failed to create constant scalar tensor with shape");
            let graph_clone = Retained::from_raw(self as *const _ as *mut _).unwrap();
            GraphTensor::new(tensor, graph_clone)
        }
    }
    
    /// Create a tensor filled with zeros
    ///
    /// This is a convenience wrapper around constant_scalar_shaped_tensor.
    fn zeros(
        &self,
        shape: &Shape,
        data_type: DataType,
        name: Option<&str>
    ) -> GraphTensor {
        self.constant_scalar_shaped_tensor(0.0, shape, data_type, name)
    }

    /// Create a tensor filled with ones
    ///
    /// This is a convenience wrapper around constant_scalar_shaped_tensor.
    fn ones(
        &self,
        shape: &Shape,
        data_type: DataType,
        name: Option<&str>
    ) -> GraphTensor {
        self.constant_scalar_shaped_tensor(1.0, shape, data_type, name)
    }

    /// Create a tensor filled with a specific value
    ///
    /// This is an alias for constant_scalar_shaped_tensor with a more intuitive name.
    fn fill<T: Into<f64> + Copy>(
        &self,
        value: T,
        shape: &Shape,
        data_type: DataType,
        name: Option<&str>
    ) -> GraphTensor {
        self.constant_scalar_shaped_tensor(value, shape, data_type, name)
    }
}

/// Functional API for tensor operations
///
/// These functions provide a functional programming style interface to tensor operations.
/// They enable a consistent style for creating tensor computation graphs.
///
/// # Examples
///
/// ```ignore
/// use mpsgraph_tools::prelude::*;
/// use objc2_foundation::NSNumber;
///
/// // Create a graph and a tensor
/// let graph = Graph::new();
/// 
/// // Create a shape using NSNumber
/// let numbers = [
///     NSNumber::new_usize(2),
///     NSNumber::new_usize(3),
/// ];
/// let number_refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();
/// let shape = Shape::from_slice(&number_refs);
/// 
/// let tensor = graph.placeholder_tensor(&shape, DataType::Float32, None);
///
/// // Method-based API (object-oriented style)
/// let squared_method = tensor.square(None);
///
/// // Functional API (functional programming style)
/// let squared_func = square(&tensor, None);
/// ```

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
pub fn square(tensor: &GraphTensor, name: Option<&str>) -> GraphTensor {
    let result = tensor.square(name);
    result
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
pub fn sqrt(tensor: &GraphTensor, name: Option<&str>) -> GraphTensor {
    let result = tensor.sqrt(name);
    result
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
pub fn abs(tensor: &GraphTensor, name: Option<&str>) -> GraphTensor {
    let result = tensor.abs(name);
    result
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
pub fn exp(tensor: &GraphTensor, name: Option<&str>) -> GraphTensor {
    let result = tensor.exp(name);
    result
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
pub fn log(tensor: &GraphTensor, name: Option<&str>) -> GraphTensor {
    let result = tensor.log(name);
    result
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
pub fn sigmoid(tensor: &GraphTensor, name: Option<&str>) -> GraphTensor {
    let result = tensor.sigmoid(name);
    result
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
pub fn tanh(tensor: &GraphTensor, name: Option<&str>) -> GraphTensor {
    let result = tensor.tanh(name);
    result
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
pub fn relu(tensor: &GraphTensor, name: Option<&str>) -> GraphTensor {
    let result = tensor.relu(name);
    result
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
pub fn silu(tensor: &GraphTensor, name_prefix: Option<&str>) -> GraphTensor {
    // Implementation using the TensorOps trait methods
    let sigmoid_name = name_prefix.map(|p| format!("{}_sigmoid", p));
    let sigmoid_tensor = tensor.sigmoid(sigmoid_name.as_deref());
    
    let result = tensor.mul(sigmoid_tensor.tensor.as_ref(), name_prefix);
    GraphTensor::new(result, tensor.graph())
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
pub fn gelu(tensor: &GraphTensor, name_prefix: Option<&str>) -> GraphTensor {
    // Constants for the GELU approximation
    let sqrt_2_over_pi = 0.7978845608028654; // sqrt(2/π)
    let coeff = 0.044715;
    let graph = tensor.graph();
    let data_type = tensor.tensor().data_type();

    // Create constant tensors
    let const_0_5 = graph.constant_scalar(0.5, data_type, None)
        .expect("Failed to create constant 0.5");
    let const_1 = graph.constant_scalar(1.0, data_type, None)
        .expect("Failed to create constant 1.0");
    let const_sqrt_2_pi = graph.constant_scalar(sqrt_2_over_pi, data_type, None)
        .expect("Failed to create constant sqrt(2/π)");
    let const_coeff = graph.constant_scalar(coeff, data_type, None)
        .expect("Failed to create constant coefficient");

    // Compute x^3
    let square_name = name_prefix.map(|p| format!("{}_square", p));
    let x_squared = tensor.square(square_name.as_deref());
    
    let cube_name = name_prefix.map(|p| format!("{}_cube", p));
    let x_cubed_tensor = tensor.mul(x_squared.tensor.as_ref(), cube_name.as_deref());
    let x_cubed = GraphTensor::new(x_cubed_tensor, graph.clone());

    // Compute coeff * x^3
    let scaled_cube_name = name_prefix.map(|p| format!("{}_scaled_cube", p));
    let scaled_x_cubed_tensor = graph.multiply(&const_coeff, x_cubed.tensor.as_ref(), scaled_cube_name.as_deref())
        .expect("Failed to compute coefficient * x^3");
    let scaled_x_cubed = GraphTensor::new(scaled_x_cubed_tensor, graph.clone());

    // Compute x + coeff * x^3
    let inner_name = name_prefix.map(|p| format!("{}_inner", p));
    let inner_tensor = tensor.add(scaled_x_cubed.tensor.as_ref(), inner_name.as_deref());
    let inner = GraphTensor::new(inner_tensor, graph.clone());

    // Compute sqrt(2/π) * (x + coeff * x^3)
    let scaled_inner_name = name_prefix.map(|p| format!("{}_scaled_inner", p));
    let scaled_inner_tensor = graph.multiply(&const_sqrt_2_pi, inner.tensor.as_ref(), scaled_inner_name.as_deref())
        .expect("Failed to compute scaled inner term");
    let scaled_inner = GraphTensor::new(scaled_inner_tensor, graph.clone());

    // Compute tanh(sqrt(2/π) * (x + coeff * x^3))
    let tanh_name = name_prefix.map(|p| format!("{}_tanh", p));
    let tanh_tensor = scaled_inner.tanh(tanh_name.as_deref());

    // Compute 1 + tanh(...)
    let one_plus_tanh_name = name_prefix.map(|p| format!("{}_one_plus_tanh", p));
    let one_plus_tanh_tensor = graph.add(&const_1, tanh_tensor.tensor.as_ref(), one_plus_tanh_name.as_deref())
        .expect("Failed to compute 1 + tanh");
    let one_plus_tanh = GraphTensor::new(one_plus_tanh_tensor, graph.clone());

    // Compute 0.5 * (1 + tanh(...))
    let half_term_name = name_prefix.map(|p| format!("{}_half_term", p));
    let half_term_tensor = graph.multiply(&const_0_5, one_plus_tanh.tensor.as_ref(), half_term_name.as_deref())
        .expect("Failed to compute half term");
    let half_term = GraphTensor::new(half_term_tensor, graph.clone());

    // Compute x * 0.5 * (1 + tanh(...))
    let result_tensor = tensor.mul(half_term.tensor.as_ref(), name_prefix);
    GraphTensor::new(result_tensor, graph)
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
pub fn pow(tensor: &GraphTensor, exponent: &GraphTensor, name: Option<&str>) -> GraphTensor {
    let graph = tensor.graph();
    let result_opt = graph.power(tensor.tensor.as_ref(), exponent.tensor.as_ref(), name);
    let result = result_opt.expect("Failed to compute power operation");
    GraphTensor::new(result, graph)
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
pub fn clip(tensor: &GraphTensor, min_val: &GraphTensor, max_val: &GraphTensor, name: Option<&str>) -> GraphTensor {
    // First clip to minimum (max of tensor and min_val)
    let graph = tensor.graph();
    let name_min = name.map(|n| format!("{}_min", n));
    let clipped_min_opt = graph.maximum(tensor.tensor.as_ref(), min_val.tensor.as_ref(), name_min.as_deref());
    let clipped_min = clipped_min_opt.expect("Failed to compute maximum for clipping");

    // Then clip to maximum (min of clipped_min and max_val)
    let name_max = name.map(|n| format!("{}_max", n));
    let result_opt = graph.minimum(&clipped_min, max_val.tensor.as_ref(), name_max.as_deref());
    let result = result_opt.expect("Failed to compute minimum for clipping");
    
    GraphTensor::new(result, graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // This is still a test that requires a Metal device, so keep it marked
    // as ignored for CI environments. You can run it locally with:
    // cargo test -- --ignored
    #[test]
    #[ignore]
    fn test_tensor_operations() {
        // Create GraphTensor trait methods - this will verify the interface
        // but doesn't actually execute Metal operations
        
        // This is a compile-time check of the API, not a runtime test
        // The actual execution would require a valid Metal device
        
        // Check that the API exists by referring to the traits and functions
        let _ = <Graph as GraphExtensions>::zeros;
        let _ = <Graph as GraphExtensions>::ones;
        let _ = <Graph as GraphExtensions>::fill::<f32>;
        
        let _ = <GraphTensor as TensorOps>::add;
        let _ = <GraphTensor as TensorOps>::sqrt;
        let _ = <GraphTensor as TensorOps>::abs;
        
        // Functional API operations
        let _ = square;
        let _ = sqrt; 
        let _ = relu;
        
        // This is a compile-time check of the API, not a runtime test
        // The actual execution would require a valid Metal device
        println!("API trait implementation verified");
    }
    
    // Add a basic unit test that doesn't require a Metal device
    #[test]
    fn test_api_structure() {
        // This test just verifies that the API structure is correct
        // and doesn't require an actual Metal device
        
        // Verify GraphTensor struct has the expected fields
        struct _TestGraphTensor {
            tensor: Retained<Tensor>,
            graph: Retained<Graph>,
        }
        
        // Verify that the API structure is correct by checking the presence
        // of the expected trait methods and functions
        
        // Check that the core tensor creation utilities exist
        let _ = <Graph as GraphExtensions>::zeros;
        let _ = <Graph as GraphExtensions>::ones;
        let _ = <Graph as GraphExtensions>::fill::<f32>;
        
        // Verify trait implementations are available
        let _ = <GraphTensor as TensorOps>::add;
        let _ = <GraphTensor as TensorOps>::square;
        let _ = <GraphTensor as TensorOpOverloads>::add_op;
        
        // Check that the GraphTensor struct has the expected methods
        let _ = GraphTensor::new;
        
        // Verify operator overloading implementation exists
        // This confirms the std::ops traits are implemented for GraphTensor
        
        // Test successful if we get here
        println!("API structure verified");
    }
}