//! Enhanced Tensor type with operator overloading
//!
//! This module provides a wrapper around mpsgraph::Tensor that adds operator
//! overloading and other ergonomic features.

use mpsgraph::tensor::Tensor as MPSTensor;
use mpsgraph::{DataType, Graph, GraphActivationOps, GraphArithmeticOps, Operation, Shape};
use objc2::msg_send;
use objc2::rc::Retained;
use std::fmt;
use std::ops::{Add, Deref, DerefMut, Div, Mul, Neg, Sub};

/// A tensor with operator overloading capabilities.
///
/// This is a wrapper around mpsgraph's Retained<Tensor> that adds
/// operator overloading and other ergonomic features.
#[derive(Clone)]
pub struct Tensor(pub Retained<MPSTensor>);

// We'll use a simpler approach for supporting a + b syntax

// Make Tensor transparently deref to Retained<MPSTensor>
impl Deref for Tensor {
    type Target = Retained<MPSTensor>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Tensor {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

// Forward Debug to the inner Retained<MPSTensor>
impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

// Conversion from Retained<MPSTensor> to Tensor
impl From<Retained<MPSTensor>> for Tensor {
    fn from(tensor: Retained<MPSTensor>) -> Self {
        Tensor(tensor)
    }
}

impl Tensor {
    /// Create a new placeholder tensor on the given graph
    pub fn placeholder(graph: &Graph, data_type: DataType, shape: &Shape) -> Self {
        Tensor(graph.placeholder(data_type, shape, None))
    }

    /// Create a new scalar constant tensor on the given graph
    pub fn constant<T: Into<f64> + Copy>(graph: &Graph, value: T, data_type: DataType) -> Self {
        unsafe {
            let value_f64 = value.into();
            let tensor: Retained<MPSTensor> = objc2::msg_send![
                &**graph,
                constantWithScalar: value_f64,
                dataType: data_type as u32
            ];

            Tensor(tensor)
        }
    }

    /// Create a tensor filled with zeros (currently a scalar, shape is ignored)
    pub fn zeros(graph: &Graph, data_type: DataType, _shape: &Shape) -> Self {
        Self::constant(graph, 0.0, data_type)
    }

    /// Create a tensor filled with ones (currently a scalar, shape is ignored)
    pub fn ones(graph: &Graph, data_type: DataType, _shape: &Shape) -> Self {
        Self::constant(graph, 1.0, data_type)
    }

    /// Get direct access to the underlying Retained<MPSTensor>
    pub fn inner(&self) -> &Retained<MPSTensor> {
        &self.0
    }

    /// Returns the operation that created this tensor
    pub fn operation(&self) -> Retained<Operation> {
        unsafe {
            let tensor_ptr = &*self.0;
            let operation_ptr: *mut Operation = msg_send![tensor_ptr, operation];
            Retained::from_raw(operation_ptr).unwrap()
        }
    }

    /// Returns the graph that this tensor belongs to
    pub fn graph(&self) -> Retained<Graph> {
        // Get the graph via the operation that created this tensor
        let operation = self.operation();
        unsafe {
            let graph_ptr: *mut Graph = msg_send![&*operation, graph];
            Retained::from_raw(graph_ptr).unwrap()
        }
    }
}

// =============================
// OPERATOR IMPLEMENTATIONS
// =============================

// ADDITION OPERATIONS

// &Tensor + &Tensor
impl<'a, 'b> Add<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    fn add(self, rhs: &'b Tensor) -> Self::Output {
        Tensor(self.graph().add(&self.0, &rhs.0, None))
    }
}

// Tensor + &Tensor
impl<'a> Add<&'a Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: &'a Tensor) -> Self::Output {
        Tensor(self.graph().add(&self.0, &rhs.0, None))
    }
}

// &Tensor + Tensor
impl<'a> Add<Tensor> for &'a Tensor {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Self::Output {
        Tensor(self.graph().add(&self.0, &rhs.0, None))
    }
}

// Tensor + Tensor
impl Add for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Self::Output {
        Tensor(self.graph().add(&self.0, &rhs.0, None))
    }
}

// SUBTRACTION OPERATIONS

// &Tensor - &Tensor
impl<'a, 'b> Sub<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    fn sub(self, rhs: &'b Tensor) -> Self::Output {
        Tensor(self.graph().subtract(&self.0, &rhs.0, None))
    }
}

// Tensor - &Tensor
impl<'a> Sub<&'a Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: &'a Tensor) -> Self::Output {
        Tensor(self.graph().subtract(&self.0, &rhs.0, None))
    }
}

// &Tensor - Tensor
impl<'a> Sub<Tensor> for &'a Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Tensor) -> Self::Output {
        Tensor(self.graph().subtract(&self.0, &rhs.0, None))
    }
}

// Tensor - Tensor
impl Sub for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Tensor) -> Self::Output {
        Tensor(self.graph().subtract(&self.0, &rhs.0, None))
    }
}

// MULTIPLICATION OPERATIONS

// &Tensor * &Tensor
impl<'a, 'b> Mul<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &'b Tensor) -> Self::Output {
        Tensor(self.graph().multiply(&self.0, &rhs.0, None))
    }
}

// Tensor * &Tensor
impl<'a> Mul<&'a Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &'a Tensor) -> Self::Output {
        Tensor(self.graph().multiply(&self.0, &rhs.0, None))
    }
}

// &Tensor * Tensor
impl<'a> Mul<Tensor> for &'a Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Self::Output {
        Tensor(self.graph().multiply(&self.0, &rhs.0, None))
    }
}

// Tensor * Tensor
impl Mul for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Self::Output {
        Tensor(self.graph().multiply(&self.0, &rhs.0, None))
    }
}

// DIVISION OPERATIONS

// &Tensor / &Tensor
impl<'a, 'b> Div<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    fn div(self, rhs: &'b Tensor) -> Self::Output {
        Tensor(self.graph().divide(&self.0, &rhs.0, None))
    }
}

// Tensor / &Tensor
impl<'a> Div<&'a Tensor> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: &'a Tensor) -> Self::Output {
        Tensor(self.graph().divide(&self.0, &rhs.0, None))
    }
}

// &Tensor / Tensor
impl<'a> Div<Tensor> for &'a Tensor {
    type Output = Tensor;

    fn div(self, rhs: Tensor) -> Self::Output {
        Tensor(self.graph().divide(&self.0, &rhs.0, None))
    }
}

// Tensor / Tensor
impl Div for Tensor {
    type Output = Tensor;

    fn div(self, rhs: Tensor) -> Self::Output {
        Tensor(self.graph().divide(&self.0, &rhs.0, None))
    }
}

// NEGATION OPERATIONS

// -&Tensor
impl<'a> Neg for &'a Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        Tensor(self.graph().negative(&self.0, None))
    }
}

// -Tensor
impl Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        Tensor(self.graph().negative(&self.0, None))
    }
}

// =============================
// SCALAR OPERATIONS
// =============================

// &Tensor + f64
impl<'a> Add<f64> for &'a Tensor {
    type Output = Tensor;

    fn add(self, rhs: f64) -> Self::Output {
        if rhs == 0.0 {
            return self.clone();
        }

        let graph = self.graph();
        let data_type = self.0.data_type();

        let const_tensor = Tensor::constant(&graph, rhs, data_type);
        Tensor(graph.add(&self.0, &const_tensor.0, None))
    }
}

// Tensor + f64
impl Add<f64> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: f64) -> Self::Output {
        if rhs == 0.0 {
            return self;
        }

        let graph = self.graph();
        let data_type = self.0.data_type();

        let const_tensor = Tensor::constant(&graph, rhs, data_type);
        Tensor(graph.add(&self.0, &const_tensor.0, None))
    }
}

// &Tensor - f64
impl<'a> Sub<f64> for &'a Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f64) -> Self::Output {
        if rhs == 0.0 {
            return self.clone();
        }

        let graph = self.graph();
        let data_type = self.0.data_type();

        let const_tensor = Tensor::constant(&graph, rhs, data_type);
        Tensor(graph.subtract(&self.0, &const_tensor.0, None))
    }
}

// Tensor - f64
impl Sub<f64> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f64) -> Self::Output {
        if rhs == 0.0 {
            return self;
        }

        let graph = self.graph();
        let data_type = self.0.data_type();

        let const_tensor = Tensor::constant(&graph, rhs, data_type);
        Tensor(graph.subtract(&self.0, &const_tensor.0, None))
    }
}

// &Tensor * f64
impl<'a> Mul<f64> for &'a Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f64) -> Self::Output {
        if rhs == 1.0 {
            return self.clone();
        }

        let graph = self.graph();
        let data_type = self.0.data_type();

        let const_tensor = Tensor::constant(&graph, rhs, data_type);
        Tensor(graph.multiply(&self.0, &const_tensor.0, None))
    }
}

// Tensor * f64
impl Mul<f64> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f64) -> Self::Output {
        if rhs == 1.0 {
            return self;
        }

        let graph = self.graph();
        let data_type = self.0.data_type();

        let const_tensor = Tensor::constant(&graph, rhs, data_type);
        Tensor(graph.multiply(&self.0, &const_tensor.0, None))
    }
}

// &Tensor / f64
impl<'a> Div<f64> for &'a Tensor {
    type Output = Tensor;

    fn div(self, rhs: f64) -> Self::Output {
        if rhs == 1.0 {
            return self.clone();
        }

        let graph = self.graph();
        let data_type = self.0.data_type();

        let const_tensor = Tensor::constant(&graph, rhs, data_type);
        Tensor(graph.divide(&self.0, &const_tensor.0, None))
    }
}

// Tensor / f64
impl Div<f64> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: f64) -> Self::Output {
        if rhs == 1.0 {
            return self;
        }

        let graph = self.graph();
        let data_type = self.0.data_type();

        let const_tensor = Tensor::constant(&graph, rhs, data_type);
        Tensor(graph.divide(&self.0, &const_tensor.0, None))
    }
}

// =============================
// COMMUTATIVE SCALAR OPERATIONS
// =============================

// f64 + &Tensor
impl<'a> Add<&'a Tensor> for f64 {
    type Output = Tensor;

    fn add(self, rhs: &'a Tensor) -> Self::Output {
        // Addition is commutative
        rhs + self
    }
}

// f64 + Tensor
impl Add<Tensor> for f64 {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Self::Output {
        // Addition is commutative
        rhs + self
    }
}

// f64 * &Tensor
impl<'a> Mul<&'a Tensor> for f64 {
    type Output = Tensor;

    fn mul(self, rhs: &'a Tensor) -> Self::Output {
        // Multiplication is commutative
        rhs * self
    }
}

// f64 * Tensor
impl Mul<Tensor> for f64 {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Self::Output {
        // Multiplication is commutative
        rhs * self
    }
}

// f64 - &Tensor
impl<'a> Sub<&'a Tensor> for f64 {
    type Output = Tensor;

    fn sub(self, rhs: &'a Tensor) -> Self::Output {
        let graph = rhs.graph();
        let data_type = rhs.0.data_type();

        let const_tensor = Tensor::constant(&graph, self, data_type);
        Tensor(graph.subtract(&const_tensor.0, &rhs.0, None))
    }
}

// f64 - Tensor
impl Sub<Tensor> for f64 {
    type Output = Tensor;

    fn sub(self, rhs: Tensor) -> Self::Output {
        let graph = rhs.graph();
        let data_type = rhs.0.data_type();

        let const_tensor = Tensor::constant(&graph, self, data_type);
        Tensor(graph.subtract(&const_tensor.0, &rhs.0, None))
    }
}

// f64 / &Tensor
impl<'a> Div<&'a Tensor> for f64 {
    type Output = Tensor;

    fn div(self, rhs: &'a Tensor) -> Self::Output {
        let graph = rhs.graph();
        let data_type = rhs.0.data_type();

        let const_tensor = Tensor::constant(&graph, self, data_type);
        Tensor(graph.divide(&const_tensor.0, &rhs.0, None))
    }
}

// f64 / Tensor
impl Div<Tensor> for f64 {
    type Output = Tensor;

    fn div(self, rhs: Tensor) -> Self::Output {
        let graph = rhs.graph();
        let data_type = rhs.0.data_type();

        let const_tensor = Tensor::constant(&graph, self, data_type);
        Tensor(graph.divide(&const_tensor.0, &rhs.0, None))
    }
}

// =============================
// ACTIVATION FUNCTIONS
// =============================

impl Tensor {
    /// Apply the sigmoid activation function
    pub fn sigmoid(&self) -> Tensor {
        Tensor(self.graph().sigmoid(&self.0, None))
    }

    /// Apply the tanh activation function
    pub fn tanh(&self) -> Tensor {
        let graph = self.graph();
        Tensor(GraphActivationOps::tanh(&*graph, &self.0, None))
    }

    /// Apply the ReLU activation function
    pub fn relu(&self) -> Tensor {
        Tensor(self.graph().relu(&self.0, None))
    }

    /// Apply the SiLU activation function: x * sigmoid(x)
    pub fn silu(&self) -> Tensor {
        let sigmoid = self.sigmoid();
        self * &sigmoid
    }

    /// Apply the GELU activation function
    /// Implementation uses the approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    pub fn gelu(&self) -> Tensor {
        // Constants for the GELU approximation
        let sqrt_2_over_pi = 0.7978845608028654; // sqrt(2/π)
        let coeff = 0.044715;
        let graph = self.graph();

        // Create constant tensors
        let data_type = self.0.data_type();
        let const_0_5 = Tensor::constant(&graph, 0.5, data_type);
        let const_1 = Tensor::constant(&graph, 1.0, data_type);
        let const_sqrt_2_pi = Tensor::constant(&graph, sqrt_2_over_pi, data_type);
        let const_coeff = Tensor::constant(&graph, coeff, data_type);

        // Compute x^3
        let x_squared = self.square();
        let x_cubed = &x_squared * self;

        // Compute coeff * x^3
        let scaled_x_cubed = &const_coeff * &x_cubed;

        // Compute x + coeff * x^3
        let inner = self + &scaled_x_cubed;

        // Compute sqrt(2/π) * (x + coeff * x^3)
        let scaled_inner: Tensor = &const_sqrt_2_pi * &inner;

        // Compute tanh(sqrt(2/π) * (x + coeff * x^3))
        let tanh_tensor = scaled_inner.tanh();

        // Compute 1 + tanh(...)
        let one_plus_tanh = &const_1 + &tanh_tensor;

        // Compute 0.5 * (1 + tanh(...))
        let half_term = &const_0_5 * &one_plus_tanh;

        // Compute x * 0.5 * (1 + tanh(...))
        self * &half_term
    }

    /// Calculate the square of this tensor
    pub fn square(&self) -> Tensor {
        self * self
    }

    /// Clip the tensor values between min and max
    pub fn clamp<T: Into<f64> + Copy>(&self, min: T, max: T) -> Tensor {
        let graph = self.graph();
        let data_type = self.0.data_type();

        let min_tensor = Tensor::constant(&graph, min, data_type);
        let max_tensor = Tensor::constant(&graph, max, data_type);

        // First clip to minimum (max of tensor and min_val)
        let clipped_min = self.maximum(&min_tensor);

        // Then clip to maximum (min of clipped_min and max_val)
        clipped_min.minimum(&max_tensor)
    }

    /// Get the minimum of this tensor and another tensor
    pub fn minimum(&self, other: &Tensor) -> Tensor {
        Tensor(self.graph().minimum(&self.0, &other.0, None))
    }

    /// Get the maximum of this tensor and another tensor
    pub fn maximum(&self, other: &Tensor) -> Tensor {
        Tensor(self.graph().maximum(&self.0, &other.0, None))
    }
}
