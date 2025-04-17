//! Tensor Operations API Example
//!
//! This example demonstrates usage of the functional tensor operations
//! provided by the mpsgraph-tools crate.
//!
//! The API includes:
//! - Operator overloading (+, -, *, /, unary -)
//! - Functional tensor operations (square, sqrt, abs, etc.)
//! - ML activation functions (relu, silu, gelu)
//! - Tensor creation utilities
//!
//! Run with: `cargo run --example tensor_ops`

use mpsgraph::{DataType, Graph, Shape};
use mpsgraph_tools::prelude::*;
use objc2_foundation::NSNumber;

// No helper needed as we use NSNumber directly

fn main() {
    // Create graph and input tensors
    let graph = Graph::new();
    
    // Create a shape using NSNumber values
    let numbers = [
        NSNumber::new_usize(2),
        NSNumber::new_usize(3),
    ];
    let number_refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();
    let shape = Shape::from_slice(&number_refs);

    // Use the GraphExtensions trait to create tensor directly with GraphTensor
    let a = graph.placeholder_tensor(&shape, DataType::Float32, Some("a"));
    let b = graph.placeholder_tensor(&shape, DataType::Float32, Some("b"));

    println!("Demonstrating tensor operations with tensors:");
    println!("Shape = [2, 3]");

    // 1. Basic arithmetic with references
    let _sum = &a + &b;
    let _diff = &a - &b;
    let _product = &a * &b;
    let _division = &a / &b;
    let _negated = -&a;

    // 2. Functional operations
    let _squared = square(&a, None);
    let _sqrt_a = sqrt(&a, None);
    
    // Use operator overloading to create a difference tensor,
    // then apply the abs function to it
    let diff_tensor = &a - &b;
    let _abs_diff = abs(&diff_tensor, None);

    // 3. Function composition
    let sum_tensor = &a + &b;
    let sqrt_sum = sqrt(&sum_tensor, None);
    let _complex_expr = abs(&sqrt_sum, None);

    // 4. Activation functions
    let _silu_a = silu(&a, None);
    // Skip gelu for now, as it relies on constant_scalar which is not working
    // let _gelu_a = gelu(&a, None);

    // 5. Additional operations
    let _exp_a = exp(&a, None);
    let _log_a = log(&a, None);

    // 6. Tensor creation using our new methods
    println!("\nCreating tensors:");
    
    // Create a scalar constant
    let _scalar = graph.constant_scalar_tensor(3.14, DataType::Float32, None);
    println!("- Created scalar constant with value 3.14");
    
    // Create a shaped constant (all elements set to the same value)
    let _shaped_const = graph.constant_scalar_shaped_tensor(2.0, &shape, DataType::Float32, None);
    println!("- Created shaped constant with value 2.0 and shape [2, 3]");
    
    // Create a tensor filled with zeros
    let _zeros = graph.zeros(&shape, DataType::Float32, None);
    println!("- Created zeros tensor with shape [2, 3]");
    
    // Create a tensor filled with ones
    let _ones = graph.ones(&shape, DataType::Float32, None);
    println!("- Created ones tensor with shape [2, 3]");
    
    // Create a tensor filled with a specific value
    let _filled = graph.fill(5.0, &shape, DataType::Float32, None);
    println!("- Created tensor filled with value 5.0 and shape [2, 3]");
    

    println!("\nSuccessfully created operations using functional style API:");
    println!("- Basic arithmetic: &a + &b, &a - &b, &a * &b, &a / &b, -&a");
    println!("- Simple operations: square(&a), sqrt(&a), abs(&tensor)");
    println!("- Complex expressions: abs(&sqrt(&(&a + &b)))");
    println!("- Activation functions: silu(&a)");
    println!("- Math operations: exp(&a), log(&a)");
    println!("- Tensor creation: zeros, ones, fill, constant tensors");
}
