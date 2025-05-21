//! Enhanced Tensor Type with Operator Overloading Example
//!
//! This example demonstrates the enhanced Tensor type from mpsgraph-tools-rs
//! which provides operator overloading for tensor operations.
//!
//! The API allows for writing expressions like:
//! - `let c = &a + &b`
//! - `let d = &c * 2.0`
//!
//! Run with: `cargo run --example tensor_operators`

use mpsgraph::shape::Shape;
use mpsgraph::tensor::DataType;
use mpsgraph::Graph;
use mpsgraph_tools::prelude::*;
use objc2_foundation::NSNumber;

fn main() {
    // Create graph and input tensors
    let graph = Graph::new();

    // Create a shape using NSNumber values
    let numbers = [NSNumber::new_usize(2), NSNumber::new_usize(3)];
    let number_refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();
    let shape = Shape::from_slice(&number_refs);

    // Create input tensors using the enhanced Tensor type
    let a = Tensor::placeholder(&graph, DataType::Float32, &shape);
    let b = Tensor::placeholder(&graph, DataType::Float32, &shape);

    println!("Demonstrating enhanced Tensor type with operator overloading:");
    println!("Shape = [2, 3]");

    // Example 1: Basic patterns for addition
    println!("\nExample 1: Basic addition patterns");
    let c1 = &a + &b;
    let c2 = a.clone() + &b;
    let c3 = &a + b.clone();
    let c4 = a.clone() + b.clone();
    println!("let c1 = &a + &b;        // References");
    println!("let c2 = a.clone() + &b; // Value + reference");
    println!("let c3 = &a + b.clone(); // Reference + value");
    println!("let c4 = a.clone() + b.clone(); // Values");
    println!("// You can also use a + b directly if a and b are moved:");
    println!("// let c5 = a + b;       // Direct value syntax (consumes values)");

    // Example 2: Scalar operations
    println!("\nExample 2: Scalar operations");
    let d = &c1 * 2.0;
    println!("let d = &c1 * 2.0;  // Scalar multiplication");

    // Example 3: Chaining operations with different operators
    println!("\nExample 3: Chaining operations");
    let e = &a + &b - &a;
    println!("let e = &a + &b - &a;  // Addition then subtraction");

    // Example 4: Using parentheses to control precedence
    println!("\nExample 4: Controlling precedence with parentheses");
    let f = (&a + &b) * &a;
    println!("let f = (&a + &b) * &a;  // Parentheses control order");

    // Example 5: More complex expressions
    println!("\nExample 5: More complex expressions");
    let g1 = (&a + &b) * (&a - &b) / (&a + 1.0);
    println!("let g1 = (&a + &b) * (&a - &b) / (&a + 1.0);  // With references");

    // Now demonstrate the same expression with direct value syntax
    let a_clone = a.clone();
    let b_clone = b.clone();
    let g2 = (a_clone + b_clone) * (a.clone() - b.clone()) / (a.clone() + 1.0);
    println!("let g2 = (a + b) * (a - b) / (a + 1.0);       // Direct syntax (requires clones)");

    // Example 6: Commutative scalar operations
    println!("\nExample 6: Commutative scalar operations");
    let h1 = &a * 3.0;
    let h2 = 3.0 * &a;
    println!("let h1 = &a * 3.0;  // Tensor * scalar");
    println!("let h2 = 3.0 * &a;  // scalar * Tensor (commutative)");

    // Example 7: Negation
    println!("\nExample 7: Negation");
    let i = -&a;
    println!("let i = -&a;  // Negation");

    // Example 8: Activation functions
    println!("\nExample 8: Activation functions");
    let j1 = a.sigmoid();
    let j2 = a.relu();
    let j3 = a.tanh();
    let j4 = a.silu();
    let j5 = a.gelu();
    println!("let j1 = a.sigmoid();  // Sigmoid activation");
    println!("let j2 = a.relu();     // ReLU activation");
    println!("let j3 = a.tanh();     // Tanh activation");
    println!("let j4 = a.silu();     // SiLU activation");
    println!("let j5 = a.gelu();     // GELU activation");

    // Example 9: Using with original mpsgraph API
    println!("\nExample 9: Interoperability with mpsgraph API");
    let k = Tensor(graph.add(&*a, &*b, Some("k")));
    println!("let k = Tensor(graph.add(&*a, &*b, Some(\"k\")));  // Wrap result from mpsgraph");

    println!("\nSuccessfully demonstrated enhanced Tensor with operator overloading:");
    println!("1. Basic pattern: let c = &a + &b");
    println!("2. Scalar operations: let d = &c * 2.0");
    println!("3. Chaining operations: let e = &a + &b - &a");
    println!("4. Parentheses for precedence: let f = (&a + &b) * &a");
    println!("5. Complex expressions: let g = (&a + &b) * (&a - &b) / (&a + 1.0)");
    println!("6. Commutative operations: &a * 3.0 and 3.0 * &a");
    println!("7. Negation: -&a");
    println!("8. Activation functions: sigmoid, relu, tanh, silu, gelu");
    println!("9. Interoperability with mpsgraph API");

    // List all the tensors we created to demonstrate the operations
    println!("\nCreated tensors:");
    let operation_tensors = [
        ("c1 (&a + &b)", &c1),
        ("c2 (a.clone() + &b)", &c2),
        ("c3 (&a + b.clone())", &c3),
        ("c4 (a.clone() + b.clone())", &c4),
        ("d (&c1 * 2.0)", &d),
        ("e (&a + &b - &a)", &e),
        ("f ((&a + &b) * &a)", &f),
        ("g1 (complex with refs)", &g1),
        ("g2 (complex without refs)", &g2),
        ("h1 (&a * 3.0)", &h1),
        ("h2 (3.0 * &a)", &h2),
        ("i (-&a)", &i),
        ("j1 (sigmoid)", &j1),
        ("j2 (relu)", &j2),
        ("j3 (tanh)", &j3),
        ("j4 (silu)", &j4),
        ("j5 (gelu)", &j5),
        ("k (mpsgraph wrapper)", &k),
    ];

    for (name, tensor) in operation_tensors.iter() {
        println!("{}: {:?}", name, tensor);
    }

    println!("\nKey Benefits:");
    println!("1. Simple, clean syntax: &a + &b instead of a.add(&b, None)");
    println!("2. Direct value syntax: a + b (consumes values)");
    println!("3. Mixed ownership: a.clone() + &b, &a + b.clone()");
    println!("4. Full support for complex expressions");
    println!("5. Scalar operations: &a * 2.0");
    println!("6. Commutative operations: 3.0 * &a");
    println!("7. Built-in activation functions");
    println!("8. Seamless interoperability with original mpsgraph API");
    println!("9. No need for feature flags or compilation options");
}
