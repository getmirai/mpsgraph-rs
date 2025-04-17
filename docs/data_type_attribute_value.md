# Using DataTypeAttributeValue in mpsgraph

The `DataTypeAttributeValue` class represents data type attributes for operations in the MPSGraph framework. This document explains how to use it effectively in your Rust code.

## Introduction

In Metal Performance Shaders Graph (MPSGraph), operations often require data type specifications as attributes. The `DataTypeAttributeValue` class provides a way to represent these data types in a type-safe manner, allowing you to specify the data types for tensors and operations.

## Basic Usage

### Creating DataTypeAttributeValue

There are several ways to create a `DataTypeAttributeValue`:

```rust
use mpsgraph::{DataTypeAttributeValue, DataType, ShapedType};
use objc2::rc::Retained;

// Create with a specific data type
let attr = DataTypeAttributeValue::with_data_type(DataType::Float32);

// Using convenience factory methods
let float32_attr = DataTypeAttributeValue::float32();
let float16_attr = DataTypeAttributeValue::float16();
let int32_attr = DataTypeAttributeValue::int32();
let int8_attr = DataTypeAttributeValue::int8();
let bool_attr = DataTypeAttributeValue::bool();

// Using the Default implementation
let default_attr: Retained<DataTypeAttributeValue> = mpsgraph::CustomDefault::custom_default();
```

### Creating with a ShapedType

You can also create a `DataTypeAttributeValue` from a `ShapedType`:

```rust
use mpsgraph::{DataTypeAttributeValue, ShapedType, ShapeHelper, DataType};

// Create a shape
let shape = ShapeHelper::tensor3d(2, 3, 4);

// Create a ShapedType with the shape and data type
let shaped_type = ShapedType::new(&shape, DataType::Float32);

// Create a DataTypeAttributeValue from the ShapedType
let attr = DataTypeAttributeValue::with_shaped_type(&shaped_type);
```

## Querying DataTypeAttributeValue

You can query properties of a `DataTypeAttributeValue`:

```rust
// Get the data type
let data_type = attr.data_type();

// Get the shaped type (if available)
if let Some(shaped_type) = attr.shaped_type() {
    let shape = shaped_type.shape();
    let rank = shaped_type.rank();
    let dt = shaped_type.data_type();
}

// Check if it's a data type attribute or a shaped type attribute
if attr.is_data_type() {
    // This is a simple data type attribute
} else if attr.is_shaped_type() {
    // This is a shaped type attribute
}

// Check the category of data type
if attr.is_floating_point() {
    // Float32 or Float16
} else if attr.is_integer() {
    // Int32, Int16, Int8, or Uint8
} else if attr.is_boolean() {
    // Boolean type
}
```

## Usage with Graph Operations

When working with graph operations that require data type attributes, you can use `DataTypeAttributeValue`:

```rust
use mpsgraph::{Graph, Tensor, DataTypeAttributeValue, DataType, ShapeHelper};

// Create a graph
let graph = Graph::new();

// Create a shape
let shape = ShapeHelper::tensor3d(2, 3, 4);

// Create a tensor
let tensor = graph.placeholder(DataType::Float32, &shape).unwrap();

// Create a data type attribute
let data_type_attr = DataTypeAttributeValue::float32();

// Use the attribute in operations
// (Note: Specific operation methods will depend on the actual API)
// let result = graph.some_operation_with_data_type(tensor, &data_type_attr);
```

## Example: Type Conversion Operations

Here's a hypothetical example of using `DataTypeAttributeValue` with a type conversion operation:

```rust
use mpsgraph::{Graph, Tensor, DataTypeAttributeValue, DataType, ShapeHelper};

// Create a graph
let graph = Graph::new();

// Create an input tensor
let shape = ShapeHelper::vector(4);
let input = graph.placeholder(DataType::Float32, &shape).unwrap();

// Create a data type attribute for the target type
let int32_type = DataTypeAttributeValue::int32();

// Convert the tensor to the target type
// (Note: This is a hypothetical API method)
// let converted = graph.cast(input, &int32_type).unwrap();
```

## Best Practices

1. Use factory methods (`float32()`, `int32()`, etc.) for clearer code when working with simple data types.
2. When working with shaped types, use `with_shaped_type()` to maintain shape information.
3. Always check the return value of `shaped_type()` as it may be None.
4. Prefer type-checking methods (`is_floating_point()`, `is_integer()`, etc.) for robust code that can handle various data types.

## Related Types

- `Type`: Base class for types used on tensors in MPSGraph.
- `ShapedType`: Subclass of `Type` that includes shape and data type information.
- `DataType`: Enum representing the different data types available in MPSGraph.
- `ShapeDescriptor`: Structure representing tensor shapes with data type information.