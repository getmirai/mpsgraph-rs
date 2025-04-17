use mpsgraph::{
    data_types::{ShapedType, Type},
    DataType, ShapeHelper,
};

fn main() {
    // Create a basic type
    let graph_type = Type::new();
    println!("Basic graph type description: {}", graph_type.description());

    // Create a shape for our shaped type using helper method
    let shape_dimensions = [2, 3, 4];
    let shape = ShapeHelper::from_dimensions(&shape_dimensions);
    println!("Shape dimensions: {:?}", shape_dimensions);

    // Create a shaped type with the shape and data type
    let shaped_type = ShapedType::new(&shape, DataType::Float32);
    println!("Shaped type: {:?}", shaped_type);
    println!("  Data type: {:?}", shaped_type.data_type());
    println!("  Rank: {}", shaped_type.rank());
    println!("  Is ranked: {}", shaped_type.is_ranked());

    // Create a tensor type with rank
    let ranked_type = ShapedType::tensor_type_with_rank(3, DataType::Float16);
    println!("Ranked tensor type: {:?}", ranked_type);

    // Create an unranked tensor type
    let unranked_type = ShapedType::unranked_tensor_type(DataType::Int32);
    println!("Unranked tensor type: {:?}", unranked_type);

    println!("Type system test completed successfully!");
}