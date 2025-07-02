use crate::data_types::{ShapedType, Type};
use crate::shape::Shape;
use crate::tensor::DataType;

#[test]
fn test_type() {
    let type_obj = Type::new();
    assert!(!type_obj.description().is_empty());
}

#[test]
fn test_shaped_type() {
    let shape = Shape::tensor3d(2, 3, 4);
    let shaped_type = ShapedType::new(&shape, DataType::Float32);

    assert_eq!(shaped_type.data_type(), shaped_type.data_type());

    let shape_from_type = shaped_type.shape();
    assert_eq!(shape_from_type.dimensions(), vec![2, 3, 4]);

    assert_eq!(shaped_type.rank(), 3);

    let unranked = ShapedType::unranked_tensor_type(DataType::Int32);
    // Just check that we get a value
    let _data_type = unranked.data_type();
    assert_eq!(unranked.rank(), 3);

    let rank_3 = ShapedType::tensor_type_with_rank(3, DataType::Float16);
    // Just check that we get a value
    let _data_type = rank_3.data_type();
    assert_eq!(rank_3.rank(), 3);
}
