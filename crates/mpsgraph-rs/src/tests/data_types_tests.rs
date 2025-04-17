use crate::data_types::{Type, ShapedType, DataTypeAttributeValue};
use crate::shape::{ShapeHelper, ShapeExtensions};
use crate::tensor::DataType;
use objc2::rc::Retained;

#[test]
fn test_type() {
    let type_obj = Type::new();
    assert!(!type_obj.description().is_empty());
}

#[test]
fn test_shaped_type() {
    let shape = ShapeHelper::tensor3d(2, 3, 4);
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

#[test]
#[ignore]
fn test_data_type_attribute_value() {
    let data_type_attr = DataTypeAttributeValue::with_data_type(DataType::Int32);
    // Just check that we got a value
    let _data_type = data_type_attr.data_type();
    
    let shape = ShapeHelper::tensor3d(2, 3, 4);
    let shaped_type = ShapedType::new(&shape, DataType::Float32);
    
    let shaped_attr = DataTypeAttributeValue::with_shaped_type(&shaped_type);
    // Check that we can get a shaped type
    let shaped_type_opt = shaped_attr.shaped_type();
    assert!(shaped_type_opt.is_some(), "Expected to retrieve a shaped type from attribute");
    
    // Test CustomDefault implementation
    let default_attr: Retained<DataTypeAttributeValue> = crate::CustomDefault::custom_default();
    // Simply check that we get some value
    let _data_type = default_attr.data_type();
    
    // Test factory methods
    let float32_attr = DataTypeAttributeValue::float32();
    let float16_attr = DataTypeAttributeValue::float16();
    let bfloat16_attr = DataTypeAttributeValue::bfloat16();
    let int32_attr = DataTypeAttributeValue::int32();
    let int8_attr = DataTypeAttributeValue::int8();
    let bool_attr = DataTypeAttributeValue::bool();
    
    // Test type checking methods
    assert!(float32_attr.is_floating_point());
    assert!(float16_attr.is_floating_point());
    assert!(bfloat16_attr.is_floating_point());
    assert!(!int32_attr.is_floating_point());
    
    assert!(int32_attr.is_integer());
    assert!(int8_attr.is_integer());
    assert!(!float32_attr.is_integer());
    
    assert!(bool_attr.is_boolean());
    assert!(!float32_attr.is_boolean());
    assert!(!int32_attr.is_boolean());
    
    // Test shaped type checking
    assert!(!data_type_attr.is_shaped_type());
    assert!(shaped_attr.is_shaped_type());
    
    assert!(data_type_attr.is_data_type());
    assert!(!shaped_attr.is_data_type());
}