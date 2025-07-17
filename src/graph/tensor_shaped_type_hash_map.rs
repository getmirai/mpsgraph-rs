use std::collections::HashMap;

use crate::ShapedType;
use crate::Tensor;

pub type TensorShapedTypeHashMap<'a> = HashMap<&'a Tensor, &'a ShapedType>;
