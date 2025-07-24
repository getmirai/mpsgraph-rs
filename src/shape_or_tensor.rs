use crate::{Shape, Tensor};

pub enum ShapeOrTensor<'a> {
    Shape(Shape),
    Tensor(&'a Tensor),
}
