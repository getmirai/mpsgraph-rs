use crate::{Shape, Tensor};

pub enum ShapeOrTensor<'a> {
    Shape(&'a Shape),
    Tensor(&'a Tensor),
}
