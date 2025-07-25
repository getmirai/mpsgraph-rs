use crate::{GraphScalar, Tensor};

#[derive(Debug, Clone, Copy)]
pub enum ScalarOrTensor<'a, T: GraphScalar> {
    Scalar(T),
    Tensor(&'a Tensor),
}
