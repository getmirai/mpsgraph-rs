use crate::Tensor;

pub enum ScalarOrTensor<'a> {
    Scalar(f64),
    Tensor(&'a Tensor),
}
