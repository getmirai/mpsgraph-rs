use crate::tensor::Tensor;

pub enum ScalarOrTensor<'a> {
    Scalar(f64),
    Tensor(&'a Tensor),
}
