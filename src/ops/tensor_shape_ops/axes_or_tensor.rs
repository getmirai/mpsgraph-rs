use crate::Tensor;

pub enum AxesOrTensor<'a> {
    Axes(&'a [isize]),
    Tensor(&'a Tensor),
}
