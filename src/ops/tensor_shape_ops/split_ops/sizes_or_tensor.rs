use crate::Tensor;

pub enum SizesOrTensor<'a> {
    Sizes(&'a [isize]),
    Tensor(&'a Tensor),
}
