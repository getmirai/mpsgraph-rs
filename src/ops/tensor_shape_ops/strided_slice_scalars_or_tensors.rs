use crate::Tensor;

pub enum StridedSliceScalarsOrTensors<'a> {
    Scalars {
        starts: &'a [u64],
        ends: &'a [u64],
        strides: &'a [u64],
    },
    Tensors {
        start_tensor: &'a Tensor,
        end_tensor: &'a Tensor,
        stride_tensor: &'a Tensor,
    },
}
