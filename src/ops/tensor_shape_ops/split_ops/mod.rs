mod sizes_or_tensor;

pub use sizes_or_tensor::SizesOrTensor;

use crate::{ns_number_array_from_slice, Graph, Tensor};
use objc2::{extern_methods, msg_send, rc::Retained};
use objc2_foundation::{NSArray, NSString};

impl Graph {
    /// Creates a split operation and returns the result tensor.
    ///
    /// Splits the input tensor along `axis` into multiple result tensors of size determined by `splitSizes`.
    /// Requires that the sum of `splitSizes` is equal to the lenth of the input along `axis`.
    ///
    /// - Parameters:
    /// - tensor: The input tensor.
    /// - splitSizes: The lengths of the result tensors along the split axis.
    /// - axis: The dimension along which MPSGraph splits the input tensor.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object.
    pub fn split_tensor<'a>(
        &self,
        tensor: &Tensor,
        split_sizes: SizesOrTensor<'a>,
        axis: i64,
        name: Option<&str>,
    ) -> Box<[Retained<Tensor>]> {
        match split_sizes {
            SizesOrTensor::Sizes(split_sizes) => {
                let result: Retained<NSArray<Tensor>> = unsafe {
                    msg_send![
                        self,
                        splitTensor: tensor,
                        splitSizes: &*ns_number_array_from_slice(split_sizes),
                        axis: axis,
                        name: name.map(NSString::from_str).as_deref(),
                    ]
                };
                result.to_vec().into_boxed_slice()
            }
            SizesOrTensor::Tensor(split_sizes_tensor) => unsafe {
                let result: Retained<NSArray<Tensor>> = unsafe {
                    msg_send![
                        self,
                        splitTensor: tensor,
                        splitSizesTensor: split_sizes_tensor,
                        axis: axis,
                        name: name.map(NSString::from_str).as_deref(),
                    ]
                };
                result.to_vec().into_boxed_slice()
            },
        }
    }

    /// Creates a split operation and returns the result tensor.
    ///
    /// Splits the input tensor along `axis` into `numsplits` result tensors of equal size.
    /// Requires that the lenth of the input along `axis` is divisible by `numSplits`.
    ///
    /// - Parameters:
    /// - tensor: The input tensor.
    /// - numSplits: The number of result tensors to split to.
    /// - axis: The dimension along which MPSGraph splits the input tensor.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object.
    pub fn split_tensor_num_splits(
        &self,
        tensor: &Tensor,
        num_splits: u64,
        axis: i64,
        name: Option<&str>,
    ) -> Box<[Retained<Tensor>]> {
        let result: Retained<NSArray<Tensor>> = unsafe {
            msg_send![
                self,
                splitTensor: tensor,
                numSplits: num_splits,
                axis: axis,
                name: name.map(NSString::from_str).as_deref(),
            ]
        };
        result.to_vec().into_boxed_slice()
    }
}
