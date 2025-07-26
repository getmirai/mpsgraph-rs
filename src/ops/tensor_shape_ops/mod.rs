mod broadcast_ops;
mod concat_ops;
mod coordinate_ops;
mod expand_dims_ops;
mod flatten_ops;
mod pad_ops;
mod reverse_tensor_ops;
mod slice_ops;
mod space_to_batch;
mod space_to_depth_ops;
mod split_ops;
mod squeeze_ops;
mod tile_ops;

pub use broadcast_ops::*;
pub use concat_ops::*;
pub use coordinate_ops::*;
pub use expand_dims_ops::*;
pub use flatten_ops::*;
pub use pad_ops::*;
pub use reverse_tensor_ops::*;
pub use slice_ops::*;
pub use space_to_batch::*;
pub use space_to_depth_ops::*;
pub use split_ops::*;
pub use squeeze_ops::*;
pub use tile_ops::*;

use crate::{DataType, Graph, ShapeOrTensor, ShapedType, Tensor};
use objc2::{extern_methods, msg_send, rc::Retained};
use objc2_foundation::{NSArray, NSNumber, NSString};

/// TensorShapeOps.
impl Graph {
    extern_methods!(
        #[cfg(feature = "MPSGraphTensor")]
        /// Creates a get-coordindate operation and returns the result tensor.
        ///
        /// See ``coordinateAlongAxis:withShape:name:``.
        ///
        /// - Parameters:
        /// - axis: The coordinate axis an element's value is set to. Negative values wrap around.
        /// - shapeTensor: A rank-1 tensor of type `MPSDataTypeInt32` or `MPSDataTypeInt64` that defines the shape of the result tensor.
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object.
        #[unsafe(method(coordinateAlongAxis:withShapeTensor:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn coordinateAlongAxis_withShapeTensor_name(
            &self,
            axis: NSInteger,
            shape_tensor: &MPSGraphTensor,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;

        #[cfg(feature = "MPSGraphTensor")]
        /// Creates a get-coordindate operation and returns the result tensor.
        ///
        /// See ``coordinateAlongAxis:withShape:name:``.
        ///
        /// - Parameters:
        /// - axisTensor: A Scalar tensor of type `MPSDataTypeInt32`, that specifies the coordinate axis an element's value is set to. Negative values wrap around.
        /// - shapeTensor: A rank-1 tensor of type `MPSDataTypeInt32` or `MPSDataTypeInt64` that defines the shape of the result tensor.
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object.
        #[unsafe(method(coordinateAlongAxisTensor:withShapeTensor:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn coordinateAlongAxisTensor_withShapeTensor_name(
            &self,
            axis_tensor: &MPSGraphTensor,
            shape_tensor: &MPSGraphTensor,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;
    );
}

impl Graph {
    /// Creates a reshape operation and returns the result tensor.
    ///
    /// This operation reshapes the input tensor to the target shape.
    /// The shape must be compatible with the input tensor shape, specifically the volume of the input tensor has to match the volume defined by the shape.
    /// The shape is allowed to contain dynamic dimensions (-1) when the result type can be inferred unambiguously.
    ///
    /// - Parameters:
    /// - tensor: The tensor to be reshaped.
    /// - shape: The result tensor shape.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object.
    pub fn reshape<'a>(
        &self,
        tensor: &Tensor,
        shape: ShapeOrTensor<'a>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match shape {
            ShapeOrTensor::Shape(shape) => unsafe {
                msg_send![self, reshapeTensor: tensor, withShape: shape, name: name.map(NSString::from_str).as_deref()]
            },
            ShapeOrTensor::Tensor(shape_tensor) => unsafe {
                msg_send![self, reshapeTensor: tensor, withShapeTensor: shape_tensor, name: name.map(NSString::from_str).as_deref()]
            },
        }
    }

    /// Creates a permutation operation and returns the result tensor.
    ///
    /// Permutes the dimensions of the input tensor according to values in `permutation`.
    ///
    /// - Parameters:
    /// - tensor: The tensor to be permuted.
    /// - permutation: An array of numbers defining the permutation, must be of length `rank(tensor)` and define a valid permutation.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object.
    pub fn transpose(
        &self,
        tensor: &Tensor,
        permutation: &[u64],
        name: Option<&str>,
    ) -> Retained<Tensor> {
        let permutation = permutation
            .iter()
            .map(|x| NSNumber::new_u64(*x))
            .collect::<Box<[Retained<NSNumber>]>>();
        let permutation_array = NSArray::from_retained_slice(&permutation);
        unsafe {
            msg_send![
                self,
                transposeTensor: tensor,
                permutation: &*permutation_array,
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }

    /// Creates a shape-of operation and returns the result tensor.
    ///
    /// Returns a rank-1 tensor of type `MPSDataTypeInt32` with the values of the static shape of the input tensor.
    ///
    /// - Parameters:
    /// - tensor: The input tensor.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object.
    pub fn shape_of_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            msg_send![self, shapeOfTensor: tensor, name: name.map(NSString::from_str).as_deref()]
        }
    }

    /// Creates a cast operation and returns the result tensor.
    ///
    /// Returns the input tensor casted to the specied data type.
    ///
    /// - Parameters:
    /// - tensor: The input tensor.
    /// - type: The datatype to which MPSGraph casts the input.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object.
    pub fn cast_tensor(
        &self,
        tensor: &Tensor,
        r#type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![self, castTensor: tensor, toType: r#type, name: name.map(NSString::from_str).as_deref()]
        }
    }

    /// Creates a reinterpret cast operation and returns the result tensor.
    ///
    /// Returns input tensor (with element type `tensor_type`) reinterpreted to element type
    /// passed in with the last dimension scaled by `sizeof(tensor_type) / sizeof(type)`.
    /// This operation is endianness agnostic and MPSGraph reinterprets the data with the endianness of the
    /// system.
    ///
    /// - Parameters:
    /// - tensor: The input tensor.
    /// - type: The element type of the returned tensor.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object.
    pub fn reinterpret_cast(
        &self,
        tensor: &Tensor,
        r#type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            msg_send![self, reinterpretCastTensor: tensor, toType: r#type, name: name.map(NSString::from_str).as_deref()]
        }
    }

    /// Creates a stack operation and returns the result tensor.
    ///
    /// Stacks all input tensors along `axis` into a result tensor of `rank + 1`. Tensors must be broadcast
    /// compatible along all dimensions except `axis`, and have the same type.
    ///
    /// - Parameters:
    /// - inputTensors: The input tensors.
    /// - axis: The dimension to stack tensors into result. Must be in range: `-rank + 1
    /// <
    /// = dimension
    /// <
    /// rank + 1`.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object.
    pub fn stack_tensors(
        &self,
        input_tensors: &[&Tensor],
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        let input_tensors_array = NSArray::from_slice(input_tensors);
        unsafe {
            msg_send![
                self,
                stackTensors: &*input_tensors_array,
                axis: axis,
                name: name.map(NSString::from_str).as_deref(),
            ]
        }
    }
}
