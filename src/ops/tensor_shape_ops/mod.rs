mod concat_ops;
mod pad_ops;
mod slice_ops;
mod space_to_batch;
mod space_to_depth_ops;
mod tile_ops;

pub use concat_ops::*;
pub use pad_ops::*;
pub use slice_ops::*;
pub use space_to_batch::*;
pub use space_to_depth_ops::*;
pub use tile_ops::*;

use crate::{Graph, ShapeOrTensor, ShapedType, Tensor};
use objc2::{extern_methods, msg_send, rc::Retained};
use objc2_foundation::{NSArray, NSNumber, NSString};

/// TensorShapeOps.
impl Graph {
    extern_methods!(
        #[cfg(feature = "MPSGraphTensor")]
        /// Creates a reverse operation and returns the result tensor.
        ///
        /// Reverses a tensor on given axes.
        /// Semantics based on [TensorFlow reverse op](https://www.tensorflow.org/api_docs/python/tf/reverse).
        ///
        /// - Parameters:
        /// - tensor: The tensor to be reversed.
        /// - axesTensor: A tensor that specifies axes to be reversed (Axes must be unique and within normal axis range).
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object.
        #[unsafe(method(reverseTensor:axesTensor:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn reverseTensor_axesTensor_name(
            &self,
            tensor: &MPSGraphTensor,
            axes_tensor: &MPSGraphTensor,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;

        #[cfg(feature = "MPSGraphTensor")]
        /// Creates a reverse operation and returns the result tensor.
        ///
        /// Reverses a tensor on given axes.
        /// Semantics based on [TensorFlow reverse op](https://www.tensorflow.org/api_docs/python/tf/reverse).
        ///
        /// - Parameters:
        /// - tensor: The tensor to be reversed.
        /// - axes: A tensor that specifies axes to be reversed (Axes must be unique and within normal axis range).
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object.
        #[unsafe(method(reverseTensor:axes:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn reverseTensor_axes_name(
            &self,
            tensor: &MPSGraphTensor,
            axes: &NSArray<NSNumber>,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;

        #[cfg(feature = "MPSGraphTensor")]
        /// Creates a reverse operation and returns the result tensor.
        ///
        /// Reverses a tensor on all axes.
        /// Semantics based on [TensorFlow reverse op](https://www.tensorflow.org/api_docs/python/tf/reverse).
        ///
        /// - Parameters:
        /// - tensor: The tensor to be reversed.
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object.
        #[unsafe(method(reverseTensor:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn reverseTensor_name(
            &self,
            tensor: &MPSGraphTensor,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;

        #[cfg(feature = "MPSGraphTensor")]
        /// Creates a flatten2D operation and returns the result tensor.
        ///
        /// Flattens dimensions before `axis` to `result[0]` and dimensions starting
        /// from `axis` to `result[1]` and returns a rank-2 tensor as result.
        ///
        /// - Parameters:
        /// - tensor: The tensor to be flattened.
        /// - axis: The axis around which to flatten.
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object.
        #[unsafe(method(flatten2DTensor:axis:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn flatten2DTensor_axis_name(
            &self,
            tensor: &MPSGraphTensor,
            axis: NSInteger,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;

        #[cfg(feature = "MPSGraphTensor")]
        /// Creates a flatten2D operation and returns the result tensor.
        ///
        /// Flattens dimensions before `axis` to `result[0]` and dimensions starting
        /// from `axis` to `result[1]` and returns a rank-2 tensor as result.
        ///
        /// - Parameters:
        /// - tensor: The tensor to be flattened.
        /// - axisTensor: A scalar tensor that contains the axis around which to flatten.
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object.
        #[unsafe(method(flatten2DTensor:axisTensor:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn flatten2DTensor_axisTensor_name(
            &self,
            tensor: &MPSGraphTensor,
            axis_tensor: &MPSGraphTensor,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;

        #[cfg(all(
            feature = "MPSGraphTensor",
            feature = "objc2-metal-performance-shaders"
        ))]
        /// Creates a broadcast operation and returns the result tensor.
        ///
        /// Broadcasts values inside the tensor, starting from the trailing dimensions, to give it the correct shape.
        /// This is equivalent to the broadcasting for arithmetic operations when operands have different shapes.
        ///
        /// - Parameters:
        /// - tensor: The tensor to be broadcasted
        /// - shape: The shape of the result tensor.
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object
        #[unsafe(method(broadcastTensor:toShape:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn broadcastTensor_toShape_name(
            &self,
            tensor: &MPSGraphTensor,
            shape: &MPSShape,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;

        #[cfg(feature = "MPSGraphTensor")]
        /// Creates a broadcast operation and returns the result tensor.
        ///
        /// Broadcasts values inside the tensor, starting from the trailing dimensions, to give it the correct shape.
        /// This is equivalent to the broadcasting for arithmetic operations when operands have different shapes.
        ///
        /// - Parameters:
        /// - tensor: The Tensor to be broadcasted.
        /// - shapeTensor: A rank-1 tensor of type `MPSDataTypeInt32` or `MPSDataTypeInt64` that defines the shape of the result tensor.
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object.
        #[unsafe(method(broadcastTensor:toShapeTensor:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn broadcastTensor_toShapeTensor_name(
            &self,
            tensor: &MPSGraphTensor,
            shape_tensor: &MPSGraphTensor,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;

        #[cfg(feature = "MPSGraphTensor")]
        /// Creates a shape-of operation and returns the result tensor.
        ///
        /// Returns a rank-1 tensor of type `MPSDataTypeInt32` with the values of the static shape of the input tensor.
        ///
        /// - Parameters:
        /// - tensor: The input tensor.
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object.
        #[unsafe(method(shapeOfTensor:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn shapeOfTensor_name(
            &self,
            tensor: &MPSGraphTensor,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;

        #[cfg(all(
            feature = "MPSGraphTensor",
            feature = "objc2-metal-performance-shaders"
        ))]
        /// Creates a cast operation and returns the result tensor.
        ///
        /// Returns the input tensor casted to the specied data type.
        ///
        /// - Parameters:
        /// - tensor: The input tensor.
        /// - type: The datatype to which MPSGraph casts the input.
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object.
        #[unsafe(method(castTensor:toType:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn castTensor_toType_name(
            &self,
            tensor: &MPSGraphTensor,
            r#type: MPSDataType,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;

        #[cfg(all(
            feature = "MPSGraphTensor",
            feature = "objc2-metal-performance-shaders"
        ))]
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
        #[unsafe(method(reinterpretCastTensor:toType:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn reinterpretCastTensor_toType_name(
            &self,
            tensor: &MPSGraphTensor,
            r#type: MPSDataType,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;

        #[cfg(feature = "MPSGraphTensor")]
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
        #[unsafe(method(stackTensors:axis:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn stackTensors_axis_name(
            &self,
            input_tensors: &NSArray<MPSGraphTensor>,
            axis: NSInteger,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;

        #[cfg(feature = "MPSGraphTensor")]
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
        #[unsafe(method(splitTensor:splitSizes:axis:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn splitTensor_splitSizes_axis_name(
            &self,
            tensor: &MPSGraphTensor,
            split_sizes: &NSArray<NSNumber>,
            axis: NSInteger,
            name: Option<&NSString>,
        ) -> Retained<NSArray<MPSGraphTensor>>;

        #[cfg(feature = "MPSGraphTensor")]
        /// Creates a split operation and returns the result tensor.
        ///
        /// Splits the input tensor along `axis` into multiple result tensors of size determined by `splitSizesTensor`.
        /// Requires that the sum of the elements of `splitSizesTensor` is equal to the lenth of the input along `axis`.
        ///
        /// - Parameters:
        /// - tensor: The input tensor
        /// - splitSizesTensor: The lengths of the result tensors along the split axis.
        /// - axis: The dimension along which MPSGraph splits the input tensor.
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object
        #[unsafe(method(splitTensor:splitSizesTensor:axis:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn splitTensor_splitSizesTensor_axis_name(
            &self,
            tensor: &MPSGraphTensor,
            split_sizes_tensor: &MPSGraphTensor,
            axis: NSInteger,
            name: Option<&NSString>,
        ) -> Retained<NSArray<MPSGraphTensor>>;

        #[cfg(feature = "MPSGraphTensor")]
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
        #[unsafe(method(splitTensor:numSplits:axis:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn splitTensor_numSplits_axis_name(
            &self,
            tensor: &MPSGraphTensor,
            num_splits: NSUInteger,
            axis: NSInteger,
            name: Option<&NSString>,
        ) -> Retained<NSArray<MPSGraphTensor>>;

        #[cfg(feature = "MPSGraphTensor")]
        /// Creates a squeeze operation and returns the result tensor.
        ///
        /// Squeezes the tensor, removing all dimensions with size 1.
        ///
        /// - Parameters:
        /// - tensor: The input tensor.
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object.
        #[unsafe(method(squeezeTensor:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn squeezeTensor_name(
            &self,
            tensor: &MPSGraphTensor,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;

        #[cfg(feature = "MPSGraphTensor")]
        /// Creates a squeeze operation and returns the result tensor.
        ///
        /// Squeezes the tensor, removing a dimension with size 1 at the specified axis.
        /// The size of the input tensor must be 1 at the specified axis.
        ///
        /// - Parameters:
        /// - tensor: The input tensor.
        /// - axis: The axis to squeeze.
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object.
        #[unsafe(method(squeezeTensor:axis:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn squeezeTensor_axis_name(
            &self,
            tensor: &MPSGraphTensor,
            axis: NSInteger,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;

        #[cfg(feature = "MPSGraphTensor")]
        /// Creates a squeeze operation and returns the result tensor.
        ///
        /// Squeezes the tensor, removing dimensions with size 1 at specified axes.
        /// The size of the input tensor must be 1 at all specified axes.
        ///
        /// - Parameters:
        /// - tensor: The input tensor.
        /// - axes: The axes to squeeze.
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object
        #[unsafe(method(squeezeTensor:axes:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn squeezeTensor_axes_name(
            &self,
            tensor: &MPSGraphTensor,
            axes: &NSArray<NSNumber>,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;

        #[cfg(feature = "MPSGraphTensor")]
        /// Creates a squeeze operation and returns the result tensor.
        ///
        /// Squeezes the tensor, removing dimensions with size 1 at specified axes.
        /// The size of the input tensor must be 1 at all specified axes.
        ///
        /// - Parameters:
        /// - tensor: The input tensor.
        /// - axesTensor: The tensor containing the axes to squeeze.
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object
        #[unsafe(method(squeezeTensor:axesTensor:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn squeezeTensor_axesTensor_name(
            &self,
            tensor: &MPSGraphTensor,
            axes_tensor: &MPSGraphTensor,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;

        #[cfg(feature = "MPSGraphTensor")]
        /// Creates an expand-dimensions operation and returns the result tensor.
        ///
        /// Expands the tensor, inserting a dimension with size 1 at the specified axis.
        ///
        /// - Parameters:
        /// - tensor: The input tensor.
        /// - axis: The axis to expand.
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object.
        #[unsafe(method(expandDimsOfTensor:axis:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn expandDimsOfTensor_axis_name(
            &self,
            tensor: &MPSGraphTensor,
            axis: NSInteger,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;

        #[cfg(feature = "MPSGraphTensor")]
        /// Creates an expand-dimensions operation and returns the result tensor.
        ///
        /// Expands the tensor, inserting dimensions with size 1 at specified axes.
        ///
        /// - Parameters:
        /// - tensor: The input tensor.
        /// - axes: The axes to expand.
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object.
        #[unsafe(method(expandDimsOfTensor:axes:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn expandDimsOfTensor_axes_name(
            &self,
            tensor: &MPSGraphTensor,
            axes: &NSArray<NSNumber>,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;

        #[cfg(feature = "MPSGraphTensor")]
        /// Creates an expand-dimensions operation and returns the result tensor.
        ///
        /// Expands the tensor, inserting dimensions with size 1 at specified axes.
        ///
        /// - Parameters:
        /// - tensor: The input tensor.
        /// - axesTensor: The tensor containing the axes to expand.
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object.
        #[unsafe(method(expandDimsOfTensor:axesTensor:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn expandDimsOfTensor_axesTensor_name(
            &self,
            tensor: &MPSGraphTensor,
            axes_tensor: &MPSGraphTensor,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;

        #[cfg(all(
            feature = "MPSGraphTensor",
            feature = "objc2-metal-performance-shaders"
        ))]
        /// Creates a get-coordindate operation and returns the result tensor.
        ///
        /// Creates a tensor of specified shape with value at index `[i_0, i_1, ... , i_N] = i_axis`
        /// For example,
        /// ```md
        /// coordinateAlongAxis(0, withShape=[5]) = [0, 1, 2, 3, 4]
        /// coordinateAlongAxis(0, withShape=[3,2]) = [[0, 0],
        /// [1, 1],
        /// [2, 2]]
        /// ```
        ///
        /// - Parameters:
        /// - axis: The coordinate axis an element's value is set to. Negative values wrap around.
        /// - shape: The shape of the result tensor.
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object.
        #[unsafe(method(coordinateAlongAxis:withShape:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn coordinateAlongAxis_withShape_name(
            &self,
            axis: NSInteger,
            shape: &MPSShape,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;

        #[cfg(all(
            feature = "MPSGraphTensor",
            feature = "objc2-metal-performance-shaders"
        ))]
        /// Creates a get-coordindate operation and returns the result tensor.
        ///
        /// See ``MPSGraph/coordinateAlongAxis:withShape:name:``.
        ///
        /// - Parameters:
        /// - axisTensor: A Scalar tensor of type `MPSDataTypeInt32`, that specifies the coordinate axis an element's value is set to. Negative values wrap around.
        /// - shape: The shape of the result tensor.
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object.
        #[unsafe(method(coordinateAlongAxisTensor:withShape:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn coordinateAlongAxisTensor_withShape_name(
            &self,
            axis_tensor: &MPSGraphTensor,
            shape: &MPSShape,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;

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
}
