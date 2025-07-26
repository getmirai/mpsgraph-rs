use crate::{Graph, ScalarOrTensor, Shape, Tensor};
use objc2::{msg_send, rc::Retained};
use objc2_foundation::NSString;

impl Graph {
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
    pub fn coordinate_along_axis<'a>(
        &self,
        axis: ScalarOrTensor<'a, i64>,
        shape: &Shape,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match axis {
            ScalarOrTensor::Scalar(axis) => unsafe {
                msg_send![
                    self,
                    coordinateAlongAxis: axis,
                    withShape: shape,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            ScalarOrTensor::Tensor(axis) => unsafe {
                msg_send![
                    self,
                    coordinateAlongAxisTensor: axis,
                    withShape: shape,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }

    /// Creates a get-coordindate operation and returns the result tensor.
    ///
    /// See ``coordinateAlongAxis:withShape:name:``.
    ///
    /// - Parameters:
    /// - axis: The coordinate axis an element's value is set to. Negative values wrap around.
    /// - shapeTensor: A rank-1 tensor of type `MPSDataTypeInt32` or `MPSDataTypeInt64` that defines the shape of the result tensor.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object.
    pub fn coordinate_along_axis_with_shape_tensor<'a>(
        &self,
        axis: ScalarOrTensor<'a, i64>,
        shape_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match axis {
            ScalarOrTensor::Scalar(axis) => unsafe {
                msg_send![
                    self,
                    coordinateAlongAxis: axis,
                    withShapeTensor: shape_tensor,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            ScalarOrTensor::Tensor(axis) => unsafe {
                msg_send![
                    self,
                    coordinateAlongAxisTensor: axis,
                    withShapeTensor: shape_tensor,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }
}
