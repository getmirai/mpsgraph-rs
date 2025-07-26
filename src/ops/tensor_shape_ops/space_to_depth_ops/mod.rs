mod scalars_or_tensors;

pub use scalars_or_tensors::WidthHeightDepthAxisScalarsOrTensors;

use crate::{Graph, ShapeOrTensor, ShapedType, Tensor};
use objc2::{extern_methods, msg_send, rc::Retained};
use objc2_foundation::{NSArray, NSNumber, NSString};

impl Graph {
    /// Creates a space-to-depth2D operation and returns the result tensor.
    ///
    /// This operation outputs a copy of the `input` tensor, where values from the
    /// `widthAxis` and `heightAxis` dimensions are moved in spatial blocks of size
    /// `blockSize` to the `depthAxis` dimension. Use the `usePixelShuffleOrder` parameter
    /// to control how the data within spatial blocks is ordered in the
    /// `depthAxis` dimension: with `usePixelShuffleOrder=YES` MPSGraph stores the
    /// values of the spatial blocks  contiguosly within the `depthAxis` dimension, whereas
    /// otherwise they are stored interleaved with existing values in the `depthAxis` dimension.
    /// This operation is the inverse of `MPSGraph/depthToSpace2DTensor:widthAxis:heightAxis:depthAxis:blockSize:usePixelShuffleOrder:name:`.
    /// - Parameters:
    /// - tensor: The input tensor.
    /// - axis: The axis that defines the fastest running dimension within the block, the 2nd fastest running dimension within the block, and the destination dimension, where to copy the blocks.
    /// - blockSize: The size of the square spatial sub-block.
    /// - usePixelShuffleOrder: A parameter that controls the layout of the sub-blocks within the depth dimension.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object
    pub fn space_to_depth_2d<'a>(
        &self,
        tensor: &Tensor,
        axis: WidthHeightDepthAxisScalarsOrTensors<'a>,
        block_size: u64,
        use_pixel_shuffle_order: bool,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match axis {
            WidthHeightDepthAxisScalarsOrTensors::Scalars {
                width_axis,
                height_axis,
                depth_axis,
            } => unsafe {
                msg_send![
                    self,
                    spaceToDepth2DTensor: tensor,
                    widthAxis: width_axis,
                    heightAxis: height_axis,
                    depthAxis: depth_axis,
                    blockSize: block_size,
                    usePixelShuffleOrder: use_pixel_shuffle_order,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            WidthHeightDepthAxisScalarsOrTensors::Tensors {
                width_axis_tensor,
                height_axis_tensor,
                depth_axis_tensor,
            } => unsafe {
                msg_send![
                    self,
                    spaceToDepth2DTensor: tensor,
                    widthAxisTensor: width_axis_tensor,
                    heightAxisTensor: height_axis_tensor,
                    depthAxisTensor: depth_axis_tensor,
                    blockSize: block_size,
                    usePixelShuffleOrder: use_pixel_shuffle_order,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }

    /// Creates a depth-to-space2D operation and returns the result tensor.
    ///
    /// This operation outputs a copy of the input tensor, where values from the
    /// `depthAxis` dimension are moved in spatial blocks of size `blockSize` to the
    /// `heightAxis` and `widthAxis` dimensions.  Use the `usePixelShuffleOrder` parameter
    /// to control how the data within spatial blocks is ordered in the
    /// `depthAxis` dimension: with `usePixelShuffleOrder = YES` MPSGraph stores the values
    /// of the spatial block contiguosly within the `depthAxis` dimension, whereas
    /// without it they are stored interleaved with existing values in the `depthAxisTensor` dimension.
    /// This operation is the inverse of
    /// ``MPSGraph/spaceToDepth2DTensor:widthAxis:heightAxis:depthAxis:blockSize:usePixelShuffleOrder:name:``.
    ///
    /// - Parameters:
    /// - tensor: The input tensor.
    /// - axis: The axis that defines the fastest running dimension within the block, the 2nd fastest running dimension within the block, and the destination dimension, where to copy the blocks.
    /// - blockSize: The size of the square spatial sub-block.
    /// - usePixelShuffleOrder: A parameter that controls the layout of the sub-blocks within the depth dimension.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object.
    pub fn depth_to_space_2d<'a>(
        &self,
        tensor: &Tensor,
        axis: WidthHeightDepthAxisScalarsOrTensors<'a>,
        block_size: u64,
        use_pixel_shuffle_order: bool,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match axis {
            WidthHeightDepthAxisScalarsOrTensors::Scalars {
                width_axis,
                height_axis,
                depth_axis,
            } => unsafe {
                msg_send![
                    self,
                    depthToSpace2DTensor: tensor,
                    widthAxis: width_axis,
                    heightAxis: height_axis,
                    depthAxis: depth_axis,
                    blockSize: block_size,
                    usePixelShuffleOrder: use_pixel_shuffle_order,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
            WidthHeightDepthAxisScalarsOrTensors::Tensors {
                width_axis_tensor,
                height_axis_tensor,
                depth_axis_tensor,
            } => unsafe {
                msg_send![
                    self,
                    depthToSpace2DTensor: tensor,
                    widthAxisTensor: width_axis_tensor,
                    heightAxisTensor: height_axis_tensor,
                    depthAxisTensor: depth_axis_tensor,
                    blockSize: block_size,
                    usePixelShuffleOrder: use_pixel_shuffle_order,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }
}
