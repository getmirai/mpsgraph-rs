mod scalars_or_tensors;

pub use scalars_or_tensors::SpatialAxesBatchAxisBlockDimensionsScalarsOrTensors;

use crate::{Graph, ShapeOrTensor, ShapedType, Tensor};
use objc2::{extern_methods, msg_send, rc::Retained};
use objc2_foundation::{NSArray, NSNumber, NSString};

impl Graph {
    /// Creates a space-to-batch operation and returns the result tensor.
    ///
    /// This operation outputs a copy of the `input` tensor, where values from the
    /// `spatialAxes` (for `usePixelShuffleOrder=YES` 1,2 or 3 axes supported, otherwise
    /// limited only by `MPSNDArray` rank limitations) dimensions are moved in spatial blocks with
    /// rectangular size defined by `blockDimensions` to the `batchAxis` dimension.
    /// Use the `usePixelShuffleOrder` parameter  to control how the data within spatial blocks is ordered
    /// in the `batchAxis` dimension: with `usePixelShuffleOrder=YES` MPSGraph stores
    /// the values of the spatial blocks contiguosly within the `batchAxis` dimension, whereas
    /// otherwise they are stored interleaved with existing values in the `batchAxis` dimension.
    /// Note: This operation is the inverse of
    /// ``MPSGraph/batchToSpaceTensor:spatialAxes:batchAxis:blockDimensions:usePixelShuffleOrder:name:``.
    /// Note: This operation is a generalization of
    /// ``MPSGraph/spaceToDepth2DTensor:widthAxis:heightAxis:depthAxis:blockSize:usePixelShuffleOrder:name:``.
    ///
    /// - Parameters:
    /// - tensor: The input tensor.
    /// - spatial_axes_batch_axis_block_dimensions: The axes that define the dimensions containing the spatial blocks, the axis that defines the destination dimension, where to copy the blocks, and the size of the rectangular spatial sub-block.
    /// - usePixelShuffleOrder: A parameter that controls layout of the sub-blocks within the batch dimension.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object.
    pub fn space_to_batch<'a>(
        &self,
        tensor: &Tensor,
        spatial_axes_batch_axis_block_dimensions: SpatialAxesBatchAxisBlockDimensionsScalarsOrTensors<'a>,
        use_pixel_shuffle_order: bool,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match spatial_axes_batch_axis_block_dimensions {
            SpatialAxesBatchAxisBlockDimensionsScalarsOrTensors::Scalars {
                spatial_axes,
                batch_axis,
                block_dimensions,
            } => {
                let spatial_axes = spatial_axes
                    .iter()
                    .map(|x| NSNumber::new_u64(*x))
                    .collect::<Box<[Retained<NSNumber>]>>();
                let block_dimensions = block_dimensions
                    .iter()
                    .map(|x| NSNumber::new_u64(*x))
                    .collect::<Box<[Retained<NSNumber>]>>();
                let spatial_axes_array = NSArray::from_retained_slice(&spatial_axes);
                let block_dimensions_array = NSArray::from_retained_slice(&block_dimensions);
                unsafe {
                    msg_send![
                        self,
                        spaceToBatchTensor: tensor,
                        spatialAxes: &*spatial_axes_array,
                        batchAxis: batch_axis,
                        blockDimensions: &*block_dimensions_array,
                        usePixelShuffleOrder: use_pixel_shuffle_order,
                        name: name.map(NSString::from_str).as_deref(),
                    ]
                }
            }
            SpatialAxesBatchAxisBlockDimensionsScalarsOrTensors::Tensors {
                spatial_axes_tensor,
                batch_axis_tensor,
                block_dimensions_tensor,
            } => unsafe {
                msg_send![
                    self,
                    spaceToBatchTensor: tensor,
                    spatialAxesTensor: spatial_axes_tensor,
                    batchAxisTensor: batch_axis_tensor,
                    blockDimensionsTensor: block_dimensions_tensor,
                    usePixelShuffleOrder: use_pixel_shuffle_order,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }

    /// Creates a batch-to-space operation and returns the result tensor.
    ///
    /// This operation outputs a copy of the input tensor, where values from the
    /// `batchAxis` dimension are moved in spatial blocks of size `blockDimensions` to the
    /// `spatialAxes` dimensions (for `usePixelShuffleOrder=YES` 1,2 or 3 axes supported,
    /// otherwise limited only by `MPSNDArray` rank limitations).  Use the `usePixelShuffleOrder` parameter
    /// to control how the data within spatial blocks is ordered in the
    /// `batchAxis` dimension: with `usePixelShuffleOrder = YES` MPSGraph stores
    /// the values of the spatial block contiguosly within the `batchAxis` dimension whereas
    /// without it they are stored interleaved with existing values in the `batchAxis` dimension.
    /// Note: This operation is the inverse of
    /// ``MPSGraph/spaceToBatchTensor:spatialAxes:batchAxis:blockDimensions:usePixelShuffleOrder:name:``.
    /// Note: This operation is a generalization of
    /// ``MPSGraph/depthToSpace2DTensor:widthAxis:heightAxis:depthAxis:blockSize:usePixelShuffleOrder:name:``.
    ///
    /// - Parameters:
    /// - tensor: The input tensor.
    /// - spatial_axes_batch_axis_block_dimensions: The axes that define the dimensions containing the spatial blocks, the axis that defines the destination dimension, where to copy the blocks, and the size of the rectangular spatial sub-block.
    /// - usePixelShuffleOrder: A parameter that controls layout of the sub-blocks within the batch dimension.
    /// - name: The name for the operation.
    /// - Returns: A valid MPSGraphTensor object.
    pub fn batch_to_space<'a>(
        &self,
        tensor: &Tensor,
        spatial_axes_batch_axis_block_dimensions: SpatialAxesBatchAxisBlockDimensionsScalarsOrTensors<'a>,
        use_pixel_shuffle_order: bool,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        match spatial_axes_batch_axis_block_dimensions {
            SpatialAxesBatchAxisBlockDimensionsScalarsOrTensors::Scalars {
                spatial_axes,
                batch_axis,
                block_dimensions,
            } => {
                let spatial_axes = spatial_axes
                    .iter()
                    .map(|x| NSNumber::new_u64(*x))
                    .collect::<Box<[Retained<NSNumber>]>>();
                let block_dimensions = block_dimensions
                    .iter()
                    .map(|x| NSNumber::new_u64(*x))
                    .collect::<Box<[Retained<NSNumber>]>>();
                let spatial_axes_array = NSArray::from_retained_slice(&spatial_axes);
                let block_dimensions_array = NSArray::from_retained_slice(&block_dimensions);
                unsafe {
                    msg_send![
                        self,
                        batchToSpaceTensor: tensor,
                        spatialAxes: &*spatial_axes_array,
                        batchAxis: batch_axis,
                        blockDimensions: &*block_dimensions_array,
                        usePixelShuffleOrder: use_pixel_shuffle_order,
                        name: name.map(NSString::from_str).as_deref(),
                    ]
                }
            }
            SpatialAxesBatchAxisBlockDimensionsScalarsOrTensors::Tensors {
                spatial_axes_tensor,
                batch_axis_tensor,
                block_dimensions_tensor,
            } => unsafe {
                msg_send![
                    self,
                    batchToSpaceTensor: tensor,
                    spatialAxesTensor: spatial_axes_tensor,
                    batchAxisTensor: batch_axis_tensor,
                    blockDimensionsTensor: block_dimensions_tensor,
                    usePixelShuffleOrder: use_pixel_shuffle_order,
                    name: name.map(NSString::from_str).as_deref(),
                ]
            },
        }
    }
}
