mod concat_ops;
mod slice_gradient_ops;
mod slice_ops;
mod slice_update_ops;
mod start_end_stride_scalars_or_tensors;
mod strided_slice_ops;

pub use concat_ops::*;
pub use slice_gradient_ops::*;
pub use slice_ops::*;
pub use slice_update_ops::*;
pub use strided_slice_ops::*;

use crate::{Graph, ShapeOrTensor, ShapedType, Tensor};
use objc2::{extern_methods, msg_send, rc::Retained};
use objc2_foundation::{NSArray, NSNumber, NSString};
use start_end_stride_scalars_or_tensors::StartEndStrideScalarsOrTensors;

/// TensorShapeOps.
impl Graph {
    extern_methods!(
        #[cfg(all(
            feature = "MPSGraphTensor",
            feature = "objc2-metal-performance-shaders"
        ))]
        /// Creates a tile operation and returns the result tensor.
        ///
        /// Creates a tensor which contains multiple copies of the input tensor along each dimension of the tensor.
        ///
        /// - Parameters:
        /// - tensor: The input tensor
        /// - multiplier: An array of numbers that specifies how many copies per dimension MPSGraph produces.
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object.
        #[unsafe(method(tileTensor:withMultiplier:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn tileTensor_withMultiplier_name(
            &self,
            tensor: &MPSGraphTensor,
            multiplier: &MPSShape,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;

        #[cfg(all(
            feature = "MPSGraphTensor",
            feature = "objc2-metal-performance-shaders"
        ))]
        /// Creates a tile gradient operation and returns the result tensor.
        ///
        /// - Parameters:
        /// - incomingGradientTensor: The input gradient tensor.
        /// - sourceTensor: The input tensor of the forward pass.
        /// - multiplier: An array of numbers that specifies how many copies per dimension MPSGraph produced in the forward pass.
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object.
        #[unsafe(method(tileGradientWithIncomingGradientTensor:sourceTensor:withMultiplier:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn tileGradientWithIncomingGradientTensor_sourceTensor_withMultiplier_name(
            &self,
            incoming_gradient_tensor: &MPSGraphTensor,
            source_tensor: &MPSGraphTensor,
            multiplier: &MPSShape,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;

        #[cfg(all(
            feature = "MPSGraphTensor",
            feature = "objc2-metal-performance-shaders"
        ))]
        /// Creates a padding operation and returns the result tensor.
        ///
        /// - Parameters:
        /// - tensor: The input tensor.
        /// - paddingMode: The parameter that defines the padding mode.
        /// - leftPadding: The parameter that defines how much padding the operation applies to the input tensor before each dimension - must be of size `rank(tensor)`.
        /// - rightPadding: The parameter that defines how much padding the operation applies to the input tensor after each dimension - must be of size `rank(tensor)`.
        /// - constantValue: The constant value the operation uses when `paddingMode = MPSGraphPaddingModeConstant`.
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object.
        #[unsafe(method(padTensor:withPaddingMode:leftPadding:rightPadding:constantValue:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn padTensor_withPaddingMode_leftPadding_rightPadding_constantValue_name(
            &self,
            tensor: &MPSGraphTensor,
            padding_mode: MPSGraphPaddingMode,
            left_padding: &MPSShape,
            right_padding: &MPSShape,
            constant_value: c_double,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;

        #[cfg(all(
            feature = "MPSGraphTensor",
            feature = "objc2-metal-performance-shaders"
        ))]
        /// Creates a padding gradient operation and returns the result tensor.
        ///
        /// - Parameters:
        /// - incomingGradientTensor: The input gradient tensor.
        /// - sourceTensor: The input tensor of the forward pass.
        /// - paddingMode: The parameter that defines the padding mode.
        /// - leftPadding: The parameter that defines how much padding the operation applies to the input tensor before each dimension - must be of size `rank(tensor)`.
        /// - rightPadding: The parameter that defines how much padding the operation applies to the input tensor after each dimension - must be of size `rank(tensor)`.
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object.
        #[unsafe(method(padGradientWithIncomingGradientTensor:sourceTensor:paddingMode:leftPadding:rightPadding:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn padGradientWithIncomingGradientTensor_sourceTensor_paddingMode_leftPadding_rightPadding_name(
            &self,
            incoming_gradient_tensor: &MPSGraphTensor,
            source_tensor: &MPSGraphTensor,
            padding_mode: MPSGraphPaddingMode,
            left_padding: &MPSShape,
            right_padding: &MPSShape,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;

        #[cfg(feature = "MPSGraphTensor")]
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
        /// - widthAxis: The axis that defines the fastest running dimension within the block.
        /// - heightAxis: The axis that defines the 2nd fastest running dimension within the block.
        /// - depthAxis: The axis that defines the destination dimension, where to copy the blocks.
        /// - blockSize: The size of the square spatial sub-block.
        /// - usePixelShuffleOrder: A parameter that controls the layout of the sub-blocks within the depth dimension.
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object
        #[unsafe(method(spaceToDepth2DTensor:widthAxis:heightAxis:depthAxis:blockSize:usePixelShuffleOrder:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn spaceToDepth2DTensor_widthAxis_heightAxis_depthAxis_blockSize_usePixelShuffleOrder_name(
            &self,
            tensor: &MPSGraphTensor,
            width_axis: NSUInteger,
            height_axis: NSUInteger,
            depth_axis: NSUInteger,
            block_size: NSUInteger,
            use_pixel_shuffle_order: bool,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;

        #[cfg(feature = "MPSGraphTensor")]
        /// Creates a space-to-depth2D operation and returns the result tensor.
        ///
        /// This operation outputs a copy of the `input` tensor, where values from the
        /// `widthAxisTensor` and `heightAxisTensor` dimensions are moved in spatial blocks of size
        /// `blockSize` to the `depthAxisTensor` dimension. Use the `usePixelShuffleOrder` parameter
        /// to control how the data within spatial blocks is ordered in the
        /// `depthAxisTensor` dimension: with `usePixelShuffleOrder=YES` MPSGraph stores the
        /// values of the spatial blocks  contiguosly within the `depthAxisTensor` dimension, whereas
        /// otherwise they are stored interleaved with existing values in the `depthAxisTensor` dimension.
        /// This operation is the inverse of ``MPSGraph/depthToSpace2DTensor:widthAxisTensor:heightAxisTensor:depthAxisTensor:blockSize:usePixelShuffleOrder:name:``.
        ///
        /// - Parameters:
        /// - tensor: The input tensor.
        /// - widthAxisTensor: A scalar tensor that contains the axis that defines the fastest running dimension within the block.
        /// - heightAxisTensor: A scalar tensor that contains the axis that defines the 2nd fastest running dimension within the block.
        /// - depthAxisTensor: A scalar tensor that contains the axis that defines the destination dimension, where to copy the blocks.
        /// - blockSize: The size of the square spatial sub-block.
        /// - usePixelShuffleOrder: A parameter that controls the layout of the sub-blocks within the depth dimension.
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object
        #[unsafe(method(spaceToDepth2DTensor:widthAxisTensor:heightAxisTensor:depthAxisTensor:blockSize:usePixelShuffleOrder:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn spaceToDepth2DTensor_widthAxisTensor_heightAxisTensor_depthAxisTensor_blockSize_usePixelShuffleOrder_name(
            &self,
            tensor: &MPSGraphTensor,
            width_axis_tensor: &MPSGraphTensor,
            height_axis_tensor: &MPSGraphTensor,
            depth_axis_tensor: &MPSGraphTensor,
            block_size: NSUInteger,
            use_pixel_shuffle_order: bool,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;

        #[cfg(feature = "MPSGraphTensor")]
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
        /// - widthAxis: The axis that defines the fastest running dimension within the block.
        /// - heightAxis: The axis that defines the 2nd fastest running dimension within the block.
        /// - depthAxis: The axis that defines the destination dimension, where to copy the blocks.
        /// - blockSize: The size of the square spatial sub-block.
        /// - usePixelShuffleOrder: A parameter that controls the layout of the sub-blocks within the depth dimension.
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object.
        #[unsafe(method(depthToSpace2DTensor:widthAxis:heightAxis:depthAxis:blockSize:usePixelShuffleOrder:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn depthToSpace2DTensor_widthAxis_heightAxis_depthAxis_blockSize_usePixelShuffleOrder_name(
            &self,
            tensor: &MPSGraphTensor,
            width_axis: NSUInteger,
            height_axis: NSUInteger,
            depth_axis: NSUInteger,
            block_size: NSUInteger,
            use_pixel_shuffle_order: bool,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;

        #[cfg(feature = "MPSGraphTensor")]
        /// Creates a depth-to-space2D operation and returns the result tensor.
        ///
        /// This operation outputs a copy of the input tensor, where values from the
        /// `depthAxisTensor` dimension are moved in spatial blocks of size `blockSize` to the
        /// `heightAxisTensor` and `widthAxisTensor` dimensions.  Use the `usePixelShuffleOrder` parameter
        /// to control how the data within spatial blocks is ordered in the
        /// `depthAxisTensor` dimension: with `usePixelShuffleOrder = YES` MPSGraph stores the values
        /// of the spatial block contiguosly within the `depthAxisTensor` dimension, whereas
        /// without it they are stored interleaved with existing values in the `depthAxisTensor` dimension.
        /// This operation is the inverse of ``MPSGraph/spaceToDepth2DTensor:widthAxisTensor:heightAxisTensor:depthAxisTensor:blockSize:usePixelShuffleOrder:name:``.
        ///
        /// - Parameters:
        /// - tensor: The input tensor.
        /// - widthAxisTensor: A scalar tensor that contains the axis that defines the fastest running dimension within the block.
        /// - heightAxisTensor: A scalar tensor that contains the axis that defines the 2nd fastest running dimension within the block.
        /// - depthAxisTensor: A scalar tensor that contains the axis that defines the destination dimension, where to copy the blocks.
        /// - blockSize: The size of the square spatial sub-block.
        /// - usePixelShuffleOrder: A parameter that controls the layout of the sub-blocks within the depth dimension.
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object.
        #[unsafe(method(depthToSpace2DTensor:widthAxisTensor:heightAxisTensor:depthAxisTensor:blockSize:usePixelShuffleOrder:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn depthToSpace2DTensor_widthAxisTensor_heightAxisTensor_depthAxisTensor_blockSize_usePixelShuffleOrder_name(
            &self,
            tensor: &MPSGraphTensor,
            width_axis_tensor: &MPSGraphTensor,
            height_axis_tensor: &MPSGraphTensor,
            depth_axis_tensor: &MPSGraphTensor,
            block_size: NSUInteger,
            use_pixel_shuffle_order: bool,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;

        #[cfg(feature = "MPSGraphTensor")]
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
        /// - spatialAxes: The axes that define the dimensions containing the spatial blocks.
        /// - batchAxis: The axis that defines the destination dimension, where to copy the blocks.
        /// - blockDimensions: An array of numbers that defines the size of the rectangular spatial sub-block.
        /// - usePixelShuffleOrder: A parameter that controls layout of the sub-blocks within the batch dimension.
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object.
        #[unsafe(method(spaceToBatchTensor:spatialAxes:batchAxis:blockDimensions:usePixelShuffleOrder:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn spaceToBatchTensor_spatialAxes_batchAxis_blockDimensions_usePixelShuffleOrder_name(
            &self,
            tensor: &MPSGraphTensor,
            spatial_axes: &NSArray<NSNumber>,
            batch_axis: NSInteger,
            block_dimensions: &NSArray<NSNumber>,
            use_pixel_shuffle_order: bool,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;

        #[cfg(feature = "MPSGraphTensor")]
        /// Creates a space-to-batch operation and returns the result tensor.
        ///
        /// This operation outputs a copy of the `input` tensor, where values from the
        /// `spatialAxesTensor` (for `usePixelShuffleOrder=YES` 1,2 or 3 axes supported, otherwise
        /// limited only by `MPSNDArray` rank limitations) dimensions are moved in spatial blocks with
        /// rectangular size defined by `blockDimensionsTensor` to the `batchAxisTensor` dimension.
        /// Use the `usePixelShuffleOrder` parameter  to control how the data within spatial blocks is ordered
        /// in the `batchAxisTensor` dimension: with `usePixelShuffleOrder=YES` MPSGraph stores
        /// the values of the spatial blocks contiguosly within the `batchAxisTensor` dimension, whereas
        /// otherwise they are stored interleaved with existing values in the `batchAxisTensor` dimension.
        /// Note: This operation is the inverse of
        /// ``MPSGraph/batchToSpaceTensor:spatialAxesTensor:batchAxisTensor:blockDimensionsTensor:usePixelShuffleOrder:name:``.
        /// Note: This operation is a generalization of
        /// ``MPSGraph/spaceToDepth2DTensor:widthAxisTensor:heightAxisTensor:depthAxisTensor:blockSize:usePixelShuffleOrder:name:``.
        ///
        /// - Parameters:
        /// - tensor: The input tensor.
        /// - spatialAxesTensor: A tensor that contains the axes that define the dimensions containing the spatial blocks.
        /// - batchAxisTensor: A tensor that contains the axis that defines the destination dimension, where to copy the blocks.
        /// - blockDimensionsTensor: A tensor that defines the size of the rectangular spatial sub-block.
        /// - usePixelShuffleOrder: A parameter that controls layout of the sub-blocks within the batch dimension.
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object.
        #[unsafe(method(spaceToBatchTensor:spatialAxesTensor:batchAxisTensor:blockDimensionsTensor:usePixelShuffleOrder:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn spaceToBatchTensor_spatialAxesTensor_batchAxisTensor_blockDimensionsTensor_usePixelShuffleOrder_name(
            &self,
            tensor: &MPSGraphTensor,
            spatial_axes_tensor: &MPSGraphTensor,
            batch_axis_tensor: &MPSGraphTensor,
            block_dimensions_tensor: &MPSGraphTensor,
            use_pixel_shuffle_order: bool,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;

        #[cfg(feature = "MPSGraphTensor")]
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
        /// - spatialAxes: The axes that define the dimensions containing the spatial blocks.
        /// - batchAxis: The axis that defines the destination dimension, where to copy the blocks.
        /// - blockDimensions: An array of numbers that defines the size of the rectangular spatial sub-block.
        /// - usePixelShuffleOrder: A parameter that controls layout of the sub-blocks within the batch dimension.
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object.
        #[unsafe(method(batchToSpaceTensor:spatialAxes:batchAxis:blockDimensions:usePixelShuffleOrder:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn batchToSpaceTensor_spatialAxes_batchAxis_blockDimensions_usePixelShuffleOrder_name(
            &self,
            tensor: &MPSGraphTensor,
            spatial_axes: &NSArray<NSNumber>,
            batch_axis: NSInteger,
            block_dimensions: &NSArray<NSNumber>,
            use_pixel_shuffle_order: bool,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;

        #[cfg(feature = "MPSGraphTensor")]
        /// Creates a batch-to-space operation and returns the result tensor.
        ///
        /// This operation outputs a copy of the input tensor, where values from the
        /// `batchAxisTensor` dimension are moved in spatial blocks of size `blockDimensionsTensor` to the
        /// `spatialAxesTensor` dimensions (for `usePixelShuffleOrder=YES` 1,2 or 3 axes supported,
        /// otherwise limited only by `MPSNDArray` rank limitations).  Use the `usePixelShuffleOrder` parameter
        /// to control how the data within spatial blocks is ordered in the
        /// `batchAxisTensor` dimension: with `usePixelShuffleOrder = YES` MPSGraph stores
        /// the values of the spatial block contiguosly within the `batchAxisTensor` dimension whereas
        /// without it they are stored interleaved with existing values in the `batchAxisTensor` dimension.
        /// Note: This operation is the inverse of
        /// ``MPSGraph/spaceToBatchTensor:spatialAxesTensor:batchAxisTensor:blockDimensionsTensor:usePixelShuffleOrder:name:``.
        /// Note: This operation is a generalization of
        /// ``MPSGraph/depthToSpace2DTensor:widthAxisTensor:heightAxisTensor:depthAxisTensor:blockSize:usePixelShuffleOrder:name:``.
        ///
        /// - Parameters:
        /// - tensor: The input tensor.
        /// - spatialAxesTensor: A tensor that contains the axes that define the dimensions containing the spatial blocks.
        /// - batchAxisTensor: A tensor that contains the axis that defines the destination dimension, where to copy the blocks.
        /// - blockDimensionsTensor: A tensor that defines the size of the rectangular spatial sub-block.
        /// - usePixelShuffleOrder: A parameter that controls layout of the sub-blocks within the batch dimension.
        /// - name: The name for the operation.
        /// - Returns: A valid MPSGraphTensor object.
        #[unsafe(method(batchToSpaceTensor:spatialAxesTensor:batchAxisTensor:blockDimensionsTensor:usePixelShuffleOrder:name:))]
        #[unsafe(method_family = none)]
        pub unsafe fn batchToSpaceTensor_spatialAxesTensor_batchAxisTensor_blockDimensionsTensor_usePixelShuffleOrder_name(
            &self,
            tensor: &MPSGraphTensor,
            spatial_axes_tensor: &MPSGraphTensor,
            batch_axis_tensor: &MPSGraphTensor,
            block_dimensions_tensor: &MPSGraphTensor,
            use_pixel_shuffle_order: bool,
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
