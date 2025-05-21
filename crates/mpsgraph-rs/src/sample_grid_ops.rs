use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::pooling_ops::TensorNamedDataLayout;
use crate::resize_ops::{ResizeMode, ResizeNearestRoundingMode};
use crate::tensor::Tensor;

/// Padding modes for sampling operations
#[repr(i64)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PaddingMode {
    /// Constant padding
    Constant = 0,
    /// Reflect padding
    Reflect = 1,
    /// Symmetric padding
    Symmetric = 2,
    /// Clamp to edge padding (PyTorch ReplicationPad)
    ClampToEdge = 3,
    /// Zero padding
    Zero = 4,
    /// Periodic padding (x[-2] -> x[L-3], where L is size of x)
    Periodic = 5,
    /// Anti-periodic padding (x[-2] -> -x[L-3])
    AntiPeriodic = 6,
}

/// Trait for performing sample grid operations on a graph
pub trait GraphSampleGridOps {
    /// Samples a tensor using the coordinates provided.
    ///
    /// Given an input tensor (N, H1, W1, C) or (N, C, H1, W1) and coordinates tensor (N, H2, W2, 2)
    /// this operation outputs a tensor of size (N, H2, W2, C) or (N, C, H2, W2) by sampling the
    /// input tensor at the coordinates provided by the coordinates tensor.
    ///
    /// # Arguments
    ///
    /// * `source` - Tensor containing source data
    /// * `coordinates` - A tensor (N, Hout, Wout, 2) that contains the coordinates of the samples in the source tensor
    /// * `layout` - Specifies what layout the provided tensor is in
    /// * `normalize_coordinates` - If true, coordinates are within [-1, 1] x [-1, 1] otherwise they are in pixels
    /// * `relative_coordinates` - If true, coordinates are relative to the position of the pixel in the output tensor
    /// * `align_corners` - If true, coordinate extrema are equal to the center of edge pixels
    /// * `padding_mode` - Determines how samples outside the inputTensor are evaluated
    /// * `sampling_mode` - Can be either MPSGraphResizeNearest or MPSGraphResizeBilinear
    /// * `constant_value` - If paddingMode is Constant, then this constant is used for samples outside the input tensor
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object or None if error
    fn sample_grid(
        &self,
        source: &Tensor,
        coordinates: &Tensor,
        layout: TensorNamedDataLayout,
        normalize_coordinates: bool,
        relative_coordinates: bool,
        align_corners: bool,
        padding_mode: PaddingMode,
        sampling_mode: ResizeMode,
        constant_value: f64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Samples a tensor using the coordinates provided, using nearest neighbor sampling with specified rounding mode.
    ///
    /// # Arguments
    ///
    /// * `source` - Tensor containing source data
    /// * `coordinates` - A tensor (N, Hout, Wout, 2) that contains the coordinates of the samples in the source tensor
    /// * `layout` - Specifies what layout the provided tensor is in
    /// * `normalize_coordinates` - If true, coordinates are within [-1, 1] x [-1, 1] otherwise they are in pixels
    /// * `relative_coordinates` - If true, coordinates are relative to the position of the pixel in the output tensor
    /// * `align_corners` - If true, coordinate extrema are equal to the center of edge pixels
    /// * `padding_mode` - Determines how samples outside the inputTensor are evaluated
    /// * `nearest_rounding_mode` - The rounding mode to use for determining the nearest neighbor
    /// * `constant_value` - If paddingMode is Constant, then this constant is used for samples outside the input tensor
    /// * `name` - The name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object or None if error
    fn sample_grid_nearest(
        &self,
        source: &Tensor,
        coordinates: &Tensor,
        layout: TensorNamedDataLayout,
        normalize_coordinates: bool,
        relative_coordinates: bool,
        align_corners: bool,
        padding_mode: PaddingMode,
        nearest_rounding_mode: ResizeNearestRoundingMode,
        constant_value: f64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;
}

/// Implementation of sample grid operations for Graph
impl GraphSampleGridOps for Graph {
    fn sample_grid(
        &self,
        source: &Tensor,
        coordinates: &Tensor,
        layout: TensorNamedDataLayout,
        normalize_coordinates: bool,
        relative_coordinates: bool,
        align_corners: bool,
        padding_mode: PaddingMode,
        sampling_mode: ResizeMode,
        constant_value: f64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = objc2::msg_send![
                self,
                sampleGridWithSourceTensor: source,
                coordinateTensor: coordinates,
                layout: layout as u64,
                normalizeCoordinates: normalize_coordinates,
                relativeCoordinates: relative_coordinates,
                alignCorners: align_corners,
                paddingMode: padding_mode as i64,
                samplingMode: sampling_mode as u64,
                constantValue: constant_value,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::retain_autoreleased(result).unwrap())
            }
        }
    }

    fn sample_grid_nearest(
        &self,
        source: &Tensor,
        coordinates: &Tensor,
        layout: TensorNamedDataLayout,
        normalize_coordinates: bool,
        relative_coordinates: bool,
        align_corners: bool,
        padding_mode: PaddingMode,
        nearest_rounding_mode: ResizeNearestRoundingMode,
        constant_value: f64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = objc2::msg_send![
                self,
                sampleGridWithSourceTensor: source,
                coordinateTensor: coordinates,
                layout: layout as u64,
                normalizeCoordinates: normalize_coordinates,
                relativeCoordinates: relative_coordinates,
                alignCorners: align_corners,
                paddingMode: padding_mode as i64,
                nearestRoundingMode: nearest_rounding_mode as u64,
                constantValue: constant_value,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::retain_autoreleased(result).unwrap())
            }
        }
    }
}

/// Extension trait for easier access to sample grid operations
pub trait GraphSampleGridOpsExtension {
    /// Get access to sample grid operations
    fn sample_grid_ops(&self) -> &dyn GraphSampleGridOps;
}

impl GraphSampleGridOpsExtension for Graph {
    fn sample_grid_ops(&self) -> &dyn GraphSampleGridOps {
        self
    }
}
