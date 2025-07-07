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

/// Sample-grid operations for [`Graph`], adapted from `MPSGraphSampleGridOps.h`.
impl Graph {
    /// Samples `source` at the spatial coordinates provided in `coordinates`.
    ///
    /// Given an input tensor with layout `(N, H₁, W₁, C)` (NHWC) or
    /// `(N, C, H₁, W₁)` (NCHW) and a **coordinates** tensor of shape
    /// `(N, H₂, W₂, 2)`, this operation produces an output tensor of shape
    /// `(N, H₂, W₂, C)` (NHWC) or `(N, C, H₂, W₂)` (NCHW). Each coordinate
    /// `[y, x]` (either normalised or absolute) selects a point in the input
    /// image which is sampled using the specified `sampling_mode` (nearest or
    /// bilinear). Behaviour outside the image is governed by `padding_mode` and
    /// `constant_value`.
    ///
    /// * `layout` must be `NHWC` or `NCHW` and the output shares the same
    ///   layout.
    /// * `normalize_coordinates` interprets coordinates in the range
    ///   `[-1, 1] × [-1, 1]` instead of pixels.
    /// * `relative_coordinates` treats coordinates as offsets relative to the
    ///   output pixel location, scaled by the input size (useful for optical
    ///   flow style sampling).
    /// * `align_corners` chooses whether coordinate extrema map to pixel
    ///   centres (true) or outer edges (false).
    ///
    /// Returns `Some(output)` or `None` if the underlying Objective-C call
    /// failed.
    pub fn sample_grid(
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
            objc2::msg_send![
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
            ]
        }
    }

    /// Variant of [`sample_grid`] that always uses **nearest-neighbour**
    /// sampling but lets you choose the rounding strategy via
    /// `nearest_rounding_mode` (e.g. ceil, floor, round-prefer-ceil, …).
    pub fn sample_grid_nearest(
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
            objc2::msg_send![
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
            ]
        }
    }
}
