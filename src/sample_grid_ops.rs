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

/// Inherent implementation of sample grid operations for Graph
impl Graph {
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
