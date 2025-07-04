use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::tensor::Tensor;

/// The non-maximum suppression coordinate mode.
///
/// This mode specifies the representation used for the 4 box coordinate values.
/// Center coordinate modes define a centered box and the box dimensions.
#[repr(u64)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum NonMaximumSuppressionCoordinateMode {
    /// [h_start, w_start, h_end, w_end]
    CornersHeightFirst = 0,
    /// [w_start, h_start, w_end, h_end]
    CornersWidthFirst = 1,
    /// [h_center, w_center, box_height, box_width]
    CentersHeightFirst = 2,
    /// [w_center, h_center, box_width, box_height]
    CentersWidthFirst = 3,
}

/// Trait for performing non-maximum suppression operations on a graph


/// Implementation of non-maximum suppression operations for Graph
impl Graph {
    pub fn non_maximum_suppression(
        &self,
        boxes_tensor: &Tensor,
        scores_tensor: &Tensor,
        iou_threshold: f32,
        score_threshold: f32,
        per_class_suppression: bool,
        coordinate_mode: NonMaximumSuppressionCoordinateMode,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: Option<Retained<Tensor>> = msg_send![
                self,
                nonMaximumSuppressionWithBoxesTensor: boxes_tensor,
                scoresTensor: scores_tensor,
                IOUThreshold: iou_threshold,
                scoreThreshold: score_threshold,
                perClassSuppression: per_class_suppression,
                coordinateMode: coordinate_mode as u64,
                name: name_ptr
            ];
            result
        }
    }

    pub fn non_maximum_suppression_with_class_indices(
        &self,
        boxes_tensor: &Tensor,
        scores_tensor: &Tensor,
        class_indices_tensor: &Tensor,
        iou_threshold: f32,
        score_threshold: f32,
        per_class_suppression: bool,
        coordinate_mode: NonMaximumSuppressionCoordinateMode,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: Option<Retained<Tensor>> = msg_send![
                self,
                nonMaximumSuppressionWithBoxesTensor: boxes_tensor,
                scoresTensor: scores_tensor,
                classIndicesTensor: class_indices_tensor,
                IOUThreshold: iou_threshold,
                scoreThreshold: score_threshold,
                perClassSuppression: per_class_suppression,
                coordinateMode: coordinate_mode as u64,
                name: name_ptr
            ];
            result
        }
    }
}

