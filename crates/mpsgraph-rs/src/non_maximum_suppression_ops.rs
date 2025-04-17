use objc2::rc::Retained;
use objc2::msg_send;
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
pub trait GraphNonMaximumSuppressionOps {
    /// Creates a nonMaximumumSuppression operation and returns the result tensor.
    ///
    /// # Arguments
    ///
    /// * `boxes_tensor` - A tensor containing the coordinates of the input boxes.
    ///                    Must be a rank 3 tensor of shape [N,B,4] of type `Float32`
    /// * `scores_tensor` - A tensor containing the scores of the input boxes.
    ///                    Must be a rank 3 tensor of shape [N,B,K] of type `Float32`
    /// * `iou_threshold` - The threshold for when to reject boxes based on their Intersection Over Union.
    ///                    Valid range is [0,1].
    /// * `score_threshold` - The threshold for when to reject boxes based on their score, before IOU suppression.
    /// * `per_class_suppression` - When this is specified a box will only suppress another box if they have the same class.
    /// * `coordinate_mode` - The coordinate mode the box coordinates are provided in.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A valid Tensor object containing the non-maximum suppression results.
    fn non_maximum_suppression(
        &self,
        boxes_tensor: &Tensor,
        scores_tensor: &Tensor,
        iou_threshold: f32,
        score_threshold: f32,
        per_class_suppression: bool,
        coordinate_mode: NonMaximumSuppressionCoordinateMode,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a nonMaximumumSuppression operation with class indices and returns the result tensor.
    ///
    /// # Arguments
    ///
    /// * `boxes_tensor` - A tensor containing the coordinates of the input boxes.
    ///                    Must be a rank 3 tensor of shape [N,B,4] of type `Float32`
    /// * `scores_tensor` - A tensor containing the scores of the input boxes.
    ///                    Must be a rank 3 tensor of shape [N,B,1] of type `Float32`
    /// * `class_indices_tensor` - A tensor containing the class indices of the input boxes.
    ///                    Must be a rank 2 tensor of shape [N,B] of type `Int32`
    /// * `iou_threshold` - The threshold for when to reject boxes based on their Intersection Over Union.
    ///                    Valid range is [0,1].
    /// * `score_threshold` - The threshold for when to reject boxes based on their score, before IOU suppression.
    /// * `per_class_suppression` - When this is specified a box will only suppress another box if they have the same class.
    /// * `coordinate_mode` - The coordinate mode the box coordinates are provided in.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A valid Tensor object containing the non-maximum suppression results.
    fn non_maximum_suppression_with_class_indices(
        &self,
        boxes_tensor: &Tensor,
        scores_tensor: &Tensor,
        class_indices_tensor: &Tensor,
        iou_threshold: f32,
        score_threshold: f32,
        per_class_suppression: bool,
        coordinate_mode: NonMaximumSuppressionCoordinateMode,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;
}

/// Implementation of non-maximum suppression operations for Graph
impl GraphNonMaximumSuppressionOps for Graph {
    fn non_maximum_suppression(
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
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self, 
                nonMaximumSuppressionWithBoxesTensor: boxes_tensor,
                scoresTensor: scores_tensor,
                IOUThreshold: iou_threshold,
                scoreThreshold: score_threshold,
                perClassSuppression: per_class_suppression,
                coordinateMode: coordinate_mode as u64,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn non_maximum_suppression_with_class_indices(
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
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
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

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }
}

/// Extension trait for easier access to non-maximum suppression operations
pub trait GraphNonMaximumSuppressionOpsExtension {
    /// Get access to non-maximum suppression operations
    fn non_maximum_suppression_ops(&self) -> &dyn GraphNonMaximumSuppressionOps;
}

impl GraphNonMaximumSuppressionOpsExtension for Graph {
    fn non_maximum_suppression_ops(&self) -> &dyn GraphNonMaximumSuppressionOps {
        self
    }
}