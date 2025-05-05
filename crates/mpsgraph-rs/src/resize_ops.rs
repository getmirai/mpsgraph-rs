use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::shape::Shape;
use crate::tensor::Tensor;
use crate::TensorNamedDataLayout;

/// The resize mode to use for resizing.
#[repr(u64)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ResizeMode {
    /// Samples the nearest neighbor to the pixel coordinate.
    Nearest = 0,
    /// Samples the 4 neighbors to the pixel coordinate and uses bilinear interpolation.
    Bilinear = 1,
}

/// The rounding mode to use when using nearest resize mode.
#[repr(u64)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ResizeNearestRoundingMode {
    /// Rounds values to the nearest integer value, with 0.5f offset rounding toward +inf.
    RoundPreferCeil = 0,
    /// Rounds values to the nearest integer value, with 0.5f rounding toward -inf.
    RoundPreferFloor = 1,
    /// Rounds values toward +inf.
    Ceil = 2,
    /// Rounds values toward -inf.
    Floor = 3,
    /// Rounds values to the nearest integer value, with 0.5f rounding toward the closest even value.
    RoundToEven = 4,
    /// Rounds values to the nearest integer value, with 0.5f rounding toward the closest odd value.
    RoundToOdd = 5,
}

/// Trait for resize operations on Graph
pub trait GraphResizeOps {
    /// Creates a Resize operation and returns the result tensor.
    ///
    /// Resamples input images to given size. Result images will be distorted if size is of different aspect ratio.
    ///
    /// - Parameters:
    ///   - images_tensor: Tensor containing input images.
    ///   - size: A 2-element shape as [newHeight, newWidth]
    ///   - mode: The resampling mode to use.
    ///   - center_result: Controls if the result image is centered on the input image. When false, the result will have the top left corner aligned.
    ///   - align_corners: When true, the result image will have the same value as the input image in the corners.
    ///   - layout: Specifies what layout the provided tensor is in.
    ///   - name: The name for the operation.
    /// - Returns: A valid Tensor object
    fn resize(
        &self,
        images_tensor: &Tensor,
        size: &Shape,
        mode: ResizeMode,
        center_result: bool,
        align_corners: bool,
        layout: TensorNamedDataLayout,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a Resize operation using a tensor for size specification and returns the result tensor.
    ///
    /// - Parameters:
    ///   - images_tensor: Tensor containing input images.
    ///   - size_tensor: 1D Int32 or Int64 tensor. A 2-element shape as [newHeight, newWidth]
    ///   - mode: The resampling mode to use.
    ///   - center_result: Controls if the result image is centered on the input image.
    ///   - align_corners: When true, the result image will have the same value as the input image in the corners.
    ///   - layout: Specifies what layout the provided tensor is in.
    ///   - name: The name for the operation.
    /// - Returns: A valid Tensor object
    fn resize_with_size_tensor(
        &self,
        images_tensor: &Tensor,
        size_tensor: &Tensor,
        mode: ResizeMode,
        center_result: bool,
        align_corners: bool,
        layout: TensorNamedDataLayout,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a Resize operation using a tensor for size specification and returns the result tensor.
    ///
    /// This is a rank-agnostic version that works with tensors of any rank (iOS 17+/macOS 14+).
    ///
    /// - Parameters:
    ///   - images_tensor: Tensor containing input images.
    ///   - size_tensor: The target size of the result tensor. 1D Int32 or Int64 tensor of size equal to rank of input.
    ///   - mode: The resampling mode to use.
    ///   - center_result: Controls if the result image is centered on the input image.
    ///   - align_corners: When true, the result image will have the same value as the input image in the corners.
    ///   - name: The name for the operation.
    /// - Returns: A valid Tensor object
    fn resize_rank_agnostic(
        &self,
        images_tensor: &Tensor,
        size_tensor: &Tensor,
        mode: ResizeMode,
        center_result: bool,
        align_corners: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a Resize operation with nearest neighbor sampling and a specific rounding mode.
    ///
    /// - Parameters:
    ///   - images_tensor: Tensor containing input images.
    ///   - size_tensor: 1D Int32 or Int64 tensor. A 2-element shape as [newHeight, newWidth]
    ///   - nearest_rounding_mode: The rounding mode to use when using nearest resampling.
    ///   - center_result: Controls if the result image is centered on the input image.
    ///   - align_corners: When true, the result image will have the same value as the input image in the corners.
    ///   - layout: Specifies what layout the provided tensor is in.
    ///   - name: The name for the operation.
    /// - Returns: A valid Tensor object
    fn resize_nearest(
        &self,
        images_tensor: &Tensor,
        size_tensor: &Tensor,
        nearest_rounding_mode: ResizeNearestRoundingMode,
        center_result: bool,
        align_corners: bool,
        layout: TensorNamedDataLayout,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a Resize operation with nearest neighbor sampling and a specific rounding mode.
    ///
    /// This is a rank-agnostic version that works with tensors of any rank (iOS 17+/macOS 14+).
    ///
    /// - Parameters:
    ///   - images_tensor: Tensor containing input images.
    ///   - size_tensor: The target size of the result tensor. 1D Int32 or Int64 tensor of size equal to rank of input.
    ///   - nearest_rounding_mode: The rounding mode to use when using nearest resampling.
    ///   - center_result: Controls if the result image is centered on the input image.
    ///   - align_corners: When true, the result image will have the same value as the input image in the corners.
    ///   - name: The name for the operation.
    /// - Returns: A valid Tensor object
    fn resize_nearest_rank_agnostic(
        &self,
        images_tensor: &Tensor,
        size_tensor: &Tensor,
        nearest_rounding_mode: ResizeNearestRoundingMode,
        center_result: bool,
        align_corners: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a Resize operation with bilinear interpolation.
    ///
    /// - Parameters:
    ///   - images_tensor: Tensor containing input images.
    ///   - size_tensor: 1D Int32 or Int64 tensor. A 2-element shape as [newHeight, newWidth]
    ///   - center_result: Controls if the result image is centered on the input image.
    ///   - align_corners: When true, the result image will have the same value as the input image in the corners.
    ///   - layout: Specifies what layout the provided tensor is in.
    ///   - name: The name for the operation.
    /// - Returns: A valid Tensor object
    fn resize_bilinear(
        &self,
        images_tensor: &Tensor,
        size_tensor: &Tensor,
        center_result: bool,
        align_corners: bool,
        layout: TensorNamedDataLayout,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a Resize operation with bilinear interpolation.
    ///
    /// This is a rank-agnostic version that works with tensors of any rank (iOS 17+/macOS 14+).
    ///
    /// - Parameters:
    ///   - images_tensor: Tensor containing input images.
    ///   - size_tensor: The target size of the result tensor. 1D Int32 or Int64 tensor of size equal to rank of input.
    ///   - center_result: Controls if the result image is centered on the input image.
    ///   - align_corners: When true, the result image will have the same value as the input image in the corners.
    ///   - name: The name for the operation.
    /// - Returns: A valid Tensor object
    fn resize_bilinear_rank_agnostic(
        &self,
        images_tensor: &Tensor,
        size_tensor: &Tensor,
        center_result: bool,
        align_corners: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a Resize operation using explicit scale and offset tensors.
    ///
    /// - Parameters:
    ///   - images_tensor: Tensor containing input images.
    ///   - size_tensor: 1D Int32 or Int64 tensor. A 2-element shape as [newHeight, newWidth]
    ///   - scale_offset_tensor: 1D float tensor. A 4-element shape as [scaleY, scaleX, offsetY, offsetX]
    ///   - mode: The resampling mode to use.
    ///   - layout: Specifies what layout the provided tensor is in.
    ///   - name: The name for the operation.
    /// - Returns: A valid Tensor object
    fn resize_with_scale_offset(
        &self,
        images_tensor: &Tensor,
        size_tensor: &Tensor,
        scale_offset_tensor: &Tensor,
        mode: ResizeMode,
        layout: TensorNamedDataLayout,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a Resize operation using separate scale and offset tensors.
    ///
    /// Available on iOS 17+/macOS 14+.
    ///
    /// - Parameters:
    ///   - images_tensor: Tensor containing input images.
    ///   - size_tensor: The target size of the result tensor. 1D Int32 or Int64 tensor of size equal to rank of input.
    ///   - scale_tensor: 1D float tensor of size equal to rank of input.
    ///   - offset_tensor: 1D float tensor of size equal to rank of input.
    ///   - mode: The resampling mode to use.
    ///   - name: The name for the operation.
    /// - Returns: A valid Tensor object
    fn resize_with_separate_scale_offset(
        &self,
        images_tensor: &Tensor,
        size_tensor: &Tensor,
        scale_tensor: &Tensor,
        offset_tensor: &Tensor,
        mode: ResizeMode,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a Resize gradient operation for backpropagation.
    ///
    /// - Parameters:
    ///   - gradient: Incoming gradient tensor
    ///   - input: Forward pass input tensor
    ///   - mode: The resampling mode to use
    ///   - center_result: Controls if the result image is centered on the input image
    ///   - align_corners: When true, the result image will have the same value as the input image in the corners
    ///   - layout: Specifies what layout the provided tensor is in
    ///   - name: The name for the operation
    /// - Returns: A valid Tensor object
    fn resize_gradient(
        &self,
        gradient: &Tensor,
        input: &Tensor,
        mode: ResizeMode,
        center_result: bool,
        align_corners: bool,
        layout: TensorNamedDataLayout,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;
}

impl GraphResizeOps for Graph {
    fn resize(
        &self,
        images_tensor: &Tensor,
        size: &Shape,
        mode: ResizeMode,
        center_result: bool,
        align_corners: bool,
        layout: TensorNamedDataLayout,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                resizeTensor: images_tensor,
                size: size,
                mode: mode as u64,
                centerResult: center_result,
                alignCorners: align_corners,
                layout: layout as u64,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::retain_autoreleased(result).unwrap())
            }
        }
    }

    fn resize_with_size_tensor(
        &self,
        images_tensor: &Tensor,
        size_tensor: &Tensor,
        mode: ResizeMode,
        center_result: bool,
        align_corners: bool,
        layout: TensorNamedDataLayout,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                resizeTensor: images_tensor,
                sizeTensor: size_tensor,
                mode: mode as u64,
                centerResult: center_result,
                alignCorners: align_corners,
                layout: layout as u64,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::retain_autoreleased(result).unwrap())
            }
        }
    }

    fn resize_rank_agnostic(
        &self,
        images_tensor: &Tensor,
        size_tensor: &Tensor,
        mode: ResizeMode,
        center_result: bool,
        align_corners: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                resizeTensor: images_tensor,
                sizeTensor: size_tensor,
                mode: mode as u64,
                centerResult: center_result,
                alignCorners: align_corners,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::retain_autoreleased(result).unwrap())
            }
        }
    }

    fn resize_nearest(
        &self,
        images_tensor: &Tensor,
        size_tensor: &Tensor,
        nearest_rounding_mode: ResizeNearestRoundingMode,
        center_result: bool,
        align_corners: bool,
        layout: TensorNamedDataLayout,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                resizeNearestWithTensor: images_tensor,
                sizeTensor: size_tensor,
                nearestRoundingMode: nearest_rounding_mode as u64,
                centerResult: center_result,
                alignCorners: align_corners,
                layout: layout as u64,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::retain_autoreleased(result).unwrap())
            }
        }
    }

    fn resize_nearest_rank_agnostic(
        &self,
        images_tensor: &Tensor,
        size_tensor: &Tensor,
        nearest_rounding_mode: ResizeNearestRoundingMode,
        center_result: bool,
        align_corners: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                resizeNearestWithTensor: images_tensor,
                sizeTensor: size_tensor,
                nearestRoundingMode: nearest_rounding_mode as u64,
                centerResult: center_result,
                alignCorners: align_corners,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::retain_autoreleased(result).unwrap())
            }
        }
    }

    fn resize_bilinear(
        &self,
        images_tensor: &Tensor,
        size_tensor: &Tensor,
        center_result: bool,
        align_corners: bool,
        layout: TensorNamedDataLayout,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                resizeBilinearWithTensor: images_tensor,
                sizeTensor: size_tensor,
                centerResult: center_result,
                alignCorners: align_corners,
                layout: layout as u64,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::retain_autoreleased(result).unwrap())
            }
        }
    }

    fn resize_bilinear_rank_agnostic(
        &self,
        images_tensor: &Tensor,
        size_tensor: &Tensor,
        center_result: bool,
        align_corners: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                resizeBilinearWithTensor: images_tensor,
                sizeTensor: size_tensor,
                centerResult: center_result,
                alignCorners: align_corners,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::retain_autoreleased(result).unwrap())
            }
        }
    }

    fn resize_with_scale_offset(
        &self,
        images_tensor: &Tensor,
        size_tensor: &Tensor,
        scale_offset_tensor: &Tensor,
        mode: ResizeMode,
        layout: TensorNamedDataLayout,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                resizeTensor: images_tensor,
                sizeTensor: size_tensor,
                scaleOffsetTensor: scale_offset_tensor,
                mode: mode as u64,
                layout: layout as u64,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::retain_autoreleased(result).unwrap())
            }
        }
    }

    fn resize_with_separate_scale_offset(
        &self,
        images_tensor: &Tensor,
        size_tensor: &Tensor,
        scale_tensor: &Tensor,
        offset_tensor: &Tensor,
        mode: ResizeMode,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                resizeTensor: images_tensor,
                sizeTensor: size_tensor,
                scaleTensor: scale_tensor,
                offsetTensor: offset_tensor,
                mode: mode as u64,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::retain_autoreleased(result).unwrap())
            }
        }
    }

    fn resize_gradient(
        &self,
        gradient: &Tensor,
        input: &Tensor,
        mode: ResizeMode,
        center_result: bool,
        align_corners: bool,
        layout: TensorNamedDataLayout,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                resizeWithGradientTensor: gradient,
                input: input,
                mode: mode as u64,
                centerResult: center_result,
                alignCorners: align_corners,
                layout: layout as u64,
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
