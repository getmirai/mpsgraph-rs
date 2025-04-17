use objc2::rc::Retained;
use objc2::msg_send;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::tensor::Tensor;
use crate::core::create_ns_array_from_i64_slice;

/// Trait for normalization operations on Graph
pub trait GraphNormalizationOps {
    /// Returns the mean of the input tensor along the specified axes.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axes` - A list of axes over which to perform the reduction
    /// * `name` - An optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn mean(
        &self,
        tensor: &Tensor,
        axes: &[i64],
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Returns the variance of the input tensor along the specified axes when the mean has been precomputed.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `mean_tensor` - The precomputed mean tensor
    /// * `axes` - A list of axes over which to perform the reduction
    /// * `name` - An optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn variance_with_mean(
        &self,
        tensor: &Tensor,
        mean_tensor: &Tensor,
        axes: &[i64],
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Returns the variance of the input tensor along the specified axes.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `axes` - A list of axes over which to perform the reduction
    /// * `name` - An optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn variance(
        &self,
        tensor: &Tensor,
        axes: &[i64],
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a batch normalization operation and returns the result tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor
    /// * `mean` - The mean tensor
    /// * `variance` - The variance tensor
    /// * `gamma` - The tensor used to scale the normalized result
    /// * `beta` - The tensor used to bias the normalized result
    /// * `epsilon` - A small value to add to the variance when normalizing the inputs
    /// * `name` - An optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn normalize(
        &self,
        tensor: &Tensor,
        mean: &Tensor,
        variance: &Tensor,
        gamma: Option<&Tensor>,
        beta: Option<&Tensor>,
        epsilon: f32,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a normalization gamma-gradient operation and returns the result tensor.
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient` - The incoming original result tensor gradient
    /// * `source` - The original input source in forward direction
    /// * `mean` - The mean tensor
    /// * `variance` - The variance tensor
    /// * `axes` - The axes of normalization
    /// * `epsilon` - A small value to add to the variance when normalizing the inputs
    /// * `name` - An optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn normalization_gamma_gradient(
        &self,
        incoming_gradient: &Tensor,
        source: &Tensor,
        mean: &Tensor,
        variance: &Tensor,
        axes: &[i64],
        epsilon: f32,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a normalization beta-gradient operation and returns the result tensor.
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient` - The incoming original result tensor gradient
    /// * `source` - The original input source in forward direction
    /// * `axes` - The axes of normalization
    /// * `name` - An optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn normalization_beta_gradient(
        &self,
        incoming_gradient: &Tensor,
        source: &Tensor,
        axes: &[i64],
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a normalization input gradient operation and returns the result tensor.
    ///
    /// # Arguments
    ///
    /// * `incoming_gradient` - The incoming original result tensor gradient
    /// * `source` - The original input source in forward direction
    /// * `mean` - The mean tensor
    /// * `variance` - The variance tensor
    /// * `gamma` - The gamma tensor
    /// * `gamma_gradient` - The gamma gradient tensor
    /// * `beta_gradient` - The beta gradient tensor
    /// * `axes` - The axes of normalization
    /// * `epsilon` - A small value to add to the variance when normalizing the inputs
    /// * `name` - An optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn normalization_gradient(
        &self,
        incoming_gradient: &Tensor,
        source: &Tensor,
        mean: &Tensor,
        variance: &Tensor,
        gamma: Option<&Tensor>,
        gamma_gradient: Option<&Tensor>,
        beta_gradient: Option<&Tensor>,
        axes: &[i64],
        epsilon: f32,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;
}

impl GraphNormalizationOps for Graph {
    fn mean(
        &self,
        tensor: &Tensor,
        axes: &[i64],
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            // Convert the axes to NSArray of NSNumbers
            let axes_array = create_ns_array_from_i64_slice(axes);
            let axes_ptr = &*axes_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                meanOfTensor: tensor,
                axes: axes_ptr,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn variance_with_mean(
        &self,
        tensor: &Tensor,
        mean_tensor: &Tensor,
        axes: &[i64],
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            // Convert the axes to NSArray of NSNumbers
            let axes_array = create_ns_array_from_i64_slice(axes);
            let axes_ptr = &*axes_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                varianceOfTensor: tensor,
                meanTensor: mean_tensor,
                axes: axes_ptr,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn variance(
        &self,
        tensor: &Tensor,
        axes: &[i64],
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            // Convert the axes to NSArray of NSNumbers
            let axes_array = create_ns_array_from_i64_slice(axes);
            let axes_ptr = &*axes_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                varianceOfTensor: tensor,
                axes: axes_ptr,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn normalize(
        &self,
        tensor: &Tensor,
        mean: &Tensor,
        variance: &Tensor,
        gamma: Option<&Tensor>,
        beta: Option<&Tensor>,
        epsilon: f32,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let gamma_ptr = gamma.map_or(std::ptr::null(), |g| g as *const _);
            let beta_ptr = beta.map_or(std::ptr::null(), |b| b as *const _);

            let result: *mut Tensor = msg_send![
                self,
                normalizationWithTensor: tensor,
                meanTensor: mean,
                varianceTensor: variance,
                gammaTensor: gamma_ptr,
                betaTensor: beta_ptr,
                epsilon: epsilon as f64,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn normalization_gamma_gradient(
        &self,
        incoming_gradient: &Tensor,
        source: &Tensor,
        mean: &Tensor,
        variance: &Tensor,
        axes: &[i64],
        epsilon: f32,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            // Convert the axes to NSArray of NSNumbers
            let axes_array = create_ns_array_from_i64_slice(axes);
            let axes_ptr = &*axes_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                normalizationGammaGradientWithIncomingGradientTensor: incoming_gradient,
                sourceTensor: source,
                meanTensor: mean,
                varianceTensor: variance,
                reductionAxes: axes_ptr,
                epsilon: epsilon as f64,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn normalization_beta_gradient(
        &self,
        incoming_gradient: &Tensor,
        source: &Tensor,
        axes: &[i64],
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            // Convert the axes to NSArray of NSNumbers
            let axes_array = create_ns_array_from_i64_slice(axes);
            let axes_ptr = &*axes_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                normalizationBetaGradientWithIncomingGradientTensor: incoming_gradient,
                sourceTensor: source,
                reductionAxes: axes_ptr,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn normalization_gradient(
        &self,
        incoming_gradient: &Tensor,
        source: &Tensor,
        mean: &Tensor,
        variance: &Tensor,
        gamma: Option<&Tensor>,
        gamma_gradient: Option<&Tensor>,
        beta_gradient: Option<&Tensor>,
        axes: &[i64],
        epsilon: f32,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let gamma_ptr = gamma.map_or(std::ptr::null(), |g| g as *const _);
            let gamma_gradient_ptr = gamma_gradient.map_or(std::ptr::null(), |g| g as *const _);
            let beta_gradient_ptr = beta_gradient.map_or(std::ptr::null(), |b| b as *const _);

            // Convert the axes to NSArray of NSNumbers
            let axes_array = create_ns_array_from_i64_slice(axes);
            let axes_ptr = &*axes_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                normalizationGradientWithIncomingGradientTensor: incoming_gradient,
                sourceTensor: source,
                meanTensor: mean,
                varianceTensor: variance,
                gammaTensor: gamma_ptr,
                gammaGradientTensor: gamma_gradient_ptr,
                betaGradientTensor: beta_gradient_ptr,
                reductionAxes: axes_ptr,
                epsilon: epsilon as f64,
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