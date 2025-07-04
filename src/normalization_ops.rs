use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSArray;
use objc2_foundation::NSString;

use crate::core::create_ns_array_from_i64_slice;
use crate::graph::Graph;
use crate::tensor::Tensor;

/// Trait for normalization operations on Graph


impl Graph {
    pub fn mean(
        &self,
        tensor: &Retained<Tensor>,
        axes: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            // Convert the axes to NSArray of NSNumbers
            let axes_array = create_ns_array_from_i64_slice(axes);
            let axes_ptr = &*axes_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                meanOfTensor: &**tensor,
                axes: axes_ptr,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create mean operation");
            } else {
                // This is a computational method that returns an autoreleased object
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    pub fn variance_with_mean(
        &self,
        tensor: &Retained<Tensor>,
        mean_tensor: &Retained<Tensor>,
        axes: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            // Convert the axes to NSArray of NSNumbers
            let axes_array = create_ns_array_from_i64_slice(axes);
            let axes_ptr = &*axes_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                varianceOfTensor: &**tensor,
                meanTensor: &**mean_tensor,
                axes: axes_ptr,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create variance with mean operation");
            } else {
                // This is a computational method that returns an autoreleased object
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    pub fn variance(
        &self,
        tensor: &Retained<Tensor>,
        axes: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            // Convert the axes to NSArray of NSNumbers
            let axes_array = create_ns_array_from_i64_slice(axes);
            let axes_ptr = &*axes_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                varianceOfTensor: &**tensor,
                axes: axes_ptr,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create variance operation");
            } else {
                // This is a computational method that returns an autoreleased object
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    pub fn normalize(
        &self,
        tensor: &Retained<Tensor>,
        mean: &Retained<Tensor>,
        variance: &Retained<Tensor>,
        gamma: Option<&Retained<Tensor>>,
        beta: Option<&Retained<Tensor>>,
        epsilon: f32,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let gamma_ptr = gamma.map_or(std::ptr::null(), |g| &**g as *const _);
            let beta_ptr = beta.map_or(std::ptr::null(), |b| &**b as *const _);

            let result: *mut Tensor = msg_send![
                self,
                normalizationWithTensor: &**tensor,
                meanTensor: &**mean,
                varianceTensor: &**variance,
                gammaTensor: gamma_ptr,
                betaTensor: beta_ptr,
                epsilon: epsilon,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create normalization operation");
            } else {
                // This is a computational method that returns an autoreleased object
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    pub fn normalization_gamma_gradient(
        &self,
        incoming_gradient: &Retained<Tensor>,
        source: &Retained<Tensor>,
        mean: &Retained<Tensor>,
        variance: &Retained<Tensor>,
        axes: &[i64],
        epsilon: f32,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            // Convert the axes to NSArray of NSNumbers
            let axes_array = create_ns_array_from_i64_slice(axes);
            let axes_ptr = &*axes_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                normalizationGammaGradientWithIncomingGradientTensor: &**incoming_gradient,
                sourceTensor: &**source,
                meanTensor: &**mean,
                varianceTensor: &**variance,
                reductionAxes: axes_ptr,
                epsilon: epsilon,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create normalization gamma gradient operation");
            } else {
                // This is a computational method that returns an autoreleased object
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    pub fn normalization_beta_gradient(
        &self,
        incoming_gradient: &Retained<Tensor>,
        source: &Retained<Tensor>,
        axes: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            // Convert the axes to NSArray of NSNumbers
            let axes_array = create_ns_array_from_i64_slice(axes);
            let axes_ptr = &*axes_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                normalizationBetaGradientWithIncomingGradientTensor: &**incoming_gradient,
                sourceTensor: &**source,
                reductionAxes: axes_ptr,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create normalization beta gradient operation");
            } else {
                // This is a computational method that returns an autoreleased object
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    pub fn normalization_gradient(
        &self,
        incoming_gradient: &Retained<Tensor>,
        source: &Retained<Tensor>,
        mean: &Retained<Tensor>,
        variance: &Retained<Tensor>,
        gamma: Option<&Retained<Tensor>>,
        gamma_gradient: Option<&Retained<Tensor>>,
        beta_gradient: Option<&Retained<Tensor>>,
        axes: &[i64],
        epsilon: f32,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let gamma_ptr = gamma.map_or(std::ptr::null(), |g| &**g as *const _);
            let gamma_gradient_ptr = gamma_gradient.map_or(std::ptr::null(), |g| &**g as *const _);
            let beta_gradient_ptr = beta_gradient.map_or(std::ptr::null(), |b| &**b as *const _);

            // Convert the axes to NSArray of NSNumbers
            let axes_array = create_ns_array_from_i64_slice(axes);
            let axes_ptr = &*axes_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                normalizationGradientWithIncomingGradientTensor: &**incoming_gradient,
                sourceTensor: &**source,
                meanTensor: &**mean,
                varianceTensor: &**variance,
                gammaTensor: gamma_ptr,
                gammaGradientTensor: gamma_gradient_ptr,
                betaGradientTensor: beta_gradient_ptr,
                reductionAxes: axes_ptr,
                epsilon: epsilon,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create normalization gradient operation");
            } else {
                // This is a computational method that returns an autoreleased object
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    pub fn layer_normalization(
        &self,
        source: &Tensor,
        axes: &[i64],
        gamma: Option<&Tensor>,
        beta: Option<&Tensor>,
        epsilon: f32,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let axes_array = crate::core::create_ns_array_from_i64_slice(axes);

            let result: Option<Retained<Tensor>> = msg_send![
                self,
                layerNormalizationWithSourceTensor: source,
                axes: &*axes_array,
                gammaTensor: gamma.map_or(std::ptr::null(), |t| t as *const Tensor),
                betaTensor: beta.map_or(std::ptr::null(), |t| t as *const Tensor),
                epsilon: epsilon,
                name: name_ptr
            ];
            result
        }
    }

    pub fn layer_normalization_gradient(
        &self,
        incoming_gradient: &Tensor,
        source: &Tensor,
        axes: &[i64],
        gamma: Option<&Tensor>,
        beta: Option<&Tensor>,
        epsilon: f32,
        name: Option<&str>,
    ) -> (
        Option<Retained<Tensor>>,
        Option<Retained<Tensor>>,
        Option<Retained<Tensor>>,
    ) {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let axes_array = crate::core::create_ns_array_from_i64_slice(axes);

            let result_array: Option<Retained<NSArray<Tensor>>> = msg_send![
                self,
                layerNormalizationGradientWithIncomingGradientTensor: incoming_gradient,
                sourceTensor: source,
                axes: &*axes_array,
                gammaTensor: gamma.map_or(std::ptr::null(), |t| t as *const Tensor),
                betaTensor: beta.map_or(std::ptr::null(), |t| t as *const Tensor),
                epsilon: epsilon,
                name: name_ptr
            ];

            if let Some(array) = result_array {
                let source_grad: Option<Retained<Tensor>> = msg_send![&*array, objectAtIndex: 0];
                let gamma_grad: Option<Retained<Tensor>> = msg_send![&*array, objectAtIndex: 1];
                let beta_grad: Option<Retained<Tensor>> = msg_send![&*array, objectAtIndex: 2];
                (source_grad, gamma_grad, beta_grad)
            } else {
                (None, None, None)
            }
        }
    }

    pub fn instance_normalization(
        &self,
        source: &Tensor,
        gamma: Option<&Tensor>,
        beta: Option<&Tensor>,
        epsilon: f32,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: Option<Retained<Tensor>> = msg_send![
                self,
                instanceNormalizationWithSourceTensor: source,
                gammaTensor: gamma.map_or(std::ptr::null(), |t| t as *const Tensor),
                betaTensor: beta.map_or(std::ptr::null(), |t| t as *const Tensor),
                epsilon: epsilon,
                name: name_ptr
            ];
            result
        }
    }

    pub fn instance_normalization_gradient(
        &self,
        incoming_gradient: &Tensor,
        source: &Tensor,
        gamma: Option<&Tensor>,
        beta: Option<&Tensor>,
        epsilon: f32,
        name: Option<&str>,
    ) -> (
        Option<Retained<Tensor>>,
        Option<Retained<Tensor>>,
        Option<Retained<Tensor>>,
    ) {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result_array: Option<Retained<NSArray<Tensor>>> = msg_send![
                self,
                instanceNormalizationGradientWithIncomingGradientTensor: incoming_gradient,
                sourceTensor: source,
                gammaTensor: gamma.map_or(std::ptr::null(), |t| t as *const Tensor),
                betaTensor: beta.map_or(std::ptr::null(), |t| t as *const Tensor),
                epsilon: epsilon,
                name: name_ptr
            ];

            if let Some(array) = result_array {
                let source_grad: Option<Retained<Tensor>> = msg_send![&*array, objectAtIndex: 0];
                let gamma_grad: Option<Retained<Tensor>> = msg_send![&*array, objectAtIndex: 1];
                let beta_grad: Option<Retained<Tensor>> = msg_send![&*array, objectAtIndex: 2];
                (source_grad, gamma_grad, beta_grad)
            } else {
                (None, None, None)
            }
        }
    }

    pub fn local_response_normalization(
        &self,
        source: &Tensor,
        size: usize,
        alpha: f64,
        beta: f64,
        delta: f64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: Option<Retained<Tensor>> = msg_send![
                self,
                localResponseNormalizationWithSourceTensor: source,
                size: size as u64,
                alpha: alpha,
                beta: beta,
                delta: delta,
                name: name_ptr
            ];
            result
        }
    }

    pub fn batch_normalization(
        &self,
        source: &Tensor,
        mean: &Tensor,
        variance: &Tensor,
        gamma: Option<&Tensor>,
        beta: Option<&Tensor>,
        epsilon: f32,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: Option<Retained<Tensor>> = msg_send![
                self,
                batchNormalizationWithSourceTensor: source,
                meanTensor: mean,
                varianceTensor: variance,
                gammaTensor: gamma.map_or(std::ptr::null(), |t| t as *const Tensor),
                betaTensor: beta.map_or(std::ptr::null(), |t| t as *const Tensor),
                epsilon: epsilon,
                name: name_ptr
            ];
            result
        }
    }

    pub fn batch_normalization_gradient(
        &self,
        incoming_gradient: &Tensor,
        source: &Tensor,
        mean: &Tensor,
        variance: &Tensor,
        gamma: Option<&Tensor>,
        beta: Option<&Tensor>,
        epsilon: f32,
        name: Option<&str>,
    ) -> (
        Option<Retained<Tensor>>,
        Option<Retained<Tensor>>,
        Option<Retained<Tensor>>,
    ) {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result_array: Option<Retained<NSArray<Tensor>>> = msg_send![
                self,
                batchNormalizationGradientWithIncomingGradientTensor: incoming_gradient,
                sourceTensor: source,
                meanTensor: mean,
                varianceTensor: variance,
                gammaTensor: gamma.map_or(std::ptr::null(), |t| t as *const Tensor),
                betaTensor: beta.map_or(std::ptr::null(), |t| t as *const Tensor),
                epsilon: epsilon,
                name: name_ptr
            ];

            if let Some(array) = result_array {
                let source_grad: Option<Retained<Tensor>> = msg_send![&*array, objectAtIndex: 0];
                let gamma_grad: Option<Retained<Tensor>> = msg_send![&*array, objectAtIndex: 1];
                let beta_grad: Option<Retained<Tensor>> = msg_send![&*array, objectAtIndex: 2];
                (source_grad, gamma_grad, beta_grad)
            } else {
                (None, None, None)
            }
        }
    }
}
