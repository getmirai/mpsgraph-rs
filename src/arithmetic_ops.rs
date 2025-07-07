//! Arithmetic operations exposed as inherent methods on `Graph`.
//!
//! These helpers were previously provided via the `GraphArithmeticOps` trait.
//! They are now implemented directly on `Graph` to simplify the public API.

use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::tensor::Tensor;

impl Graph {
    // MARK: - Unary operations -------------------------------------------------

    pub fn identity_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, identityWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn erf_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, erfWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn exponent_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, exponentWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn exponent_base2_with_tensor(
        &self,
        tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, exponentBase2WithTensor: tensor, name: name_ptr]
        }
    }

    pub fn logarithm_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, logarithmWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn logarithm_base2_with_tensor(
        &self,
        tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, logarithmBase2WithTensor: tensor, name: name_ptr]
        }
    }

    pub fn square_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, squareWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn square_root_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, squareRootWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn reciprocal_square_root_with_tensor(
        &self,
        tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, reciprocalSquareRootWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn reciprocal_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, reciprocalWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn absolute_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, absoluteWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn negative_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, negativeWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn sign_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, signWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn ceil_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, ceilWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn sin_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, sineWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn cos_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, cosineWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn tan_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, tangentWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn sinh_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, hyperbolicSineWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn cosh_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, hyperbolicCosineWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn floor_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, floorWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn round_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, roundWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn asin_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, asinWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn acos_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, acosWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn atan_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, atanWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn asinh_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, asinhWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn acosh_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, acoshWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn atanh_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, atanhWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn atan2_with_primary_tensor_secondary_tensor(
        &self,
        primary_tensor: &Tensor,
        secondary_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, atan2WithPrimaryTensor: primary_tensor, secondaryTensor: secondary_tensor, name: name_ptr]
        }
    }

    // MARK: - Binary helpers (add / subtract / multiply / divide are already inherent)

    pub fn division_no_nan_with_primary_tensor_secondary_tensor(
        &self,
        primary_tensor: &Tensor,
        secondary_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, divisionNoNaNWithPrimaryTensor: primary_tensor, secondaryTensor: secondary_tensor, name: name_ptr]
        }
    }

    pub fn modulo_with_primary_tensor_secondary_tensor(
        &self,
        primary_tensor: &Tensor,
        secondary_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, moduloWithPrimaryTensor: primary_tensor, secondaryTensor: secondary_tensor, name: name_ptr]
        }
    }

    pub fn power_with_primary_tensor_secondary_tensor(
        &self,
        primary_tensor: &Tensor,
        secondary_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, powerWithPrimaryTensor: primary_tensor, secondaryTensor: secondary_tensor, name: name_ptr]
        }
    }

    pub fn minimum_with_primary_tensor_secondary_tensor(
        &self,
        primary_tensor: &Tensor,
        secondary_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, minimumWithPrimaryTensor: primary_tensor, secondaryTensor: secondary_tensor, name: name_ptr]
        }
    }

    pub fn maximum_with_primary_tensor_secondary_tensor(
        &self,
        primary_tensor: &Tensor,
        secondary_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, maximumWithPrimaryTensor: primary_tensor, secondaryTensor: secondary_tensor, name: name_ptr]
        }
    }

    // MARK: - Comparisons ------------------------------------------------------

    pub fn equal_with_primary_tensor_secondary_tensor(
        &self,
        primary_tensor: &Tensor,
        secondary_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, equalWithPrimaryTensor: primary_tensor, secondaryTensor: secondary_tensor, name: name_ptr]
        }
    }

    pub fn not_equal_with_primary_tensor_secondary_tensor(
        &self,
        primary_tensor: &Tensor,
        secondary_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, notEqualWithPrimaryTensor: primary_tensor, secondaryTensor: secondary_tensor, name: name_ptr]
        }
    }

    pub fn less_than_with_primary_tensor_secondary_tensor(
        &self,
        primary_tensor: &Tensor,
        secondary_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, lessThanWithPrimaryTensor: primary_tensor, secondaryTensor: secondary_tensor, name: name_ptr]
        }
    }

    pub fn less_than_or_equal_with_primary_tensor_secondary_tensor(
        &self,
        primary_tensor: &Tensor,
        secondary_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, lessThanOrEqualWithPrimaryTensor: primary_tensor, secondaryTensor: secondary_tensor, name: name_ptr]
        }
    }

    pub fn greater_than_with_primary_tensor_secondary_tensor(
        &self,
        primary_tensor: &Tensor,
        secondary_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, greaterThanWithPrimaryTensor: primary_tensor, secondaryTensor: secondary_tensor, name: name_ptr]
        }
    }

    pub fn greater_than_or_equal_with_primary_tensor_secondary_tensor(
        &self,
        primary_tensor: &Tensor,
        secondary_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, greaterThanOrEqualWithPrimaryTensor: primary_tensor, secondaryTensor: secondary_tensor, name: name_ptr]
        }
    }

    // MARK: - Logical ---------------------------------------------------------

    pub fn logical_and_with_primary_tensor_secondary_tensor(
        &self,
        primary_tensor: &Tensor,
        secondary_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, logicalANDWithPrimaryTensor: primary_tensor, secondaryTensor: secondary_tensor, name: name_ptr]
        }
    }

    pub fn logical_or_with_primary_tensor_secondary_tensor(
        &self,
        primary_tensor: &Tensor,
        secondary_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, logicalORWithPrimaryTensor: primary_tensor, secondaryTensor: secondary_tensor, name: name_ptr]
        }
    }

    pub fn not_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, logicalNOTWithTensor: tensor, name: name_ptr]
        }
    }

    // MARK: - Ternary ----------------------------------------------------------

    pub fn select_with_predicate_tensor_true_predicate_tensor_false_predicate_tensor(
        &self,
        predicate_tensor: &Tensor,
        true_predicate_tensor: &Tensor,
        false_predicate_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self,
                selectWithPredicateTensor: predicate_tensor,
                truePredicateTensor: true_predicate_tensor,
                falsePredicateTensor: false_predicate_tensor,
                name: name_ptr]
        }
    }

    pub fn clamp_with_tensor_min_value_tensor_max_value_tensor(
        &self,
        tensor: &Tensor,
        min_value_tensor: &Tensor,
        max_value_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, clampWithTensor: tensor, minValueTensor: min_value_tensor, maxValueTensor: max_value_tensor, name: name_ptr]
        }
    }

    // Additional unary operations to fully mirror Objective-C header ---------

    pub fn exponent_base10_with_tensor(
        &self,
        tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, exponentBase10WithTensor: tensor, name: name_ptr]
        }
    }

    pub fn logarithm_base10_with_tensor(
        &self,
        tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, logarithmBase10WithTensor: tensor, name: name_ptr]
        }
    }

    pub fn absolute_square_with_tensor(
        &self,
        tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, absoluteSquareWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn signbit_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, signbitWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn tanh_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, tanhWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn rint_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, rintWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn truncate_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, truncateWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn bitwise_not_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, bitwiseNOTWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn bitwise_population_count_with_tensor(
        &self,
        tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, bitwisePopulationCountWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn conjugate_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, conjugateWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn is_infinite_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, isInfiniteWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn is_finite_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, isFiniteWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn is_nan_with_tensor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, isNaNWithTensor: tensor, name: name_ptr]
        }
    }
}
