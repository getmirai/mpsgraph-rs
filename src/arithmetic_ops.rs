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
    pub fn identity(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, identityWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn erf(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, erfWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn exponent(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, exponentWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn exponent_base2(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, exponentBase2WithTensor: tensor, name: name_ptr]
        }
    }

    pub fn logarithm(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, logarithmWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn logarithm_base2(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, logarithmBase2WithTensor: tensor, name: name_ptr]
        }
    }

    pub fn square(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, squareWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn square_root(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, squareRootWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn reciprocal_square_root(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, reciprocalSquareRootWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn reciprocal(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, reciprocalWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn absolute(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, absoluteWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn negative(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, negativeWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn sign(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, signWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn ceil(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, ceilWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn sine(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, sineWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn cosine(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, cosineWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn tangent(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, tangentWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn hyperbolic_sine(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, hyperbolicSineWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn hyperbolic_cosine(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, hyperbolicCosineWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn floor(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, floorWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn round(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, roundWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn asin(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, asinWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn acos(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, acosWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn atan(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, atanWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn asinh(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, asinhWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn acosh(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, acoshWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn atanh(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, atanhWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn atan2(
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

    pub fn division_no_nan(
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

    pub fn modulo(
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

    pub fn power(
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

    pub fn minimum(
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

    pub fn maximum(
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

    pub fn equal(
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

    pub fn not_equal(
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

    pub fn less_than(
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

    pub fn less_than_or_equal(
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

    pub fn greater_than(
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

    pub fn greater_than_or_equal(
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

    pub fn logical_and(
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

    pub fn logical_or(
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

    pub fn logical_not(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, logicalNOTWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn select(
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

    pub fn clamp(
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

    pub fn exponent_base10(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, exponentBase10WithTensor: tensor, name: name_ptr]
        }
    }

    pub fn logarithm_base10(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, logarithmBase10WithTensor: tensor, name: name_ptr]
        }
    }

    pub fn absolute_square(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, absoluteSquareWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn signbit(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, signbitWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn tanh(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, tanhWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn rint(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, rintWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn truncate(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, truncateWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn bitwise_not(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, bitwiseNOTWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn bitwise_population_count(
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

    pub fn conjugate(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, conjugateWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn is_infinite(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, isInfiniteWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn is_finite(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, isFiniteWithTensor: tensor, name: name_ptr]
        }
    }

    pub fn is_nan(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, isNaNWithTensor: tensor, name: name_ptr]
        }
    }
}
