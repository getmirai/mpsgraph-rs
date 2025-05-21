use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::tensor::Tensor;

/// Trait for arithmetic operations on Graph
pub trait GraphArithmeticOps {
    // MARK: - Unary Operations

    /// Creates an identity operation
    fn identity(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    /// Returns e raised to the power of the input tensor
    fn exp(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    /// Returns 2 raised to the power of the input tensor
    fn exp2(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    /// Returns the natural logarithm of the input tensor
    fn log(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    /// Returns the base-2 logarithm of the input tensor
    fn log2(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    /// Returns the square of the input tensor
    fn square(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    /// Returns the square root of the input tensor
    fn sqrt(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    /// Returns the reciprocal square root of the input tensor
    fn rsqrt(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    /// Returns the reciprocal of the input tensor
    fn reciprocal(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    /// Returns the absolute value of the input tensor
    fn abs(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    /// Returns the negation of the input tensor
    fn negative(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    /// Returns a tensor with the sign of each element in the input tensor
    /// -1 for negative, 0 for zero, 1 for positive
    fn sign(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    /// Returns the ceiling of the input tensor
    fn ceil(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    /// Returns the sine of the input tensor
    fn sin(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    /// Returns the cosine of the input tensor
    fn cos(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    /// Returns the tangent of the input tensor
    fn tan(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    /// Returns the hyperbolic sine of the input tensor
    fn sinh(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    /// Returns the hyperbolic cosine of the input tensor
    fn cosh(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    /// Returns the hyperbolic tangent of the input tensor
    fn tanh(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    /// Returns the inverse sine of the input tensor
    fn asin(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    /// Returns the inverse cosine of the input tensor
    fn acos(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    /// Returns the inverse tangent of the input tensor
    fn atan(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    /// Returns the inverse hyperbolic sine of the input tensor
    fn asinh(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    /// Returns the inverse hyperbolic cosine of the input tensor
    fn acosh(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    /// Returns the inverse hyperbolic tangent of the input tensor
    fn atanh(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    /// Returns the four-quadrant arctangent of y/x; atan2(y,x)
    fn atan2(&self, y: &Tensor, x: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    /// Returns the floor of the input tensor
    fn floor(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    /// Returns the rounded value of the input tensor
    fn round(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    // MARK: - Binary Operations

    /// Performs addition of two tensors
    fn add(&self, primary: &Tensor, secondary: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    /// Performs subtraction of two tensors
    fn subtract(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Performs multiplication of two tensors
    fn multiply(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Performs division of two tensors
    fn divide(&self, primary: &Tensor, secondary: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    /// Performs division of two tensors but returns 0 if secondary is 0
    fn division_no_nan(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Performs modulo operation between two tensors
    fn modulo(&self, primary: &Tensor, secondary: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    /// Raises primary tensor to the power of secondary tensor
    fn power(&self, primary: &Tensor, secondary: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    /// Returns the minimum of two tensors
    fn minimum(&self, primary: &Tensor, secondary: &Tensor, name: Option<&str>)
        -> Retained<Tensor>;

    /// Returns the maximum of two tensors
    fn maximum(&self, primary: &Tensor, secondary: &Tensor, name: Option<&str>)
        -> Retained<Tensor>;

    // MARK: - Comparison Operations

    /// Returns element-wise equal comparison
    fn equal(&self, primary: &Tensor, secondary: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    /// Returns element-wise not equal comparison
    fn not_equal(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Returns element-wise less than comparison
    fn less_than(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Returns element-wise less than or equal comparison
    fn less_than_or_equal(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Returns element-wise greater than comparison
    fn greater_than(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Returns element-wise greater than or equal comparison
    fn greater_than_or_equal(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    // MARK: - Logical Operations

    /// Performs logical AND of two tensors
    fn logical_and(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Performs logical OR of two tensors
    fn logical_or(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Returns the logical NOT of the input tensor
    fn logical_not(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor>;

    // MARK: - Ternary Operations

    /// Creates a select operation which chooses values from true or false tensor based on predicate
    fn select(
        &self,
        predicate: &Tensor,
        true_tensor: &Tensor,
        false_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a clamp operation that clamps values to the given min and max
    fn clamp(
        &self,
        tensor: &Tensor,
        min_tensor: &Tensor,
        max_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor>;
}

impl GraphArithmeticOps for Graph {
    fn identity(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, identityWithTensor: x, name: name_ptr]
        }
    }

    fn asin(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, asinWithTensor: x, name: name_ptr]
        }
    }

    fn acos(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, acosWithTensor: x, name: name_ptr]
        }
    }

    fn atan(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, atanWithTensor: x, name: name_ptr]
        }
    }

    fn asinh(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, asinhWithTensor: x, name: name_ptr]
        }
    }

    fn acosh(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, acoshWithTensor: x, name: name_ptr]
        }
    }

    fn atanh(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, atanhWithTensor: x, name: name_ptr]
        }
    }

    fn atan2(&self, y: &Tensor, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, atan2WithPrimaryTensor: y, secondaryTensor: x, name: name_ptr]
        }
    }

    fn exp(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, exponentWithTensor: x, name: name_ptr]
        }
    }

    fn exp2(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, exponentBase2WithTensor: x, name: name_ptr]
        }
    }

    fn log(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, logarithmWithTensor: x, name: name_ptr]
        }
    }

    fn log2(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, logarithmBase2WithTensor: x, name: name_ptr]
        }
    }

    fn square(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, squareWithTensor: x, name: name_ptr]
        }
    }

    fn sqrt(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, squareRootWithTensor: x, name: name_ptr]
        }
    }

    fn rsqrt(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, reciprocalSquareRootWithTensor: x, name: name_ptr]
        }
    }

    fn reciprocal(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, reciprocalWithTensor: x, name: name_ptr]
        }
    }

    fn abs(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, absoluteWithTensor: x, name: name_ptr]
        }
    }

    fn negative(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, negativeWithTensor: x, name: name_ptr]
        }
    }

    fn sign(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, signWithTensor: x, name: name_ptr]
        }
    }

    fn ceil(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, ceilWithTensor: x, name: name_ptr]
        }
    }

    fn sin(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, sineWithTensor: x, name: name_ptr]
        }
    }

    fn cos(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, cosineWithTensor: x, name: name_ptr]
        }
    }

    fn tan(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, tangentWithTensor: x, name: name_ptr]
        }
    }

    fn sinh(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, hyperbolicSineWithTensor: x, name: name_ptr]
        }
    }

    fn cosh(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, hyperbolicCosineWithTensor: x, name: name_ptr]
        }
    }

    fn tanh(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, hyperbolicTangentWithTensor: x, name: name_ptr]
        }
    }

    fn floor(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, floorWithTensor: x, name: name_ptr]
        }
    }

    fn round(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, roundWithTensor: x, name: name_ptr]
        }
    }

    fn add(&self, primary: &Tensor, secondary: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, additionWithPrimaryTensor: primary, secondaryTensor: secondary, name: name_ptr]
        }
    }

    fn subtract(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, subtractionWithPrimaryTensor: primary, secondaryTensor: secondary, name: name_ptr]
        }
    }

    fn multiply(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, multiplicationWithPrimaryTensor: primary, secondaryTensor: secondary, name: name_ptr]
        }
    }

    fn divide(&self, primary: &Tensor, secondary: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, divisionWithPrimaryTensor: primary, secondaryTensor: secondary, name: name_ptr]
        }
    }

    fn division_no_nan(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, divisionNoNaNWithPrimaryTensor: primary, secondaryTensor: secondary, name: name_ptr]
        }
    }

    fn modulo(&self, primary: &Tensor, secondary: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, moduloWithPrimaryTensor: primary, secondaryTensor: secondary, name: name_ptr]
        }
    }

    fn power(&self, primary: &Tensor, secondary: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, powerWithPrimaryTensor: primary, secondaryTensor: secondary, name: name_ptr]
        }
    }

    fn minimum(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, minimumWithPrimaryTensor: primary, secondaryTensor: secondary, name: name_ptr]
        }
    }

    fn maximum(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, maximumWithPrimaryTensor: primary, secondaryTensor: secondary, name: name_ptr]
        }
    }

    fn equal(&self, primary: &Tensor, secondary: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, equalWithPrimaryTensor: primary, secondaryTensor: secondary, name: name_ptr]
        }
    }

    fn not_equal(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, notEqualWithPrimaryTensor: primary, secondaryTensor: secondary, name: name_ptr]
        }
    }

    fn less_than(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, lessThanWithPrimaryTensor: primary, secondaryTensor: secondary, name: name_ptr]
        }
    }

    fn less_than_or_equal(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, lessThanOrEqualWithPrimaryTensor: primary, secondaryTensor: secondary, name: name_ptr]
        }
    }

    fn greater_than(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, greaterThanWithPrimaryTensor: primary, secondaryTensor: secondary, name: name_ptr]
        }
    }

    fn greater_than_or_equal(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, greaterThanOrEqualWithPrimaryTensor: primary, secondaryTensor: secondary, name: name_ptr]
        }
    }

    fn logical_and(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, logicalANDWithPrimaryTensor: primary, secondaryTensor: secondary, name: name_ptr]
        }
    }

    fn logical_or(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, logicalORWithPrimaryTensor: primary, secondaryTensor: secondary, name: name_ptr]
        }
    }

    fn logical_not(&self, x: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, logicalNOTWithTensor: x, name: name_ptr]
        }
    }

    fn select(
        &self,
        predicate: &Tensor,
        true_tensor: &Tensor,
        false_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, selectWithPredicateTensor: predicate, truePredicateTensor: true_tensor, falsePredicateTensor: false_tensor, name: name_ptr]
        }
    }

    fn clamp(
        &self,
        tensor: &Tensor,
        min_tensor: &Tensor,
        max_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, clampWithTensor: tensor, minValueTensor: min_tensor, maxValueTensor: max_tensor, name: name_ptr]
        }
    }
}

/// Extension trait providing a method for Graph to access arithmetic operations
pub trait GraphArithmeticOpsExtension {
    /// Access arithmetic operations for this graph
    fn arithmetic_ops(&self) -> &dyn GraphArithmeticOps;
}

impl GraphArithmeticOpsExtension for Graph {
    fn arithmetic_ops(&self) -> &dyn GraphArithmeticOps {
        self
    }
}
