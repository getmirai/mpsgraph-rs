use objc2::rc::Retained;
use objc2::msg_send;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::tensor::Tensor;

/// Trait for arithmetic operations on Graph
pub trait GraphArithmeticOps {
    // MARK: - Unary Operations

    /// Creates an identity operation
    fn identity(&self, x: &Retained<Tensor>, name: Option<&str>) -> Option<Retained<Tensor>>;

    /// Returns e raised to the power of the input tensor
    fn exp(&self, x: &Retained<Tensor>, name: Option<&str>) -> Option<Retained<Tensor>>;

    /// Returns 2 raised to the power of the input tensor
    fn exp2(&self, x: &Retained<Tensor>, name: Option<&str>) -> Option<Retained<Tensor>>;

    /// Returns the natural logarithm of the input tensor
    fn log(&self, x: &Retained<Tensor>, name: Option<&str>) -> Option<Retained<Tensor>>;

    /// Returns the base-2 logarithm of the input tensor
    fn log2(&self, x: &Retained<Tensor>, name: Option<&str>) -> Option<Retained<Tensor>>;

    /// Returns the square of the input tensor
    fn square(&self, x: &Retained<Tensor>, name: Option<&str>) -> Option<Retained<Tensor>>;

    /// Returns the square root of the input tensor
    fn sqrt(&self, x: &Retained<Tensor>, name: Option<&str>) -> Option<Retained<Tensor>>;

    /// Returns the reciprocal square root of the input tensor
    fn rsqrt(&self, x: &Retained<Tensor>, name: Option<&str>) -> Option<Retained<Tensor>>;

    /// Returns the reciprocal of the input tensor
    fn reciprocal(&self, x: &Retained<Tensor>, name: Option<&str>) -> Option<Retained<Tensor>>;

    /// Returns the absolute value of the input tensor
    fn abs(&self, x: &Retained<Tensor>, name: Option<&str>) -> Option<Retained<Tensor>>;

    /// Returns the negation of the input tensor
    fn negative(&self, x: &Retained<Tensor>, name: Option<&str>) -> Option<Retained<Tensor>>;

    /// Returns a tensor with the sign of each element in the input tensor
    /// -1 for negative, 0 for zero, 1 for positive
    fn sign(&self, x: &Retained<Tensor>, name: Option<&str>) -> Option<Retained<Tensor>>;

    /// Returns the ceiling of the input tensor
    fn ceil(&self, x: &Retained<Tensor>, name: Option<&str>) -> Option<Retained<Tensor>>;

    /// Returns the floor of the input tensor
    fn floor(&self, x: &Retained<Tensor>, name: Option<&str>) -> Option<Retained<Tensor>>;

    /// Returns the rounded value of the input tensor
    fn round(&self, x: &Retained<Tensor>, name: Option<&str>) -> Option<Retained<Tensor>>;

    // MARK: - Binary Operations

    /// Performs addition of two tensors
    fn add(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Performs subtraction of two tensors
    fn subtract(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Performs multiplication of two tensors
    fn multiply(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Performs division of two tensors
    fn divide(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Performs division of two tensors but returns 0 if secondary is 0
    fn division_no_nan(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Performs modulo operation between two tensors
    fn modulo(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Raises primary tensor to the power of secondary tensor
    fn power(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Returns the minimum of two tensors
    fn minimum(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Returns the maximum of two tensors
    fn maximum(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    // MARK: - Comparison Operations

    /// Returns element-wise equal comparison
    fn equal(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Returns element-wise not equal comparison
    fn not_equal(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Returns element-wise less than comparison
    fn less_than(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Returns element-wise less than or equal comparison
    fn less_than_or_equal(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Returns element-wise greater than comparison
    fn greater_than(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Returns element-wise greater than or equal comparison
    fn greater_than_or_equal(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    // MARK: - Logical Operations

    /// Performs logical AND of two tensors
    fn logical_and(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Performs logical OR of two tensors
    fn logical_or(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Returns the logical NOT of the input tensor
    fn logical_not(&self, x: &Retained<Tensor>, name: Option<&str>) -> Option<Retained<Tensor>>;

    // MARK: - Ternary Operations

    /// Creates a select operation which chooses values from true or false tensor based on predicate
    fn select(
        &self,
        predicate: &Retained<Tensor>,
        true_tensor: &Retained<Tensor>,
        false_tensor: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a clamp operation that clamps values to the given min and max
    fn clamp(
        &self,
        tensor: &Retained<Tensor>,
        min_tensor: &Retained<Tensor>,
        max_tensor: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;
}

impl GraphArithmeticOps for Graph {
    fn identity(&self, x: &Retained<Tensor>, name: Option<&str>) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                identityWithTensor: &**x,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn exp(&self, x: &Retained<Tensor>, name: Option<&str>) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                exponentWithTensor: &**x,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn exp2(&self, x: &Retained<Tensor>, name: Option<&str>) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                exponentBase2WithTensor: &**x,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn log(&self, x: &Retained<Tensor>, name: Option<&str>) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                logarithmWithTensor: &**x,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn log2(&self, x: &Retained<Tensor>, name: Option<&str>) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                logarithmBase2WithTensor: &**x,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn square(&self, x: &Retained<Tensor>, name: Option<&str>) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                squareWithTensor: &**x,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn sqrt(&self, x: &Retained<Tensor>, name: Option<&str>) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                squareRootWithTensor: &**x,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn rsqrt(&self, x: &Retained<Tensor>, name: Option<&str>) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                reciprocalSquareRootWithTensor: &**x,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn reciprocal(&self, x: &Retained<Tensor>, name: Option<&str>) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                reciprocalWithTensor: &**x,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn abs(&self, x: &Retained<Tensor>, name: Option<&str>) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                absoluteWithTensor: &**x,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn negative(&self, x: &Retained<Tensor>, name: Option<&str>) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                negativeWithTensor: &**x,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn sign(&self, x: &Retained<Tensor>, name: Option<&str>) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                signWithTensor: &**x,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn ceil(&self, x: &Retained<Tensor>, name: Option<&str>) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                ceilWithTensor: &**x,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn floor(&self, x: &Retained<Tensor>, name: Option<&str>) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                floorWithTensor: &**x,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn round(&self, x: &Retained<Tensor>, name: Option<&str>) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                roundWithTensor: &**x,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn add(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                additionWithPrimaryTensor: &**primary,
                secondaryTensor: &**secondary,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn subtract(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                subtractionWithPrimaryTensor: &**primary,
                secondaryTensor: &**secondary,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn multiply(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                multiplicationWithPrimaryTensor: &**primary,
                secondaryTensor: &**secondary,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn divide(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                divisionWithPrimaryTensor: &**primary,
                secondaryTensor: &**secondary,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn division_no_nan(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                divisionNoNaNWithPrimaryTensor: &**primary,
                secondaryTensor: &**secondary,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn modulo(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                moduloWithPrimaryTensor: &**primary,
                secondaryTensor: &**secondary,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn power(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                powerWithPrimaryTensor: &**primary,
                secondaryTensor: &**secondary,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn minimum(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                minimumWithPrimaryTensor: &**primary,
                secondaryTensor: &**secondary,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn maximum(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                maximumWithPrimaryTensor: &**primary,
                secondaryTensor: &**secondary,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn equal(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                equalWithPrimaryTensor: &**primary,
                secondaryTensor: &**secondary,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn not_equal(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                notEqualWithPrimaryTensor: &**primary,
                secondaryTensor: &**secondary,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn less_than(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                lessThanWithPrimaryTensor: &**primary,
                secondaryTensor: &**secondary,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn less_than_or_equal(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                lessThanOrEqualWithPrimaryTensor: &**primary,
                secondaryTensor: &**secondary,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn greater_than(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                greaterThanWithPrimaryTensor: &**primary,
                secondaryTensor: &**secondary,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn greater_than_or_equal(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                greaterThanOrEqualWithPrimaryTensor: &**primary,
                secondaryTensor: &**secondary,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn logical_and(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                logicalANDWithPrimaryTensor: &**primary,
                secondaryTensor: &**secondary,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn logical_or(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                logicalORWithPrimaryTensor: &**primary,
                secondaryTensor: &**secondary,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn logical_not(&self, x: &Retained<Tensor>, name: Option<&str>) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                logicalNOTWithTensor: &**x,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn select(
        &self,
        predicate: &Retained<Tensor>,
        true_tensor: &Retained<Tensor>,
        false_tensor: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                selectWithPredicateTensor: &**predicate,
                truePredicateTensor: &**true_tensor,
                falsePredicateTensor: &**false_tensor,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn clamp(
        &self,
        tensor: &Retained<Tensor>,
        min_tensor: &Retained<Tensor>,
        max_tensor: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                clampWithTensor: &**tensor,
                minValueTensor: &**min_tensor,
                maxValueTensor: &**max_tensor,
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