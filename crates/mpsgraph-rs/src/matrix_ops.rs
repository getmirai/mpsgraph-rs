use objc2::rc::Retained;
use objc2::msg_send;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::tensor::Tensor;

/// Trait for matrix operations on Graph
pub trait GraphMatrixOps {
    /// Creates a transpose operation
    ///
    /// # Arguments
    ///
    /// * `x` - The tensor to transpose
    /// * `dimensions` - The permutation of dimensions to apply
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object.
    fn transpose(
        &self,
        x: &Retained<Tensor>,
        dimensions: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a matrix multiplication operation.
    ///
    /// # Arguments
    ///
    /// * `primary` - First tensor input
    /// * `secondary` - Second tensor input
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object.
    fn matmul(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a matrix multiplication operation with transposed operands.
    ///
    /// # Arguments
    ///
    /// * `primary` - First tensor input
    /// * `primary_transpose` - Whether to transpose the first tensor
    /// * `secondary` - Second tensor input
    /// * `secondary_transpose` - Whether to transpose the second tensor
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object.
    fn matmul_with_transpose(
        &self,
        primary: &Retained<Tensor>,
        primary_transpose: bool,
        secondary: &Retained<Tensor>,
        secondary_transpose: bool,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a vector inner product operation.
    ///
    /// # Arguments
    ///
    /// * `primary` - First vector tensor
    /// * `secondary` - Second vector tensor
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object.
    fn inner_product(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a vector outer product operation.
    ///
    /// # Arguments
    ///
    /// * `primary` - First vector tensor
    /// * `secondary` - Second vector tensor
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object.
    fn outer_product(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a batch matrix multiplication operation.
    ///
    /// # Arguments
    ///
    /// * `primary` - First tensor input
    /// * `secondary` - Second tensor input
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object.
    fn batch_matmul(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a batch matrix multiplication operation with transposed operands.
    ///
    /// # Arguments
    ///
    /// * `primary` - First tensor input
    /// * `primary_transpose` - Whether to transpose the first tensor
    /// * `secondary` - Second tensor input
    /// * `secondary_transpose` - Whether to transpose the second tensor
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object.
    fn batch_matmul_with_transpose(
        &self,
        primary: &Retained<Tensor>,
        primary_transpose: bool,
        secondary: &Retained<Tensor>,
        secondary_transpose: bool,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a tensor with a band part extracted from the input tensor.
    ///
    /// This operation extracts a band of rows and columns from the input 2D tensor,
    /// where values outside the band are set to zero. The lower and upper diagonals
    /// specify the distance below and above the main diagonal that should be retained.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor (at least rank 2)
    /// * `num_lower` - The lower diagonal index value. If negative, retain the entire lower triangle.
    /// * `num_upper` - The upper diagonal index value. If negative, retain the entire upper triangle.
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object with the band part extracted.
    fn band_part(
        &self,
        input: &Retained<Tensor>,
        num_lower: &Retained<Tensor>,
        num_upper: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a tensor with a band part extracted from the input tensor using scalar values.
    ///
    /// This operation extracts a band of rows and columns from the input 2D tensor,
    /// where values outside the band are set to zero. The lower and upper diagonals
    /// specify the distance below and above the main diagonal that should be retained.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor (at least rank 2)
    /// * `num_lower` - The lower diagonal scalar value. If negative, retain the entire lower triangle.
    /// * `num_upper` - The upper diagonal scalar value. If negative, retain the entire upper triangle.
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object with the band part extracted.
    fn band_part_with_scalars(
        &self,
        input: &Retained<Tensor>,
        num_lower: i64,
        num_upper: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;
}

impl GraphMatrixOps for Graph {
    fn transpose(
        &self,
        x: &Retained<Tensor>,
        dimensions: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            // Create a shape object from the dimensions
            let dimensions_shape = crate::ShapeHelper::from_dimensions(dimensions);

            // Create the operation
            let result: *mut Tensor = msg_send![
                self,
                transposeTensor: &**x,
                permutation: &*dimensions_shape,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create transpose tensor");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn matmul(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                matrixMultiplicationWithPrimaryTensor: &**primary,
                secondaryTensor: &**secondary,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create matrix multiplication tensor");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn matmul_with_transpose(
        &self,
        primary: &Retained<Tensor>,
        primary_transpose: bool,
        secondary: &Retained<Tensor>,
        secondary_transpose: bool,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                matrixMultiplicationWithPrimaryTensor: &**primary,
                transposePrimary: primary_transpose,
                secondaryTensor: &**secondary,
                transposeSecondary: secondary_transpose,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create matrix multiplication with transpose tensor");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn inner_product(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                innerProductWithPrimaryTensor: &**primary,
                secondaryTensor: &**secondary,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create inner product tensor");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn outer_product(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                outerProductWithPrimaryTensor: &**primary,
                secondaryTensor: &**secondary,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create outer product tensor");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn batch_matmul(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                matrixMultiplicationWithPrimaryTensor: &**primary,
                secondaryTensor: &**secondary,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create batch matrix multiplication tensor");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn batch_matmul_with_transpose(
        &self,
        primary: &Retained<Tensor>,
        primary_transpose: bool,
        secondary: &Retained<Tensor>,
        secondary_transpose: bool,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                matrixMultiplicationWithPrimaryTensor: &**primary,
                transposePrimary: primary_transpose,
                secondaryTensor: &**secondary,
                transposeSecondary: secondary_transpose,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create batch matrix multiplication with transpose tensor");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn band_part(
        &self,
        input: &Retained<Tensor>,
        num_lower: &Retained<Tensor>,
        num_upper: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                bandPartWithTensor: &**input,
                numLower: &**num_lower,
                numUpper: &**num_upper,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create band part tensor");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn band_part_with_scalars(
        &self,
        input: &Retained<Tensor>,
        num_lower: i64,
        num_upper: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                bandPartWithTensor: &**input,
                numLowerScalar: num_lower,
                numUpperScalar: num_upper,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create band part with scalars tensor");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }
}