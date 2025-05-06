use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::tensor::DataType;
use crate::tensor::Tensor;

/// Linear algebra operations for Graph
pub trait GraphLinearAlgebraOps {
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

    /// Calculates the Hamming distance between two tensors.
    ///
    /// The Hamming distance between two tensors is the number of positions at which the corresponding elements
    /// are different. This operation computes the Hamming distance along the innermost dimension.
    ///
    /// # Arguments
    ///
    /// * `primary` - The first input tensor
    /// * `secondary` - The second input tensor
    /// * `result_data_type` - The result data type
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A new tensor with the Hamming distance between the inputs.
    fn hamming_distance(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        result_data_type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Performs scaled dot-product attention on the input tensors.
    ///
    /// Scaled dot-product attention is a key operation in transformer architectures. This operation
    /// computes attention weights by scaling the dot product of query and key, then applying softmax,
    /// and finally multiplying with value.
    ///
    /// The formula is: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
    ///
    /// # Arguments
    ///
    /// * `query` - The query tensor
    /// * `key` - The key tensor
    /// * `value` - The value tensor
    /// * `scale` - The scaling factor tensor (usually 1/sqrt(d_k))
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A new tensor containing the result of the attention operation
    fn scaled_dot_product_attention(
        &self,
        query: &Retained<Tensor>,
        key: &Retained<Tensor>,
        value: &Retained<Tensor>,
        scale: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Performs scaled dot-product attention with a scalar scaling factor.
    ///
    /// This is a variant of scaled dot-product attention where the scaling factor is a scalar value
    /// instead of a tensor.
    ///
    /// # Arguments
    ///
    /// * `query` - The query tensor
    /// * `key` - The key tensor
    /// * `value` - The value tensor
    /// * `scale` - The scalar scaling factor (usually 1/sqrt(d_k))
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A new tensor containing the result of the attention operation
    fn scaled_dot_product_attention_with_scalar(
        &self,
        query: &Retained<Tensor>,
        key: &Retained<Tensor>,
        value: &Retained<Tensor>,
        scale: f32,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Performs masked scaled dot-product attention.
    ///
    /// This is a variant of scaled dot-product attention where certain attention weights
    /// can be masked out (e.g., for causal attention).
    ///
    /// # Arguments
    ///
    /// * `query` - The query tensor
    /// * `key` - The key tensor
    /// * `value` - The value tensor
    /// * `mask` - The mask tensor (often used for causal attention)
    /// * `scale` - The scaling factor tensor (usually 1/sqrt(d_k))
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A new tensor containing the result of the attention operation
    fn masked_scaled_dot_product_attention(
        &self,
        query: &Retained<Tensor>,
        key: &Retained<Tensor>,
        value: &Retained<Tensor>,
        mask: &Retained<Tensor>,
        scale: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Performs masked scaled dot-product attention with a scalar scaling factor.
    ///
    /// This is a variant of masked scaled dot-product attention where the scaling factor is a scalar value
    /// instead of a tensor.
    ///
    /// # Arguments
    ///
    /// * `query` - The query tensor
    /// * `key` - The key tensor
    /// * `value` - The value tensor
    /// * `mask` - The mask tensor (often used for causal attention)
    /// * `scale` - The scalar scaling factor (usually 1/sqrt(d_k))
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A new tensor containing the result of the attention operation
    fn masked_scaled_dot_product_attention_with_scalar(
        &self,
        query: &Retained<Tensor>,
        key: &Retained<Tensor>,
        value: &Retained<Tensor>,
        mask: &Retained<Tensor>,
        scale: f32,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Calculates the determinant of a square matrix.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor representing a square matrix
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A tensor containing the determinant of the input matrix
    fn determinant(&self, tensor: &Retained<Tensor>, name: Option<&str>) -> Retained<Tensor>;

    /// Calculates the determinant for each matrix in a batch of matrices.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor containing a batch of matrices
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A tensor containing determinants for each matrix in the batch
    fn batched_determinant(
        &self,
        tensor: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Solves the triangular linear system with multiple right-hand sides.
    ///
    /// Solves a system of linear equations AX = B, where A is a triangular matrix.
    ///
    /// # Arguments
    ///
    /// * `matrix` - A tensor representing the triangular matrix A
    /// * `rhs` - A tensor representing the right-hand side B
    /// * `lower_triangular` - Whether A is lower triangular (true) or upper triangular (false)
    /// * `right_side` - Whether to solve A*X = B (false) or X*A = B (true)
    /// * `unit_diagonal` - Whether to assume the diagonal elements of A are all 1
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A tensor containing the solution X
    fn triangular_solve(
        &self,
        matrix: &Retained<Tensor>,
        rhs: &Retained<Tensor>,
        lower_triangular: bool,
        right_side: bool,
        unit_diagonal: bool,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Einstein summation (einsum) operation.
    ///
    /// Computes a generalized contraction between tensors according to the Einstein summation convention.
    ///
    /// # Arguments
    ///
    /// * `tensors` - A list of tensors to contract
    /// * `equation` - The einsum equation in the form of a string
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A tensor containing the result of the einsum operation
    fn einsum(
        &self,
        tensors: &[&Retained<Tensor>],
        equation: &str,
        name: Option<&str>,
    ) -> Retained<Tensor>;
}

/// Implementation of linear algebra operations for Graph
impl GraphLinearAlgebraOps for Graph {
    fn matmul(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                matrixMultiplicationWithPrimaryTensor: &**primary,
                secondaryTensor: &**secondary,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create matrix multiplication operation");
            } else {
                Retained::retain_autoreleased(result).unwrap()
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
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                matrixMultiplicationWithPrimaryTensor: &**primary,
                transposePrimary: primary_transpose,
                secondaryTensor: &**secondary,
                transposeSecondary: secondary_transpose,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create matrix multiplication with transpose operation");
            } else {
                Retained::retain_autoreleased(result).unwrap()
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
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                innerProductWithPrimaryTensor: &**primary,
                secondaryTensor: &**secondary,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create inner product operation");
            } else {
                Retained::retain_autoreleased(result).unwrap()
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
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                outerProductWithPrimaryTensor: &**primary,
                secondaryTensor: &**secondary,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create outer product operation");
            } else {
                Retained::retain_autoreleased(result).unwrap()
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
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                matrixMultiplicationWithPrimaryTensor: &**primary,
                secondaryTensor: &**secondary,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create batch matrix multiplication operation");
            } else {
                Retained::retain_autoreleased(result).unwrap()
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
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                matrixMultiplicationWithPrimaryTensor: &**primary,
                transposePrimary: primary_transpose,
                secondaryTensor: &**secondary,
                transposeSecondary: secondary_transpose,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create batch matrix multiplication with transpose operation");
            } else {
                Retained::retain_autoreleased(result).unwrap()
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
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                bandPartWithTensor: &**input,
                numLowerTensor: &**num_lower,
                numUpperTensor: &**num_upper,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create band part operation");
            } else {
                Retained::retain_autoreleased(result).unwrap()
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
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                bandPartWithTensor: &**input,
                numLower: num_lower,
                numUpper: num_upper,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create band part with scalars operation");
            } else {
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    fn hamming_distance(
        &self,
        primary: &Retained<Tensor>,
        secondary: &Retained<Tensor>,
        result_data_type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                HammingDistanceWithPrimaryTensor: &**primary,
                secondaryTensor: &**secondary,
                resultDataType: result_data_type as u32,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create hamming distance operation");
            } else {
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    fn scaled_dot_product_attention(
        &self,
        query: &Retained<Tensor>,
        key: &Retained<Tensor>,
        value: &Retained<Tensor>,
        scale: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                scaledDotProductAttentionWithQueryTensor: &**query,
                keyTensor: &**key,
                valueTensor: &**value,
                scaleTensor: &**scale,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create scaled dot product attention operation");
            } else {
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    fn scaled_dot_product_attention_with_scalar(
        &self,
        query: &Retained<Tensor>,
        key: &Retained<Tensor>,
        value: &Retained<Tensor>,
        scale: f32,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                scaledDotProductAttentionWithQueryTensor: &**query,
                keyTensor: &**key,
                valueTensor: &**value,
                scale: scale,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create scaled dot product attention with scalar operation");
            } else {
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    fn masked_scaled_dot_product_attention(
        &self,
        query: &Retained<Tensor>,
        key: &Retained<Tensor>,
        value: &Retained<Tensor>,
        mask: &Retained<Tensor>,
        scale: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                scaledDotProductAttentionWithQueryTensor: &**query,
                keyTensor: &**key,
                valueTensor: &**value,
                maskTensor: &**mask,
                scaleTensor: &**scale,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create masked scaled dot product attention operation");
            } else {
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    fn masked_scaled_dot_product_attention_with_scalar(
        &self,
        query: &Retained<Tensor>,
        key: &Retained<Tensor>,
        value: &Retained<Tensor>,
        mask: &Retained<Tensor>,
        scale: f32,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                scaledDotProductAttentionWithQueryTensor: &**query,
                keyTensor: &**key,
                valueTensor: &**value,
                maskTensor: &**mask,
                scale: scale,
                name: name_ptr
            ];

            if result.is_null() {
                panic!(
                    "Failed to create masked scaled dot product attention with scalar operation"
                );
            } else {
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    fn determinant(&self, tensor: &Retained<Tensor>, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                determinant: &**tensor,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create determinant operation");
            } else {
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    fn batched_determinant(
        &self,
        tensor: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                batchedDeterminant: &**tensor,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create batched determinant operation");
            } else {
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    fn triangular_solve(
        &self,
        matrix: &Retained<Tensor>,
        rhs: &Retained<Tensor>,
        lower_triangular: bool,
        right_side: bool,
        unit_diagonal: bool,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                triangularSolve: &**matrix,
                rhs: &**rhs,
                lowerTriangular: lower_triangular,
                rightSide: right_side,
                unitDiagonal: unit_diagonal,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create triangular solve operation");
            } else {
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }

    fn einsum(
        &self,
        tensors: &[&Retained<Tensor>],
        equation: &str,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);

            let equation_ns = NSString::from_str(equation);

            // Create NSArray by dereferencing each Retained<Tensor> to get the Tensor object
            let tensor_ptrs: Vec<*const Tensor> =
                tensors.iter().map(|t| &***t as *const _).collect();

            let result: *mut Tensor = msg_send![
                self,
                einsumWithTensors: tensor_ptrs.as_ptr(),
                equation: &*equation_ns,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create einsum operation");
            } else {
                Retained::retain_autoreleased(result).unwrap()
            }
        }
    }
}

/// Extension trait for easier access to linear algebra operations
pub trait GraphLinearAlgebraOpsExtension {
    /// Get access to linear algebra operations
    fn linear_algebra_ops(&self) -> &dyn GraphLinearAlgebraOps;
}

impl GraphLinearAlgebraOpsExtension for Graph {
    fn linear_algebra_ops(&self) -> &dyn GraphLinearAlgebraOps {
        self
    }
}
