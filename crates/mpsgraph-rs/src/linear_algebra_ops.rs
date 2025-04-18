use objc2::rc::Retained;
use objc2::msg_send;
use objc2_foundation::NSString;

use crate::graph::Graph;
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
    ) -> Option<Retained<Tensor>>;

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
    ) -> Option<Retained<Tensor>>;

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
    ) -> Option<Retained<Tensor>>;

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
    ) -> Option<Retained<Tensor>>;

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
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

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
        primary: &Tensor,
        primary_transpose: bool,
        secondary: &Tensor,
        secondary_transpose: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

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
        input: &Tensor,
        num_lower: &Tensor,
        num_upper: &Tensor,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

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
        input: &Tensor,
        num_lower: i64,
        num_upper: i64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Calculates the Hamming distance between two tensors.
    ///
    /// The Hamming distance between two tensors is the number of positions at which the corresponding elements
    /// are different. This operation computes the Hamming distance along the innermost dimension.
    ///
    /// # Arguments
    ///
    /// * `primary` - The first input tensor
    /// * `secondary` - The second input tensor
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A new tensor with the Hamming distance between the inputs.
    fn hamming_distance(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

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
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        scale: &Tensor,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

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
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        scale: f32,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

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
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: &Tensor,
        scale: &Tensor,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

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
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: &Tensor,
        scale: f32,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;
}

/// Implementation of linear algebra operations for Graph
impl GraphLinearAlgebraOps for Graph {
    fn matmul(
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
                matrixMultiplicationWithPrimaryTensor: &**primary,
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

    fn matmul_with_transpose(
        &self,
        primary: &Retained<Tensor>,
        primary_transpose: bool,
        secondary: &Retained<Tensor>,
        secondary_transpose: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
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
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn inner_product(
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
                innerProductWithPrimaryTensor: &**primary,
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

    fn outer_product(
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
                outerProductWithPrimaryTensor: &**primary,
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

    fn batch_matmul(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self, 
                matrixMultiplicationWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn batch_matmul_with_transpose(
        &self,
        primary: &Tensor,
        primary_transpose: bool,
        secondary: &Tensor,
        secondary_transpose: bool,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                matrixMultiplicationWithPrimaryTensor: primary,
                transposePrimary: primary_transpose,
                secondaryTensor: secondary,
                transposeSecondary: secondary_transpose,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn band_part(
        &self,
        input: &Tensor,
        num_lower: &Tensor,
        num_upper: &Tensor,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                bandPartWithTensor: input,
                numLowerTensor: num_lower,
                numUpperTensor: num_upper,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn band_part_with_scalars(
        &self,
        input: &Tensor,
        num_lower: i64,
        num_upper: i64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                bandPartWithTensor: input,
                numLower: num_lower,
                numUpper: num_upper,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn hamming_distance(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                hammingDistanceWithPrimaryTensor: primary,
                secondaryTensor: secondary,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn scaled_dot_product_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        scale: &Tensor,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                scaledDotProductAttentionWithQueryTensor: query,
                keyTensor: key,
                valueTensor: value,
                scaleTensor: scale,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn scaled_dot_product_attention_with_scalar(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        scale: f32,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                scaledDotProductAttentionWithQueryTensor: query,
                keyTensor: key,
                valueTensor: value,
                scaleScalar: scale as f64,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn masked_scaled_dot_product_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: &Tensor,
        scale: &Tensor,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                scaledDotProductAttentionWithQueryTensor: query,
                keyTensor: key,
                valueTensor: value,
                maskTensor: mask,
                scaleTensor: scale,
                name: name_ptr
            ];

            if result.is_null() {
                None
            } else {
                Some(Retained::from_raw(result).unwrap())
            }
        }
    }

    fn masked_scaled_dot_product_attention_with_scalar(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: &Tensor,
        scale: f32,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                scaledDotProductAttentionWithQueryTensor: query,
                keyTensor: key,
                valueTensor: value,
                maskTensor: mask,
                scaleScalar: scale as f64,
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