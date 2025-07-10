//! Matrix multiplication and linear algebra helpers implemented on `Graph`.

use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::tensor::{DataType, Tensor};

impl Graph {
    /// Computes the matrix multiplication of 2 input tensors with support for broadcasting.
    ///
    /// - Parameters:
    ///   - primary: The left-hand side tensor.
    ///   - secondary: The right-hand side tensor.
    ///   - name: The name for the operation.
    /// - Returns: A valid tensor containing the product of the input matrices.
    pub fn matrix_multiplication(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, matrixMultiplicationWithPrimaryTensor: primary, secondaryTensor: secondary, name: name_ptr]
        }
    }

    /// Computes the hamming distance of two input tensors with support for broadcasting.
    ///
    /// The hamming distance is computed between 2 sets of vectors and the last dimension(s) of each
    /// input tensor is considered a vector.
    ///
    /// - Parameters:
    ///   - primary: The first input tensor.
    ///   - secondary: The second input tensor.
    ///   - result_data_type: The datatype of the return Tensor. Must be either `DataType::UInt32` or `DataType::UInt16`.
    ///   - name: The name for the operation.
    /// - Returns: A valid tensor containing the hamming distance between the input tensors.
    pub fn hamming_distance(
        &self,
        primary: &Tensor,
        secondary: &Tensor,
        result_data_type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, HammingDistanceWithPrimaryTensor: primary, secondaryTensor: secondary, resultDataType: result_data_type as u32, name: name_ptr]
        }
    }

    /// Creates a scaled dot product attention (SDPA) operation and returns the result tensor.
    ///
    /// SDPA Op computes attention by computing softmax(scale * QK^T + M)V.
    /// query_tensor Q with shape [B, Hq, Nq, F] and key_tensor K with shape [B, Hq, Nkv, F],
    /// with Q's H dimension expandable to satisfy matmul QK^T. mask_tensor M's shape
    /// should be broadcast compatible to satisfy (QK^T + M). value_tensor V with shape
    /// [B, Hv, Nkv, F] should satisfy the matmul (QK^T + M)V.
    ///
    /// - Parameters:
    ///   - query: A tensor that represents the query projection.
    ///   - key: A tensor that represents the key projection.
    ///   - value: A tensor that represents the value projection.
    ///   - mask_tensor: An optional tensor that contains a mask that is applied to the scaled, matrix
    ///   multiplied query and value matrices. If mask tensor is nil, the QK^T is not element-wise masked.
    ///   - scale: A scale that is applied to the result of query and value matrix multiply.
    ///   - name: The name for the operation.
    /// - Returns: A valid Tensor object.
    pub fn scaled_dot_product_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask_tensor: &Tensor,
        scale: f32,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                scaledDotProductAttentionWithQueryTensor: query,
                keyTensor: key,
                valueTensor: value,
                maskTensor: mask_tensor,
                scale: scale,
                name: name_ptr
            ]
        }
    }
}

pub trait ScaledDotProductAttentionExt {
    fn scaled_dot_product_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        scale: f32,
        name: Option<&str>,
    ) -> Retained<Tensor>;
}

impl ScaledDotProductAttentionExt for Graph {
    /// Creates a scaled dot product attention (SDPA) operation (without a mask) and returns the result tensor.
    ///
    /// - Parameters:
    ///   - query: A tensor that represents the query projection.
    ///   - key: A tensor that represents the key projection.
    ///   - value: A tensor that represents the value projection.
    ///   - scale: A scale that is applied on the result of query and value matrix multiply.
    ///   - name: The name for the operation.
    /// - Returns: A valid Tensor object.
    fn scaled_dot_product_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        scale: f32,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                scaledDotProductAttentionWithQueryTensor: query,
                keyTensor: key,
                valueTensor: value,
                scale: scale,
                name: name_ptr
            ]
        }
    }
}
