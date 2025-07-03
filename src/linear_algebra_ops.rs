//! Linear algebra helpers now implemented directly on `Graph`.

use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::{NSArray, NSString};

use crate::graph::Graph;
use crate::tensor::{DataType, Tensor};

impl Graph {
    pub fn matmul(
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

    pub fn matmul_with_transpose(
        &self,
        primary: &Tensor,
        primary_transpose: bool,
        secondary: &Tensor,
        secondary_transpose: bool,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, matrixMultiplicationWithPrimaryTensor: primary, transposePrimary: primary_transpose, secondaryTensor: secondary, transposeSecondary: secondary_transpose, name: name_ptr]
        }
    }

    pub fn inner_product(
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
            msg_send![self, innerProductWithPrimaryTensor: primary, secondaryTensor: secondary, name: name_ptr]
        }
    }

    pub fn outer_product(
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
            msg_send![self, outerProductWithPrimaryTensor: primary, secondaryTensor: secondary, name: name_ptr]
        }
    }

    pub fn batch_matmul(
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

    pub fn batch_matmul_with_transpose(
        &self,
        primary: &Tensor,
        primary_transpose: bool,
        secondary: &Tensor,
        secondary_transpose: bool,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, matrixMultiplicationWithPrimaryTensor: primary, transposePrimary: primary_transpose, secondaryTensor: secondary, transposeSecondary: secondary_transpose, name: name_ptr]
        }
    }

    pub fn band_part(
        &self,
        input: &Tensor,
        num_lower: &Tensor,
        num_upper: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, bandPartWithTensor: input, numLowerTensor: num_lower, numUpperTensor: num_upper, name: name_ptr]
        }
    }

    pub fn band_part_with_scalars(
        &self,
        input: &Tensor,
        num_lower: i64,
        num_upper: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, bandPartWithTensor: input, numLower: num_lower, numUpper: num_upper, name: name_ptr]
        }
    }

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

    pub fn scaled_dot_product_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        scale: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, scaledDotProductAttentionWithQueryTensor: query, keyTensor: key, valueTensor: value, scaleTensor: scale, name: name_ptr]
        }
    }

    pub fn scaled_dot_product_attention_with_scalar(
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
            msg_send![self, scaledDotProductAttentionWithQueryTensor: query, keyTensor: key, valueTensor: value, scale: scale, name: name_ptr]
        }
    }

    pub fn masked_scaled_dot_product_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: &Tensor,
        scale: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, scaledDotProductAttentionWithQueryTensor: query, keyTensor: key, valueTensor: value, maskTensor: mask, scaleTensor: scale, name: name_ptr]
        }
    }

    pub fn masked_scaled_dot_product_attention_with_scalar(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: &Tensor,
        scale: f32,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, scaledDotProductAttentionWithQueryTensor: query, keyTensor: key, valueTensor: value, maskTensor: mask, scale: scale, name: name_ptr]
        }
    }

    pub fn determinant(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, determinant: tensor, name: name_ptr]
        }
    }

    pub fn batched_determinant(&self, tensor: &Tensor, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, batchedDeterminant: tensor, name: name_ptr]
        }
    }

    pub fn triangular_solve(
        &self,
        matrix: &Tensor,
        rhs: &Tensor,
        lower_triangular: bool,
        right_side: bool,
        unit_diagonal: bool,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![self, triangularSolve: matrix, rhs: rhs, lowerTriangular: lower_triangular, rightSide: right_side, unitDiagonal: unit_diagonal, name: name_ptr]
        }
    }

    pub fn einsum(
        &self,
        tensors: &[&Tensor],
        equation: &str,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ptr = name
                .map(NSString::from_str)
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            let equation_ns = NSString::from_str(equation);
            let tensors_array = NSArray::from_slice(tensors);
            msg_send![self, einsumWithTensors: &*tensors_array, equation: &*equation_ns, name: name_ptr]
        }
    }
}
