use objc2::msg_send;
use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::core::DataType;
use crate::graph::Graph;
use crate::tensor::Tensor;

/// One-hot operations for Graph
pub trait GraphOneHotOps {
    /// Creates a oneHot operation and returns the result tensor.
    ///
    /// Creates a tensor of rank equal to the indicesTensor rank + 1.
    /// Inserts a new axis at the axis specified, or the minor axis if axis is -1.
    /// The values at the indices in the indicesTensor will have the onValue,
    /// and all other values will be set to the offValue.
    ///
    /// # Arguments
    ///
    /// * `indices_tensor` - Tensor of indices for on values
    /// * `depth` - Depth of the oneHot vector along the axis
    /// * `axis` - The axis to insert the new oneHot vector at
    /// * `data_type` - DataType of the result tensor
    /// * `on_value` - The value for indices designated by the indicesTensor
    /// * `off_value` - The value for indices not designated by the indicesTensor
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn one_hot(
        &self,
        indices_tensor: &Tensor,
        depth: usize,
        axis: usize,
        data_type: DataType,
        on_value: f64,
        off_value: f64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a oneHot operation with default axis (the minor dimension).
    ///
    /// # Arguments
    ///
    /// * `indices_tensor` - Tensor of indices for on values
    /// * `depth` - Depth of the oneHot vector along the axis
    /// * `data_type` - DataType of the result tensor
    /// * `on_value` - The value for indices designated by the indicesTensor
    /// * `off_value` - The value for indices not designated by the indicesTensor
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn one_hot_default_axis(
        &self,
        indices_tensor: &Tensor,
        depth: usize,
        data_type: DataType,
        on_value: f64,
        off_value: f64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a oneHot operation with default on/off values (1.0 and 0.0).
    ///
    /// # Arguments
    ///
    /// * `indices_tensor` - Tensor of indices for on values
    /// * `depth` - Depth of the oneHot vector along the axis
    /// * `axis` - The axis to insert the new oneHot vector at
    /// * `data_type` - DataType of the result tensor
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn one_hot_default_values(
        &self,
        indices_tensor: &Tensor,
        depth: usize,
        axis: usize,
        data_type: DataType,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a oneHot operation with default axis and float32 data type (simplest version).
    ///
    /// # Arguments
    ///
    /// * `indices_tensor` - Tensor of indices for on values
    /// * `depth` - Depth of the oneHot vector along the axis
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn one_hot_simple(
        &self,
        indices_tensor: &Tensor,
        depth: usize,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a oneHot operation with default axis and default values.
    ///
    /// # Arguments
    ///
    /// * `indices_tensor` - Tensor of indices for on values
    /// * `depth` - Depth of the oneHot vector along the axis
    /// * `data_type` - DataType of the result tensor
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn one_hot_default_axis_values(
        &self,
        indices_tensor: &Tensor,
        depth: usize,
        data_type: DataType,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;

    /// Creates a oneHot operation with default data type (Float32) and default values.
    ///
    /// # Arguments
    ///
    /// * `indices_tensor` - Tensor of indices for on values
    /// * `depth` - Depth of the oneHot vector along the axis
    /// * `axis` - The axis to insert the new oneHot vector at
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn one_hot_default_type_values(
        &self,
        indices_tensor: &Tensor,
        depth: usize,
        axis: usize,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>>;
}

/// Implementation of one-hot operations for Graph
impl GraphOneHotOps for Graph {
    fn one_hot(
        &self,
        indices_tensor: &Tensor,
        depth: usize,
        axis: usize,
        data_type: DataType,
        on_value: f64,
        off_value: f64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                oneHotWithIndicesTensor: indices_tensor,
                depth: depth,
                axis: axis,
                dataType: data_type as u64,
                onValue: on_value,
                offValue: off_value,
                name: name_ptr
            ]
        }
    }

    fn one_hot_default_axis(
        &self,
        indices_tensor: &Tensor,
        depth: usize,
        data_type: DataType,
        on_value: f64,
        off_value: f64,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                oneHotWithIndicesTensor: indices_tensor,
                depth: depth,
                dataType: data_type as u64,
                onValue: on_value,
                offValue: off_value,
                name: name_ptr
            ]
        }
    }

    fn one_hot_default_values(
        &self,
        indices_tensor: &Tensor,
        depth: usize,
        axis: usize,
        data_type: DataType,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                oneHotWithIndicesTensor: indices_tensor,
                depth: depth,
                axis: axis,
                dataType: data_type as u64,
                name: name_ptr
            ]
        }
    }

    fn one_hot_simple(
        &self,
        indices_tensor: &Tensor,
        depth: usize,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                oneHotWithIndicesTensor: indices_tensor,
                depth: depth,
                name: name_ptr
            ]
        }
    }

    fn one_hot_default_axis_values(
        &self,
        indices_tensor: &Tensor,
        depth: usize,
        data_type: DataType,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                oneHotWithIndicesTensor: indices_tensor,
                depth: depth,
                dataType: data_type as u64,
                name: name_ptr
            ]
        }
    }

    fn one_hot_default_type_values(
        &self,
        indices_tensor: &Tensor,
        depth: usize,
        axis: usize,
        name: Option<&str>,
    ) -> Option<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns
                .as_deref()
                .map_or(std::ptr::null(), |s| s as *const _);
            msg_send![
                self,
                oneHotWithIndicesTensor: indices_tensor,
                depth: depth,
                axis: axis,
                name: name_ptr
            ]
        }
    }
}

/// Extension trait for easier access to one-hot operations
pub trait GraphOneHotOpsExtension {
    /// Get access to one-hot operations
    fn one_hot_ops(&self) -> &dyn GraphOneHotOps;
}

impl GraphOneHotOpsExtension for Graph {
    fn one_hot_ops(&self) -> &dyn GraphOneHotOps {
        self
    }
}
