use objc2::rc::Retained;
use objc2::msg_send;
use objc2_foundation::NSString;

use crate::graph::Graph;
use crate::tensor::Tensor;
use crate::tensor::DataType;
use crate::core::create_ns_array_from_i64_slice;
use crate::core::create_ns_array_from_slice;

/// Trait for tensor shape operations on Graph
pub trait GraphTensorShapeOps {
    /// Creates a reshape operation
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `shape` - New shape for the tensor
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn reshape(
        &self,
        x: &Retained<Tensor>,
        shape: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a flatten2D operation
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `axis` - Axis to flatten at
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn flatten2d(
        &self,
        x: &Retained<Tensor>,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a broadcast operation
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `shape` - Target shape to broadcast to
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn broadcast(
        &self,
        x: &Retained<Tensor>,
        shape: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a shape-of operation
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object containing the shape
    fn shape_of(
        &self,
        x: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a cast operation to change tensor data type
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `data_type` - Target data type
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object with the new data type
    fn cast(
        &self,
        x: &Retained<Tensor>,
        data_type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a stack operation to combine tensors along a new axis
    ///
    /// # Arguments
    ///
    /// * `tensors` - Array of tensors to stack
    /// * `axis` - Axis along which to stack
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn stack(
        &self,
        tensors: &[&Retained<Tensor>],
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a split operation to split a tensor into multiple parts
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `num_splits` - Number of splits to create
    /// * `axis` - Axis along which to split
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A vector of Tensor objects
    fn split(
        &self,
        x: &Retained<Tensor>,
        num_splits: i64,
        axis: i64,
        name: Option<&str>,
    ) -> Vec<Retained<Tensor>>;

    /// Creates a squeeze operation to remove dimensions of size 1 at specified axes
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `axes` - Axes to squeeze
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn squeeze(
        &self,
        x: &Retained<Tensor>,
        axes: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor>;
    
    /// Creates a squeeze operation to remove all dimensions of size 1
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn squeeze_all(
        &self,
        x: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor>;
    
    /// Creates a squeeze operation to remove a dimension of size 1 at the specified axis
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `axis` - The axis to squeeze
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn squeeze_axis(
        &self,
        x: &Retained<Tensor>,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;
    
    /// Creates a squeeze operation to remove dimensions with size 1 specified by a tensor
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `axes_tensor` - The tensor containing the axes to squeeze
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn squeeze_with_tensor(
        &self,
        x: &Retained<Tensor>,
        axes_tensor: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates an expand_dims operation to insert dimensions of size 1
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `axes` - Axes at which to insert new dimensions
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn expand_dims(
        &self,
        x: &Retained<Tensor>,
        axes: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor>;
    
    /// Creates an expand_dims operation to insert a dimension of size 1 at the specified axis
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `axis` - The axis to expand
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn expand_dims_axis(
        &self,
        x: &Retained<Tensor>,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;
    
    /// Creates an expand_dims operation to insert dimensions with size 1 specified by a tensor
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `axes_tensor` - The tensor containing the axes to expand
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn expand_dims_with_tensor(
        &self,
        x: &Retained<Tensor>,
        axes_tensor: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a tile operation to repeat a tensor along specified dimensions
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `multiples` - Number of times to repeat in each dimension
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn tile(
        &self,
        x: &Retained<Tensor>,
        multiples: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a pad operation to pad a tensor
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `padding` - Padding specification
    /// * `constant` - Value to use for padding
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn pad(
        &self,
        x: &Retained<Tensor>,
        padding: &[i64],
        constant: f32,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a space-to-depth operation
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `block_size` - Size of spatial blocks
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn space_to_depth(
        &self,
        x: &Retained<Tensor>,
        block_size: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a depth-to-space operation
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `block_size` - Size of spatial blocks
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn depth_to_space(
        &self,
        x: &Retained<Tensor>,
        block_size: i64,
        name: Option<&str>,
    ) -> Retained<Tensor>;

    /// Creates a reverse operation to reverse a tensor along specified dimensions
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    /// * `axes` - Axes along which to reverse
    /// * `name` - Optional name for the operation
    ///
    /// # Returns
    ///
    /// A valid Tensor object
    fn reverse(
        &self,
        x: &Retained<Tensor>,
        axes: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor>;
}

impl GraphTensorShapeOps for Graph {
    fn reshape(
        &self,
        x: &Retained<Tensor>,
        shape: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let shape_array = create_ns_array_from_i64_slice(shape);
            let shape_ptr = &*shape_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                reshapeTensor: &**x,
                withShape: shape_ptr,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create reshape operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn flatten2d(
        &self,
        x: &Retained<Tensor>,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                flatten2DTensor: &**x,
                axis: axis,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create flatten2d operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn broadcast(
        &self,
        x: &Retained<Tensor>,
        shape: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let shape_array = create_ns_array_from_i64_slice(shape);
            let shape_ptr = &*shape_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                broadcastTensor: &**x,
                toShape: shape_ptr,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create broadcast operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn shape_of(
        &self,
        x: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                shapeOfTensor: &**x,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create shape_of operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn cast(
        &self,
        x: &Retained<Tensor>,
        data_type: DataType,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                castTensor: &**x,
                toType: data_type as u32,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create cast operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn stack(
        &self,
        tensors: &[&Retained<Tensor>],
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            // Create array of tensors
            let tensor_ptrs: Vec<*const Tensor> = tensors
                .iter()
                .map(|t| &***t as *const Tensor)
                .collect();
            
            let tensor_array = create_ns_array_from_slice(&tensor_ptrs);
            let tensor_array_ptr = &*tensor_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                stackTensors: tensor_array_ptr,
                axis: axis,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create stack operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn split(
        &self,
        x: &Retained<Tensor>,
        num_splits: i64,
        axis: i64,
        name: Option<&str>,
    ) -> Vec<Retained<Tensor>> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result_array: *mut objc2_foundation::NSArray<Tensor> = msg_send![
                self,
                splitTensor: &**x,
                numSplits: num_splits,
                axis: axis,
                name: name_ptr
            ];

            if result_array.is_null() {
                return Vec::new();
            }

            // Convert NSArray to Vec<Retained<Tensor>>
            let count = (*result_array).count();
            let mut tensors = Vec::with_capacity(count);

            for i in 0..count {
                let tensor: *mut Tensor = msg_send![result_array, objectAtIndex: i];
                if !tensor.is_null() {
                    tensors.push(Retained::from_raw(tensor).unwrap());
                }
            }

            tensors
        }
    }

    fn squeeze(
        &self,
        x: &Retained<Tensor>,
        axes: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let axes_array = create_ns_array_from_i64_slice(axes);
            let axes_ptr = &*axes_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                squeezeTensor: &**x,
                axes: axes_ptr,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create squeeze operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn expand_dims(
        &self,
        x: &Retained<Tensor>,
        axes: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let axes_array = create_ns_array_from_i64_slice(axes);
            let axes_ptr = &*axes_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                expandDimsTensor: &**x,
                axes: axes_ptr,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create expand_dims operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn tile(
        &self,
        x: &Retained<Tensor>,
        multiples: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let multiples_array = create_ns_array_from_i64_slice(multiples);
            let multiples_ptr = &*multiples_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                tileTensor: &**x,
                withMultiples: multiples_ptr,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create tile operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn pad(
        &self,
        x: &Retained<Tensor>,
        padding: &[i64],
        constant: f32,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let padding_array = create_ns_array_from_i64_slice(padding);
            let padding_ptr = &*padding_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                padTensor: &**x,
                paddings: padding_ptr,
                constantValue: constant as f64,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create pad operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn space_to_depth(
        &self,
        x: &Retained<Tensor>,
        block_size: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                spaceToDepthWithTensor: &**x,
                blockSize: block_size,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create space_to_depth operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn depth_to_space(
        &self,
        x: &Retained<Tensor>,
        block_size: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                depthToSpaceWithTensor: &**x,
                blockSize: block_size,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create depth_to_space operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn reverse(
        &self,
        x: &Retained<Tensor>,
        axes: &[i64],
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let axes_array = create_ns_array_from_i64_slice(axes);
            let axes_ptr = &*axes_array as *const _;

            let result: *mut Tensor = msg_send![
                self,
                reverseTensor: &**x,
                axes: axes_ptr,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create reverse operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn squeeze_all(
        &self,
        x: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                squeezeTensor: &**x,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create squeeze_all operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn squeeze_axis(
        &self,
        x: &Retained<Tensor>,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                squeezeTensor: &**x,
                axis: axis,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create squeeze_axis operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }

    fn squeeze_with_tensor(
        &self,
        x: &Retained<Tensor>,
        axes_tensor: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                squeezeTensor: &**x,
                axesTensor: &**axes_tensor,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create squeeze_with_tensor operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }
    
    fn expand_dims_axis(
        &self,
        x: &Retained<Tensor>,
        axis: i64,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                expandDimsOfTensor: &**x,
                axis: axis,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create expand_dims_axis operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }
    
    fn expand_dims_with_tensor(
        &self,
        x: &Retained<Tensor>,
        axes_tensor: &Retained<Tensor>,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_ns = name.map(NSString::from_str);
            let name_ptr = name_ns.as_deref().map_or(std::ptr::null(), |s| s as *const _);

            let result: *mut Tensor = msg_send![
                self,
                expandDimsOfTensor: &**x,
                axesTensor: &**axes_tensor,
                name: name_ptr
            ];

            if result.is_null() {
                panic!("Failed to create expand_dims_with_tensor operation");
            } else {
                Retained::from_raw(result).unwrap()
            }
        }
    }
}

/// Extension trait providing a method for Graph to access tensor shape operations
pub trait GraphTensorShapeOpsExtension {
    /// Access tensor shape operations for this graph
    fn tensor_shape_ops(&self) -> &dyn GraphTensorShapeOps;
}

impl GraphTensorShapeOpsExtension for Graph {
    fn tensor_shape_ops(&self) -> &dyn GraphTensorShapeOps {
        self
    }
}