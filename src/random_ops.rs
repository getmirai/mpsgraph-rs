use crate::core::DataType;
use crate::graph::Graph;
use crate::shape::Shape;
use crate::tensor::Tensor;
use objc2::extern_class;
use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::AnyClass;
use objc2_foundation::{NSArray, NSObject, NSObjectProtocol, NSString};

/// Random distribution types supported by MPSGraph
#[repr(u64)]
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum RandomDistribution {
    /// Uniform distribution, with samples drawn uniformly from [min, max) for float types,
    /// and [min, max] for integer types
    Uniform = 0,
    /// Normal distribution defined by mean and standard deviation
    Normal = 1,
    /// Normal distribution defined by mean and standard deviation, truncated to range [min, max)
    TruncatedNormal = 2,
}

/// Sampling methods for normal distributions
#[repr(u64)]
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum RandomNormalSamplingMethod {
    /// Use inverse erf to convert uniform values to values in the normal distribution
    InverseCDF = 0,
    /// Use Box Muller transform to convert uniform values to values in the normal distribution
    BoxMuller = 1,
}

extern_class!(
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraphRandomOpDescriptor"]
    /// Descriptor for random operations in MPSGraph
    pub struct RandomOpDescriptor;
);

unsafe impl NSObjectProtocol for RandomOpDescriptor {}

impl RandomOpDescriptor {
    /// Creates a new random operation descriptor with the specified distribution and data type
    pub fn new(distribution: RandomDistribution, data_type: DataType) -> Retained<Self> {
        unsafe {
            let cls = AnyClass::get(c"MPSGraphRandomOpDescriptor").unwrap();
            msg_send![
                cls,
                descriptorWithDistribution: distribution as u64,
                dataType: data_type as u32
            ]
        }
    }

    /// Sets the minimum value (for float data types)
    pub fn set_min(&self, min: f32) -> &Self {
        unsafe {
            let _: () = msg_send![self, setMin: min];
        }
        self
    }

    /// Sets the maximum value (for float data types)
    pub fn set_max(&self, max: f32) -> &Self {
        unsafe {
            let _: () = msg_send![self, setMax: max];
        }
        self
    }

    /// Sets the minimum integer value (for integer data types)
    pub fn set_min_integer(&self, min: i64) -> &Self {
        unsafe {
            let _: () = msg_send![self, setMinInteger: min];
        }
        self
    }

    /// Sets the maximum integer value (for integer data types)
    pub fn set_max_integer(&self, max: i64) -> &Self {
        unsafe {
            let _: () = msg_send![self, setMaxInteger: max];
        }
        self
    }

    /// Sets the mean (for normal distributions)
    pub fn set_mean(&self, mean: f32) -> &Self {
        unsafe {
            let _: () = msg_send![self, setMean: mean];
        }
        self
    }

    /// Sets the standard deviation (for normal distributions)
    pub fn set_standard_deviation(&self, std_dev: f32) -> &Self {
        unsafe {
            let _: () = msg_send![self, setStandardDeviation: std_dev];
        }
        self
    }

    /// Sets the sampling method (for normal distributions)
    pub fn set_sampling_method(&self, method: RandomNormalSamplingMethod) -> &Self {
        unsafe {
            let _: () = msg_send![self, setSamplingMethod: method as u64];
        }
        self
    }
}

/// Random operations for Graph
impl Graph {
    /// Creates a tensor representing state using the Philox algorithm with given seed
    pub fn random_philox_state_tensor_with_seed(
        &self,
        seed: usize,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };

            msg_send![
                self,
                randomPhiloxStateTensorWithSeed: seed as u64,
                name: name_obj
            ]
        }
    }

    /// Creates a tensor representing state using the Philox algorithm with given counter and key values
    pub fn random_philox_state_tensor_with_counter(
        &self,
        counter_low: usize,
        counter_high: usize,
        key: usize,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };

            msg_send![
                self,
                randomPhiloxStateTensorWithCounterLow: counter_low as u64,
                counterHigh: counter_high as u64,
                key: key as u64,
                name: name_obj
            ]
        }
    }

    /// Creates a random tensor with the specified shape and distribution
    pub fn random_tensor(
        &self,
        shape: &Shape,
        descriptor: &RandomOpDescriptor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };

            msg_send![
                self,
                randomTensorWithShape: shape.as_ptr(),
                descriptor: descriptor,
                name: name_obj
            ]
        }
    }

    /// Creates a random tensor with the specified shape and distribution, using a specific seed
    pub fn random_tensor_with_seed(
        &self,
        shape: &Shape,
        descriptor: &RandomOpDescriptor,
        seed: usize,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };

            msg_send![
                self,
                randomTensorWithShape: shape.as_ptr(),
                descriptor: descriptor,
                seed: seed as u64,
                name: name_obj
            ]
        }
    }

    /// Creates a random tensor with specified shape and distribution, using and updating the state tensor
    pub fn random_tensor_with_state(
        &self,
        shape: &Shape,
        descriptor: &RandomOpDescriptor,
        state: &Tensor,
        name: Option<&str>,
    ) -> (Retained<Tensor>, Retained<Tensor>) {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };

            let result_array: Retained<NSArray<Tensor>> = msg_send![
                self,
                randomTensorWithShape: shape.as_ptr(),
                descriptor: descriptor,
                stateTensor: state,
                name: name_obj
            ];

            let count = result_array.count();
            assert_eq!(
                count, 2,
                "Random tensor with state should return an array of 2 tensors"
            );

            let random_tensor: Retained<Tensor> = msg_send![&*result_array, objectAtIndex: 0];
            let updated_state: Retained<Tensor> = msg_send![&*result_array, objectAtIndex: 1];
            (random_tensor, updated_state)
        }
    }

    /// Creates a random uniform tensor with values in range [0.0, 1.0)
    pub fn random_uniform_tensor(&self, shape: &Shape, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };

            msg_send![
                self,
                randomUniformTensorWithShape: shape.as_ptr(),
                name: name_obj
            ]
        }
    }

    /// Creates a random uniform tensor with values in range [0.0, 1.0) using a specific seed
    pub fn random_uniform_tensor_with_seed(
        &self,
        shape: &Shape,
        seed: usize,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };

            msg_send![
                self,
                randomUniformTensorWithShape: shape.as_ptr(),
                seed: seed as u64,
                name: name_obj
            ]
        }
    }

    /// Creates a random uniform tensor with values in range [0.0, 1.0), using and updating the state tensor
    pub fn random_uniform_tensor_with_state(
        &self,
        shape: &Shape,
        state: &Tensor,
        name: Option<&str>,
    ) -> (Retained<Tensor>, Retained<Tensor>) {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };

            let result_array: Retained<NSArray<Tensor>> = msg_send![
                self,
                randomUniformTensorWithShape: shape.as_ptr(),
                stateTensor: state,
                name: name_obj
            ];

            let count = result_array.count();
            assert_eq!(
                count, 2,
                "Random uniform tensor with state should return an array of 2 tensors"
            );

            let random_tensor: Retained<Tensor> = msg_send![&*result_array, objectAtIndex: 0];
            let updated_state: Retained<Tensor> = msg_send![&*result_array, objectAtIndex: 1];
            (random_tensor, updated_state)
        }
    }

    /// Creates a dropout operation which zeros out elements of the input tensor randomly with probability equal to rate
    pub fn dropout(&self, tensor: &Tensor, rate: f64, name: Option<&str>) -> Retained<Tensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };

            msg_send![
                self,
                dropoutTensor: tensor,
                rate: rate,
                name: name_obj
            ]
        }
    }

    /// Creates a dropout operation using a tensor to specify the dropout rate
    pub fn dropout_with_rate_tensor(
        &self,
        tensor: &Tensor,
        rate_tensor: &Tensor,
        name: Option<&str>,
    ) -> Retained<Tensor> {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };

            msg_send![
                self,
                dropoutTensor: tensor,
                rateTensor: rate_tensor,
                name: name_obj
            ]
        }
    }
}
