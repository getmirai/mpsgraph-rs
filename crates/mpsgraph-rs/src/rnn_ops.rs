use crate::graph::Graph;
use crate::tensor::Tensor;
use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::AnyClass;
use objc2::extern_class;
use objc2_foundation::{NSArray, NSObject, NSObjectProtocol, NSString};

/// Activation functions for RNN operations
#[repr(u64)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum RNNActivation {
    /// No activation
    None = 0,
    /// ReLU activation
    ReLU = 1,
    /// TanH activation
    TanH = 2,
    /// Sigmoid activation
    Sigmoid = 3,
    /// Hard Sigmoid activation
    HardSigmoid = 4,
}

extern_class!(
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraphSingleGateRNNDescriptor"]
    /// Descriptor for single-gate RNN operations
    pub struct SingleGateRNNDescriptor;
);

unsafe impl NSObjectProtocol for SingleGateRNNDescriptor {}

extern_class!(
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraphLSTMDescriptor"]
    /// Descriptor for LSTM operations
    pub struct LSTMDescriptor;
);

unsafe impl NSObjectProtocol for LSTMDescriptor {}

extern_class!(
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraphGRUDescriptor"]
    /// Descriptor for GRU operations
    pub struct GRUDescriptor;
);

unsafe impl NSObjectProtocol for GRUDescriptor {}

impl SingleGateRNNDescriptor {
    /// Creates a new single-gate RNN descriptor with default settings
    pub fn new() -> Retained<Self> {
        unsafe {
            let cls = AnyClass::get(c"MPSGraphSingleGateRNNDescriptor").unwrap();
            msg_send![cls, descriptor]
        }
    }

    /// A parameter that defines time direction of the input sequence.
    ///
    /// If set to `true` then the input sequence is passed in reverse time order to the layer.
    /// Note: Ignored when `bidirectional = true`.
    /// Default value: `false`.
    pub fn set_reverse(&self, reverse: bool) -> &Self {
        unsafe {
            let _: () = msg_send![self, setReverse: reverse];
        }
        self
    }

    /// A parameter that defines a bidirectional RNN layer.
    ///
    /// If set to `true` then the input sequence is traversed in both directions and the two results
    /// are concatenated together on the channel-axis.
    /// Default value: `false`.
    pub fn set_bidirectional(&self, bidirectional: bool) -> &Self {
        unsafe {
            let _: () = msg_send![self, setBidirectional: bidirectional];
        }
        self
    }

    /// A parameter that makes the RNN layer support training.
    ///
    /// If set to `true` then the layer will produce training state tensor as a secondary output.
    /// Default value: `false`.
    pub fn set_training(&self, training: bool) -> &Self {
        unsafe {
            let _: () = msg_send![self, setTraining: training];
        }
        self
    }

    /// A parameter that defines the activation function to use with the RNN operation.
    ///
    /// Default value: `RNNActivation::ReLU`.
    pub fn set_activation(&self, activation: RNNActivation) -> &Self {
        unsafe {
            let _: () = msg_send![self, setActivation: activation as u64];
        }
        self
    }

    /// Returns the reverse setting.
    pub fn reverse(&self) -> bool {
        unsafe { msg_send![self, reverse] }
    }

    /// Returns the bidirectional setting.
    pub fn bidirectional(&self) -> bool {
        unsafe { msg_send![self, bidirectional] }
    }

    /// Returns the training setting.
    pub fn training(&self) -> bool {
        unsafe { msg_send![self, training] }
    }

    /// Returns the activation function setting.
    pub fn activation(&self) -> RNNActivation {
        unsafe {
            let activation_val: u64 = msg_send![self, activation];
            std::mem::transmute(activation_val)
        }
    }
}

// Instead of implementing Default directly, we can use a CustomDefault trait
impl crate::device::CustomDefault for SingleGateRNNDescriptor {
    fn custom_default() -> Retained<Self> {
        Self::new()
    }
}

impl LSTMDescriptor {
    /// Creates a new LSTM descriptor with default settings
    pub fn new() -> Retained<Self> {
        unsafe {
            let cls = AnyClass::get(c"MPSGraphLSTMDescriptor").unwrap();
            msg_send![cls, descriptor]
        }
    }

    /// A parameter that defines time direction of the input sequence.
    ///
    /// If set to `true` then the input sequence is passed in reverse time order to the layer.
    /// Note: Ignored when `bidirectional = true`.
    /// Default value: `false`.
    pub fn set_reverse(&self, reverse: bool) -> &Self {
        unsafe {
            let _: () = msg_send![self, setReverse: reverse];
        }
        self
    }

    /// A parameter that defines a bidirectional LSTM layer.
    ///
    /// If set to `true` then the input sequence is traversed in both directions and the two results
    /// are concatenated together on the channel-axis.
    /// Default value: `false`.
    pub fn set_bidirectional(&self, bidirectional: bool) -> &Self {
        unsafe {
            let _: () = msg_send![self, setBidirectional: bidirectional];
        }
        self
    }

    /// A parameter that controls whether or not to return the output cell from the LSTM layer.
    ///
    /// If set to `true` then this layer will produce the internal cell of the LSTM unit as secondary output.
    /// Default value: `false`.
    pub fn set_produce_cell(&self, produce_cell: bool) -> &Self {
        unsafe {
            let _: () = msg_send![self, setProduceCell: produce_cell];
        }
        self
    }

    /// A parameter that enables the LSTM layer to support training.
    ///
    /// If set to `true` then the layer will produce training state tensor as a secondary output.
    /// Default value: `false`.
    pub fn set_training(&self, training: bool) -> &Self {
        unsafe {
            let _: () = msg_send![self, setTraining: training];
        }
        self
    }

    /// A parameter that controls the internal order of the LSTM gates.
    ///
    /// If set to `true` then the layer will use the gate-ordering `[ i, z, f, o ]` instead of default `[ i, f, z, o ]`.
    /// Default value: `false`
    pub fn set_forget_gate_last(&self, forget_gate_last: bool) -> &Self {
        unsafe {
            let _: () = msg_send![self, setForgetGateLast: forget_gate_last];
        }
        self
    }

    /// A parameter that defines the activation function used with the input gate of the LSTM operation.
    ///
    /// Default value: `RNNActivation::Sigmoid`.
    pub fn set_input_gate_activation(&self, activation: RNNActivation) -> &Self {
        unsafe {
            let _: () = msg_send![self, setInputGateActivation: activation as u64];
        }
        self
    }

    /// A parameter that defines the activation function used with the forget gate of the LSTM operation.
    ///
    /// Default value: `RNNActivation::Sigmoid`.
    pub fn set_forget_gate_activation(&self, activation: RNNActivation) -> &Self {
        unsafe {
            let _: () = msg_send![self, setForgetGateActivation: activation as u64];
        }
        self
    }

    /// A parameter that defines the activation function used with the cell gate of the LSTM operation.
    ///
    /// Default value: `RNNActivation::TanH`.
    pub fn set_cell_gate_activation(&self, activation: RNNActivation) -> &Self {
        unsafe {
            let _: () = msg_send![self, setCellGateActivation: activation as u64];
        }
        self
    }

    /// A parameter that defines the activation function used with the output gate of the LSTM operation.
    ///
    /// Default value: `RNNActivation::Sigmoid`.
    pub fn set_output_gate_activation(&self, activation: RNNActivation) -> &Self {
        unsafe {
            let _: () = msg_send![self, setOutputGateActivation: activation as u64];
        }
        self
    }

    /// A parameter that defines the activation function used with the current cell value of the LSTM operation.
    ///
    /// Default value: `RNNActivation::TanH`.
    pub fn set_activation(&self, activation: RNNActivation) -> &Self {
        unsafe {
            let _: () = msg_send![self, setActivation: activation as u64];
        }
        self
    }

    // Getter methods

    /// Returns the reverse setting.
    pub fn reverse(&self) -> bool {
        unsafe { msg_send![self, reverse] }
    }

    /// Returns the bidirectional setting.
    pub fn bidirectional(&self) -> bool {
        unsafe { msg_send![self, bidirectional] }
    }

    /// Returns the produce cell setting.
    pub fn produce_cell(&self) -> bool {
        unsafe { msg_send![self, produceCell] }
    }

    /// Returns the training setting.
    pub fn training(&self) -> bool {
        unsafe { msg_send![self, training] }
    }

    /// Returns the forget gate last setting.
    pub fn forget_gate_last(&self) -> bool {
        unsafe { msg_send![self, forgetGateLast] }
    }

    /// Returns the input gate activation setting.
    pub fn input_gate_activation(&self) -> RNNActivation {
        unsafe {
            let activation_val: u64 = msg_send![self, inputGateActivation];
            std::mem::transmute(activation_val)
        }
    }

    /// Returns the forget gate activation setting.
    pub fn forget_gate_activation(&self) -> RNNActivation {
        unsafe {
            let activation_val: u64 = msg_send![self, forgetGateActivation];
            std::mem::transmute(activation_val)
        }
    }

    /// Returns the cell gate activation setting.
    pub fn cell_gate_activation(&self) -> RNNActivation {
        unsafe {
            let activation_val: u64 = msg_send![self, cellGateActivation];
            std::mem::transmute(activation_val)
        }
    }

    /// Returns the output gate activation setting.
    pub fn output_gate_activation(&self) -> RNNActivation {
        unsafe {
            let activation_val: u64 = msg_send![self, outputGateActivation];
            std::mem::transmute(activation_val)
        }
    }

    /// Returns the activation setting.
    pub fn activation(&self) -> RNNActivation {
        unsafe {
            let activation_val: u64 = msg_send![self, activation];
            std::mem::transmute(activation_val)
        }
    }
}

impl crate::device::CustomDefault for LSTMDescriptor {
    fn custom_default() -> Retained<Self> {
        Self::new()
    }
}

impl GRUDescriptor {
    /// Creates a new GRU descriptor with default settings
    pub fn new() -> Retained<Self> {
        unsafe {
            let cls = AnyClass::get(c"MPSGraphGRUDescriptor").unwrap();
            msg_send![cls, descriptor]
        }
    }

    /// A parameter that defines the time direction of the input sequence.
    ///
    /// If set to `true` then the input sequence is passed in reverse time order to the layer.
    /// Note: Ignored when `bidirectional = true`.
    /// Default value: `false`.
    pub fn set_reverse(&self, reverse: bool) -> &Self {
        unsafe {
            let _: () = msg_send![self, setReverse: reverse];
        }
        self
    }

    /// A parameter that defines a bidirectional GRU layer.
    ///
    /// If set to `true` then the input sequence is traversed in both directions and the two results
    /// are concatenated together on the channel-axis.
    /// Default value: `false`.
    pub fn set_bidirectional(&self, bidirectional: bool) -> &Self {
        unsafe {
            let _: () = msg_send![self, setBidirectional: bidirectional];
        }
        self
    }

    /// A parameter that enables the GRU layer to support training.
    ///
    /// If set to `true` then the layer will produce training state tensor as a secondary output.
    /// Default value: `false`.
    pub fn set_training(&self, training: bool) -> &Self {
        unsafe {
            let _: () = msg_send![self, setTraining: training];
        }
        self
    }

    /// A parameter that controls the internal order of the GRU gates.
    ///
    /// If set to `true` then the layer will use the gate-ordering `[ r, z, o ]` instead of default `[ z, r, o ]`.
    /// Default value: `false`.
    pub fn set_reset_gate_first(&self, reset_gate_first: bool) -> &Self {
        unsafe {
            let _: () = msg_send![self, setResetGateFirst: reset_gate_first];
        }
        self
    }

    /// A parameter that chooses between two variants for the reset gate computation.
    ///
    /// If set to `true` then the layer will compute the intermediate value as `c[t] = ( b + (h[t-1] m ) R^T) r[t]`.
    /// Otherwise it's computed as `c[t] = (h[t-1] r[t] m) R^T`.
    /// Default value: `false`.
    pub fn set_reset_after(&self, reset_after: bool) -> &Self {
        unsafe {
            let _: () = msg_send![self, setResetAfter: reset_after];
        }
        self
    }

    /// A parameter that chooses between two variants for the final output computation.
    ///
    /// If set to `true` then the layer will compute the final value as `h[t] = z[t] h[t-1] + (1-z[t]) o[t]`.
    /// Otherwise it's computed as `h[t] = (1-z[t]) h[t-1] + z[t] o[t]`.
    /// Default value: `false`.
    pub fn set_flip_z(&self, flip_z: bool) -> &Self {
        unsafe {
            let _: () = msg_send![self, setFlipZ: flip_z];
        }
        self
    }

    /// A parameter that defines the activation function to use with the update-gate of the GRU operation.
    ///
    /// Default value: `RNNActivation::Sigmoid`.
    pub fn set_update_gate_activation(&self, activation: RNNActivation) -> &Self {
        unsafe {
            let _: () = msg_send![self, setUpdateGateActivation: activation as u64];
        }
        self
    }

    /// A parameter that defines the activation function to use with the reset-gate of the GRU operation.
    ///
    /// Default value: `RNNActivation::Sigmoid`.
    pub fn set_reset_gate_activation(&self, activation: RNNActivation) -> &Self {
        unsafe {
            let _: () = msg_send![self, setResetGateActivation: activation as u64];
        }
        self
    }

    /// A parameter that defines the activation function to use with the output-gate of the GRU operation.
    ///
    /// Default value: `RNNActivation::TanH`.
    pub fn set_output_gate_activation(&self, activation: RNNActivation) -> &Self {
        unsafe {
            let _: () = msg_send![self, setOutputGateActivation: activation as u64];
        }
        self
    }

    // Getter methods

    /// Returns the reverse setting.
    pub fn reverse(&self) -> bool {
        unsafe { msg_send![self, reverse] }
    }

    /// Returns the bidirectional setting.
    pub fn bidirectional(&self) -> bool {
        unsafe { msg_send![self, bidirectional] }
    }

    /// Returns the training setting.
    pub fn training(&self) -> bool {
        unsafe { msg_send![self, training] }
    }

    /// Returns the reset gate first setting.
    pub fn reset_gate_first(&self) -> bool {
        unsafe { msg_send![self, resetGateFirst] }
    }

    /// Returns the reset after setting.
    pub fn reset_after(&self) -> bool {
        unsafe { msg_send![self, resetAfter] }
    }

    /// Returns the flip Z setting.
    pub fn flip_z(&self) -> bool {
        unsafe { msg_send![self, flipZ] }
    }

    /// Returns the update gate activation setting.
    pub fn update_gate_activation(&self) -> RNNActivation {
        unsafe {
            let activation_val: u64 = msg_send![self, updateGateActivation];
            std::mem::transmute(activation_val)
        }
    }

    /// Returns the reset gate activation setting.
    pub fn reset_gate_activation(&self) -> RNNActivation {
        unsafe {
            let activation_val: u64 = msg_send![self, resetGateActivation];
            std::mem::transmute(activation_val)
        }
    }

    /// Returns the output gate activation setting.
    pub fn output_gate_activation(&self) -> RNNActivation {
        unsafe {
            let activation_val: u64 = msg_send![self, outputGateActivation];
            std::mem::transmute(activation_val)
        }
    }
}

impl crate::device::CustomDefault for GRUDescriptor {
    fn custom_default() -> Retained<Self> {
        Self::new()
    }
}

/// RNN operation for Graph
impl Graph {
    /// Creates a single-gate RNN operation with mask support.
    ///
    /// This operation returns tensors `h` and optionally `z` that are defined recursively as follows:
    /// ```text
    /// for t = 0 to T-1
    ///   z[t] = x[t] W^T + (h[t-1]m) R^T + b
    ///   h[t] = activation( z[t] ), where
    /// ```
    /// `W` is input_weight, `R` is recurrent_weight, `b` is bias, `m` is mask,
    /// `x[t]` is input, `h[t]` is the first output, `z[t]` is the second output (optional) and `h[-1]` is init_state.
    ///
    /// # Arguments
    ///
    /// * `input` - A tensor that contains the source data `x[t]` with the data layout [T,N,I].
    ///             In case `input_weight = None` and `bidirectional = false` then the layout is [T,N,H] and
    ///             for `input_weight = None` and `bidirectional = true` the layout is [T,N,2H].
    /// * `recurrent_weight` - A tensor containing the recurrent weights `R`. For `bidirectional` the layout is [2,H,H] and otherwise it is [H,H].
    /// * `input_weight` - A tensor containing the input weights matrix `W` - optional, if missing the operation assumes a diagonal unit-matrix.
    ///                  For `bidirectional` the layout is [2H,I] and otherwise it is [H,I].
    /// * `bias` - A tensor containing the bias `b` - optional, if missing the operation assumes zeroes. For `bidirectional` the layout is [2H] and otherwise it is [H].
    /// * `init_state` - The initial internal state of the RNN `h[-1]` - optional, if missing the operation assumes zeroes. For `bidirectional` the layout is [N,2H] and otherwise it is [N,H].
    /// * `mask` - A tensor containing the mask `m` - optional, if missing the operation assumes ones. This is useful for dropout support.
    /// * `descriptor` - A descriptor that defines the parameters for the RNN operation.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A vector of Tensor objects of size 1 or 2, depending on value of `descriptor.training`.
    /// The layout of both outputs is [T,N,H] or [T,N,2H] for bidirectional.
    pub fn single_gate_rnn_with_mask(
        &self,
        input: &Tensor,
        recurrent_weight: &Tensor,
        input_weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        init_state: Option<&Tensor>,
        mask: Option<&Tensor>,
        descriptor: &SingleGateRNNDescriptor,
        name: Option<&str>,
    ) -> Vec<Retained<Tensor>> {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };
            
            let input_weight_obj = match input_weight {
                Some(w) => w as *const _,
                None => std::ptr::null(),
            };
            
            let bias_obj = match bias {
                Some(b) => b as *const _,
                None => std::ptr::null(),
            };
            
            let init_state_obj = match init_state {
                Some(s) => s as *const _,
                None => std::ptr::null(),
            };
            
            let mask_obj = match mask {
                Some(m) => m as *const _,
                None => std::ptr::null(),
            };

            let result: Retained<NSArray<Tensor>> = msg_send![
                self,
                singleGateRNNWithSourceTensor: input,
                recurrentWeight: recurrent_weight,
                inputWeight: input_weight_obj,
                bias: bias_obj,
                initState: init_state_obj,
                mask: mask_obj,
                descriptor: descriptor,
                name: name_obj
            ];

            // Count the number of result tensors
            let count = result.count();
            let mut tensors = Vec::with_capacity(count);

            // Extract all tensors from the array
            for i in 0..count {
                let tensor_ptr: *mut Tensor = msg_send![&*result, objectAtIndex: i];
                let tensor = Retained::retain(tensor_ptr).unwrap();
                tensors.push(tensor);
            }

            tensors
        }
    }

    /// Creates a single-gate RNN operation without mask support.
    ///
    /// This operation returns tensors `h` and optionally `z` that are defined recursively as follows:
    /// ```text
    /// for t = 0 to T-1
    ///   z[t] = x[t] W^T + (h[t-1]) R^T + b
    ///   h[t] = activation( z[t] ), where
    /// ```
    /// `W` is input_weight, `R` is recurrent_weight, `b` is bias,
    /// `x[t]` is input, `h[t]` is the first output, `z[t]` is the second output (optional) and `h[-1]` is init_state.
    ///
    /// # Arguments
    ///
    /// * `input` - A tensor that contains the source data `x[t]` with the data layout [T,N,I].
    ///             In case `input_weight = None` and `bidirectional = false` then the layout is [T,N,H] and
    ///             for `input_weight = None` and `bidirectional = true` the layout is [T,N,2H].
    /// * `recurrent_weight` - A tensor containing the recurrent weights `R`. For `bidirectional` the layout is [2,H,H] and otherwise it is [H,H].
    /// * `input_weight` - A tensor containing the input weights matrix `W` - optional, if missing the operation assumes a diagonal unit-matrix.
    ///                  For `bidirectional` the layout is [2H,I] and otherwise it is [H,I].
    /// * `bias` - A tensor containing the bias `b` - optional, if missing the operation assumes zeroes. For `bidirectional` the layout is [2H] and otherwise it is [H].
    /// * `init_state` - The initial internal state of the RNN `h[-1]` - optional, if missing the operation assumes zeroes. For `bidirectional` the layout is [N,2H] and otherwise it is [N,H].
    /// * `descriptor` - A descriptor that defines the parameters for the RNN operation.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A vector of Tensor objects of size 1 or 2, depending on value of `descriptor.training`.
    /// The layout of both outputs is [T,N,H] or [T,N,2H] for bidirectional.
    pub fn single_gate_rnn(
        &self,
        input: &Tensor,
        recurrent_weight: &Tensor,
        input_weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        init_state: Option<&Tensor>,
        descriptor: &SingleGateRNNDescriptor,
        name: Option<&str>,
    ) -> Vec<Retained<Tensor>> {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };
            
            let input_weight_obj = match input_weight {
                Some(w) => w as *const _,
                None => std::ptr::null(),
            };
            
            let bias_obj = match bias {
                Some(b) => b as *const _,
                None => std::ptr::null(),
            };
            
            let init_state_obj = match init_state {
                Some(s) => s as *const _,
                None => std::ptr::null(),
            };

            let result: Retained<NSArray<Tensor>> = msg_send![
                self,
                singleGateRNNWithSourceTensor: input,
                recurrentWeight: recurrent_weight,
                inputWeight: input_weight_obj,
                bias: bias_obj,
                initState: init_state_obj,
                descriptor: descriptor,
                name: name_obj
            ];

            // Count the number of result tensors
            let count = result.count();
            let mut tensors = Vec::with_capacity(count);

            // Extract all tensors from the array
            for i in 0..count {
                let tensor_ptr: *mut Tensor = msg_send![&*result, objectAtIndex: i];
                let tensor = Retained::retain(tensor_ptr).unwrap();
                tensors.push(tensor);
            }

            tensors
        }
    }

    /// Creates a single-gate RNN operation with minimal parameters.
    ///
    /// This operation returns tensors `h` and optionally `z` that are defined recursively as follows:
    /// ```text
    /// for t = 0 to T-1
    ///   z[t] = x[t] R^T
    ///   h[t] = activation( z[t] ), where
    /// ```
    /// `R` is recurrent_weight, `x[t]` is input, `h[t]` is the first output,
    /// `z[t]` is the second output (optional) and `h[-1]` is init_state.
    ///
    /// # Arguments
    ///
    /// * `input` - A tensor that contains the source data `x[t]` with the data layout [T,N,H] or [T,N,2H] for bidirectional.
    /// * `recurrent_weight` - A tensor containing the recurrent weights `R`. For `bidirectional` the layout is [2,H,H] and otherwise it is [H,H].
    /// * `init_state` - The initial internal state of the RNN `h[-1]` - optional, if missing the operation assumes zeroes. For `bidirectional` the layout is [N,2H] and otherwise it is [N,H].
    /// * `descriptor` - A descriptor that defines the parameters for the RNN operation.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A vector of Tensor objects of size 1 or 2, depending on value of `descriptor.training`.
    /// The layout of both outputs is [T,N,H] or [T,N,2H] for bidirectional.
    pub fn single_gate_rnn_minimal(
        &self,
        input: &Tensor,
        recurrent_weight: &Tensor,
        init_state: Option<&Tensor>,
        descriptor: &SingleGateRNNDescriptor,
        name: Option<&str>,
    ) -> Vec<Retained<Tensor>> {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };
            
            let init_state_obj = match init_state {
                Some(s) => s as *const _,
                None => std::ptr::null(),
            };

            let result: Retained<NSArray<Tensor>> = msg_send![
                self,
                singleGateRNNWithSourceTensor: input,
                recurrentWeight: recurrent_weight,
                initState: init_state_obj,
                descriptor: descriptor,
                name: name_obj
            ];

            // Count the number of result tensors
            let count = result.count();
            let mut tensors = Vec::with_capacity(count);

            // Extract all tensors from the array
            for i in 0..count {
                let tensor_ptr: *mut Tensor = msg_send![&*result, objectAtIndex: i];
                let tensor = Retained::retain(tensor_ptr).unwrap();
                tensors.push(tensor);
            }

            tensors
        }
    }

    /// Creates a single-gate RNN gradient operation with all parameters.
    ///
    /// # Arguments
    ///
    /// * `input` - A tensor that contains the source data `x[t]` with the data layout [T,N,I].
    /// * `recurrent_weight` - A tensor containing the recurrent weights `R`. For `bidirectional` the layout is [2,H,H] and otherwise it is [H,H].
    /// * `source_gradient` - The input gradient, that is the gradient of a tensor with respect to the first output of the forward pass.
    /// * `z_state` - The second output of the forward pass with `descriptor.training = true`.
    /// * `state_gradient` - The input gradient coming from the future timestep - optional, if missing the operation assumes zeroes.
    /// * `input_weight` - A tensor containing the input weights matrix `W` - optional.
    /// * `bias` - A tensor containing the bias `b` - optional.
    /// * `init_state` - The initial internal state of the RNN `h[-1]` - optional.
    /// * `mask` - A tensor containing the mask `m` - optional.
    /// * `descriptor` - A descriptor that defines the parameters for the RNN operation.
    /// * `name` - The name for the operation.
    ///
    /// # Returns
    ///
    /// A vector of Tensor objects containing gradients for each input tensor, except for `source_gradient` and `mask`.
    /// In case an input is `None`, no gradient will be returned for it.
    /// The order of the gradients will be: for `input`, for `recurrent_weight`, for `input_weight`, for `bias` and finally for `init_state`.
    pub fn single_gate_rnn_gradients(
        &self,
        input: &Tensor,
        recurrent_weight: &Tensor,
        source_gradient: &Tensor,
        z_state: &Tensor,
        state_gradient: Option<&Tensor>,
        input_weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        init_state: Option<&Tensor>,
        mask: Option<&Tensor>,
        descriptor: &SingleGateRNNDescriptor,
        name: Option<&str>,
    ) -> Vec<Retained<Tensor>> {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };
            
            let state_gradient_obj = match state_gradient {
                Some(sg) => sg as *const _,
                None => std::ptr::null(),
            };
            
            let input_weight_obj = match input_weight {
                Some(w) => w as *const _,
                None => std::ptr::null(),
            };
            
            let bias_obj = match bias {
                Some(b) => b as *const _,
                None => std::ptr::null(),
            };
            
            let init_state_obj = match init_state {
                Some(s) => s as *const _,
                None => std::ptr::null(),
            };
            
            let mask_obj = match mask {
                Some(m) => m as *const _,
                None => std::ptr::null(),
            };

            let result: Retained<NSArray<Tensor>> = msg_send![
                self,
                singleGateRNNGradientsWithSourceTensor: input,
                recurrentWeight: recurrent_weight,
                sourceGradient: source_gradient,
                zState: z_state,
                stateGradient: state_gradient_obj,
                inputWeight: input_weight_obj,
                bias: bias_obj,
                initState: init_state_obj,
                mask: mask_obj,
                descriptor: descriptor,
                name: name_obj
            ];

            // Count the number of result tensors
            let count = result.count();
            let mut tensors = Vec::with_capacity(count);

            // Extract all tensors from the array
            for i in 0..count {
                let tensor_ptr: *mut Tensor = msg_send![&*result, objectAtIndex: i];
                let tensor = Retained::retain(tensor_ptr).unwrap();
                tensors.push(tensor);
            }

            tensors
        }
    }

    /// Creates an LSTM operation.
    ///
    /// # Arguments
    ///
    /// * `input` - Input sequence tensor of shape [T,N,C] or [N,T,C]
    /// * `initial_hidden_state` - Initial hidden state tensor of shape [N,H]
    /// * `initial_cell_state` - Initial cell state tensor of shape [N,H]
    /// * `weights` - Kernel tensor of shape [C+H,4*H]
    /// * `recurrent_weights` - Recurrent kernel tensor of shape [H,4*H]
    /// * `biases` - Bias tensor of shape [4*H], may be NULL if descriptor.useBiasVectors is false
    /// * `descriptor` - LSTM descriptor
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// Tuple containing (output tensor of shape [T,N,H] or [N,T,H], output hidden state tensor of shape [N,H], output cell state tensor of shape [N,H])
    pub fn lstm(
        &self,
        input: &Tensor,
        initial_hidden_state: &Tensor,
        initial_cell_state: &Tensor,
        weights: &Tensor,
        recurrent_weights: &Tensor,
        biases: Option<&Tensor>,
        descriptor: &LSTMDescriptor,
        name: Option<&str>,
    ) -> (Retained<Tensor>, Retained<Tensor>, Retained<Tensor>) {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };
            
            let biases_obj = match biases {
                Some(b) => b as *const _,
                None => std::ptr::null(),
            };

            let result: Retained<NSArray<Tensor>> = msg_send![
                self,
                LSTMWithSourceTensor: input,
                recurrentSourceTensor: initial_hidden_state,
                cellSourceTensor: initial_cell_state,
                weightsTensor: weights,
                recurrentWeightsTensor: recurrent_weights,
                biasesTensor: biases_obj,
                descriptor: descriptor,
                name: name_obj
            ];

            // This returns an NSArray with three tensors: output, output_hidden_state, and output_cell_state
            let count = result.count();
            assert_eq!(count, 3, "Expected 3 result tensors from LSTM");

            let output_tensor_ptr: *mut Tensor = msg_send![&*result, objectAtIndex: 0];
            let output_hidden_state_tensor_ptr: *mut Tensor = msg_send![&*result, objectAtIndex: 1];
            let output_cell_state_tensor_ptr: *mut Tensor = msg_send![&*result, objectAtIndex: 2];

            let output_tensor = Retained::retain(output_tensor_ptr).unwrap();
            let output_hidden_state_tensor = Retained::retain(output_hidden_state_tensor_ptr).unwrap();
            let output_cell_state_tensor = Retained::retain(output_cell_state_tensor_ptr).unwrap();

            (
                output_tensor,
                output_hidden_state_tensor,
                output_cell_state_tensor,
            )
        }
    }

    /// Creates a GRU operation.
    ///
    /// # Arguments
    ///
    /// * `input` - Input sequence tensor of shape [T,N,C] or [N,T,C]
    /// * `initial_state` - Initial hidden state tensor of shape [N,H]
    /// * `weights` - Kernel tensor of shape [C+H,3*H]
    /// * `recurrent_weights` - Recurrent kernel tensor of shape [H,3*H]
    /// * `biases` - Bias tensor of shape [3*H], may be NULL if descriptor.useBiasVectors is false
    /// * `descriptor` - GRU descriptor
    /// * `name` - Name for the operation
    ///
    /// # Returns
    ///
    /// Tuple containing (output tensor of shape [T,N,H] or [N,T,H], output state tensor of shape [N,H])
    pub fn gru(
        &self,
        input: &Tensor,
        initial_state: &Tensor,
        weights: &Tensor,
        recurrent_weights: &Tensor,
        biases: Option<&Tensor>,
        descriptor: &GRUDescriptor,
        name: Option<&str>,
    ) -> (Retained<Tensor>, Retained<Tensor>) {
        unsafe {
            let name_obj = match name {
                Some(s) => &*NSString::from_str(s),
                None => std::ptr::null(),
            };
            
            let biases_obj = match biases {
                Some(b) => b as *const _,
                None => std::ptr::null(),
            };

            let result: Retained<NSArray<Tensor>> = msg_send![
                self,
                GRUWithSourceTensor: input,
                recurrentSourceTensor: initial_state,
                weightsTensor: weights,
                recurrentWeightsTensor: recurrent_weights,
                biasesTensor: biases_obj,
                descriptor: descriptor,
                name: name_obj
            ];

            // This returns an NSArray with two tensors: output and output_state
            let count = result.count();
            assert_eq!(count, 2, "Expected 2 result tensors from GRU");

            let output_tensor_ptr: *mut Tensor = msg_send![&*result, objectAtIndex: 0];
            let output_state_tensor_ptr: *mut Tensor = msg_send![&*result, objectAtIndex: 1];

            let output_tensor = Retained::retain(output_tensor_ptr).unwrap();
            let output_state_tensor = Retained::retain(output_state_tensor_ptr).unwrap();

            (
                output_tensor,
                output_state_tensor,
            )
        }
    }
}