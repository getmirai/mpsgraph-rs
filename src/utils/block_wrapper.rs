use block2::{Block, RcBlock};
use objc2::encode::{Encoding, RefEncode};
use objc2::msg_send;
use objc2::rc::Retained;
use objc2::{ClassType, Encode, Message};
use objc2_foundation::NSArray;
use std::ffi::c_void;
use std::fmt;

/// Helper to create NSArray from vector of Retained<Tensor>
pub fn create_ns_array_from_tensors<T>(tensors: Vec<Retained<T>>) -> Retained<NSArray<T>>
where
    T: Message + ClassType,
{
    let mut refs: Vec<&T> = Vec::with_capacity(tensors.len());
    for tensor in &tensors {
        refs.push(tensor);
    }
    NSArray::from_slice(&refs)
}

/// A wrapper for no-arg blocks that return a void pointer (typically array of tensors)
#[repr(transparent)]
pub struct TensorBlock {
    block: RcBlock<dyn Fn() -> *mut c_void>,
}

impl TensorBlock {
    pub fn new(block: RcBlock<dyn Fn() -> *mut c_void>) -> Self {
        Self { block }
    }

    pub fn as_block_ptr(&self) -> *const Block<dyn Fn() -> *mut c_void> {
        &*self.block
    }
}

// Debug implementation
impl fmt::Debug for TensorBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorBlock")
            .field("block", &"<block>")
            .finish()
    }
}

// Implement necessary encoding traits
unsafe impl Encode for TensorBlock {
    const ENCODING: Encoding = Encoding::Block;
}

unsafe impl RefEncode for TensorBlock {
    const ENCODING_REF: Encoding = Encoding::Block;
}

/// A wrapper for blocks that take a void pointer and return a void pointer
#[repr(transparent)]
pub struct TensorArrayBlock {
    block: RcBlock<dyn Fn(*mut c_void) -> *mut c_void>,
}

impl TensorArrayBlock {
    pub fn new(block: RcBlock<dyn Fn(*mut c_void) -> *mut c_void>) -> Self {
        Self { block }
    }

    pub fn as_block_ptr(&self) -> *const Block<dyn Fn(*mut c_void) -> *mut c_void> {
        &*self.block
    }
}

// Debug implementation
impl fmt::Debug for TensorArrayBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorArrayBlock")
            .field("block", &"<block>")
            .finish()
    }
}

// Implement necessary encoding traits
unsafe impl Encode for TensorArrayBlock {
    const ENCODING: Encoding = Encoding::Block;
}

unsafe impl RefEncode for TensorArrayBlock {
    const ENCODING_REF: Encoding = Encoding::Block;
}

/// A wrapper for blocks that take an index and array of tensors and return an array
#[repr(transparent)]
pub struct IndexTensorArrayBlock {
    block: RcBlock<dyn Fn(*mut c_void, *mut c_void) -> *mut c_void>,
}

impl IndexTensorArrayBlock {
    pub fn new(block: RcBlock<dyn Fn(*mut c_void, *mut c_void) -> *mut c_void>) -> Self {
        Self { block }
    }

    pub fn as_block_ptr(&self) -> *const Block<dyn Fn(*mut c_void, *mut c_void) -> *mut c_void> {
        &*self.block
    }
}

// Debug implementation
impl fmt::Debug for IndexTensorArrayBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IndexTensorArrayBlock")
            .field("block", &"<block>")
            .finish()
    }
}

// Implement necessary encoding traits
unsafe impl Encode for IndexTensorArrayBlock {
    const ENCODING: Encoding = Encoding::Block;
}

unsafe impl RefEncode for IndexTensorArrayBlock {
    const ENCODING_REF: Encoding = Encoding::Block;
}

/// A wrapper for condition blocks in while loops
#[repr(transparent)]
pub struct ConditionBlock {
    block: RcBlock<dyn Fn(*mut c_void, *mut c_void) -> *mut c_void>,
}

impl ConditionBlock {
    pub fn new(block: RcBlock<dyn Fn(*mut c_void, *mut c_void) -> *mut c_void>) -> Self {
        Self { block }
    }

    pub fn as_block_ptr(&self) -> *const Block<dyn Fn(*mut c_void, *mut c_void) -> *mut c_void> {
        &*self.block
    }
}

// Debug implementation
impl fmt::Debug for ConditionBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConditionBlock")
            .field("block", &"<block>")
            .finish()
    }
}

// Implement necessary encoding traits
unsafe impl Encode for ConditionBlock {
    const ENCODING: Encoding = Encoding::Block;
}

unsafe impl RefEncode for ConditionBlock {
    const ENCODING_REF: Encoding = Encoding::Block;
}

/// Helper to create a block that takes no arguments and returns tensors
pub fn create_tensor_block<F>(closure: F) -> TensorBlock
where
    F: Fn() -> *mut c_void + 'static,
{
    let block = RcBlock::new(closure);
    TensorBlock::new(block)
}

/// Helper to create a block that takes a tensor array and returns a tensor array
pub fn create_tensor_array_block<F>(closure: F) -> TensorArrayBlock
where
    F: Fn(*mut c_void) -> *mut c_void + 'static,
{
    let block = RcBlock::new(closure);
    TensorArrayBlock::new(block)
}

/// Helper to create a block that takes an index tensor and tensor array and returns a tensor array
pub fn create_index_tensor_array_block<F>(closure: F) -> IndexTensorArrayBlock
where
    F: Fn(*mut c_void, *mut c_void) -> *mut c_void + 'static,
{
    let block = RcBlock::new(closure);
    IndexTensorArrayBlock::new(block)
}

/// Helper to create a block that takes tensors and a mutable array and returns a condition tensor
pub fn create_condition_block<F>(closure: F) -> ConditionBlock
where
    F: Fn(*mut c_void, *mut c_void) -> *mut c_void + 'static,
{
    let block = RcBlock::new(closure);
    ConditionBlock::new(block)
}
