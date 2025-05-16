use metal::foreign_types::ForeignType;
use metal::{CommandBuffer as MTLCommandBuffer, CommandQueue};
use objc2::rc::Retained;
use objc2::runtime::NSObject;
use objc2::{extern_class, msg_send, ClassType};
use objc2_foundation::NSObjectProtocol;

/// Command buffer status
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CommandBufferStatus {
    /// The command buffer has not been enqueued yet
    NotEnqueued = 0,
    /// The command buffer is enqueued, but not committed
    Enqueued = 1,
    /// Committed to the command queue, but not yet scheduled for execution
    Committed = 2,
    /// All dependencies have been resolved and scheduled for execution
    Scheduled = 3,
    /// The command buffer has finished executing successfully
    Completed = 4,
    /// Execution was aborted due to an error
    Error = 5,
}

extern_class!(
    #[derive(Debug, PartialEq, Eq)]
    #[unsafe(super = NSObject)]
    #[name = "MPSCommandBuffer"]
    pub struct CommandBuffer;
);

unsafe impl NSObjectProtocol for CommandBuffer {}

impl CommandBuffer {
    /// Creates a new MPSCommandBuffer from a Metal CommandBuffer
    ///
    /// This initializes a MPSCommandBuffer with an existing MTLCommandBuffer.
    pub fn from_command_buffer(command_buffer: &MTLCommandBuffer) -> Retained<Self> {
        unsafe {
            let class = Self::class();
            let alloc: *mut Self = msg_send![class, alloc];
            let buffer_ptr = command_buffer.as_ptr() as *mut objc2::runtime::AnyObject;

            // Pass the MTLCommandBuffer pointer as an Objective-C object
            let mps_command_buffer: *mut Self = msg_send![
                alloc,
                initWithCommandBuffer:buffer_ptr
            ];

            // -[MPSCommandBuffer initWithCommandBuffer:] returns an *autoreleased* object.
            // We must use retain_autoreleased to correctly manage its lifecycle.
            Retained::retain_autoreleased(mps_command_buffer).unwrap()
        }
    }

    /// Creates a new MPSCommandBuffer from a command queue
    ///
    /// This is a factory method that creates a new MPSCommandBuffer with a new
    /// underlying MTLCommandBuffer from the provided command queue.
    pub fn from_command_queue(command_queue: &CommandQueue) -> Retained<Self> {
        unsafe {
            let class = Self::class();
            let queue_ptr = command_queue.as_ptr() as *mut objc2::runtime::AnyObject;

            // Pass the MTLCommandQueue pointer as an Objective-C object
            let mps_command_buffer: *mut Self = msg_send![
                class,
                commandBufferFromCommandQueue:queue_ptr
            ];

            // `commandBufferFromCommandQueue:` returns an *autoreleased*
            // object. We must explicitly retain it to own one reference.
            // Otherwise, when the surrounding autorelease pool drains, the
            // object is freed while still in use by MPSGraph, leading to
            // "message sent to deallocated instance" crashes.
            Retained::retain_autoreleased(mps_command_buffer).unwrap()
        }
    }

    /// Gets the underlying command buffer.
    ///
    /// This method returns the Metal CommandBuffer that was used to initialize this object.
    pub fn command_buffer(&self) -> &<MTLCommandBuffer as ForeignType>::Ref {
        unsafe {
            let cmd_buffer_ptr: *mut objc2::runtime::AnyObject = msg_send![self, commandBuffer];
            // Assuming the lifetime of the returned pointer is tied to `self`,
            // and that CommandBuffer::Ref is a transparent wrapper or suitable for this cast.
            &*(cmd_buffer_ptr as *const <MTLCommandBuffer as ForeignType>::Ref)
        }
    }

    /// Gets the root command buffer.
    ///
    /// MPSCommandBuffers may wrap other MPSCommandBuffers, in the process
    /// creating what is in effect a stack of predicate objects that may be
    /// pushed or popped by making new MPSCommandBuffers or by calling command_buffer().
    /// In some circumstances, it is preferable to use the root command buffer,
    /// particularly when trying to identify the command buffer that will be commited
    /// by commit_and_continue().
    pub fn root_command_buffer(&self) -> &<MTLCommandBuffer as ForeignType>::Ref {
        unsafe {
            let cmd_buffer_ptr: *mut objc2::runtime::AnyObject = msg_send![self, rootCommandBuffer];
            &*(cmd_buffer_ptr as *const <MTLCommandBuffer as ForeignType>::Ref)
        }
    }

    /// Commits a command buffer so it can be executed as soon as possible.
    ///
    /// This method calls `commit` on the underlying MTLCommandBuffer.
    pub fn commit(&self) {
        unsafe {
            let _: () = msg_send![self, commit];
        }
    }

    /// Commits the current command buffer and continues with a new one.
    ///
    /// This method commits the underlying root MTLCommandBuffer and creates
    /// a new one on the same command queue.
    ///
    /// This is a MPSCommandBuffer-specific method not available in MTLCommandBuffer.
    pub fn commit_and_continue(&self) {
        unsafe {
            let _: () = msg_send![self, commitAndContinue];
        }
    }

    /// Prefetches heap into the MPS command buffer heap cache.
    ///
    /// If there is not sufficient free storage in the MPS heap for the command buffer for allocations
    /// of total size, pre-warm the MPS heap with a new MTLHeap allocation of sufficient size.
    ///
    /// This is a MPSCommandBuffer-specific method not available in MTLCommandBuffer.
    pub fn prefetch_heap_for_workload_size(&self, size: usize) {
        unsafe {
            let _: () = msg_send![self, prefetchHeapForWorkloadSize:size];
        }
    }
}
