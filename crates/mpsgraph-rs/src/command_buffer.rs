use metal::foreign_types::ForeignType;
use metal::{CommandBuffer as MetalCommandBuffer, CommandQueue};
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
    pub fn from_command_buffer(command_buffer: &MetalCommandBuffer) -> Retained<Self> {
        unsafe {
            let class = Self::class();
            let alloc: *mut Self = msg_send![class, alloc];
            let buffer_ptr = command_buffer.as_ptr() as *mut objc2::runtime::AnyObject;

            // Pass the MTLCommandBuffer pointer as an Objective-C object
            let mps_command_buffer: *mut Self = msg_send![
                alloc,
                initWithCommandBuffer:buffer_ptr
            ];

            // init methods return retained objects, so from_raw is correct here
            Retained::from_raw(mps_command_buffer).unwrap()
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

            // Class methods return autoreleased objects, so we need to retain
            Retained::retain_autoreleased(mps_command_buffer).unwrap()
        }
    }

    /// Gets the underlying command buffer.
    ///
    /// This method returns the Metal CommandBuffer that was used to initialize this object.
    pub fn command_buffer(&self) -> MetalCommandBuffer {
        unsafe {
            let cmd_buffer: *mut objc2::runtime::AnyObject = msg_send![self, commandBuffer];
            // Create a metal::CommandBuffer from the raw pointer
            MetalCommandBuffer::from_ptr(cmd_buffer as _)
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
    pub fn root_command_buffer(&self) -> MetalCommandBuffer {
        unsafe {
            let cmd_buffer: *mut objc2::runtime::AnyObject = msg_send![self, rootCommandBuffer];
            // Create a metal::CommandBuffer from the raw pointer
            MetalCommandBuffer::from_ptr(cmd_buffer as _)
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
