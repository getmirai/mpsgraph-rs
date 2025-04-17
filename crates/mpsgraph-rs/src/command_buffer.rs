use metal::{CommandBuffer as MetalCommandBuffer, CommandQueue, Device as MetalDevice};
use metal::foreign_types::ForeignType;
use objc2::rc::Retained;
use objc2::{extern_class, msg_send, ClassType};
use objc2::runtime::NSObject;
use objc2_foundation::{NSObjectProtocol, NSString};

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
            
            Retained::from_raw(mps_command_buffer).unwrap()
        }
    }

    /// Creates a new MPSCommandBuffer from a command queue
    pub fn from_command_queue(command_queue: &CommandQueue) -> Retained<Self> {
        unsafe {
            let class = Self::class();
            let queue_ptr = command_queue.as_ptr() as *mut objc2::runtime::AnyObject;
            
            // Pass the MTLCommandQueue pointer as an Objective-C object
            let mps_command_buffer: *mut Self = msg_send![
                class, 
                commandBufferFromCommandQueue:queue_ptr
            ];
            
            Retained::from_raw(mps_command_buffer).unwrap()
        }
    }
    
    /// Returns the current status of the command buffer
    ///
    /// Since MPSCommandBuffer doesn't have a status method, this attempts
    /// to get the status from the underlying Metal command buffer.
    pub fn status(&self) -> CommandBufferStatus {
        unsafe {
            // Try direct status first
            let status_sel = objc2::sel!(status);
            let status_exists = objc2::ffi::class_getInstanceMethod(
                Self::class(), 
                status_sel
            ) != std::ptr::null_mut();
            
            if status_exists {
                let status: u32 = msg_send![self, status];
                return match status {
                    0 => CommandBufferStatus::NotEnqueued,
                    1 => CommandBufferStatus::Enqueued,
                    2 => CommandBufferStatus::Committed,
                    3 => CommandBufferStatus::Scheduled,
                    4 => CommandBufferStatus::Completed,
                    5 => CommandBufferStatus::Error,
                    _ => CommandBufferStatus::NotEnqueued,
                };
            }
            
            // Try to get the underlying metal command buffer
            let metal_command_buffer_sel = objc2::sel!(metalCommandBuffer);
            let method_exists = objc2::ffi::class_getInstanceMethod(
                Self::class(), 
                metal_command_buffer_sel
            ) != std::ptr::null_mut();
            
            if method_exists {
                let metal_buffer: *mut objc2::runtime::AnyObject = msg_send![self, metalCommandBuffer];
                if !metal_buffer.is_null() {
                    let status: u32 = msg_send![metal_buffer, status];
                    return match status {
                        0 => CommandBufferStatus::NotEnqueued,
                        1 => CommandBufferStatus::Enqueued,
                        2 => CommandBufferStatus::Committed,
                        3 => CommandBufferStatus::Scheduled,
                        4 => CommandBufferStatus::Completed,
                        5 => CommandBufferStatus::Error,
                        _ => CommandBufferStatus::NotEnqueued,
                    };
                }
            }
            
            // Default to NotEnqueued if we can't get the status
            CommandBufferStatus::NotEnqueued
        }
    }
    
    /// Commits the current command buffer and continues with a new one
    ///
    /// Since MPSCommandBuffer might not have a commitAndContinue method, this attempts
    /// to call the method on the underlying Metal command buffer.
    pub fn commit_and_continue(&self) {
        unsafe {
            // Try direct commit first
            let commit_continue_sel = objc2::sel!(commitAndContinue);
            let commit_continue_exists = objc2::ffi::class_getInstanceMethod(
                Self::class(), 
                commit_continue_sel
            ) != std::ptr::null_mut();
            
            if commit_continue_exists {
                let _: () = msg_send![self, commitAndContinue];
                return;
            }
            
            // If that fails, try to get the underlying metal command buffer
            let metal_command_buffer_sel = objc2::sel!(metalCommandBuffer);
            let method_exists = objc2::ffi::class_getInstanceMethod(
                Self::class(), 
                metal_command_buffer_sel
            ) != std::ptr::null_mut();
            
            if method_exists {
                let metal_buffer: *mut objc2::runtime::AnyObject = msg_send![self, metalCommandBuffer];
                if !metal_buffer.is_null() {
                    // Check if the metal buffer has commitAndContinue
                    let commit_continue_exists = objc2::ffi::class_getInstanceMethod(
                        (*metal_buffer).class(), 
                        commit_continue_sel
                    ) != std::ptr::null_mut();
                    
                    if commit_continue_exists {
                        let _: () = msg_send![metal_buffer, commitAndContinue];
                        return;
                    }
                    
                    // If no commitAndContinue, just commit
                    let _: () = msg_send![metal_buffer, commit];
                }
            }
        }
    }
    
    /// Prefetches heap for workload size
    ///
    /// This is a specialized method that may not exist on all MPSCommandBuffer implementations.
    pub fn prefetch_heap_for_workload_size(&self, size: usize) {
        unsafe {
            // Check if our own implementation exists
            let prefetch_sel = objc2::sel!(prefetchHeapForWorkloadSize:);
            let prefetch_method_exists = objc2::ffi::class_getInstanceMethod(
                Self::class(), 
                prefetch_sel
            ) != std::ptr::null_mut();
            
            if prefetch_method_exists {
                let _: () = msg_send![self, prefetchHeapForWorkloadSize:size];
            }
            // Silently ignore if the method doesn't exist
        }
    }
    
    /// Commits the command buffer for execution as soon as possible
    ///
    /// Since MPSCommandBuffer might not have a commit method, this attempts
    /// to call the method on the underlying Metal command buffer.
    pub fn commit(&self) {
        unsafe {
            // Try direct commit first
            let commit_sel = objc2::sel!(commit);
            let commit_exists = objc2::ffi::class_getInstanceMethod(
                Self::class(), 
                commit_sel
            ) != std::ptr::null_mut();
            
            if commit_exists {
                let _: () = msg_send![self, commit];
                return;
            }
            
            // If that fails, try to get the underlying metal command buffer
            let metal_command_buffer_sel = objc2::sel!(metalCommandBuffer);
            let method_exists = objc2::ffi::class_getInstanceMethod(
                Self::class(), 
                metal_command_buffer_sel
            ) != std::ptr::null_mut();
            
            if method_exists {
                let metal_buffer: *mut objc2::runtime::AnyObject = msg_send![self, metalCommandBuffer];
                if !metal_buffer.is_null() {
                    let _: () = msg_send![metal_buffer, commit];
                }
            }
        }
    }
    
    /// Waits until this command buffer has completed execution
    ///
    /// Since MPSCommandBuffer doesn't have a waitUntilCompleted method, this attempts
    /// to call the method on the underlying Metal command buffer.
    pub fn wait_until_completed(&self) {
        unsafe {
            // Try to get the underlying metal command buffer
            let metal_command_buffer_sel = objc2::sel!(metalCommandBuffer);
            let method_exists = objc2::ffi::class_getInstanceMethod(
                Self::class(), 
                metal_command_buffer_sel
            ) != std::ptr::null_mut();
            
            if method_exists {
                let metal_buffer: *mut objc2::runtime::AnyObject = msg_send![self, metalCommandBuffer];
                if !metal_buffer.is_null() {
                    let _: () = msg_send![metal_buffer, waitUntilCompleted];
                }
            }
        }
    }
    
    /// Sets a label to help identify this object
    /// 
    /// This attempts to get the underlying Metal command buffer and set its label,
    /// since MPSCommandBuffer itself doesn't have a setLabel method.
    pub fn set_label(&self, label: &str) {
        unsafe {
            // Try to get the underlying metal command buffer
            let metal_command_buffer_sel = objc2::sel!(metalCommandBuffer);
            let method_exists = objc2::ffi::class_getInstanceMethod(
                Self::class(), 
                metal_command_buffer_sel
            ) != std::ptr::null_mut();
            
            if method_exists {
                let metal_buffer: *mut objc2::runtime::AnyObject = msg_send![self, metalCommandBuffer];
                if !metal_buffer.is_null() {
                    let ns_string = NSString::from_str(label);
                    let _: () = msg_send![metal_buffer, setLabel:&*ns_string];
                }
            }
            // Silently ignore if the method doesn't exist
        }
    }
    
    /// Returns the current label
    /// 
    /// Since MPSCommandBuffer doesn't have a label method, this attempts to get the
    /// label from the underlying Metal command buffer.
    pub fn label(&self) -> String {
        unsafe {
            // Try to get the underlying metal command buffer
            let metal_command_buffer_sel = objc2::sel!(metalCommandBuffer);
            let method_exists = objc2::ffi::class_getInstanceMethod(
                Self::class(), 
                metal_command_buffer_sel
            ) != std::ptr::null_mut();
            
            if method_exists {
                let metal_buffer: *mut objc2::runtime::AnyObject = msg_send![self, metalCommandBuffer];
                if !metal_buffer.is_null() {
                    let label_ptr: *mut NSString = msg_send![metal_buffer, label];
                    if !label_ptr.is_null() {
                        let ns_string = Retained::from_raw(label_ptr).unwrap();
                        return ns_string.to_string();
                    }
                }
            }
            
            // Return empty string if we can't get the label
            String::new()
        }
    }
}

impl crate::CustomDefault for CommandBuffer {
    fn custom_default() -> Retained<Self> {
        // Get the default Metal device and create a command queue
        let device = MetalDevice::system_default().expect("No Metal device found");
        let command_queue = device.new_command_queue();
        
        // Create a command buffer from the queue
        Self::from_command_queue(&command_queue)
    }
}