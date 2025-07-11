use super::ExecutionStage;
use metal::foreign_types::ForeignType;
use metal::SharedEvent;
use objc2::rc::{Allocated, Retained};
use objc2::runtime::NSObject;
use objc2::{extern_class, msg_send, ClassType};
use objc2_foundation::NSObjectProtocol;

extern_class!(
    #[derive(Debug, PartialEq, Eq)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraphExecutableExecutionDescriptor"]
    pub struct ExecutableExecutionDescriptor;
);

unsafe impl NSObjectProtocol for ExecutableExecutionDescriptor {}

impl ExecutableExecutionDescriptor {
    /// Create a new executable execution descriptor
    pub fn new() -> Retained<Self> {
        unsafe {
            let class = Self::class();
            let allocated: Allocated<Self> = msg_send![class, alloc];
            // alloc/init returns a +1 retained object.
            let initialized: Retained<Self> = msg_send![allocated, init];
            initialized
        }
    }

    /// Set wait until completed flag
    pub fn set_wait_until_completed(&self, wait: bool) {
        unsafe {
            let _: () = msg_send![self, setWaitUntilCompleted: wait];
        }
    }

    /// Set the state of the execution descriptor to prefer synchronous execution
    pub fn prefer_synchronous_execution(&self) {
        self.set_wait_until_completed(true);
    }

    /// Set the state of the execution descriptor to prefer asynchronous execution
    pub fn prefer_asynchronous_execution(&self) {
        self.set_wait_until_completed(false);
    }

    /// Wait for a Metal shared event with a specific value before scheduling execution
    pub fn wait_for_event(&self, event: &SharedEvent, value: u64) {
        unsafe {
            let event_ptr = event.as_ptr() as *mut std::ffi::c_void;
            let _: () = msg_send![self, waitForEvent: event_ptr, value: value];
        }
    }

    /// Signal a Metal shared event with a value at a specific execution stage
    pub fn signal_event(&self, event: &SharedEvent, execution_stage: ExecutionStage, value: u64) {
        unsafe {
            let event_ptr = event.as_ptr() as *mut std::ffi::c_void;
            let _: () = msg_send![self, signalEvent: event_ptr, atExecutionEvent: execution_stage as u64, value: value];
        }
    }

    pub fn set_enable_commit_and_continue(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setEnableCommitAndContinue: enable];
        }
    }

    pub fn set_simulate_ane_compile_failure(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setSimulateANECompileFailure: enable];
        }
    }

    pub fn set_simulate_ane_load_model_failure(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setSimulateANELoadModelFailure: enable];
        }
    }

    pub fn set_disable_synchronize_results(&self, disable: bool) {
        unsafe {
            let _: () = msg_send![self, setDisableSynchronizeResults: disable];
        }
    }

    pub fn set_disable_ane_caching(&self, disable: bool) {
        unsafe {
            let _: () = msg_send![self, setDisableANECaching: disable];
        }
    }

    pub fn set_disable_ane_fallback(&self, disable: bool) {
        unsafe {
            let _: () = msg_send![self, setDisableANEFallback: disable];
        }
    }

    pub fn set_enable_profiling_op_names(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setEnableProfilingOpNames: enable];
        }
    }

    pub fn set_brief_profiling_op_names(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setBriefProfilingOpNames: enable];
        }
    }

    pub fn set_break_up_metal_encoders(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setBreakUpMetalEncoders: enable];
        }
    }

    pub fn set_generate_runtime_execution_report(&self, enable: bool) {
        unsafe {
            let _: () = msg_send![self, setGenerateRuntimeExecutionReport: enable];
        }
    }

    pub fn set_maximum_number_of_encoding_threads(&self, value: u64) {
        unsafe {
            let _: () = msg_send![self, setMaximumNumberOfEncodingThreads: value];
        }
    }

    pub fn number_of_commits_by_mps_graph(&self) -> u64 {
        unsafe { msg_send![self, numberOfCommitsByMPSGraph] }
    }
}
