use crate::graph::Graph;
use crate::tensor::Tensor;
use objc2::msg_send;
use objc2::runtime::AnyObject;
use std::fmt;
use std::ptr;

/// A wrapper for Metal Performance Shaders Graph operation objects
pub struct Operation(pub(crate) *mut AnyObject);

impl Operation {
    /// Returns the input tensors of this operation
    pub fn input_tensors(&self) -> Vec<Tensor> {
        unsafe {
            let input_tensors: *mut AnyObject = msg_send![self.0, inputTensors];
            let count: usize = msg_send![input_tensors, count];
            let mut result = Vec::with_capacity(count);

            for i in 0..count {
                let tensor: *mut AnyObject = msg_send![input_tensors, objectAtIndex: i,];
                let tensor = objc2::ffi::objc_retain(tensor as *mut _);
                result.push(Tensor(tensor));
            }

            result
        }
    }

    /// Returns the output tensors of this operation
    pub fn output_tensors(&self) -> Vec<Tensor> {
        unsafe {
            let output_tensors: *mut AnyObject = msg_send![self.0, outputTensors];
            let count: usize = msg_send![output_tensors, count];
            let mut result = Vec::with_capacity(count);

            for i in 0..count {
                let tensor: *mut AnyObject = msg_send![output_tensors, objectAtIndex: i,];
                let tensor = objc2::ffi::objc_retain(tensor as *mut _);
                result.push(Tensor(tensor));
            }

            result
        }
    }

    /// Returns the graph this operation belongs to
    pub fn graph(&self) -> Graph {
        unsafe {
            let graph: *mut AnyObject = msg_send![self.0, graph];
            let graph = objc2::ffi::objc_retain(graph as *mut _);
            Graph(graph)
        }
    }

    /// Returns the name of this operation
    pub fn name(&self) -> String {
        unsafe {
            let name: *mut AnyObject = msg_send![self.0, name];
            let utf8: *const i8 = msg_send![name, UTF8String];
            std::ffi::CStr::from_ptr(utf8).to_string_lossy().to_string()
        }
    }

    /// Returns the control dependencies of this operation
    pub fn control_dependencies(&self) -> Vec<Operation> {
        unsafe {
            let dependencies: *mut AnyObject = msg_send![self.0, controlDependencies];
            let count: usize = msg_send![dependencies, count];
            let mut result = Vec::with_capacity(count);

            for i in 0..count {
                let op: *mut AnyObject = msg_send![dependencies, objectAtIndex: i,];
                let op = objc2::ffi::objc_retain(op as *mut _);
                result.push(Operation(op));
            }

            result
        }
    }
}

impl Drop for Operation {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                objc2::ffi::objc_release(self.0 as *mut _);
            }
        }
    }
}

impl Clone for Operation {
    fn clone(&self) -> Self {
        unsafe {
            if !self.0.is_null() {
                let obj = objc2::ffi::objc_retain(self.0 as *mut _);
                Operation(obj)
            } else {
                Operation(ptr::null_mut())
            }
        }
    }
}

impl fmt::Debug for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Operation")
            .field("name", &self.name())
            .finish()
    }
}
