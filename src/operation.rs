use objc2::rc::Retained;
use objc2::runtime::NSObject;
use objc2::{extern_class, msg_send};
use objc2_foundation::{NSArray, NSObjectProtocol, NSString};

use crate::graph::Graph;
use crate::tensor::Tensor;

extern_class!(
    #[derive(Debug, PartialEq, Eq)]
    #[unsafe(super = NSObject)]
    #[name = "MPSGraphOperation"]
    /// A wrapper for MPSGraph operation objects
    pub struct Operation;
);

unsafe impl NSObjectProtocol for Operation {}

impl Operation {
    /// Returns the input tensors of this operation
    pub fn input_tensors(&self) -> Vec<Retained<Tensor>> {
        unsafe {
            let array_opt: Option<Retained<NSArray<Tensor>>> = msg_send![self, inputTensors];
            if let Some(array) = array_opt {
                let count: usize = array.len();
                let mut result = Vec::with_capacity(count);
                for i in 0..count {
                    let tensor: Retained<Tensor> = msg_send![&*array, objectAtIndex: i];
                    result.push(tensor);
                }
                result
            } else {
                Vec::new()
            }
        }
    }

    /// Returns the output tensors of this operation
    pub fn output_tensors(&self) -> Vec<Retained<Tensor>> {
        unsafe {
            let array_opt: Option<Retained<NSArray<Tensor>>> = msg_send![self, outputTensors];
            if let Some(array) = array_opt {
                let count: usize = array.len();
                let mut result = Vec::with_capacity(count);
                for i in 0..count {
                    let tensor: Retained<Tensor> = msg_send![&*array, objectAtIndex: i];
                    result.push(tensor);
                }
                result
            } else {
                Vec::new()
            }
        }
    }

    /// Returns the graph this operation belongs to
    pub fn graph(&self) -> Retained<Graph> {
        unsafe {
            let graph: Retained<Graph> = msg_send![self, graph];
            graph
        }
    }

    /// Returns the name of this operation
    pub fn name(&self) -> Option<String> {
        unsafe {
            let name_ptr: *mut NSString = msg_send![self, name];
            if name_ptr.is_null() {
                None
            } else {
                let name = Retained::retain_autoreleased(name_ptr).unwrap();
                Some(name.to_string())
            }
        }
    }

    /// Returns the control dependencies of this operation
    pub fn control_dependencies(&self) -> Vec<Retained<Operation>> {
        unsafe {
            let array_opt: Option<Retained<NSArray<Operation>>> =
                msg_send![self, controlDependencies];
            if let Some(array) = array_opt {
                let count = array.len();
                let mut result = Vec::with_capacity(count);
                for i in 0..count {
                    let op: Retained<Operation> = msg_send![&*array, objectAtIndex: i];
                    result.push(op);
                }
                result
            } else {
                Vec::new()
            }
        }
    }
}
