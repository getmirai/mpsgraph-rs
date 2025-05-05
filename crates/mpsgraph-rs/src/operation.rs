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
            let input_tensors: *mut NSArray<Tensor> = msg_send![self, inputTensors];
            let array_ptr = input_tensors;
            let count: usize = msg_send![array_ptr, count];
            let mut result = Vec::with_capacity(count);

            for i in 0..count {
                let tensor_ptr: *mut Tensor = msg_send![array_ptr, objectAtIndex: i];
                let tensor = Retained::retain_autoreleased(tensor_ptr).unwrap();
                result.push(tensor);
            }

            let _ = Retained::retain_autoreleased(array_ptr).unwrap();

            result
        }
    }

    /// Returns the output tensors of this operation
    pub fn output_tensors(&self) -> Vec<Retained<Tensor>> {
        unsafe {
            let output_tensors: *mut NSArray<Tensor> = msg_send![self, outputTensors];
            let array_ptr = output_tensors;
            let count: usize = msg_send![array_ptr, count];
            let mut result = Vec::with_capacity(count);

            for i in 0..count {
                let tensor_ptr: *mut Tensor = msg_send![array_ptr, objectAtIndex: i];
                let tensor = Retained::retain_autoreleased(tensor_ptr).unwrap();
                result.push(tensor);
            }

            let _ = Retained::retain_autoreleased(array_ptr).unwrap();

            result
        }
    }

    /// Returns the graph this operation belongs to
    pub fn graph(&self) -> Retained<Graph> {
        unsafe {
            let graph_ptr: *mut Graph = msg_send![self, graph];
            let graph = Retained::retain_autoreleased(graph_ptr).unwrap();
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
            let dependencies_ptr: *mut NSArray<Operation> = msg_send![self, controlDependencies];
            let array = Retained::retain_autoreleased(dependencies_ptr).unwrap();
            let count = array.len();
            let mut result = Vec::with_capacity(count);

            for i in 0..count {
                let op_ptr: *mut Operation = msg_send![&*array, objectAtIndex: i];
                let op = Retained::retain_autoreleased(op_ptr).unwrap();
                result.push(op);
            }

            result
        }
    }
}
