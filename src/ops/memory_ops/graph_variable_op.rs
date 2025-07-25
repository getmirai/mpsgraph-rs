use objc2::{
    extern_class, extern_conformance, extern_methods,
    rc::{Allocated, Retained},
    runtime::NSObject,
};
use objc2_foundation::{CopyingHelper, NSCopying, NSObjectProtocol};

use crate::{DataType, Graph, GraphObject, Operation, Shape, ShapeOrTensor, Tensor};

extern_class!(
    /// The class that defines the parameters for a variable.
    ///
    /// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphvariableop?language=objc)
    #[unsafe(super(Operation, GraphObject, NSObject))]
    #[derive(Debug, PartialEq, Eq, Hash)]
    pub struct GraphVariableOp;
);

extern_conformance!(
    unsafe impl NSCopying for GraphVariableOp {}
);

unsafe impl CopyingHelper for GraphVariableOp {
    type Result = Self;
}

extern_conformance!(
    unsafe impl NSObjectProtocol for GraphVariableOp {}
);

impl GraphVariableOp {
    extern_methods!(
        /// The shape of the variable.
        #[unsafe(method(shape))]
        #[unsafe(method_family = none)]
        pub fn shape(&self) -> Retained<Shape>;

        /// The data type of the variable.
        #[unsafe(method(dataType))]
        #[unsafe(method_family = none)]
        pub fn data_type(&self) -> DataType;
    );
}

/// Methods declared on superclass `MPSGraphOperation`.
impl GraphVariableOp {
    extern_methods!(
        /// Unavailable, please utilize graph methods to create and initialize operations.
        #[unsafe(method(init))]
        #[unsafe(method_family = init)]
        pub fn init(this: Allocated<Self>) -> Retained<Self>;
    );
}

/// Methods declared on superclass `NSObject`.
impl GraphVariableOp {
    extern_methods!(
        #[unsafe(method(new))]
        #[unsafe(method_family = new)]
        pub fn new() -> Retained<Self>;
    );
}
