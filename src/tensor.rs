use crate::{DataType, GraphObject, Operation, Shape};
use objc2::rc::Retained;
use objc2::runtime::NSObject;
use objc2::{extern_class, extern_conformance, extern_methods};
use objc2_foundation::CopyingHelper;
use objc2_foundation::NSCopying;
use objc2_foundation::NSObjectProtocol;

extern_class!(
    /// The symbolic representation of a compute data type.
    ///
    /// `NSCopy` will take a refrence, this is so `NSDictionary` can work with the tensor.
    /// All tensors are created, owned and destroyed by the MPSGraph
    ///
    /// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphtensor?language=objc)
    #[unsafe(super(GraphObject, NSObject))]
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[name = "MPSGraphTensor"]
    pub struct Tensor;
);

extern_conformance!(
    unsafe impl NSCopying for Tensor {}
);

unsafe impl CopyingHelper for Tensor {
    type Result = Self;
}

extern_conformance!(
    unsafe impl NSObjectProtocol for Tensor {}
);

impl Tensor {
    extern_methods!(
        /// The shape of the tensor.
        ///
        /// nil shape represents an unranked tensor.
        /// -1 value for a dimension represents that it will be resolved via shape inference at runtime and it can be anything.
        #[unsafe(method(shape))]
        #[unsafe(method_family = none)]
        pub unsafe fn shape(&self) -> Option<Retained<Shape>>;

        /// The data type of the tensor.
        #[unsafe(method(dataType))]
        #[unsafe(method_family = none)]
        pub unsafe fn data_type(&self) -> DataType;

        /// The operation responsible for creating this tensor.
        #[unsafe(method(operation))]
        #[unsafe(method_family = none)]
        pub unsafe fn operation(&self) -> Retained<Operation>;
    );
}
