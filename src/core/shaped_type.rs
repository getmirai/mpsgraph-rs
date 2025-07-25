use super::{DataType, GraphObject, GraphType, Shape};
use objc2::extern_class;
use objc2::extern_conformance;
use objc2::extern_methods;
use objc2::rc::{Allocated, Retained};
use objc2_foundation::CopyingHelper;
use objc2_foundation::{NSCopying, NSObject, NSObjectProtocol};

extern_class!(
    /// The shaped type class for types on tensors with a shape and data type.
    ///
    /// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphshapedtype?language=objc)
    #[unsafe(super(GraphType, GraphObject, NSObject))]
    #[derive(Debug, PartialEq, Eq, Hash)]
    pub struct ShapedType;
);

extern_conformance!(
    unsafe impl NSCopying for ShapedType {}
);

unsafe impl CopyingHelper for ShapedType {
    type Result = Self;
}

extern_conformance!(
    unsafe impl NSObjectProtocol for ShapedType {}
);

impl ShapedType {
    extern_methods!(
        /// The Shape of the shaped type.
        #[unsafe(method(shape))]
        #[unsafe(method_family = none)]
        pub fn shape(&self) -> Option<Retained<Shape>>;

        /// Setter for [`shape`][Self::shape].
        #[unsafe(method(setShape:))]
        #[unsafe(method_family = none)]
        pub fn set_shape(&self, shape: Option<&Shape>);

        /// The data type of the shaped type.
        #[unsafe(method(dataType))]
        #[unsafe(method_family = none)]
        pub fn data_type(&self) -> DataType;

        /// Setter for [`dataType`][Self::dataType].
        #[unsafe(method(setDataType:))]
        #[unsafe(method_family = none)]
        pub fn set_data_type(&self, data_type: DataType);

        /// Initializes a shaped type.
        ///
        /// - Parameters:
        /// - shape: The shape of the shaped type.
        /// - dataType: The dataType of the shaped type.
        /// - Returns: A valid MPSGraphShapedType, or nil if allocation failure.
        #[unsafe(method(initWithShape:dataType:))]
        #[unsafe(method_family = init)]
        pub fn init_with_shape_data_type(
            this: Allocated<Self>,
            shape: Option<&Shape>,
            data_type: DataType,
        ) -> Retained<Self>;

        /// Checks if shapes and element data type are the same as the input shaped type.
        ///
        /// - Parameters:
        /// - object: shapedType to compare to
        /// - Returns: true if equal, false if unequal
        #[unsafe(method(isEqualTo:))]
        #[unsafe(method_family = none)]
        pub fn is_equal_to(&self, object: Option<&ShapedType>) -> bool;
    );
}

/// Methods declared on superclass `NSObject`.
impl ShapedType {
    extern_methods!(
        #[unsafe(method(init))]
        #[unsafe(method_family = init)]
        pub fn init(this: Allocated<Self>) -> Retained<Self>;

        #[unsafe(method(new))]
        #[unsafe(method_family = new)]
        pub fn new() -> Retained<Self>;
    );
}
