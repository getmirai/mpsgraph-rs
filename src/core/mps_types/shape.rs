use objc2::{rc::Retained, runtime::AnyObject, Encode, RefEncode};
use objc2_foundation::{NSArray, NSNumber};
use std::ops::Deref;

/// An array of NSNumbers where dimension lengths provided by the user goes from slowest moving to fastest moving dimension.
/// This is same order as MLMultiArray in coreML and most frameworks in Python.
///
/// ```text
///   A shape @[5, 4, 2] would mean fastest moving 0th dimension is one with size 2,
///   1st dimension is size 4 finally slowest moving 2nd dimension is size 5.
/// ```
///
/// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshaders/mpsshape?language=objc)
#[repr(transparent)]
pub struct Shape(Retained<NSArray<NSNumber>>);

impl Shape {
    pub fn new(values: &[i64]) -> Self {
        let ns_numbers = values
            .iter()
            .map(|&x| NSNumber::new_i64(x))
            .collect::<Box<[Retained<NSNumber>]>>();
        let shape = NSArray::from_retained_slice(&ns_numbers);
        Self(shape)
    }
}

impl From<&[i64]> for Shape {
    fn from(values: &[i64]) -> Self {
        Self::new(values)
    }
}

impl Into<Box<[i64]>> for Shape {
    fn into(self) -> Box<[i64]> {
        self.0
            .to_vec()
            .into_iter()
            .map(|x| x.longLongValue())
            .collect::<Box<[i64]>>()
    }
}

impl From<&Shape> for Box<[i64]> {
    fn from(shape: &Shape) -> Self {
        shape
            .deref()
            .to_vec()
            .into_iter()
            .map(|x| x.longLongValue())
            .collect()
    }
}

impl<const N: usize> From<&[i64; N]> for Shape {
    fn from(values: &[i64; N]) -> Self {
        Self::new(values)
    }
}

impl From<Shape> for Retained<NSArray<NSNumber>> {
    fn from(shape: Shape) -> Self {
        shape.0
    }
}

impl From<Retained<NSArray<NSNumber>>> for Shape {
    fn from(array: Retained<NSArray<NSNumber>>) -> Self {
        Shape(array)
    }
}

impl Deref for Shape {
    type Target = NSArray<NSNumber>;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

unsafe impl Encode for Shape {
    const ENCODING: objc2::Encoding = <*mut AnyObject>::ENCODING;
}

unsafe impl RefEncode for Shape {
    const ENCODING_REF: objc2::Encoding = <*mut AnyObject>::ENCODING_REF;
}
