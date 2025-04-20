use objc2::encode::{Encode, RefEncode};
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2_foundation::{NSArray, NSNumber};
use std::ops::Deref;

/// Shape wrapper for Metal Performance Shaders Graph shape objects that represent tensor dimensions
#[derive(Debug)]
pub struct Shape(Retained<NSArray<NSNumber>>);

impl Shape {
    /// Create a new Shape from an NSArray
    pub fn new(array: Retained<NSArray<NSNumber>>) -> Self {
        Self(array)
    }

    /// Create a Shape from a slice of NSNumber references
    pub fn from_slice(numbers: &[&NSNumber]) -> Self {
        Self(NSArray::from_slice(numbers))
    }

    /// Create a Shape from a slice of i64 values
    pub fn from_dimensions(dimensions: &[i64]) -> Self {
        let numbers: Vec<Retained<NSNumber>> =
            dimensions.iter().map(|&d| NSNumber::new_i64(d)).collect();

        let refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();
        Self(NSArray::from_slice(&refs))
    }

    /// Create a Shape representing a scalar
    pub fn scalar() -> Self {
        Self::from_slice(&[&NSNumber::new_i64(1)])
    }

    /// Create a Shape representing a vector
    pub fn vector(length: i64) -> Self {
        Self::from_slice(&[&NSNumber::new_i64(length)])
    }

    /// Create a Shape representing a matrix
    pub fn matrix(rows: i64, columns: i64) -> Self {
        Self::from_slice(&[&NSNumber::new_i64(rows), &NSNumber::new_i64(columns)])
    }

    /// Create a Shape representing a 3D tensor
    pub fn tensor3d(dim1: i64, dim2: i64, dim3: i64) -> Self {
        Self::from_slice(&[
            &NSNumber::new_i64(dim1),
            &NSNumber::new_i64(dim2),
            &NSNumber::new_i64(dim3),
        ])
    }

    /// Create a Shape representing a 4D tensor
    pub fn tensor4d(dim1: i64, dim2: i64, dim3: i64, dim4: i64) -> Self {
        Self::from_slice(&[
            &NSNumber::new_i64(dim1),
            &NSNumber::new_i64(dim2),
            &NSNumber::new_i64(dim3),
            &NSNumber::new_i64(dim4),
        ])
    }

    /// Get the dimensions as a vector
    pub fn dimensions(&self) -> Vec<i64> {
        let count = self.0.len();
        let mut result = Vec::with_capacity(count);

        for i in 0..count {
            let num = self.0.objectAtIndex(i);
            let value = num.as_i64();
            result.push(value);
        }

        result
    }

    /// Get the inner NSArray
    pub fn as_array(&self) -> &NSArray<NSNumber> {
        &self.0
    }

    /// Unwrap the Shape into the inner NSArray
    pub fn into_inner(self) -> Retained<NSArray<NSNumber>> {
        self.0
    }
    
    /// Get the inner NSArray pointer for Objective-C interop
    pub fn as_ptr(&self) -> *mut AnyObject {
        self.0.as_ref() as *const NSArray<NSNumber> as *mut AnyObject
    }
}

impl Deref for Shape {
    type Target = NSArray<NSNumber>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// Allow conversion from Shape to Retained<NSArray<NSNumber>>
impl From<Shape> for Retained<NSArray<NSNumber>> {
    fn from(shape: Shape) -> Self {
        shape.0
    }
}

// Allow conversion from Retained<NSArray<NSNumber>> to Shape
impl From<Retained<NSArray<NSNumber>>> for Shape {
    fn from(array: Retained<NSArray<NSNumber>>) -> Self {
        Shape(array)
    }
}

impl AsRef<NSArray<NSNumber>> for Shape {
    fn as_ref(&self) -> &NSArray<NSNumber> {
        &self.0
    }
}

// Objective-C encoding for NSArray compatibility
unsafe impl Encode for Shape {
    const ENCODING: objc2::Encoding = <*mut AnyObject>::ENCODING;
}

unsafe impl RefEncode for Shape {
    const ENCODING_REF: objc2::Encoding = <*mut AnyObject>::ENCODING_REF;
}

// Remove ShapeHelper struct as its methods are now integrated into Shape
// Keep for backwards compatibility as deprecated
#[deprecated(since = "0.2.0", note = "Use Shape methods directly instead")]
pub struct ShapeHelper;

#[allow(deprecated)]
impl ShapeHelper {
    /// Create a Shape representing a scalar
    pub fn scalar() -> Shape {
        Shape::scalar()
    }

    /// Create a Shape representing a vector
    pub fn vector(length: i64) -> Shape {
        Shape::vector(length)
    }

    /// Create a Shape representing a matrix
    pub fn matrix(rows: i64, columns: i64) -> Shape {
        Shape::matrix(rows, columns)
    }

    /// Create a Shape representing a 3D tensor
    pub fn tensor3d(dim1: i64, dim2: i64, dim3: i64) -> Shape {
        Shape::tensor3d(dim1, dim2, dim3)
    }

    /// Create a Shape representing a 4D tensor
    pub fn tensor4d(dim1: i64, dim2: i64, dim3: i64, dim4: i64) -> Shape {
        Shape::tensor4d(dim1, dim2, dim3, dim4)
    }

    /// Create a Shape from a slice of dimensions
    pub fn from_dimensions(dimensions: &[i64]) -> Shape {
        Shape::from_dimensions(dimensions)
    }
}

