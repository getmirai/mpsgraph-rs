use objc2::rc::Retained;
use objc2_foundation::{NSArray, NSNumber};

/// Type for Metal Performance Shaders Graph shape objects (NSArray) that represent tensor dimensions
pub type Shape = NSArray<NSNumber>;

/// Extension trait for Shape to add helper methods
pub trait ShapeHelpers {
    /// Creates a Shape from a slice of i64 values
    fn from_slice(dimensions: &[i64]) -> Retained<Self>;
}

impl ShapeHelpers for Shape {
    fn from_slice(dimensions: &[i64]) -> Retained<Self> {
        let numbers: Vec<Retained<NSNumber>> = dimensions
            .iter()
            .map(|&d| NSNumber::new_i64(d))
            .collect();
            
        let refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();
        NSArray::from_slice(&refs)
    }
}

/// Shape helper struct with static methods to create common shapes
pub struct ShapeHelper;

impl ShapeHelper {
    /// Create a Shape representing a scalar
    pub fn scalar() -> Retained<Shape> {
        Shape::from_slice(&[&NSNumber::new_i64(1)])
    }

    /// Create a Shape representing a vector
    pub fn vector(length: i64) -> Retained<Shape> {
        Shape::from_slice(&[&NSNumber::new_i64(length)])
    }

    /// Create a Shape representing a matrix
    pub fn matrix(rows: i64, columns: i64) -> Retained<Shape> {
        Shape::from_slice(&[&NSNumber::new_i64(rows), &NSNumber::new_i64(columns)])
    }

    /// Create a Shape representing a 3D tensor
    pub fn tensor3d(dim1: i64, dim2: i64, dim3: i64) -> Retained<Shape> {
        Shape::from_slice(&[
            &NSNumber::new_i64(dim1),
            &NSNumber::new_i64(dim2),
            &NSNumber::new_i64(dim3),
        ])
    }

    /// Create a Shape representing a 4D tensor
    pub fn tensor4d(dim1: i64, dim2: i64, dim3: i64, dim4: i64) -> Retained<Shape> {
        Shape::from_slice(&[
            &NSNumber::new_i64(dim1),
            &NSNumber::new_i64(dim2),
            &NSNumber::new_i64(dim3),
            &NSNumber::new_i64(dim4),
        ])
    }

    /// Create a Shape from a slice of dimensions
    pub fn from_dimensions(dimensions: &[i64]) -> Retained<Shape> {
        let numbers: Vec<Retained<NSNumber>> = dimensions
            .iter()
            .map(|&d| NSNumber::new_i64(d))
            .collect();
            
        let refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();
        Shape::from_slice(&refs)
    }
}

/// Extension trait for NSArray to add shape-specific methods
pub trait ShapeExtensions {
    /// Get the dimensions as a vector
    fn dimensions(&self) -> Vec<i64>;
}

impl ShapeExtensions for Shape {
    fn dimensions(&self) -> Vec<i64> {
        let count = self.len();
        let mut result = Vec::with_capacity(count);

        for i in 0..count {
            // Use objectAtIndex to get the number at this index
            let num = self.objectAtIndex(i);
            let value = num.as_i64();
            result.push(value);
        }

        result
    }
}