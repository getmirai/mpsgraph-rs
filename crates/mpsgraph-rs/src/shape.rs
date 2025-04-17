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
        Shape::from_slice(&[&NSNumber::new_usize(1)])
    }

    /// Create a Shape representing a vector
    pub fn vector(length: usize) -> Retained<Shape> {
        Shape::from_slice(&[&NSNumber::new_usize(length)])
    }

    /// Create a Shape representing a matrix
    pub fn matrix(rows: usize, columns: usize) -> Retained<Shape> {
        Shape::from_slice(&[&NSNumber::new_usize(rows), &NSNumber::new_usize(columns)])
    }

    /// Create a Shape representing a 3D tensor
    pub fn tensor3d(dim1: usize, dim2: usize, dim3: usize) -> Retained<Shape> {
        Shape::from_slice(&[
            &NSNumber::new_usize(dim1),
            &NSNumber::new_usize(dim2),
            &NSNumber::new_usize(dim3),
        ])
    }

    /// Create a Shape representing a 4D tensor
    pub fn tensor4d(dim1: usize, dim2: usize, dim3: usize, dim4: usize) -> Retained<Shape> {
        Shape::from_slice(&[
            &NSNumber::new_usize(dim1),
            &NSNumber::new_usize(dim2),
            &NSNumber::new_usize(dim3),
            &NSNumber::new_usize(dim4),
        ])
    }

    /// Create a Shape from a slice of dimensions
    pub fn from_dimensions(dimensions: &[usize]) -> Retained<Shape> {
        let numbers: Vec<Retained<NSNumber>> = dimensions
            .iter()
            .map(|&d| NSNumber::new_usize(d))
            .collect();
            
        let refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();
        Shape::from_slice(&refs)
    }
}

/// Extension trait for NSArray to add shape-specific methods
pub trait ShapeExtensions {
    /// Get the dimensions as a vector
    fn dimensions(&self) -> Vec<usize>;
    
    /// Get the total number of elements in this shape
    fn element_count(&self) -> usize;
}

impl ShapeExtensions for Shape {
    fn dimensions(&self) -> Vec<usize> {
        let count = self.len();
        let mut result = Vec::with_capacity(count);

        for i in 0..count {
            // Use objectAtIndex to get the number at this index
            let num = self.objectAtIndex(i);
            let value = num.as_usize();
            result.push(value);
        }

        result
    }

    fn element_count(&self) -> usize {
        self.dimensions().iter().product()
    }
}