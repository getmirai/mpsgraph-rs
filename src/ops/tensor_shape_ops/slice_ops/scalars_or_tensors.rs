use crate::Tensor;

/// Represents the *start*, *end*, and *stride* parameters required by slice
/// operations.
///
/// Callers may supply these parameters either as plain Rust slices (`&[u64]`)—
/// one entry per tensor dimension—or as pre-curated [`Tensor`] objects (e.g.
/// computed at runtime).
///
/// The helper variants make this polymorphism explicit.
///
/// # Examples
///
/// Using compile-time scalars:
/// ```rust
/// let args = StartEndStrideScalarsOrTensors::Scalars {
///     starts: &[0, 0],
///     ends: &[10, 10],
///     strides: &[1, 1],
/// };
/// ```
///
/// Using tensors (dynamic at graph-build time):
/// ```rust
/// let args = StartEndStrideScalarsOrTensors::Tensors {
///     start_tensor,
///     end_tensor,
///     stride_tensor,
/// };
/// ```
pub enum StartEndStrideScalarsOrTensors<'a> {
    Scalars {
        starts: &'a [u64],
        ends: &'a [u64],
        strides: &'a [u64],
    },
    Tensors {
        start_tensor: &'a Tensor,
        end_tensor: &'a Tensor,
        stride_tensor: &'a Tensor,
    },
}
