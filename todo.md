# ToDo

This checklist tracks our progress syncing Rust wrappers with the MetalPerformanceShadersGraph Objective-C headers.
The all the functions from headers should be represented in rust in the same order. Rust functions that use api that was not exposed in headers should be deleted.
Each item is marked once its wrapper follows the new naming convention and required extension traits are in place.


Code-mapping convention (recap):

• Rust method name = snake_case of the Obj-C selector **up to the first argument label** (`With`, If`, `While`, `For`, …).
  Example: `convolutionTranspose2DWeightsGradientWithIncomingGradientTensor` → convolution_transpose_2d_weights_gradient`.

• If multiple selectors map to the same Rust name, keep ONE inherent method on `Graph`; expose dditional overloads via extension traits.
  All such traits are publicly re-exported from the crate root so users get them automatically with use mpsgraph_rs::*;`.

• For overload traits we follow `<BaseName><Specifics>Ext` naming, e.g. Convolution2DDataGradientTensorShapeExt`.

• Optional `name` parameters are passed as
```rust
let name = name.map(NSString::from_str)
    .as_deref()
    .map_or(std::ptr::null(), |s| s as *const _);
```

• `Shape` objects are passed via `shape.as_ptr()` when Objective-C API expects an `MPSShape*`.

                         
- [x] ActivationOps
- [x] ArithmeticOps
- [x] CallOps
- [ ] ConvolutionOps
- [ ] ConvolutionTransposeOps
- [ ] ControlFlowOps
- [ ] CumulativeOps
- [ ] DepthwiseConvolutionOps
- [ ] FourierTransformOps
- [ ] GatherOps
- [ ] ImToColOps
- [ ] LinearAlgebraOps
- [ ] LossOps
- [ ] MatrixInverseOps
- [x] MatrixMultiplicationOps
- [ ] MemoryOps
- [ ] NonMaximumSuppressionOps
- [ ] NonZeroOps
- [ ] NormalizationOps
- [ ] OneHotOps
- [ ] OptimizerOps
- [ ] PoolingOps
- [ ] QuantizationOps
- [ ] RandomOps
- [ ] ReductionOps
- [ ] ResizeOps
- [ ] RNNOps
- [ ] SampleGridOps
- [ ] ScatterNDOps
- [ ] SortOps
- [ ] SparseOps
- [ ] StencilOps
- [ ] TensorShapeOps
- [ ] TopKOps


- [ ] MPSGraphTensorData
- [ ] MPSGraph
- [ ] MPSGraphCore
- [ ] MetalPerformanceShadersGraph
- [ ] MPSGraphExecutable
- [ ] MPSGraphOperation
- [ ] MPSGraphAutomaticDifferentiation
- [ ] MPSGraphDevice
- [ ] MPSGraphTensor