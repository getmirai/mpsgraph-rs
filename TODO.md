# Metal Performance Shaders Graph Operations - Implementation Status

This document catalogs the operations available in Metal Performance Shaders Graph and their implementation status in the mpsgraph-rs Rust bindings.

Legend:
- ✓ - Implemented
- ☐ - Not implemented

## Activation Operations (MPSGraphActivationOps.h)

- ✓ ReLU
- ✓ Sigmoid
- ✓ Tanh
- ✓ Softmax
- ✓ GELU (Gaussian Error Linear Unit)
- ✓ ELU (Exponential Linear Unit)
- ✓ LeakyReLU
- ✓ SoftPlus
- ✓ SoftSign
- ✓ HardSigmoid
- ✓ HardSwish
- ☐ ReLUN (with custom computation precision)
- ☐ SigmoidN (with custom computation precision)
- ☐ TanhN (with custom computation precision)
- ☐ SoftmaxN (with custom computation precision)

## Arithmetic Operations (MPSGraphArithmeticOps.h)

- ✓ Add (binary operation)
- ✓ Subtract
- ✓ Multiply
- ✓ Divide
- ✓ Power
- ✓ Modulo
- ✓ Minimum
- ✓ Maximum
- ✓ Equal
- ✓ NotEqual
- ✓ GreaterThan
- ✓ GreaterThanOrEqual
- ✓ LessThan
- ✓ LessThanOrEqual
- ✓ LogicalAND
- ✓ LogicalOR
- ✓ LogicalNOT
- ✓ LogicalXOR
- ✓ BitwiseAND
- ✓ BitwiseOR
- ✓ BitwiseXOR
- ✓ BitwiseNOT
- ✓ LeftShift
- ✓ RightShift
- ✓ Absolute
- ✓ Exponent
- ✓ Logarithm (base e, 2, 10)
- ✓ Square
- ✓ SquareRoot
- ✓ ReciprocalSquareRoot
- ✓ Sign
- ✓ Ceiling
- ✓ Floor
- ✓ Round
- ✓ Sin
- ✓ Cos
- ✓ Tan
- ✓ Asin
- ✓ Acos
- ✓ Atan
- ✓ Sinh
- ✓ Cosh
- ✓ Tanh
- ✓ Asinh
- ✓ Acosh
- ✓ Atanh
- ☐ AddN (multiple tensor addition)
- ☐ MultiplyN (multiple tensor multiplication)
- ☐ Clamp (range clamping)

## Call Operations (MPSGraphCallOps.h)

- ✓ Call (basic function call)
- ☐ CallForward
- ☐ CallBackward
- ☐ OptimizedCall
- ☐ CustomCall

## Control Flow Operations (MPSGraphControlFlowOps.h)

- ✓ Condition
- ✓ If
- ✓ While
- ☐ ForLoop
- ☐ Switch
- ☐ Case

## Convolution Operations (MPSGraphConvolutionOps.h)

- ✓ Convolution2D
- ✓ Convolution1D
- ✓ Convolution3D
- ☐ DilatedConvolution
- ☐ GroupedConvolution
- ☐ ConvolutionGradient
- ☐ ConvolutionDataGradient
- ☐ ConvolutionWeightsGradient
- ☐ ConvolutionBiasGradient

## Convolution Transpose Operations (MPSGraphConvolutionTransposeOps.h)

- ✓ ConvolutionTranspose2D
- ✓ ConvolutionTranspose1D
- ✓ ConvolutionTranspose3D
- ☐ DilatedConvolutionTranspose
- ☐ GroupedConvolutionTranspose
- ☐ ConvolutionTransposeGradient
- ☐ ConvolutionTransposeDataGradient
- ☐ ConvolutionTransposeWeightsGradient
- ☐ ConvolutionTransposeBiasGradient

## Cumulative Operations (MPSGraphCumulativeOps.h)

- ✓ CumulativeSum
- ✓ CumulativeProduct
- ✓ CumulativeMinimum
- ✓ CumulativeMaximum
- ☐ CumulativeLogSumExp

## Depthwise Convolution Operations (MPSGraphDepthwiseConvolutionOps.h)

- ✓ DepthwiseConvolution2D
- ☐ DepthwiseConvolution1D
- ☐ DepthwiseConvolution3D
- ☐ DepthwiseConvolutionGradient
- ☐ DepthwiseConvolutionDataGradient
- ☐ DepthwiseConvolutionWeightsGradient
- ☐ DepthwiseConvolutionBiasGradient

## Executable Operations (MPSGraphExecutable.h)

- ✓ CreateExecutable
- ✓ RunExecutable
- ✓ WaitForCompletion
- ✓ ExecutableDescriptor
- ☐ ExecutableProfiling
- ☐ ExecutableOptimization

## Fourier Transform Operations (MPSGraphFourierTransformOps.h)

- ✓ ForwardFourierTransform1D
- ✓ InverseFourierTransform1D
- ✓ ForwardFourierTransform2D
- ✓ InverseFourierTransform2D
- ☐ ForwardFourierTransform3D
- ☐ InverseFourierTransform3D
- ☐ ForwardFourierTransformNormalized
- ☐ InverseFourierTransformNormalized

## Gather Operations (MPSGraphGatherOps.h)

- ✓ Gather (basic)
- ✓ GatherND
- ✓ GatherAlongAxis
- ☐ GatherGradient
- ☐ GatherNDGradient
- ☐ GatherAlongAxisGradient

## ImToCol Operations (MPSGraphImToColOps.h)

- ✓ ImageToColumn
- ✓ ColumnToImage
- ☐ ImageToColumnGradient
- ☐ ColumnToImageGradient

## Linear Algebra Operations (MPSGraphLinearAlgebraOps.h)

- ✓ MatrixMultiplication
- ✓ MatrixTranspose
- ✓ BatchMatrixMultiplication
- ☐ MatrixSolve
- ☐ MatrixDeterminant
- ☐ MatrixLUFactorization
- ☐ MatrixSVDecomposition

## Loss Operations (MPSGraphLossOps.h)

- ✓ SoftmaxCrossEntropy
- ✓ CategoricalCrossEntropy
- ✓ MeanSquaredError
- ✓ HingeLoss
- ☐ HuberLoss
- ☐ L1Loss
- ☐ LogLoss
- ☐ SigmoidCrossEntropy

## Matrix Inverse Operations (MPSGraphMatrixInverseOps.h)

- ✓ MatrixInverse
- ☐ PseudoInverse
- ☐ MatrixInverseGradient

## Memory Operations (MPSGraphMemoryOps.h)

- ✓ Load
- ✓ Store
- ✓ Read
- ✓ Write
- ✓ Placeholder
- ✓ Constant
- ✓ Variable
- ☐ TemporaryStorage
- ☐ PersistentStorage

## Non-Maximum Suppression Operations (MPSGraphNonMaximumSuppressionOps.h)

- ✓ NonMaximumSuppression
- ☐ NonMaximumSuppressionWithScores
- ☐ NonMaximumSuppressionWithIndices

## Non-Zero Operations (MPSGraphNonZeroOps.h)

- ✓ NonZero
- ☐ NonZeroIndices
- ☐ NonZeroCount

## Normalization Operations (MPSGraphNormalizationOps.h)

- ✓ BatchNormalization
- ✓ LayerNormalization
- ✓ InstanceNormalization
- ✓ GroupNormalization
- ☐ BatchNormalizationGradient
- ☐ LayerNormalizationGradient
- ☐ InstanceNormalizationGradient
- ☐ GroupNormalizationGradient

## One-Hot Operations (MPSGraphOneHotOps.h)

- ✓ OneHot
- ☐ OneHotCategorical
- ☐ OneHotWithProbabilities

## Optimization Operations (MPSGraphOptimizerOps.h)

- ✓ SGD (Stochastic Gradient Descent)
- ✓ Adam
- ✓ RMSProp
- ✓ AdaGrad
- ☐ AdaDelta
- ☐ AMSGrad
- ☐ Momentum
- ☐ Nesterov

## Pooling Operations (MPSGraphPoolingOps.h)

- ✓ MaxPooling2D
- ✓ AveragePooling2D
- ✓ MaxPooling1D
- ✓ AveragePooling1D
- ✓ MaxPooling3D
- ✓ AveragePooling3D
- ✓ L2Pooling2D
- ☐ MaxPoolingGradient
- ☐ AveragePoolingGradient
- ☐ L2PoolingGradient
- ☐ AdaptivePooling

## Quantization Operations (MPSGraphQuantizationOps.h)

- ✓ Quantize
- ✓ Dequantize
- ✓ QuantizePerChannel
- ✓ DequantizePerChannel
- ☐ QuantizedMatrixMultiplication
- ☐ QuantizedConvolution
- ☐ QuantizedDepthwiseConvolution

## Random Operations (MPSGraphRandomOps.h)

- ✓ RandomUniform
- ✓ RandomNormal
- ✓ RandomBernoulli
- ✓ RandomCategorical
- ☐ RandomExponential
- ☐ RandomGamma
- ☐ RandomPoisson
- ☐ RandomBinomial

## Reduction Operations (MPSGraphReductionOps.h)

- ✓ ReduceSum
- ✓ ReduceProduct
- ✓ ReduceMin
- ✓ ReduceMax
- ✓ ReduceMean
- ✓ ReduceArgMin
- ✓ ReduceArgMax
- ✓ ReduceAll
- ✓ ReduceAny
- ☐ ReduceLogSumExp
- ☐ ReduceL1Norm
- ☐ ReduceL2Norm

## Resize Operations (MPSGraphResizeOps.h)

- ✓ ResizeNearest
- ✓ ResizeBilinear
- ✓ ResizeBicubic
- ☐ ResizeTrilinear
- ☐ ResizeNearestGradient
- ☐ ResizeBilinearGradient
- ☐ ResizeBicubicGradient

## RNN Operations (MPSGraphRNNOps.h)

- ✓ VanillaRNN
- ✓ LSTM
- ✓ GRU
- ☐ BidirectionalRNN
- ☐ BidirectionalLSTM
- ☐ BidirectionalGRU
- ☐ StackedRNN
- ☐ StackedLSTM
- ☐ StackedGRU
- ☐ RNNGradient
- ☐ LSTMGradient
- ☐ GRUGradient

## Sample Grid Operations (MPSGraphSampleGridOps.h)

- ✓ SampleGrid
- ✓ AffineTransform
- ✓ PerspectiveTransform
- ☐ SampleGridGradient
- ☐ AffineTransformGradient
- ☐ PerspectiveTransformGradient

## ScatterND Operations (MPSGraphScatterNDOps.h)

- ✓ ScatterND
- ✓ ScatterAlongAxis
- ✓ ScatterToIndex
- ☐ ScatterNDAdd
- ☐ ScatterNDSubtract
- ☐ ScatterNDMultiply
- ☐ ScatterNDDivide
- ☐ ScatterNDMin
- ☐ ScatterNDMax

## Sort Operations (MPSGraphSortOps.h)

- ✓ Sort
- ✓ ArgSort
- ✓ TopK
- ☐ ArgTopK
- ☐ BottomK
- ☐ ArgBottomK

## Sparse Operations (MPSGraphSparseOps.h)

- ✓ SparseToDense
- ✓ DenseToSparse
- ✓ SparseMatrixMultiplication
- ☐ SparseConvolution
- ☐ SparseGather
- ☐ SparseSegmentSum
- ☐ SparseSegmentMean

## Stencil Operations (MPSGraphStencilOps.h)

- ✓ StencilKernel2D
- ✓ StencilKernel1D
- ✓ StencilKernel3D
- ☐ StencilGradient
- ☐ StencilDataGradient
- ☐ StencilWeightsGradient

## Tensor Shape Operations (MPSGraphTensorShapeOps.h)

- ✓ Reshape
- ✓ Flatten
- ✓ Concatenate
- ✓ Stack
- ✓ Split
- ✓ Slice
- ✓ Tile
- ✓ Transpose
- ✓ Permute
- ✓ ExpandDims
- ✓ Squeeze
- ✓ Pad
- ✓ BroadcastTo
- ✓ Reverse
- ✓ ReverseSequence
- ☐ Unfold
- ☐ Roll
- ☐ SliceGradient
- ☐ PadGradient

## TopK Operations (MPSGraphTopKOps.h)

- ✓ TopK
- ✓ TopKIndices
- ✓ TopKValues
- ☐ TopKGradient