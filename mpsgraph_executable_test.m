#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        NSLog(@"Starting MPSGraphExecutable encodeToCommandBuffer test...");

        // 1. Get Metal Device and Command Queue
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            NSLog(@"Error: Failed to get Metal device.");
            return 1;
        }
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        NSLog(@"Using device: %@", device.name);

        // 2. Create a simple graph (A + B = C)
        MPSGraph *graph = [[MPSGraph alloc] init];
        NSArray<NSNumber *> *shape = @[@2, @2]; // Simple 2x2 shape
        MPSGraphTensor *a = [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:@"a"];
        MPSGraphTensor *b = [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:@"b"];
        MPSGraphTensor *c = [graph additionWithPrimaryTensor:a secondaryTensor:b name:@"c"];
        NSLog(@"Graph created.");

        // 3. Compile the graph
        MPSGraphCompilationDescriptor *compDesc = [[MPSGraphCompilationDescriptor alloc] init];
        MPSGraphDevice *mpsGraphDevice = [MPSGraphDevice deviceWithMTLDevice:device];
        MPSGraphShapedType *shapedType = [[MPSGraphShapedType alloc] initWithShape:shape dataType:MPSDataTypeFloat32];
        NSDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feeds = @{a: shapedType, b: shapedType};

        MPSGraphExecutable *executable = [graph compileWithDevice:mpsGraphDevice
                                                             feeds:feeds
                                                       targetTensors:@[c]
                                                  targetOperations:nil
                                             compilationDescriptor:compDesc];
        if (!executable) {
             NSLog(@"Error: Failed to compile graph.");
             return 1;
        }
        NSLog(@"Graph compiled successfully.");


        // 4. Prepare Data & Buffers
        float aData[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float bData[] = {0.5f, 0.5f, 0.5f, 0.5f};
        NSUInteger bufferSize = 2 * 2 * sizeof(float);

        id<MTLBuffer> aBuffer = [device newBufferWithBytes:aData length:bufferSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuffer = [device newBufferWithBytes:bData length:bufferSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> cBuffer = [device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
        NSLog(@"Metal buffers created.");

        // 5. Create MPSGraphTensorData
        MPSGraphTensorData *aTensorData = [[MPSGraphTensorData alloc] initWithMTLBuffer:aBuffer shape:shape dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *bTensorData = [[MPSGraphTensorData alloc] initWithMTLBuffer:bBuffer shape:shape dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *cTensorData = [[MPSGraphTensorData alloc] initWithMTLBuffer:cBuffer shape:shape dataType:MPSDataTypeFloat32];
        NSLog(@"MPSGraphTensorData created.");

        // 6. Create Input/Output NSArrays
        NSArray<MPSGraphTensorData *> *inputs = @[aTensorData, bTensorData];
        NSArray<MPSGraphTensorData *> *results = @[cTensorData];
        NSLog(@"Input/Output arrays created.");

        // 7. Create Execution Descriptor
        MPSGraphExecutableExecutionDescriptor *descriptor = [[MPSGraphExecutableExecutionDescriptor alloc] init];
        descriptor.waitUntilCompleted = YES; // Make it synchronous for testing
        NSLog(@"Execution descriptor created.");

        // 8. Get Command Buffer
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        NSLog(@"Command buffer created.");

        // 9. *** Call the target method ***
        NSLog(@"Calling [executable encodeToCommandBuffer:...] ");
        @try {
            NSArray<MPSGraphTensorData*>* execResults = [executable encodeToCommandBuffer:commandBuffer
                                                                            inputsArray:inputs
                                                                           resultsArray:results // Optional: could be nil if we only want side effects
                                                                    executionDescriptor:descriptor];

             // Check if the returned array is what we expect (optional, depends on API contract)
             if (execResults) {
                 NSLog(@"encodeToCommandBuffer returned an array with %lu elements.", (unsigned long)execResults.count);
             } else {
                 NSLog(@"encodeToCommandBuffer returned nil (as expected if resultsArray was provided).");
             }

        } @catch (NSException *exception) {
            NSLog(@"*** CRASH/EXCEPTION during encodeToCommandBuffer: %@ ***", exception);
            NSLog(@"Reason: %@", exception.reason);
            // You might want to inspect exception.callStackSymbols here too
             return 1; // Indicate failure
        }
         NSLog(@"encodeToCommandBuffer call completed (no immediate crash).");


        // 10. Commit and Wait
        NSLog(@"Committing command buffer...");
        [commandBuffer commit];
        NSLog(@"Waiting for command buffer completion...");
        [commandBuffer waitUntilCompleted]; // Important if not using descriptor.waitUntilCompleted=YES

        // 11. Check for GPU errors (optional but recommended)
        if (commandBuffer.status == MTLCommandBufferStatusError) {
            NSLog(@"*** Error during command buffer execution: %@ ***", commandBuffer.error);
             return 1; // Indicate failure
        }

        NSLog(@"Command buffer executed successfully!");

        // Optional: Verify results
        float* cResultPtr = (float*)cBuffer.contents;
        NSLog(@"Result buffer contents: [%f, %f, %f, %f]", cResultPtr[0], cResultPtr[1], cResultPtr[2], cResultPtr[3]);
        // Expected: [1.5, 2.5, 3.5, 4.5]

    }
    NSLog(@"Test finished.");
    return 0;
} 