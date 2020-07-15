| OpKernel | Unsupported context methods |
| :---------- | :---------- |
| _HostConstantOp |  |
| AbortOp |  |
| AccumulatorApplyGradientOp | tensorflow::OpKernelContext::MatchSignature,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::resource_manager |
| AccumulatorNumAccumulatedOp | tensorflow::OpKernelContext::MatchSignature,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::resource_manager |
| AccumulatorSetGlobalStepOp | tensorflow::OpKernelContext::MatchSignature,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::resource_manager |
| AccumulatorTakeGradientOp | tensorflow::OpKernelContext::MatchSignature,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::resource_manager |
| AddManySparseToTensorsMapOp | tensorflow::OpKernelContext::resource_manager |
| AddNOp | tensorflow::OpKernelContext::forward_input_to_output_with_shape |
| AddSparseToTensorsMapOp | tensorflow::OpKernelContext::resource_manager |
| AdjustContrastOp | tensorflow::OpKernelContext::allocate_temp |
| AdjustContrastOpv2 | tensorflow::OpKernelContext::allocate_temp |
| AdjustContrastOpV2Base |  |
| AdjustHueOp | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::eigen_gpu_device,tensorflow::OpKernelContext::forward_input_or_allocate_output |
| AdjustHueOpBase | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| AdjustSaturationOp | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::eigen_gpu_device |
| AdjustSaturationOpBase |  |
| AllCandidateSamplerOp |  |
| AnonymousIteratorHandleOp | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::function_library,tensorflow::OpKernelContext::resource_manager |
| AnonymousMemoryCacheHandleOp | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::function_library,tensorflow::OpKernelContext::resource_manager |
| AnonymousMultiDeviceIteratorOp | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::env,tensorflow::OpKernelContext::function_library,tensorflow::OpKernelContext::resource_manager |
| AnonymousResourceOp | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::function_library,tensorflow::OpKernelContext::resource_manager |
| AnonymousSeedGeneratorHandleOp | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::function_library,tensorflow::OpKernelContext::resource_manager |
| ApplyAdadeltaOp | tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input |
| ApplyAdagradDAOp | tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::mutable_input |
| ApplyAdagradOp | tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::mutable_input |
| ApplyAdagradV2Op | tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::mutable_input |
| ApplyAdaMaxOp | tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::mutable_input |
| ApplyAdamOp | tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::mutable_input |
| ApplyAdamWithAmsgradOp | tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::mutable_input |
| ApplyAddSignOp | tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::mutable_input |
| ApplyCenteredRMSPropOp | tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::mutable_input |
| ApplyFtrlOp | tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::mutable_input |
| ApplyGradientDescentOp | tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::mutable_input |
| ApplyKerasMomentumOp | tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::mutable_input |
| ApplyMomentumOp | tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::mutable_input |
| ApplyPowerSignOp | tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::mutable_input |
| ApplyProximalAdagradOp | tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::mutable_input |
| ApplyProximalGradientDescentOp | tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::mutable_input |
| ApplyRMSPropOp | tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::mutable_input |
| ApproximateEqualOp |  |
| AreAllKernelsInlineOp | tensorflow::OpKernelContext::run_all_kernels_inline |
| ArgMaxOp |  |
| ArgMinOp |  |
| ArgOp |  |
| AssertCardinalityDatasetOp |  |
| AssertNextDatasetOp |  |
| AssertOp |  |
| AssignOp | tensorflow::OpKernelConstruction::input_type,tensorflow::OpKernelContext::allocate_persistent,tensorflow::OpKernelContext::clear_recorded_memory,tensorflow::OpKernelContext::forward_input,tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::replace_ref_input |
| AssignOpT | tensorflow::OpKernelConstruction::input_type,tensorflow::OpKernelContext::allocate_persistent,tensorflow::OpKernelContext::clear_recorded_memory,tensorflow::OpKernelContext::forward_input,tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::replace_ref_input |
| AssignUpdateVariableOp | tensorflow::OpKernelContext::allocate_persistent,tensorflow::OpKernelContext::resource_manager |
| AssignVariableOp | tensorflow::OpKernelContext::allocate_persistent,tensorflow::OpKernelContext::forward_input,tensorflow::OpKernelContext::resource_manager |
| AsStringOp |  |
| AsyncOpKernel |  |
| AudioSpectrogramOp |  |
| AutoShardDatasetOp |  |
| AvgPooling3dGradOp | tensorflow::OpKernelConstruction::device_type |
| AvgPoolingGradOp | tensorflow::OpKernelConstruction::device_type,tensorflow::OpKernelContext::device |
| AvgPoolingGradOpCustomGPUKernel |  |
| AvgPoolingOp | tensorflow::OpKernelConstruction::device_type,tensorflow::OpKernelContext::device |
| BarrierCloseOp | tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::resource_manager |
| BarrierIncompleteSizeOp | tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::resource_manager |
| BarrierOp | tensorflow::OpKernelConstruction::allocate_persistent,tensorflow::OpKernelConstruction::output_type,tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::resource_manager,tensorflow::OpKernelContext::set_output_ref |
| BarrierOpKernel | tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::resource_manager |
| BarrierReadySizeOp | tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::resource_manager |
| BaseBatchMatMulOp |  |
| BaseCandidateSamplerOp |  |
| BaseDebugOp |  |
| BaseKernel |  |
| BaseMatrixTriangularSolveOp |  |
| BatchDatasetOp |  |
| BatchFunctionKernel | tensorflow::OpKernelConstruction::function_library,tensorflow::OpKernelContext::resource_manager |
| BatchKernel | tensorflow::OpKernelContext::resource_manager |
| BatchMatMulOp |  |
| BatchMatMulV2Op |  |
| BatchNormGradOp | tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::forward_input_or_allocate_output |
| BatchNormOp |  |
| BatchToSpaceNDOp |  |
| BatchToSpaceOp |  |
| BCastArgsOp |  |
| BCastGradArgsOp |  |
| BeginEpochOp |  |
| BetaincOp | tensorflow::OpKernelContext::SetStatus |
| BiasGradOp | tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::op_device_context |
| BiasOp | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| BinaryDatasetOpKernel |  |
| BinaryElementWiseOp | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::forward_input_or_allocate_output |
| BinaryOp | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| BinaryOpBase |  |
| BinaryOpShared |  |
| BincountOp |  |
| BlockingOp |  |
| BlockLSTMGradOp | tensorflow::OpKernelContext::allocate_temp |
| BlockLSTMOp | tensorflow::OpKernelConstruction::HasAttr,tensorflow::OpKernelContext::allocate_temp |
| BoostedTreesAggregateStatsOp | tensorflow::OpKernelContext::allocate_temp |
| BoostedTreesBucketizeOp | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::input_list,tensorflow::OpKernelContext::output_list |
| BoostedTreesCalculateBestFeatureSplitOp |  |
| BoostedTreesCalculateBestFeatureSplitV2 | tensorflow::OpKernelContext::input_list |
| BoostedTreesCalculateBestGainsPerFeatureOp | tensorflow::OpKernelContext::input_list,tensorflow::OpKernelContext::output_list |
| BoostedTreesCenterBiasOp | tensorflow::OpKernelContext::resource_manager |
| BoostedTreesCreateEnsembleOp | tensorflow::OpKernelContext::resource_manager |
| BoostedTreesCreateQuantileStreamResourceOp | tensorflow::OpKernelContext::resource_manager |
| BoostedTreesDeserializeEnsembleOp | tensorflow::OpKernelContext::resource_manager |
| BoostedTreesExampleDebugOutputsOp | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::input_list,tensorflow::OpKernelContext::resource_manager |
| BoostedTreesFlushQuantileSummariesOp | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::output_list,tensorflow::OpKernelContext::resource_manager |
| BoostedTreesGetEnsembleStatesOp | tensorflow::OpKernelContext::resource_manager |
| BoostedTreesMakeQuantileSummariesOp | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::input_list,tensorflow::OpKernelContext::output_list |
| BoostedTreesMakeStatsSummaryOp | tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::input_list |
| BoostedTreesPredictOp | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::input_list,tensorflow::OpKernelContext::resource_manager |
| BoostedTreesQuantileStreamResourceAddSummariesOp | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::input_list,tensorflow::OpKernelContext::resource_manager |
| BoostedTreesQuantileStreamResourceDeserializeOp | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::input_list,tensorflow::OpKernelContext::resource_manager |
| BoostedTreesQuantileStreamResourceFlushOp | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::resource_manager |
| BoostedTreesQuantileStreamResourceGetBucketBoundariesOp | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::output_list,tensorflow::OpKernelContext::resource_manager |
| BoostedTreesSerializeEnsembleOp | tensorflow::OpKernelContext::resource_manager |
| BoostedTreesSparseAggregateStatsOp |  |
| BoostedTreesSparseCalculateBestFeatureSplitOp |  |
| BoostedTreesTrainingPredictOp | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::input_list,tensorflow::OpKernelContext::resource_manager |
| BoostedTreesUpdateEnsembleOp | tensorflow::OpKernelContext::input_list,tensorflow::OpKernelContext::resource_manager |
| BoostedTreesUpdateEnsembleV2Op | tensorflow::OpKernelContext::input_list,tensorflow::OpKernelContext::resource_manager |
| BroadcastToOp |  |
| BrokenOp | tensorflow::OpKernelConstruction::SetStatus,tensorflow::OpKernelContext::SetStatus |
| BucketizeOp |  |
| BytesProducedStatsDatasetOp |  |
| CacheDatasetOp |  |
| CallOp | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::cancellation_manager,tensorflow::OpKernelContext::collective_executor,tensorflow::OpKernelContext::function_library,tensorflow::OpKernelContext::rendezvous,tensorflow::OpKernelContext::run_all_kernels_inline,tensorflow::OpKernelContext::runner,tensorflow::OpKernelContext::stats_collector,tensorflow::OpKernelContext::step_container,tensorflow::OpKernelContext::step_id |
| CancellationMgrPollingOp | tensorflow::OpKernelContext::cancellation_manager,tensorflow::OpKernelContext::env |
| CaseOp | tensorflow::OpKernelConstruction::function_library,tensorflow::OpKernelContext::cancellation_manager,tensorflow::OpKernelContext::function_library,tensorflow::OpKernelContext::rendezvous,tensorflow::OpKernelContext::run_all_kernels_inline,tensorflow::OpKernelContext::runner,tensorflow::OpKernelContext::stats_collector,tensorflow::OpKernelContext::step_container |
| CastOpBase |  |
| CheckNumericsOp | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::op_device_context |
| CheckNumericsV2Op | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::op_device_context |
| CholeskyGrad |  |
| CholeskyOp |  |
| CholeskyOpGpu | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| ChooseFastestBranchDatasetOp | tensorflow::OpKernelContext::input_list |
| ChooseFastestDatasetOp | tensorflow::OpKernelContext::input_list |
| ClipOp | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| CloseSummaryWriterOp | tensorflow::OpKernelContext::resource_manager |
| CollectiveBcastRecvOpKernel | tensorflow::OpKernelConstruction::device_type,tensorflow::OpKernelContext::collective_executor,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::frame_iter,tensorflow::OpKernelContext::mutable_output,tensorflow::OpKernelContext::op_kernel |
| CollectiveBcastSendOpKernel | tensorflow::OpKernelConstruction::device_type,tensorflow::OpKernelContext::collective_executor,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::forward_input_or_allocate_output,tensorflow::OpKernelContext::frame_iter,tensorflow::OpKernelContext::mutable_output,tensorflow::OpKernelContext::op_kernel |
| CollectiveGatherOpKernel | tensorflow::OpKernelConstruction::def,tensorflow::OpKernelConstruction::device_type,tensorflow::OpKernelContext::collective_executor,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::frame_iter,tensorflow::OpKernelContext::mutable_output,tensorflow::OpKernelContext::op_kernel |
| CollectiveOpKernel | tensorflow::OpKernelContext::frame_iter |
| CollectiveReduceOpKernel | tensorflow::OpKernelConstruction::def,tensorflow::OpKernelConstruction::device,tensorflow::OpKernelConstruction::device_type,tensorflow::OpKernelConstruction::graph_def_version,tensorflow::OpKernelContext::collective_executor,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::forward_input_or_allocate_output,tensorflow::OpKernelContext::frame_iter,tensorflow::OpKernelContext::mutable_output,tensorflow::OpKernelContext::op_kernel |
| CombinedNonMaxSuppressionOp |  |
| CompareAndBitpackOp |  |
| ComputeAccidentalHitsOp |  |
| ConcatBaseOp | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::eigen_gpu_device |
| ConcatenateDatasetOp |  |
| ConcatOffsetOp |  |
| ConditionalAccumulatorBaseApplyGradientOp | tensorflow::OpKernelContext::MatchSignature,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::resource_manager |
| ConditionalAccumulatorBaseAsyncOpKernel | tensorflow::OpKernelContext::MatchSignature,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::resource_manager |
| ConditionalAccumulatorBaseOp | tensorflow::OpKernelConstruction::allocate_persistent,tensorflow::OpKernelContext::resource_manager |
| ConditionalAccumulatorBaseSyncOpKernel | tensorflow::OpKernelContext::MatchSignature,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::resource_manager |
| ConditionalAccumulatorBaseTakeGradientOp | tensorflow::OpKernelContext::MatchSignature,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::resource_manager |
| ConditionalAccumulatorOp | tensorflow::OpKernelConstruction::allocate_persistent,tensorflow::OpKernelContext::MatchSignature,tensorflow::OpKernelContext::resource_manager |
| ConjugateTransposeCpuOp |  |
| ConjugateTransposeGpuOp |  |
| ConstantOp |  |
| ConsumeMutexLockOp |  |
| ControlTriggerOp |  |
| Conv2DBackpropFilterOp |  |
| Conv2DBackpropInputOp |  |
| Conv2DCustomBackpropFilterOp | tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::eigen_cpu_device |
| Conv2DCustomBackpropInputOp | tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::eigen_cpu_device |
| Conv2DOp | tensorflow::OpKernelConstruction::HasAttr |
| Conv3DBackpropFilterOp | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::op_device_context |
| Conv3DBackpropInputOp | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::op_device_context |
| Conv3DCustomBackpropFilterOp | tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::eigen_cpu_device |
| Conv3DCustomBackpropInputOp | tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::eigen_cpu_device |
| Conv3DOp |  |
| CopyFromGpuToHostKernel | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::input_alloc_attr,tensorflow::OpKernelContext::op_device_context |
| CopyFromHostToGpuKernel | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::input_alloc_attr,tensorflow::OpKernelContext::op_device_context |
| CopyOp | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::input_alloc_attr,tensorflow::OpKernelContext::op_device_context |
| CopyOpBase |  |
| CountUpToOp | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input |
| CpuCastOp |  |
| CreateSummaryDbWriterOp | tensorflow::OpKernelContext::env,tensorflow::OpKernelContext::resource_manager |
| CreateSummaryFileWriterOp | tensorflow::OpKernelContext::env,tensorflow::OpKernelContext::resource_manager |
| CreateTestVariantOp |  |
| CropAndResizeGradBoxesOp | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::op_device_context |
| CropAndResizeGradImageOp | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::op_device_context |
| CropAndResizeOp | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::op_device_context |
| CrossOp |  |
| CSRAddOp |  |
| CSRMatMulCPUOp | tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::device |
| CSRMatMulGPUOp | tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::forward_input_or_allocate_temp |
| CSRMatMulOp |  |
| CSRMulOp |  |
| CSRNNZOp |  |
| CSROrderingAMDCPUOp | tensorflow::OpKernelContext::device |
| CSRSoftmaxGradOp | tensorflow::OpKernelContext::allocate_persistent |
| CSRSoftmaxOp | tensorflow::OpKernelContext::allocate_temp |
| CSRSparseCholeskyCPUOp | tensorflow::OpKernelContext::device |
| CSRSparseMatMulCPUOp | tensorflow::OpKernelContext::device |
| CSRSparseMatMulGPUOp | tensorflow::OpKernelContext::allocate_temp |
| CSRSparseMatrixComponentsOp |  |
| CSRSparseMatrixToDenseCPUOp | tensorflow::OpKernelContext::device |
| CSRSparseMatrixToDenseGPUOp | tensorflow::OpKernelContext::allocate_temp |
| CSRSparseMatrixToSparseTensorCPUOp | tensorflow::OpKernelContext::device |
| CSRSparseMatrixToSparseTensorGPUOp | tensorflow::OpKernelContext::allocate_temp |
| CSRTransposeOp |  |
| CSRZerosOp |  |
| CSVDatasetOp | tensorflow::OpKernelContext::input_list |
| CTCBeamSearchDecoderOp |  |
| CTCGreedyDecoderOp | tensorflow::OpKernelContext::device |
| CTCLossOp | tensorflow::OpKernelContext::device |
| CTCLossOpGPU | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::op_device_context |
| CudnnRNNBackwardOp | tensorflow::OpKernelContext::op_device_context |
| CudnnRNNBackwardOpV2 | tensorflow::OpKernelContext::op_device_context |
| CudnnRNNBackwardOpV3 | tensorflow::OpKernelConstruction::HasAttr,tensorflow::OpKernelContext::op_device_context |
| CudnnRNNCanonicalToParams | tensorflow::OpKernelConstruction::HasAttr,tensorflow::OpKernelContext::input_list,tensorflow::OpKernelContext::op_device_context |
| CudnnRNNForwardOp | tensorflow::OpKernelContext::op_device_context |
| CudnnRNNForwardOpV2 | tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::op_device_context |
| CudnnRNNForwardOpV3 | tensorflow::OpKernelConstruction::HasAttr,tensorflow::OpKernelContext::op_device_context |
| CudnnRNNKernelCommon | tensorflow::OpKernelContext::op_device_context |
| CudnnRNNParamsSizeOp | tensorflow::OpKernelConstruction::HasAttr,tensorflow::OpKernelContext::op_device_context |
| CudnnRNNParamsToCanonical | tensorflow::OpKernelConstruction::HasAttr,tensorflow::OpKernelContext::op_device_context |
| DarthOp |  |
| DataFormatDimMapOp |  |
| DataFormatVecPermuteOp |  |
| DataServiceDatasetOp |  |
| DatasetCardinalityOp |  |
| DatasetFromGraphOp |  |
| DatasetOpKernel |  |
| DatasetToGraphOp |  |
| DebugIdentityOp |  |
| DebugIdentityV2Op | tensorflow::OpKernelConstruction::device |
| DebugNanCountOp |  |
| DebugNumericSummaryOp |  |
| DebugNumericSummaryV2Op | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::op_device_context |
| DecodeBase64Op |  |
| DecodeBmpOp |  |
| DecodeCompressedOp |  |
| DecodeCSVOp | tensorflow::OpKernelContext::input_list,tensorflow::OpKernelContext::output_list |
| DecodeImageOp | tensorflow::OpKernelContext::SetStatus |
| DecodePaddedRawOp |  |
| DecodeProtoOp | tensorflow::OpKernelConstruction::env |
| DecodeRawOp |  |
| DecodeWavOp |  |
| DelayOp | tensorflow::OpKernelContext::env |
| DeleteIteratorOp |  |
| DeleteMemoryCacheOp |  |
| DeleteMultiDeviceIteratorOp | tensorflow::OpKernelContext::resource_manager |
| DeleteSeedGeneratorOp |  |
| DeleteSessionTensorOp | tensorflow::OpKernelContext::session_state |
| DenseToCSRSparseMatrixCPUOp | tensorflow::OpKernelContext::allocate_temp |
| DenseToCSRSparseMatrixGPUOp | tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::op_device_context |
| DenseToDenseSetOperationOp |  |
| DenseToSparseBatchDatasetOp |  |
| DenseToSparseSetOperationOp |  |
| DenseUpdateOp | tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_ref_mutex |
| DepthToSpaceOp |  |
| DepthwiseConv2dGroupedConvBackpropFilterOp | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| DepthwiseConv2dGroupedConvBackpropInputOp | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| DepthwiseConv2dGroupedConvOp |  |
| DepthwiseConv2dNativeBackpropFilterOp | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| DepthwiseConv2dNativeBackpropInputOp | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| DepthwiseConv2dNativeOp |  |
| DequantizeOp | tensorflow::OpKernelConstruction::output_type |
| DequeueManyOp |  |
| DequeueOp |  |
| DequeueUpToOp |  |
| DeserializeIteratorOp |  |
| DeserializeSparseOp |  |
| DestroyResourceOp |  |
| DestroyTemporaryVariableOp | tensorflow::OpKernelConstruction::input_type,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::record_persistent_memory_allocation,tensorflow::OpKernelContext::resource_manager,tensorflow::OpKernelContext::step_container,tensorflow::OpKernelContext::track_allocations |
| DeterminantOp |  |
| DeterminantOpGpu |  |
| DiagOp |  |
| DiagPartOp |  |
| DilationBackpropFilterOp |  |
| DilationBackpropInputOp |  |
| DilationOp |  |
| DirectedInterleaveDatasetOp |  |
| DrawBoundingBoxesOp |  |
| DummyKernel |  |
| DummyOp |  |
| DynamicPartitionOp | tensorflow::OpKernelContext::output_list |
| DynamicPartitionOp_Shared | tensorflow::OpKernelContext::output_list |
| DynamicStitchOpCPU | tensorflow::OpKernelContext::device |
| DynamicStitchOpGPU | tensorflow::OpKernelContext::eigen_gpu_device,tensorflow::OpKernelContext::input_list |
| DynamicStitchOpImplBase | tensorflow::OpKernelContext::input_list |
| DynamicStitchOpImplCPU | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::input_list |
| EditDistanceOp |  |
| EigOp |  |
| EinsumOp | tensorflow::OpKernelContext::input_list |
| EluGradOp | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::forward_input_or_allocate_output |
| EluOp | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| EmptyOp |  |
| EmptyTensorList |  |
| EncodeBase64Op |  |
| EncodeJpegOp |  |
| EncodeJpegVariableQualityOp |  |
| EncodePngOp | tensorflow::OpKernelConstruction::input_type |
| EncodeProtoOp | tensorflow::OpKernelConstruction::env,tensorflow::OpKernelContext::input_list |
| EncodeWavOp |  |
| EnqueueManyOp |  |
| EnqueueOp |  |
| EnsureShapeOp | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_dtype |
| EnterOp |  |
| ErrorOp | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::cancellation_manager |
| ExitOp |  |
| ExpandDimsOp | tensorflow::OpKernelContext::SetStatus |
| ExpensiveNoopOp |  |
| ExtractGlimpseOp | tensorflow::OpKernelContext::eigen_cpu_device |
| ExtractImagePatchesOp |  |
| ExtractJpegShapeOp |  |
| ExtractVolumePatchesOp |  |
| FactOp |  |
| FactOpKernel | tensorflow::OpKernelContext::env |
| FactOpKernel1 | tensorflow::OpKernelContext::env |
| FactOpKernel2 | tensorflow::OpKernelContext::env |
| FailureKernel |  |
| FakeParamOp | tensorflow::OpKernelConstruction::allocate_persistent |
| FakeQuantWithMinMaxArgsGradientOp | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::forward_input_or_allocate_output |
| FakeQuantWithMinMaxArgsOp | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| FakeQuantWithMinMaxVarsGradientOp |  |
| FakeQuantWithMinMaxVarsOp |  |
| FakeQuantWithMinMaxVarsPerChannelGradientOp |  |
| FakeQuantWithMinMaxVarsPerChannelOp |  |
| FakeQueueOp | tensorflow::OpKernelConstruction::allocate_persistent,tensorflow::OpKernelContext::set_output_ref |
| FFTBase |  |
| FFTCPU | tensorflow::OpKernelContext::allocate_temp |
| FFTGPU | tensorflow::OpKernelContext::op_device_context |
| FFTGPUBase | tensorflow::OpKernelContext::op_device_context |
| FIFOQueueOp | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::resource_manager,tensorflow::OpKernelContext::set_output_ref |
| FillOp |  |
| FilterDatasetOp |  |
| FindDeviceOpKernel | tensorflow::OpKernelContext::function_library |
| FingerprintOp | tensorflow::OpKernelContext::allocate_temp |
| FixedLengthRecordDatasetOp |  |
| FixedLengthRecordReaderOp | tensorflow::OpKernelConstruction::env,tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::cancellation_manager |
| FixedUnigramCandidateSamplerOp | tensorflow::OpKernelConstruction::env |
| FlatMapDatasetOp |  |
| FlushSummaryWriterOp | tensorflow::OpKernelContext::resource_manager |
| ForOp | tensorflow::OpKernelConstruction::function_library,tensorflow::OpKernelContext::cancellation_manager,tensorflow::OpKernelContext::function_library,tensorflow::OpKernelContext::rendezvous,tensorflow::OpKernelContext::run_all_kernels_inline,tensorflow::OpKernelContext::runner,tensorflow::OpKernelContext::stats_collector,tensorflow::OpKernelContext::step_container |
| FractionalAvgPoolGradOp | tensorflow::OpKernelContext::forward_input_or_allocate_output,tensorflow::OpKernelContext::forward_input_or_allocate_temp |
| FractionalAvgPoolOp |  |
| FractionalMaxPoolGradOp | tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::forward_input_or_allocate_output,tensorflow::OpKernelContext::forward_input_or_allocate_temp |
| FractionalMaxPoolOp |  |
| FusedBatchNormGradOp |  |
| FusedBatchNormGradOpBase |  |
| FusedBatchNormGradOpV3 |  |
| FusedBatchNormOp | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| FusedBatchNormOpBase | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| FusedBatchNormOpEx | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| FusedBatchNormOpV3 | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| FusedConv2DOp | tensorflow::OpKernelConstruction::HasAttr |
| FusedMatMulOp |  |
| FusedResizeConv2DUsingGemmOp |  |
| GatherNdOp | tensorflow::OpKernelContext::allocate_temp |
| GatherOp | tensorflow::OpKernelConstruction::HasAttr |
| GenerateVocabRemappingOp | tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::env |
| GeneratorDatasetOp |  |
| GetAttrKernel |  |
| GetSessionHandleOp | tensorflow::OpKernelConstruction::device,tensorflow::OpKernelConstruction::resource_manager,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::expected_output_dtype,tensorflow::OpKernelContext::resource_manager,tensorflow::OpKernelContext::session_state,tensorflow::OpKernelContext::tensor_store |
| GetSessionTensorOp | tensorflow::OpKernelContext::session_state |
| GpuCastOp |  |
| GroupByReducerDatasetOp |  |
| GroupByWindowDatasetOp |  |
| GRUBlockCellGradOp | tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::forward_input_or_allocate_output |
| GRUCellBlockOp | tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::forward_input_or_allocate_output |
| GuaranteeConstOp | tensorflow::OpKernelContext::forward_input_to_output_with_shape,tensorflow::OpKernelContext::input_dtype |
| HistogramFixedWidthOp |  |
| HSVToRGBOp |  |
| HybridAsyncOpKernel |  |
| IdentityNOp | tensorflow::OpKernelContext::input_list,tensorflow::OpKernelContext::output_list |
| IdentityOp | tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_dtype |
| IdentityReaderOp | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::cancellation_manager |
| IfOp | tensorflow::OpKernelConstruction::function_library,tensorflow::OpKernelContext::cancellation_manager,tensorflow::OpKernelContext::function_library,tensorflow::OpKernelContext::rendezvous,tensorflow::OpKernelContext::run_all_kernels_inline,tensorflow::OpKernelContext::runner,tensorflow::OpKernelContext::stats_collector,tensorflow::OpKernelContext::step_container |
| IgnoreErrorsDatasetOp |  |
| ImageProjectiveTransformV2 |  |
| ImmutableConstantOp |  |
| ImportEventOp | tensorflow::OpKernelContext::resource_manager |
| InitializeTableFromDatasetOp | tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::resource_manager |
| InitializeTableFromTextFileOp | tensorflow::OpKernelContext::MatchSignature,tensorflow::OpKernelContext::env,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::record_persistent_memory_allocation,tensorflow::OpKernelContext::resource_manager,tensorflow::OpKernelContext::track_allocations |
| InitializeTableOp | tensorflow::OpKernelContext::MatchSignature,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::record_persistent_memory_allocation,tensorflow::OpKernelContext::resource_manager,tensorflow::OpKernelContext::track_allocations |
| InplaceOp |  |
| InplaceOpBase |  |
| InsertManyOp | tensorflow::OpKernelContext::MatchSignature,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::resource_manager |
| InterleaveDatasetOp |  |
| InTopK |  |
| InvalidRefType | tensorflow::OpKernelContext::set_output_ref |
| InvertPermutationOp |  |
| IsResourceInitialized | tensorflow::OpKernelContext::resource_manager |
| IsVariableInitializedOp | tensorflow::OpKernelContext::mutable_input |
| IteratorFromStringHandleOp |  |
| IteratorGetDeviceOp | tensorflow::OpKernelContext::device |
| IteratorGetNextAsOptionalOp |  |
| IteratorGetNextOp |  |
| IteratorHandleOp |  |
| IteratorToStringHandleOp |  |
| KMC2ChainInitializationOp |  |
| KmeansPlusPlusInitializationOp |  |
| L2LossOp |  |
| LabeledKernel |  |
| LatencyStatsDatasetOp |  |
| LeakyReluGradOp | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::forward_input_or_allocate_output |
| LeakyReluOp | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| LegacyStringToHashBucketOp |  |
| LinearAlgebraOp | tensorflow::OpKernelContext::device |
| LinSpaceOp |  |
| ListDiffOp |  |
| LMDBDatasetOp |  |
| LMDBReaderOp | tensorflow::OpKernelConstruction::env,tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::cancellation_manager |
| LoadAndRemapMatrixOp | tensorflow::OpKernelContext::env |
| LogDeterminantOp |  |
| LogDeterminantOpGpu |  |
| LookupTableExportOp | tensorflow::OpKernelConstruction::input_type,tensorflow::OpKernelContext::resource_manager |
| LookupTableFindOp | tensorflow::OpKernelConstruction::input_type,tensorflow::OpKernelContext::MatchSignature,tensorflow::OpKernelContext::resource_manager |
| LookupTableImportOp | tensorflow::OpKernelConstruction::input_type,tensorflow::OpKernelContext::MatchSignature,tensorflow::OpKernelContext::record_persistent_memory_allocation,tensorflow::OpKernelContext::resource_manager,tensorflow::OpKernelContext::track_allocations |
| LookupTableInsertOp | tensorflow::OpKernelConstruction::input_type,tensorflow::OpKernelContext::MatchSignature,tensorflow::OpKernelContext::record_persistent_memory_allocation,tensorflow::OpKernelContext::resource_manager,tensorflow::OpKernelContext::track_allocations |
| LookupTableOp | tensorflow::OpKernelConstruction::allocate_persistent,tensorflow::OpKernelConstruction::device,tensorflow::OpKernelConstruction::output_type,tensorflow::OpKernelConstruction::resource_manager,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::expected_output_dtype,tensorflow::OpKernelContext::record_persistent_memory_allocation,tensorflow::OpKernelContext::resource_manager,tensorflow::OpKernelContext::set_output_ref,tensorflow::OpKernelContext::track_allocations |
| LookupTableOpKernel | tensorflow::OpKernelConstruction::input_type,tensorflow::OpKernelContext::resource_manager |
| LookupTableRemoveOp | tensorflow::OpKernelConstruction::input_type,tensorflow::OpKernelContext::MatchSignature,tensorflow::OpKernelContext::record_persistent_memory_allocation,tensorflow::OpKernelContext::resource_manager,tensorflow::OpKernelContext::track_allocations |
| LookupTableSizeOp | tensorflow::OpKernelConstruction::input_type,tensorflow::OpKernelContext::resource_manager |
| LoopCondOp |  |
| LowerBoundOp |  |
| LRNGradOp |  |
| LRNOp |  |
| LSTMBlockCellGradOp | tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::forward_input_or_allocate_output |
| LSTMBlockCellOp | tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::forward_input_or_allocate_output |
| LuOp | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::forward_input_or_allocate_output |
| MakeDataServiceIteratorOp |  |
| MakeIteratorOp |  |
| MapAndBatchDatasetOp |  |
| MapClearOp | tensorflow::OpKernelContext::resource_manager |
| MapDatasetOp |  |
| MapDefunOp |  |
| MapIncompleteSizeOp | tensorflow::OpKernelContext::resource_manager |
| MapPeekOp | tensorflow::OpKernelContext::resource_manager |
| MapSizeOp | tensorflow::OpKernelContext::resource_manager |
| MapStageOp | tensorflow::OpKernelContext::input_list,tensorflow::OpKernelContext::resource_manager |
| MapUnstageNoKeyOp | tensorflow::OpKernelContext::resource_manager |
| MapUnstageOp | tensorflow::OpKernelContext::resource_manager |
| MatchingFilesDatasetOp |  |
| MatchingFilesOp | tensorflow::OpKernelContext::env |
| MatMulOp | tensorflow::OpKernelContext::allocate_temp |
| MatrixBandPartOp | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| MatrixDiagOp | tensorflow::OpKernelConstruction::HasAttr |
| MatrixDiagPartOp | tensorflow::OpKernelConstruction::HasAttr |
| MatrixExponentialOp |  |
| MatrixInverseOp |  |
| MatrixInverseOpGpu | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| MatrixLogarithmOp |  |
| MatrixSetDiagOp | tensorflow::OpKernelConstruction::HasAttr,tensorflow::OpKernelContext::forward_input_or_allocate_output |
| MatrixSolveLsOp |  |
| MatrixSolveOp |  |
| MatrixSolveOpGpu | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| MatrixSquareRootOp |  |
| MatrixTriangularSolveOp |  |
| MaxIntraOpParallelismDatasetOp |  |
| MaxPooling3dGradGradOp | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| MaxPooling3dGradOp | tensorflow::OpKernelConstruction::device_type |
| MaxPoolingGradGradOp | tensorflow::OpKernelConstruction::device_type,tensorflow::OpKernelContext::forward_input_or_allocate_output |
| MaxPoolingGradGradWithArgmaxOp | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| MaxPoolingGradOp | tensorflow::OpKernelConstruction::device_type,tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::forward_input_or_allocate_output,tensorflow::OpKernelContext::forward_input_or_allocate_temp |
| MaxPoolingGradWithArgmaxOp | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| MaxPoolingNoMaskOp | tensorflow::OpKernelConstruction::device_type |
| MaxPoolingNoMaskV2Op | tensorflow::OpKernelConstruction::device_type |
| MaxPoolingOp | tensorflow::OpKernelConstruction::device_type,tensorflow::OpKernelContext::device |
| MaxPoolingV2Op | tensorflow::OpKernelContext::device |
| MaxPoolingWithArgmaxOp |  |
| MergeOp |  |
| MergeV2Checkpoints |  |
| MfccOp |  |
| MirrorPadGradOp | tensorflow::OpKernelContext::allocate_temp |
| MirrorPadOp |  |
| ModelDatasetOp | tensorflow::OpKernelConstruction::HasAttr |
| MultiDeviceIteratorFromStringHandleOp | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::resource_manager |
| MultiDeviceIteratorGetNextFromShardOp | tensorflow::OpKernelConstruction::env,tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::resource_manager |
| MultiDeviceIteratorHandleOp | tensorflow::OpKernelConstruction::graph_def_version,tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::env,tensorflow::OpKernelContext::function_library,tensorflow::OpKernelContext::resource_manager |
| MultiDeviceIteratorInitOp | tensorflow::OpKernelContext::cancellation_manager,tensorflow::OpKernelContext::resource_manager |
| MultiDeviceIteratorToStringHandleOp | tensorflow::OpKernelContext::resource_manager |
| MultiIdentity |  |
| MultinomialOp | tensorflow::OpKernelContext::allocate_temp |
| MutexLockOp | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::resource_manager |
| NcclAllReduceOpKernel | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::forward_input_or_allocate_output,tensorflow::OpKernelContext::frame_iter,tensorflow::OpKernelContext::op_device_context,tensorflow::OpKernelContext::step_id |
| NcclAsyncOpBase | tensorflow::OpKernelContext::frame_iter,tensorflow::OpKernelContext::step_id |
| NcclBroadcastRecvKernel | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::frame_iter,tensorflow::OpKernelContext::op_device_context,tensorflow::OpKernelContext::step_id |
| NcclBroadcastSendKernel | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::frame_iter,tensorflow::OpKernelContext::op_device_context,tensorflow::OpKernelContext::step_id |
| NcclReduceOpBase | tensorflow::OpKernelContext::frame_iter,tensorflow::OpKernelContext::step_id |
| NcclReduceRecvKernel | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::frame_iter,tensorflow::OpKernelContext::op_device_context,tensorflow::OpKernelContext::step_id |
| NcclReduceSendKernel | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::frame_iter,tensorflow::OpKernelContext::op_device_context,tensorflow::OpKernelContext::step_id |
| NcclStubKernel | tensorflow::OpKernelContext::SetStatus |
| NearestNeighborsOp | tensorflow::OpKernelContext::device |
| NegTrainOp | tensorflow::OpKernelContext::mutable_input |
| NeonDepthwiseConv2dNativeOp |  |
| NextIterationOp |  |
| NonDeterministicIntsOp |  |
| NonMaxSuppressionOp |  |
| NonMaxSuppressionV2Op |  |
| NonMaxSuppressionV3Op |  |
| NonMaxSuppressionV4Op |  |
| NonMaxSuppressionV5Op |  |
| NonMaxSuppressionWithOverlapsOp |  |
| NonSerializableDatasetOp |  |
| NoOp |  |
| NthElementOp |  |
| OneHotOp |  |
| OneShotIteratorOp | tensorflow::OpKernelConstruction::device,tensorflow::OpKernelConstruction::env,tensorflow::OpKernelConstruction::resource_manager,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::op_kernel,tensorflow::OpKernelContext::resource_manager |
| OnesLikeOp | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| OptimizeDatasetOp |  |
| OptionalFromValueOp |  |
| OptionalGetValueOp |  |
| OptionalHasValueOp |  |
| OptionalNoneOp |  |
| PackOp | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::eigen_gpu_device |
| PaddedBatchDatasetOp |  |
| PaddingFIFOQueueOp | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::resource_manager,tensorflow::OpKernelContext::set_output_ref |
| PadOp | tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::input_memory_type |
| ParallelConcatStart |  |
| ParallelConcatUpdate |  |
| ParallelDynamicStitchOpCPU | tensorflow::OpKernelContext::device |
| ParallelInterleaveDatasetOp |  |
| ParallelMapDatasetOp |  |
| ParameterizedTruncatedNormalOp |  |
| ParseExampleDatasetOp | tensorflow::OpKernelConstruction::graph_def_version,tensorflow::OpKernelContext::input_list |
| ParseExampleOp | tensorflow::OpKernelConstruction::def,tensorflow::OpKernelContext::input_list |
| ParseSequenceExampleOp | tensorflow::OpKernelConstruction::def,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::input_list |
| ParseSingleExampleOp | tensorflow::OpKernelContext::input_list,tensorflow::OpKernelContext::output_list |
| ParseSingleSequenceExampleOp | tensorflow::OpKernelContext::input_list,tensorflow::OpKernelContext::output_list |
| ParseTensorOp | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::output_alloc_attr |
| PartitionedCallOp |  |
| PassOn |  |
| PhiloxRandomOp |  |
| PlaceholderOp |  |
| Pooling3DOp | tensorflow::OpKernelConstruction::device_type |
| PopulationCountOp |  |
| PrefetchDatasetOp |  |
| PrintOp | tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_dtype |
| PrintV2Op | tensorflow::OpKernelContext::env |
| PriorityQueueOp | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::resource_manager,tensorflow::OpKernelContext::set_output_ref |
| PrivateThreadPoolDatasetOp | tensorflow::OpKernelContext::env |
| QrOp |  |
| QrOpGpu |  |
| QuantizeAndDequantizeOp |  |
| QuantizeAndDequantizeV2Op | tensorflow::OpKernelContext::allocate_temp |
| QuantizeAndDequantizeV3Op | tensorflow::OpKernelContext::allocate_temp |
| QuantizedAddOp | tensorflow::OpKernelContext::SetStatus |
| QuantizedAvgPoolingOp | tensorflow::OpKernelContext::device |
| QuantizedBatchNormOp |  |
| QuantizedBiasAddOp |  |
| QuantizedConcatOp | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::input_list |
| QuantizedConv2DOp |  |
| QuantizedInstanceNorm |  |
| QuantizedMatMulOp | tensorflow::OpKernelContext::device |
| QuantizedMaxPoolingOp | tensorflow::OpKernelConstruction::device_type,tensorflow::OpKernelContext::device |
| QuantizedMulOp | tensorflow::OpKernelContext::SetStatus |
| QuantizeDownAndShrinkRangeOp |  |
| QuantizedRelu6Op | tensorflow::OpKernelContext::eigen_cpu_device |
| QuantizedReluOp | tensorflow::OpKernelContext::eigen_cpu_device |
| QuantizedReshapeOp |  |
| QuantizedResizeBilinearOp |  |
| QuantizeV2Op |  |
| QueueAccessOpKernel |  |
| QueueCloseOp |  |
| QueueIsClosedOp |  |
| QueueOp | tensorflow::OpKernelConstruction::allocate_persistent,tensorflow::OpKernelConstruction::output_type,tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::resource_manager,tensorflow::OpKernelContext::set_output_ref |
| QueueOpKernel |  |
| QueueSizeOp |  |
| RaggedCrossOp | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::input_list |
| RaggedGatherOp | tensorflow::OpKernelContext::input_list |
| RaggedGatherOpBase | tensorflow::OpKernelContext::input_list |
| RaggedRangeOp |  |
| RaggedTensorFromVariantOp | tensorflow::OpKernelContext::output_list |
| RaggedTensorToSparseOp | tensorflow::OpKernelContext::input_list |
| RaggedTensorToTensorBaseOp |  |
| RaggedTensorToTensorOp | tensorflow::OpKernelContext::allocate_temp |
| RaggedTensorToVariantOp | tensorflow::OpKernelContext::input_list |
| RandomBinomialOp | tensorflow::OpKernelContext::allocate_persistent,tensorflow::OpKernelContext::resource_manager |
| RandomCropOp |  |
| RandomDatasetOp |  |
| RandomGammaOp | tensorflow::OpKernelContext::device |
| RandomPoissonOp |  |
| RandomShuffleOp | tensorflow::OpKernelContext::mutable_output |
| RandomShuffleQueueOp | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::resource_manager,tensorflow::OpKernelContext::set_output_ref |
| RandomUniformIntOp |  |
| RangeDatasetOp |  |
| RangeOp |  |
| RankOp |  |
| ReaderNumRecordsProducedOp | tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::resource_manager |
| ReaderNumWorkUnitsCompletedOp | tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::resource_manager |
| ReaderOpKernel | tensorflow::OpKernelConstruction::allocate_persistent,tensorflow::OpKernelConstruction::output_type,tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::cancellation_manager,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::resource_manager,tensorflow::OpKernelContext::set_output_ref |
| ReaderReadOp | tensorflow::OpKernelConstruction::env,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::resource_manager |
| ReaderReadUpToOp | tensorflow::OpKernelConstruction::env,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::resource_manager |
| ReaderResetOp | tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::resource_manager |
| ReaderRestoreStateOp | tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::resource_manager |
| ReaderSerializeStateOp | tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::resource_manager |
| ReaderVerbAsyncOpKernel | tensorflow::OpKernelConstruction::env,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::resource_manager |
| ReaderVerbSyncOpKernel | tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::resource_manager |
| ReadFileOp | tensorflow::OpKernelContext::env |
| ReadVariableOp |  |
| ReadVariablesOp |  |
| RebatchDatasetOp |  |
| RecordInputOp |  |
| RecvOp |  |
| ReduceDatasetOp | tensorflow::OpKernelContext::cancellation_manager,tensorflow::OpKernelContext::input_list,tensorflow::OpKernelContext::op_kernel |
| ReduceJoinOp |  |
| ReductionOp | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::expected_output_dtype,tensorflow::OpKernelContext::output_alloc_attr |
| RefSelectOp | tensorflow::OpKernelContext::forward_ref_input_to_ref_output |
| RegexFullMatchOp |  |
| RegexReplaceOp | tensorflow::OpKernelContext::forward_input,tensorflow::OpKernelContext::input_alloc_attr,tensorflow::OpKernelContext::input_memory_type |
| RegisterDatasetOp |  |
| Relu6GradOp | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::forward_input_or_allocate_output |
| Relu6Op | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| ReluGradOp | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::forward_input_or_allocate_output |
| ReluOp | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| RemoteCallOp |  |
| RemoteFusedGraphExecuteOp |  |
| RepeatDatasetOp |  |
| RequantizationRangeOp |  |
| RequantizeOp |  |
| ReshapeOp |  |
| ResizeAreaOp |  |
| ResizeBicubicOp |  |
| ResizeBicubicOpGrad |  |
| ResizeBilinearOp |  |
| ResizeBilinearOpGrad |  |
| ResizeNearestNeighborOp | tensorflow::OpKernelContext::SetStatus |
| ResizeNearestNeighborOpGrad | tensorflow::OpKernelContext::SetStatus |
| ResourceAccumulatorApplyGradientOp | tensorflow::OpKernelContext::MatchSignature,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::resource_manager |
| ResourceAccumulatorNumAccumulatedOp | tensorflow::OpKernelContext::MatchSignature |
| ResourceAccumulatorSetGlobalStepOp | tensorflow::OpKernelContext::MatchSignature,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::resource_manager |
| ResourceAccumulatorTakeGradientOp | tensorflow::OpKernelContext::MatchSignature,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::resource_manager |
| ResourceConditionalAccumulatorOp | tensorflow::OpKernelConstruction::allocate_persistent,tensorflow::OpKernelContext::MatchSignature,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::resource_manager |
| ResourceCountUpToOp | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::allocate_persistent,tensorflow::OpKernelContext::resource_manager |
| ResourceGatherNdOp | tensorflow::OpKernelContext::allocate_persistent,tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::resource_manager |
| ResourceGatherOp | tensorflow::OpKernelContext::allocate_persistent,tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::resource_manager |
| ResourceHandleOp | tensorflow::OpKernelConstruction::device,tensorflow::OpKernelConstruction::resource_manager,tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::resource_manager |
| ResourceHandlesOp | tensorflow::OpKernelConstruction::device,tensorflow::OpKernelConstruction::resource_manager,tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::resource_manager |
| ResourceOpKernel | tensorflow::OpKernelConstruction::allocate_persistent,tensorflow::OpKernelConstruction::output_type,tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::resource_manager,tensorflow::OpKernelContext::set_output_ref |
| ResourceScatterUpdateOp | tensorflow::OpKernelContext::allocate_persistent,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::resource_manager |
| RestoreOp | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::expected_output_dtype,tensorflow::OpKernelContext::slice_reader_cache |
| RestoreSliceOp | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::expected_output_dtype,tensorflow::OpKernelContext::slice_reader_cache |
| RestoreV2 | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::expected_output_dtype,tensorflow::OpKernelContext::mutable_output,tensorflow::OpKernelContext::slice_reader_cache |
| RetvalOp |  |
| ReverseOp |  |
| ReverseSequenceOp |  |
| ReverseV2Op |  |
| RGBToHSVOp | tensorflow::OpKernelContext::allocate_temp |
| RngSkipOp | tensorflow::OpKernelContext::allocate_persistent,tensorflow::OpKernelContext::resource_manager |
| RollOp |  |
| RpcOp |  |
| SampleDistortedBoundingBoxV2Op |  |
| SamplingDatasetOp |  |
| SaveOp | tensorflow::OpKernelContext::SetStatus |
| SaveSlicesOp | tensorflow::OpKernelContext::SetStatus |
| SaveV2 |  |
| ScalarAdd |  |
| ScaleAndTranslateGradOp | tensorflow::OpKernelContext::allocate_temp |
| ScaleAndTranslateOp | tensorflow::OpKernelContext::allocate_temp |
| ScanDatasetOp | tensorflow::OpKernelConstruction::HasAttr,tensorflow::OpKernelContext::input_list |
| ScanOp |  |
| ScatterNdOp | tensorflow::OpKernelContext::allocate_temp |
| ScatterNdUpdateOp | tensorflow::OpKernelConstruction::input_type,tensorflow::OpKernelContext::allocate_persistent,tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::resource_manager |
| ScatterUpdateOp | tensorflow::OpKernelContext::input_ref_mutex |
| ScopedAllocatorConcatOp | tensorflow::OpKernelConstruction::device,tensorflow::OpKernelContext::op_kernel |
| ScopedAllocatorOp | tensorflow::OpKernelConstruction::device,tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::op_kernel,tensorflow::OpKernelContext::output_alloc_attr,tensorflow::OpKernelContext::step_id |
| ScopedAllocatorSplitOp | tensorflow::OpKernelConstruction::device,tensorflow::OpKernelContext::op_kernel |
| SdcaFprint |  |
| SdcaOptimizer | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::eigen_cpu_device |
| SdcaShrinkL1 | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::eigen_cpu_device,tensorflow::OpKernelContext::mutable_input_list |
| SegmentReductionOp |  |
| SegmentSumGPUOp | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::op_device_context |
| SelectOp | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| SelectV2Op | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::forward_input_or_allocate_output |
| SelfAdjointEigOp |  |
| SelfAdjointEigV2Op |  |
| SelfAdjointEigV2OpGpu |  |
| SeluGradOp | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::forward_input_or_allocate_output |
| SeluOp | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| SendOp |  |
| SerializeIteratorOp |  |
| SerializeManySparseOp |  |
| SerializeSparseOp |  |
| SerializeTensorOp |  |
| SessionMetadataReaderOp | tensorflow::OpKernelContext::session_metadata |
| SetOperationOp |  |
| SetSizeOp | tensorflow::OpKernelContext::eigen_cpu_device |
| SetStatsAggregatorDatasetOp | tensorflow::OpKernelContext::resource_manager |
| ShapeNOp |  |
| ShapeOp |  |
| ShardDatasetOp |  |
| ShardedFilenameOp |  |
| ShardedFilespecOp |  |
| ShuffleAndRepeatDatasetOp |  |
| ShuffleDatasetOp |  |
| ShuffleDatasetOpBase |  |
| SimpleBinaryOp | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| SimpleCandidateSamplerOp |  |
| SizeOp |  |
| SkipDatasetOp |  |
| SkipgramOp | tensorflow::OpKernelConstruction::env |
| SleepDatasetOp |  |
| SliceOp |  |
| SlidingWindowDatasetOp |  |
| SnapshotDatasetOp | tensorflow::OpKernelConstruction::graph_def_version |
| SnapshotOp | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| SobolSampleOp | tensorflow::OpKernelContext::device |
| SoftmaxOp | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| SoftmaxXentWithLogitsOp | tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::forward_input_or_allocate_output |
| SoftplusGradOp | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::forward_input_or_allocate_output |
| SoftplusOp | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| SoftsignGradOp | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::forward_input_or_allocate_output |
| SoftsignOp | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| SpaceToBatchNDOp |  |
| SpaceToBatchOp |  |
| SpaceToDepthOp |  |
| SparseAccumulatorApplyGradientOp | tensorflow::OpKernelContext::MatchSignature,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::resource_manager |
| SparseAccumulatorTakeGradientOp | tensorflow::OpKernelContext::MatchSignature,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::resource_manager |
| SparseAddGradOp |  |
| SparseAddOp |  |
| SparseApplyAdadeltaOp | tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input |
| SparseApplyAdagradDAOp | tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::mutable_input |
| SparseApplyAdagradOp | tensorflow::OpKernelContext::eigen_cpu_device,tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::mutable_input |
| SparseApplyAdagradV2Op | tensorflow::OpKernelContext::eigen_cpu_device,tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::mutable_input |
| SparseApplyCenteredRMSPropOp | tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::mutable_input |
| SparseApplyFtrlOp | tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::mutable_input |
| SparseApplyKerasMomentumOp | tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::mutable_input |
| SparseApplyMomentumOp | tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::mutable_input |
| SparseApplyProximalAdagradOp | tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::mutable_input |
| SparseApplyProximalGradientDescentOp | tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::mutable_input |
| SparseApplyRMSPropOp | tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::mutable_input |
| SparseConcatOp | tensorflow::OpKernelContext::input_list |
| SparseConditionalAccumulatorOp | tensorflow::OpKernelConstruction::allocate_persistent,tensorflow::OpKernelContext::MatchSignature,tensorflow::OpKernelContext::resource_manager |
| SparseCrossOp | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::input_list |
| SparseDenseBinaryOpShared | tensorflow::OpKernelContext::allocate_temp |
| SparseFillEmptyRowsGradOp | tensorflow::OpKernelContext::allocate_temp |
| SparseFillEmptyRowsOp | tensorflow::OpKernelContext::output_required |
| SparseMatMulOp | tensorflow::OpKernelContext::device |
| SparseReduceOp | tensorflow::OpKernelContext::allocate_temp |
| SparseReduceSparseOp | tensorflow::OpKernelContext::allocate_temp |
| SparseReorderOp |  |
| SparseReshapeOp |  |
| SparseSegmentGradOpBase |  |
| SparseSegmentMeanGradOp |  |
| SparseSegmentReductionMeanOp |  |
| SparseSegmentReductionMeanWithNumSegmentsOp |  |
| SparseSegmentReductionOpBase |  |
| SparseSegmentReductionSqrtNOp |  |
| SparseSegmentReductionSqrtNWithNumSegmentsOp |  |
| SparseSegmentReductionSumOp |  |
| SparseSegmentReductionSumWithNumSegmentsOp |  |
| SparseSegmentSqrtNGradOp |  |
| SparseSliceGradOp |  |
| SparseSliceOp |  |
| SparseSoftmaxOp | tensorflow::OpKernelContext::allocate_temp |
| SparseSoftmaxXentWithLogitsOp | tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::forward_input_or_allocate_output |
| SparseSparseBinaryOpShared |  |
| SparseSplitOp |  |
| SparseTensorAccessingOp | tensorflow::OpKernelContext::resource_manager |
| SparseTensorDenseAddOp |  |
| SparseTensorDenseMatMulOp |  |
| SparseTensorSliceDatasetOp |  |
| SparseTensorToCSRSparseMatrixCPUOp |  |
| SparseTensorToCSRSparseMatrixGPUOp | tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::op_device_context |
| SparseToDense |  |
| SparseToSparseSetOperationOp |  |
| SplitOpBase |  |
| SplitOpCPU |  |
| SplitOpGPU | tensorflow::OpKernelContext::op_device_context |
| SplitVOpBase |  |
| SplitVOpCPU |  |
| SplitVOpGPU | tensorflow::OpKernelContext::op_device_context |
| SqlDatasetOp |  |
| SqueezeOp | tensorflow::OpKernelContext::SetStatus |
| StackCloseOp |  |
| StackOp |  |
| StackPopOp |  |
| StackPushOp |  |
| StageClearOp | tensorflow::OpKernelContext::resource_manager |
| StageOp | tensorflow::OpKernelContext::resource_manager |
| StagePeekOp | tensorflow::OpKernelContext::resource_manager |
| StageSizeOp | tensorflow::OpKernelContext::resource_manager |
| StatefulMultinomialOp | tensorflow::OpKernelContext::allocate_temp |
| StatefulOutputRequiredOp | tensorflow::OpKernelContext::output_required |
| StatefulRandomOp |  |
| StatefulRandomOpV2 |  |
| StatefulUniformFullIntOp |  |
| StatefulUniformIntOp |  |
| StatelessMultinomialOp | tensorflow::OpKernelContext::allocate_temp |
| StatelessRandomBinomialOp |  |
| StatelessRandomGammaOp | tensorflow::OpKernelContext::device |
| StatelessRandomOp |  |
| StatelessRandomOpBase |  |
| StatelessRandomPoissonOp |  |
| StatelessRandomUniformFullIntOp |  |
| StatelessRandomUniformIntOp |  |
| StaticRegexFullMatchOp |  |
| StaticRegexReplaceOp | tensorflow::OpKernelContext::forward_input,tensorflow::OpKernelContext::input_alloc_attr,tensorflow::OpKernelContext::input_memory_type |
| StatsAggregatorHandleOp | tensorflow::OpKernelConstruction::allocate_persistent,tensorflow::OpKernelConstruction::output_type,tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::resource_manager,tensorflow::OpKernelContext::set_output_ref |
| StatsAggregatorHandleOpV2 | tensorflow::OpKernelConstruction::allocate_persistent,tensorflow::OpKernelConstruction::output_type,tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::resource_manager,tensorflow::OpKernelContext::set_output_ref |
| StatsAggregatorSetSummaryWriterOp | tensorflow::OpKernelContext::resource_manager |
| StatsAggregatorSummaryOp | tensorflow::OpKernelContext::resource_manager |
| StridedSliceAssignOp | tensorflow::OpKernelContext::allocate_persistent,tensorflow::OpKernelContext::forward_input,tensorflow::OpKernelContext::forward_ref_input_to_ref_output,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::resource_manager |
| StridedSliceGradOp |  |
| StridedSliceOp |  |
| StringFormatOp |  |
| StringJoinOp | tensorflow::OpKernelContext::input_list |
| StringLengthOp |  |
| StringLowerOp |  |
| StringNGramsOp |  |
| StringSplitV2Op |  |
| StringStripOp |  |
| StringToHashBucketOp |  |
| StringToKeyedHashBucketOp |  |
| StringToNumberOp |  |
| StringUpperOp |  |
| StubResourceOpKernel | tensorflow::OpKernelConstruction::allocate_persistent,tensorflow::OpKernelConstruction::output_type,tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::resource_manager,tensorflow::OpKernelContext::set_output_ref |
| SubstrOp | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::allocate_temp |
| SummaryAudioOp |  |
| SummaryHistoOp | tensorflow::OpKernelContext::SetStatus |
| SummaryImageOp | tensorflow::OpKernelConstruction::device |
| SummaryMergeOp | tensorflow::OpKernelContext::SetStatus |
| SummaryScalarOp |  |
| SummaryTensorOp | tensorflow::OpKernelContext::op_kernel |
| SummaryTensorOpV2 |  |
| SvdOp |  |
| SwitchNOp |  |
| SwitchOp |  |
| SymbolicGradientOp | tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::cancellation_manager,tensorflow::OpKernelContext::collective_executor,tensorflow::OpKernelContext::function_library,tensorflow::OpKernelContext::rendezvous,tensorflow::OpKernelContext::run_all_kernels_inline,tensorflow::OpKernelContext::runner,tensorflow::OpKernelContext::stats_collector,tensorflow::OpKernelContext::step_container,tensorflow::OpKernelContext::step_id |
| TakeDatasetOp |  |
| TakeManyOp | tensorflow::OpKernelContext::MatchSignature,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::input_ref_mutex,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::output_list,tensorflow::OpKernelContext::resource_manager |
| TakeManySparseFromTensorsMapOp | tensorflow::OpKernelContext::resource_manager |
| TakeWhileDatasetOp |  |
| TemplatedStackPushOp |  |
| TemporaryVariableOp | tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::record_persistent_memory_allocation,tensorflow::OpKernelContext::resource_manager,tensorflow::OpKernelContext::set_output_ref,tensorflow::OpKernelContext::step_container,tensorflow::OpKernelContext::track_allocations |
| TensorArrayCloseOp | tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::resource_manager,tensorflow::OpKernelContext::step_container |
| TensorArrayConcatOp | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::eigen_gpu_device,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::resource_manager,tensorflow::OpKernelContext::step_container |
| TensorArrayCreationOp | tensorflow::OpKernelConstruction::device_type,tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::expected_output_dtype,tensorflow::OpKernelContext::resource_manager,tensorflow::OpKernelContext::set_output_ref |
| TensorArrayGradOp | tensorflow::OpKernelConstruction::device_type,tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::expected_output_dtype,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::mutable_input,tensorflow::OpKernelContext::resource_manager,tensorflow::OpKernelContext::set_output_ref,tensorflow::OpKernelContext::step_container |
| TensorArrayOp | tensorflow::OpKernelConstruction::HasAttr,tensorflow::OpKernelConstruction::device_type,tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::expected_output_dtype,tensorflow::OpKernelContext::resource_manager,tensorflow::OpKernelContext::set_output_ref,tensorflow::OpKernelContext::step_container |
| TensorArrayPackOrGatherOp | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::eigen_gpu_device,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::resource_manager,tensorflow::OpKernelContext::step_container |
| TensorArrayReadOp | tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::resource_manager,tensorflow::OpKernelContext::step_container |
| TensorArraySizeOp | tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::resource_manager,tensorflow::OpKernelContext::step_container |
| TensorArraySplitOp | tensorflow::OpKernelContext::allocate_persistent,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::resource_manager,tensorflow::OpKernelContext::step_container |
| TensorArrayUnpackOrScatterOp | tensorflow::OpKernelContext::allocate_persistent,tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::resource_manager,tensorflow::OpKernelContext::step_container |
| TensorArrayWriteOp | tensorflow::OpKernelContext::input_dtype,tensorflow::OpKernelContext::resource_manager,tensorflow::OpKernelContext::step_container |
| TensorDatasetOp |  |
| TensorForestCreateTreeVariableOp | tensorflow::OpKernelContext::resource_manager |
| TensorForestTreeDeserializeOp | tensorflow::OpKernelContext::resource_manager |
| TensorForestTreePredictOp | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::resource_manager |
| TensorForestTreeSerializeOp | tensorflow::OpKernelContext::resource_manager |
| TensorForestTreeSizeOp | tensorflow::OpKernelContext::resource_manager |
| TensorListConcat | tensorflow::OpKernelConstruction::HasAttr,tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::eigen_gpu_device |
| TensorListConcatLists | tensorflow::OpKernelContext::forward_input |
| TensorListElementShape |  |
| TensorListFromTensor | tensorflow::OpKernelContext::allocate_temp |
| TensorListGather | tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::eigen_gpu_device |
| TensorListGetItem |  |
| TensorListLength |  |
| TensorListPopBack | tensorflow::OpKernelContext::forward_input,tensorflow::OpKernelContext::input_memory_type |
| TensorListPushBack | tensorflow::OpKernelContext::forward_input,tensorflow::OpKernelContext::input_memory_type |
| TensorListPushBackBatch | tensorflow::OpKernelContext::allocate_persistent,tensorflow::OpKernelContext::forward_input |
| TensorListReserve |  |
| TensorListResize | tensorflow::OpKernelContext::forward_input,tensorflow::OpKernelContext::input_memory_type |
| TensorListScatter | tensorflow::OpKernelContext::allocate_temp |
| TensorListScatterIntoExistingList | tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::forward_input,tensorflow::OpKernelContext::input_memory_type |
| TensorListSetItem | tensorflow::OpKernelContext::forward_input,tensorflow::OpKernelContext::input_memory_type |
| TensorListSplit | tensorflow::OpKernelContext::allocate_temp |
| TensorListStack | tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::eigen_gpu_device |
| TensorScatterOp | tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::forward_input |
| TensorSliceDatasetOp |  |
| TestKernel |  |
| TestOp |  |
| TestOp2 | tensorflow::OpKernelConstruction::SetStatus |
| TestOp3Cpu |  |
| TestOp3Gpu |  |
| TestOp5Cpu |  |
| TestOp5Gpu |  |
| TextLineDatasetOp |  |
| TextLineReaderOp | tensorflow::OpKernelConstruction::env,tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::cancellation_manager |
| TFRecordDatasetOp |  |
| TFRecordReaderOp | tensorflow::OpKernelConstruction::env,tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::cancellation_manager |
| ThreadIDOp |  |
| ThreadPoolDatasetOp | tensorflow::OpKernelContext::resource_manager |
| ThreadPoolHandleOp | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::env,tensorflow::OpKernelContext::resource_manager |
| TileGradientOp |  |
| TileOp |  |
| TimestampOp |  |
| ToBoolOp |  |
| TopK |  |
| ToSingleElementOp | tensorflow::OpKernelContext::cancellation_manager,tensorflow::OpKernelContext::op_kernel |
| ToTFRecordOp | tensorflow::OpKernelConstruction::env,tensorflow::OpKernelContext::cancellation_manager,tensorflow::OpKernelContext::env,tensorflow::OpKernelContext::op_kernel |
| TransposeCpuOp |  |
| TransposeGpuOp |  |
| TransposeOp |  |
| TridiagonalMatMulOp |  |
| TridiagonalSolveOp |  |
| TypedQueueOp | tensorflow::OpKernelConstruction::allocate_persistent,tensorflow::OpKernelConstruction::output_type,tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::resource_manager,tensorflow::OpKernelContext::set_output_ref |
| UnaryDatasetOpKernel |  |
| UnaryElementWiseOp | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| UnaryOp | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| UnaryOpBase |  |
| UnaryOpsComposition | tensorflow::OpKernelContext::forward_input_or_allocate_output |
| UnaryVariantOp | tensorflow::OpKernelContext::device |
| UnbatchDatasetOp |  |
| UnbatchGradKernel | tensorflow::OpKernelContext::resource_manager |
| UnbatchKernel | tensorflow::OpKernelContext::resource_manager |
| UnicodeDecodeBaseOp | tensorflow::OpKernelConstruction::HasAttr |
| UnicodeDecodeOp | tensorflow::OpKernelConstruction::HasAttr |
| UnicodeDecodeWithOffsetsOp | tensorflow::OpKernelConstruction::HasAttr |
| UnicodeEncodeOp | tensorflow::OpKernelConstruction::HasAttr |
| UnicodeScriptOp |  |
| UnicodeTranscodeOp | tensorflow::OpKernelConstruction::HasAttr,tensorflow::OpKernelContext::forward_input,tensorflow::OpKernelContext::input_alloc_attr,tensorflow::OpKernelContext::input_memory_type |
| UniqueDatasetOp |  |
| UniqueOp |  |
| UnpackOp | tensorflow::OpKernelContext::output_required |
| UnravelIndexOp | tensorflow::OpKernelContext::allocate_temp |
| UnsortedSegmentJoinOp |  |
| UnsortedSegmentReductionOp |  |
| UnstageOp | tensorflow::OpKernelContext::resource_manager |
| UnwrapDatasetVariantOp |  |
| UpperBoundOp |  |
| VarHandleOp |  |
| VariableOp |  |
| VariableShapeOp | tensorflow::OpKernelContext::resource_manager |
| VarIsInitializedOp | tensorflow::OpKernelContext::resource_manager |
| WhereCPUOp |  |
| WhereGPUOp | tensorflow::OpKernelContext::allocate_temp,tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::op_device_context |
| WhileOp | tensorflow::OpKernelContext::cancellation_manager,tensorflow::OpKernelContext::function_library,tensorflow::OpKernelContext::rendezvous,tensorflow::OpKernelContext::run_all_kernels_inline,tensorflow::OpKernelContext::runner,tensorflow::OpKernelContext::stats_collector,tensorflow::OpKernelContext::step_container |
| WholeFileReaderOp | tensorflow::OpKernelConstruction::env,tensorflow::OpKernelContext::SetStatus,tensorflow::OpKernelContext::cancellation_manager |
| WindowDatasetOp |  |
| WrapDatasetVariantOp |  |
| WriteAudioSummaryOp | tensorflow::OpKernelContext::resource_manager |
| WriteFileOp | tensorflow::OpKernelContext::env |
| WriteGraphSummaryOp | tensorflow::OpKernelContext::resource_manager |
| WriteHistogramSummaryOp | tensorflow::OpKernelContext::resource_manager |
| WriteImageSummaryOp | tensorflow::OpKernelContext::resource_manager |
| WriteRawProtoSummaryOp | tensorflow::OpKernelContext::env,tensorflow::OpKernelContext::resource_manager |
| WriteScalarSummaryOp | tensorflow::OpKernelContext::resource_manager |
| WriteSummaryOp | tensorflow::OpKernelContext::resource_manager |
| ZerosLikeOp | tensorflow::OpKernelContext::device,tensorflow::OpKernelContext::forward_input_or_allocate_output |
| ZipDatasetOp |  |

