#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_KERNEL_INIT_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_KERNEL_INIT_H_

#include "tensorflow_plugin/src/xpu_core/device/gpu/gpu_device_plugin.h"

void RegisterGPUAdd(const char* device_type);
void RegisterGPUAddV2(const char* device_type);
void RegisterGPUArgMax(const char* device_type);
void RegisterGPUAvgPooling(const char* device_type);
void RegisterGPUBiasAdd(const char* device_type);
void RegisterGPUBroadcastTo(const char* device_type);
void RegisterGPUCast(const char* device_type);
void RegisterGPUConcatV2(const char* device_type);
void RegisterGPUConvOps(const char* device_type);
void RegisterGPUDataFormatDimMap(const char* device_type);
void RegisterGPUDataFormatVecPermute(const char* device_type);
void RegisterGPUDiv(const char* device_type);
void RegisterGPUDivNoNan(const char* device_type);
void RegisterGPUEqual(const char* device_type);
void RegisterGPUErf(const char* device_type);
void RegisterGPUExp(const char* device_type);
void RegisterGPUFill(const char* device_type);
void RegisterGPUFloorDiv(const char* device_type);
void RegisterGPUFusedBatchNorm(const char* device_type);
void RegisterGPUFusedBatchNormV2(const char* device_type);
void RegisterGPUFusedBatchNormV3(const char* device_type);
void RegisterGPUGather(const char* device_type);
void RegisterGPUGelu(const char* device_type);
void RegisterGPUGreater(const char* device_type);
void RegisterGPUGreaterEqual(const char* device_type);
void RegisterGPUIsFinite(const char* device_type);
void RegisterGPUL2Loss(const char* device_type);
void RegisterGPULeakyRelu(const char* device_type);
void RegisterGPULess(const char* device_type);
void RegisterGPULessEqual(const char* device_type);
void RegisterGPULogicalAnd(const char* device_type);
void RegisterGPULogicalNot(const char* device_type);
void RegisterGPULogSoftmax(const char* device_type);
void RegisterGPUMatMul(const char* device_type);
void RegisterGPUMaximum(const char* device_type);
void RegisterGPUMaxPooling(const char* device_type);
void RegisterGPUMean(const char* device_type);
void RegisterGPUMinimum(const char* device_type);
void RegisterGPUMul(const char* device_type);
void RegisterGPUMulNoNan(const char* device_type);
void RegisterGPUNeg(const char* device_type);
void RegisterGPUNoOp(const char* device_type);
void RegisterGPUOneHot(const char* device_type);
void RegisterGPUPack(const char* device_type);
void RegisterGPUPad(const char* device_type);
void RegisterGPUPow(const char* device_type);
void RegisterGPURandom(const char* device_type);
void RegisterGPURealDiv(const char* device_type);
void RegisterGPUReciprocal(const char* device_type);
void RegisterGPURelu(const char* device_type);
void RegisterGPURelu6(const char* device_type);
void RegisterGPUReluGrad(const char* device_type);
void RegisterGPURsqrt(const char* device_type);
void RegisterGPURsqrtGrad(const char* device_type);
void RegisterGPUSelect(const char* device_type);
void RegisterGPUSlice(const char* device_type);
void RegisterGPUSoftmax(const char* device_type);
void RegisterGPUSqrt(const char* device_type);
void RegisterGPUSquare(const char* device_type);
void RegisterGPUSquaredDifference(const char* device_type);
void RegisterGPUStridedSliceOps(const char* device_type);
void RegisterGPUSub(const char* device_type);
void RegisterGPUSum(const char* device_type);
void RegisterGPUTanh(const char* device_type);
void RegisterGPUTanhGrad(const char* device_type);
void RegisterGPUTransposeOp(const char* device_type);
void RegisterGPUTruncateDiv(const char* device_type);
void RegisterGPUUnpack(const char* device_type);
void RegisterGPUUnsortedSegmentReduction(const char* device_type);

void RegisterGPUKernels(const char* device_type);
#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_KERNEL_INIT_H_
