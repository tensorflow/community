#include "absl/container/inlined_vector.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

#include <iostream>
#include <memory>
#include <vector>

namespace demo_plugin {

struct StatusDeleter {
  void operator()(TF_Status* s) {
    if (s != nullptr) {
      TF_DeleteStatus(s);
    }
  }
};

struct TensorDeleter {
  void operator()(TF_Tensor* t) {
    if (t != nullptr) {
      TF_DeleteTensor(t);
    }
  }
};

using StatusSafePtr = std::unique_ptr<TF_Status, StatusDeleter>;
using TensorSafePtr = std::unique_ptr<TF_Tensor, TensorDeleter>;

struct ReluOp {
  float alpha_;
  float beta_;
};

void* ReluOp_Create(ReluOp* kernel, float alpha, float beta) {
  kernel->alpha_ = alpha;
  kernel->beta_ = beta;
  return kernel;
}

template <typename T>
void ReluOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  ReluOp* relu = static_cast<ReluOp*>(kernel);
  StatusSafePtr status(TF_NewStatus());
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, status.get());
  TensorSafePtr input_safe_ptr(input);
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    return;
  }
  if (TF_TensorElementCount(input_safe_ptr.get()) == 0) return;
  absl::InlinedVector<int64_t, 4> dims(TF_NumDims(input_safe_ptr.get()));
  for (auto i = 0; i < TF_NumDims(input_safe_ptr.get()); ++i) {
    dims[i] = TF_Dim(input_safe_ptr.get(), i);
  }

  TensorSafePtr output_safe_ptr(TF_AllocateOutput(
      ctx, 0, TF_ExpectedOutputDataType(ctx, 0), dims.data(), dims.size(),
      TF_TensorElementCount(input_safe_ptr.get()) * sizeof(T), status.get()));
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    return;
  }

  auto input_ptr = static_cast<T*>(TF_TensorData(input_safe_ptr.get()));
  auto output_ptr = static_cast<T*>(TF_TensorData(output_safe_ptr.get()));
  for (auto i = 0; i < TF_TensorElementCount(input_safe_ptr.get()); ++i) {
    output_ptr[i] = input_ptr[i] > 0 ? input_ptr[i] : 0;
  }
}

template <typename T>
void RegisterReluOpKernel(const char* device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder("Relu", device_type, nullptr,
                                      &ReluOp_Compute<T>, nullptr);
  TF_KernelBuilder_TypeConstraint(builder, "T", TF_FLOAT, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering relu kernel with attribute T";
  TF_RegisterKernelBuilder("ReluOp", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering relu kernel";
}

}  // namespace demo_plugin

void RegisterDeviceRelu(const char* device_type) {
  demo_plugin::RegisterReluOpKernel<float>(device_type);
}
