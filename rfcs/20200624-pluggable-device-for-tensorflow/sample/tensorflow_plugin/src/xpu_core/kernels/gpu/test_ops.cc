#include "gpu_kernel_init.h"
#include "iostream"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_tensor.h"
struct TestOp {
  std::string op_node_name;
};

void* TestOp_Create(TF_OpKernelConstruction* ctx) {
  TestOp* kernel = new TestOp;
  TF_StringView string_view_name = TF_OpKernelConstruction_GetName(ctx);
  kernel->op_node_name =
      std::string(string_view_name.data, string_view_name.len);
  return kernel;
}

void TestOp_Delete(void* kernel) { delete static_cast<TestOp*>(kernel); }

void TestOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TestOp* k = static_cast<TestOp*>(kernel);
  std::cout << "TEST OP" << k->op_node_name << std::endl;
  DPCPPStream* dpcpp_stream = TF_GetStream(ctx)->stream_handle;
  printf("tf get stream = %p\n", dpcpp_stream);
  TF_Tensor* src_tensor;
  TF_Status* status = TF_NewStatus();
  TF_GetInput(ctx, 0, &src_tensor, status);
  TF_Tensor* dst_tensor = nullptr;

  int candidata_input_indices[1] = {0};
  int forward_input;
  int64_t output_dims[1] = {};
  TF_Tensor* output =
      TF_ForwardInputOrAllocateOutput(ctx, candidata_input_indices, 1, 0,
                                      output_dims, 0, &forward_input, status);
}

void RegisterReluKernel() {
  StatusUniquePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder("Relu", "GPU", &TestOp_Create,
                                      &TestOp_Compute, &TestOp_Delete);
  TF_RegisterKernelBuilder("Relu", builder, status.get());
}
