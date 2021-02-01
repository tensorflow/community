#include "tensorflow_plugin/src/xpu_core/device/gpu/eigen_stream_device.h"
#include "tensorflow_plugin/src/xpu_core/device/gpu/gpu_device_plugin.h"
#include "tensorflow_plugin/src/xpu_core/util/dnnl_util.h"
#include "tensorflow_plugin/src/xpu_core/util/errors.h"
#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/op_requires.h"
#include "tensorflow_plugin/src/xpu_core/util/plugin_tensor.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

using dnnl::algorithm;
using dnnl::eltwise_backward;
using dnnl::eltwise_forward;
using dnnl::engine;
using dnnl::memory;
using dnnl::primitive;
using dnnl::prop_kind;
using dnnl::stream;

namespace intel_plugin {
typedef Eigen::GpuDevice GPUDevice;

struct ReluOpBase {
  float alpha_;
  float beta_;
};

void* ReluOpBase_Create(ReluOpBase* kernel, float alpha, float beta) {
  kernel->alpha_ = alpha;
  kernel->beta_ = beta;
  return kernel;
}

template <typename T, algorithm alg_kind>
void ReluOpBase_Compute(void* kernel, TF_OpKernelContext* ctx) {
  OpKernelContext context(ctx);
  try {
    auto op_kernel = static_cast<ReluOpBase*>(kernel);
    auto onednn_engine = CreateDnnlEngine(context);

    const size_t src_index = 0;  // index of src input tensor
    const size_t dst_index = 0;  // index of dst output tensor
    const Tensor& src_tensor = context.input(src_index);

    if (std::is_same<T, qint8>::value) {
      OP_REQUIRES(
          &context, src_tensor.NumElements() % 4 == 0,
          errors::InvalidArgument(
              "Tensor size must be a multiple of 4 for Relu<qint8>. Got ",
              src_tensor.NumElements()));
    }

    Tensor* dst_tensor = nullptr;
    // Nothing to compute, return.
    if (src_tensor.shape().num_elements() == 0) {
      OP_REQUIRES_OK(&context, context.allocate_output(
                                   dst_index, src_tensor.shape(), &dst_tensor));
      return;
    }

    // memory desc
    memory::desc src_md({}, memory::data_type::undef,
                        memory::format_tag::undef);
    memory::desc dst_md;
    memory::dims src_dims = TFShapeToMklDnnDims(src_tensor.shape());
    auto src_strides = CalculateTFStrides(src_dims);
    // Create blocked memory descriptor
    src_md = CreateBlockedMemDesc<T>(src_dims, src_strides);

    // Create an eltwise forward descriptor and primitive descriptor
    std::shared_ptr<eltwise_forward::desc> fwd_desc(
        new eltwise_forward::desc(prop_kind::forward, alg_kind, src_md,
                                  op_kernel->alpha_, op_kernel->beta_));
    std::shared_ptr<eltwise_forward::primitive_desc> fwd_pd(
        new eltwise_forward::primitive_desc(*fwd_desc, onednn_engine));
    std::shared_ptr<primitive> fwd_primitive(new eltwise_forward(*fwd_pd));

    // Create memory primitive
    T* src_data = static_cast<T*>(const_cast<T*>(src_tensor.flat<T>().data()));
    auto src_mem = CreateDnnlMemory(fwd_pd->src_desc(), onednn_engine,
                                    static_cast<void*>(src_data));

    OP_REQUIRES_OK(&context, context.forward_input_or_allocate_output(
                                 {static_cast<const int>(src_index)},
                                 static_cast<const int>(dst_index),
                                 src_tensor.shape(), &dst_tensor));

    T* dst_data = dst_tensor->flat<T>().data();
    auto dst_mem = CreateDnnlMemory(fwd_pd->dst_desc(), onednn_engine,
                                    static_cast<void*>(dst_data));

    // execute eltwise
    auto onednn_stream = CreateDnnlStream(context, onednn_engine);
    std::unordered_map<int, memory> fwd_primitive_args = {
        {DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}};
    fwd_primitive->execute(onednn_stream, fwd_primitive_args);
  } catch (dnnl::error& e) {
    string error_msg = "Status: " + std::to_string(e.status) +
                       ", message: " + string(e.message) + ", in file " +
                       string(__FILE__) + ":" + std::to_string(__LINE__);
    OP_REQUIRES_OK(&context, errors::Aborted("Operation received an exception:",
                                             error_msg));
  }
}

struct ReluOp : public ReluOpBase {};

void* ReluOp_Create(TF_OpKernelConstruction* ctx) {
  OpKernelConstruction context(ctx);
  auto* kernel = new ReluOp;
  ReluOpBase_Create(kernel, 0.0f, 0.0f);
  return kernel;
}

void ReluOp_Delete(void* kernel) {
  if (kernel) delete static_cast<ReluOp*>(kernel);
}

template <typename T>
void RegisterReluOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder = TF_NewKernelBuilder(
        "Relu", device_type, &ReluOp_Create,
        &ReluOpBase_Compute<T, algorithm::eltwise_relu>, &ReluOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering relu kernel with attribute T";
    TF_RegisterKernelBuilder("ReluOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering relu kernel";
  }
}

/// LeakyRelu impl.

struct LeakyReluOp : public ReluOpBase {};

void* LeakyReluOp_Create(TF_OpKernelConstruction* ctx) {
  OpKernelConstruction context(ctx);
  auto* kernel = new LeakyReluOp;
  float alpha = 0.0f;
  TF_CHECK_OK(context.GetAttr("alpha", &alpha))
      << " Error can't find the attribute alpha of leaky relu";
  CHECK_LE(alpha, 1) << "LeakyRelu only support alpha <=1. But now alpha is: "
                     << alpha;
  ReluOpBase_Create(kernel, alpha, 0.0f);
  return kernel;
}

void LeakyReluOp_Delete(void* kernel) {
  if (kernel) delete static_cast<LeakyReluOp*>(kernel);
}

template <typename T>
void RegisterLeakyReluOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto builder = TF_NewKernelBuilder(
        "LeakyRelu", device_type, &LeakyReluOp_Create,
        &ReluOpBase_Compute<T, algorithm::eltwise_relu>, &LeakyReluOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering leaky relu kernel with attribute T";
    TF_RegisterKernelBuilder("LeakyReluOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering leaky relu kernel";
  }
}

#define RELU6_UPPER_BOUND 6.0f
struct Relu6Op : public ReluOpBase {};

void* Relu6Op_Create(TF_OpKernelConstruction* ctx) {
  OpKernelConstruction context(ctx);
  Relu6Op* kernel = new Relu6Op;
  ReluOpBase_Create(kernel, RELU6_UPPER_BOUND, 0.0f);
  return kernel;
}

void Relu6Op_Delete(void* kernel) {
  if (kernel) {
    delete static_cast<Relu6Op*>(kernel);
  }
}

template <typename T>
void RegisterRelu6OpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto builder = TF_NewKernelBuilder(
        "Relu6", device_type, &Relu6Op_Create,
        &ReluOpBase_Compute<T, algorithm::eltwise_bounded_relu>,
        &Relu6Op_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering Relu6 kernel with attribute T";
    TF_RegisterKernelBuilder("Relu6Op", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering Relu6 kernel";
  }
}

// Gelu impl.
struct GeluOp : public ReluOpBase {};

void* GeluOp_Create(TF_OpKernelConstruction* ctx) {
  OpKernelConstruction context(ctx);
  auto* kernel = new GeluOp;
  ReluOpBase_Create(kernel, 0.0f, 0.0f);
  return kernel;
}

void GeluOp_Delete(void* kernel) {
  if (kernel) delete static_cast<GeluOp*>(kernel);
}

template <typename T>
void RegisterGeluOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto builder = TF_NewKernelBuilder(
        "Gelu", device_type, &GeluOp_Create,
        &ReluOpBase_Compute<T, algorithm::eltwise_gelu>, &GeluOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering gelu kernel with attribute T";
    TF_RegisterKernelBuilder("GeluOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering gelu kernel";
  }
}

// ReluGrad impl.
template <typename T, algorithm alg_kind>
void ReluGradOpBase_Compute(void* kernel, TF_OpKernelContext* ctx) {
  OpKernelContext context(ctx);
  try {
    auto op_kernel = static_cast<ReluOpBase*>(kernel);
    auto onednn_engine_ = CreateDnnlEngine(context);

    const size_t diff_dst_index = 0;  // index of diff_dst input tensor
    const size_t src_index = 1;       // index of src input tensor
    const size_t diff_src_index = 0;  // index of diff_src output tensor

    const Tensor& src_tensor = context.input(src_index);
    const Tensor& diff_dst_tensor = context.input(diff_dst_index);
    Tensor* diff_src_tensor = nullptr;

    int src_dims_size = src_tensor.dims();

    // Nothing to compute, return.
    if (src_tensor.shape().num_elements() == 0) {
      OP_REQUIRES_OK(&context,
                     context.allocate_output(
                         diff_src_index, context.input(diff_src_index).shape(),
                         &diff_src_tensor));
      return;
    }

    memory::dims src_dims = {};
    memory::desc src_md, diff_dst_md;

    src_dims = TFShapeToMklDnnDims(src_tensor.shape());
    auto src_strides = CalculateTFStrides(src_dims);
    src_md = CreateBlockedMemDesc<T>(src_dims, src_strides);
    diff_dst_md = src_md;

    // Create forward eltwise primitive based on src/diff_dst md
    eltwise_forward::desc fwd_desc(prop_kind::forward_training, alg_kind,
                                   src_md, op_kernel->alpha_, op_kernel->beta_);
    eltwise_forward::primitive_desc fwd_pd(fwd_desc, onednn_engine_);

    eltwise_backward::desc bwd_desc(alg_kind, src_md, diff_dst_md,
                                    op_kernel->alpha_, op_kernel->beta_);
    eltwise_backward::primitive_desc bwd_pd(bwd_desc, onednn_engine_, fwd_pd);
    primitive bwd_primitive(bwd_pd);

    T* src_data = static_cast<T*>(const_cast<T*>(src_tensor.flat<T>().data()));
    memory src_mem = CreateDnnlMemory(bwd_pd.src_desc(), onednn_engine_,
                                      static_cast<void*>(src_data));

    T* diff_dst_data =
        static_cast<T*>(const_cast<T*>(diff_dst_tensor.flat<T>().data()));
    memory diff_dst_mem =
        CreateDnnlMemory(bwd_pd.diff_dst_desc(), onednn_engine_,
                         static_cast<void*>(diff_dst_data));

    OP_REQUIRES_OK(&context, context.forward_input_or_allocate_output(
                                 {static_cast<const int>(diff_dst_index)},
                                 static_cast<const int>(diff_src_index),
                                 src_tensor.shape(), &diff_src_tensor));

    T* diff_src_data = diff_src_tensor->flat<T>().data();
    memory diff_src_mem =
        CreateDnnlMemory(bwd_pd.diff_src_desc(), onednn_engine_,
                         static_cast<void*>(diff_src_data));

    // execute eltwise bwd
    stream onednn_stream = CreateDnnlStream(context, onednn_engine_);
    std::unordered_map<int, memory> bwd_primitive_args = {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_DIFF_DST, diff_dst_mem},
        {DNNL_ARG_DIFF_SRC, diff_src_mem}};
    bwd_primitive.execute(onednn_stream, bwd_primitive_args);
  } catch (dnnl::error& e) {
    string error_msg = "Status: " + std::to_string(e.status) +
                       ", message: " + string(e.message) + ", in file " +
                       string(__FILE__) + ":" + std::to_string(__LINE__);
    OP_REQUIRES_OK(&context, errors::Aborted("Operation received an exception:",
                                             error_msg));
  }
}

template <typename T>
void RegisterReluGradOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder = TF_NewKernelBuilder(
        "ReluGrad", device_type, &ReluOp_Create,
        &ReluGradOpBase_Compute<T, algorithm::eltwise_relu>, &ReluOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering reluGrad kernel with attribute T";
    TF_RegisterKernelBuilder("ReluGradOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering reluGrad kernel";
  }
}

}  // namespace intel_plugin

void RegisterGPURelu(const char* device_type) {
  intel_plugin::RegisterReluOpKernel<float>(device_type);
  intel_plugin::RegisterReluOpKernel<Eigen::bfloat16>(device_type);
  intel_plugin::RegisterReluOpKernel<Eigen::half>(device_type);
  // TODO(yangshe1): Add qint8
  // intel_plugin::RegisterReluOpKernel<intel_plugin::qint8>();
}

void RegisterGPULeakyRelu(const char* device_type) {
  intel_plugin::RegisterLeakyReluOpKernel<float>(device_type);
  intel_plugin::RegisterLeakyReluOpKernel<Eigen::bfloat16>(device_type);
  intel_plugin::RegisterLeakyReluOpKernel<Eigen::half>(device_type);
}

void RegisterGPURelu6(const char* device_type) {
  intel_plugin::RegisterRelu6OpKernel<float>(device_type);
  intel_plugin::RegisterRelu6OpKernel<Eigen::bfloat16>(device_type);
  intel_plugin::RegisterRelu6OpKernel<Eigen::half>(device_type);
}

void RegisterGPUGelu(const char* device_type) {
  intel_plugin::RegisterGeluOpKernel<float>(device_type);
  intel_plugin::RegisterGeluOpKernel<Eigen::bfloat16>(device_type);
  intel_plugin::RegisterGeluOpKernel<Eigen::half>(device_type);
}

void RegisterGPUReluGrad(const char* device_type) {
  intel_plugin::RegisterReluGradOpKernel<float>(device_type);
  intel_plugin::RegisterReluGradOpKernel<Eigen::bfloat16>(device_type);
}
