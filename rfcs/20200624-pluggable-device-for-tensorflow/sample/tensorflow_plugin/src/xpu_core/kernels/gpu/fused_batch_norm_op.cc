#include "tensorflow_plugin/src/xpu_core/kernels/gpu/fused_batch_norm_op.h"
#include "dpcpp_runtime.h"
#include "tensorflow_plugin/src/xpu_core/device/gpu/eigen_stream_device.h"
#include "tensorflow_plugin/src/xpu_core/device/gpu/gpu_device_plugin.h"
#include "tensorflow_plugin/src/xpu_core/util/dnnl_util.h"
#include "tensorflow_plugin/src/xpu_core/util/errors.h"
#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/op_requires.h"
#include "tensorflow_plugin/src/xpu_core/util/plugin_tensor.h"
#include "tensorflow_plugin/src/xpu_core/util/register_types.h"
#include "tensorflow_plugin/src/xpu_core/util/tensor_format.h"
#include "tensorflow_plugin/src/xpu_core/util/tensor_types.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
using dnnl::batch_normalization_backward;
using dnnl::batch_normalization_forward;
using dnnl::engine;
using dnnl::memory;
using dnnl::primitive;
using dnnl::prop_kind;
using dnnl::stream;

namespace intel_plugin {
using GPUDevice = Eigen::GpuDevice;

namespace functor {

string ToString(FusedBatchNormActivationMode activation_mode) {
  switch (activation_mode) {
    case FusedBatchNormActivationMode::kIdentity:
      return "Identity";
    case FusedBatchNormActivationMode::kRelu:
      return "Relu";
  }
}
}  // namespace functor

template <typename T, typename U>
struct FusedBatchNormOp {
  U epsilon_;
  U exponential_avg_factor_;
  TensorFormat tensor_format_;
  bool is_training_;
  U* mean_values_;
  U* variance_values_;
  size_t depth_;
  functor::FusedBatchNormActivationMode activation_mode_;
};

template <bool reserved_space>
void AllocateTFOutputs(OpKernelContext* context, TensorShape tf_shape_scale,
                       TensorShape workspace_tf_shape,
                       Tensor** batch_mean_tensor,
                       Tensor** batch_variance_tensor,
                       Tensor** saved_mean_tensor,
                       Tensor** saved_variance_tensor,
                       Tensor** reserved_space_tensor) {
  DCHECK(batch_mean_tensor);
  DCHECK(batch_variance_tensor);
  DCHECK(saved_mean_tensor);
  DCHECK(saved_variance_tensor);

  OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                              {3}, 1, tf_shape_scale, batch_mean_tensor));
  OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                              {4}, 2, tf_shape_scale, batch_variance_tensor));
  OP_REQUIRES_OK(
      context, context->allocate_output(3, tf_shape_scale, saved_mean_tensor));
  OP_REQUIRES_OK(context, context->allocate_output(4, tf_shape_scale,
                                                   saved_variance_tensor));
  if (reserved_space)
    OP_REQUIRES_OK(context, context->allocate_output(5, workspace_tf_shape,
                                                     reserved_space_tensor));
}

template <typename T, bool reserved_space>
void HandleEmptyInput(OpKernelContext* context, TensorShape tf_shape_src,
                      TensorShape workspace_tf_shape,
                      TensorShape tf_shape_scale, Tensor** dst_tensor,
                      DPCPPStream* stream) {
  DCHECK(dst_tensor);

  OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                              {0}, 0, tf_shape_src, dst_tensor));

  DCHECK(*dst_tensor);

  Tensor* batch_mean_tensor = nullptr;
  Tensor* batch_variance_tensor = nullptr;
  Tensor* saved_mean_tensor = nullptr;
  Tensor* saved_variance_tensor = nullptr;
  Tensor* reserved_space_tensor = nullptr;
  AllocateTFOutputs<reserved_space>(context, tf_shape_scale, workspace_tf_shape,
                                    &batch_mean_tensor, &batch_variance_tensor,
                                    &saved_mean_tensor, &saved_variance_tensor,
                                    &reserved_space_tensor);
}

Status ParseActivationMode(
    OpKernelConstruction* context,
    functor::FusedBatchNormActivationMode* activation_mode) {
  string activation_mode_str;
  context->GetAttr("activation_mode", &activation_mode_str);

  if (activation_mode_str == "Identity") {
    *activation_mode = functor::FusedBatchNormActivationMode::kIdentity;
    return Status::OK();
  }
  if (activation_mode_str == "Relu") {
    *activation_mode = functor::FusedBatchNormActivationMode::kRelu;
    return Status::OK();
  }
  return errors::InvalidArgument("Unsupported activation mode: ",
                                 activation_mode_str);
}

template <typename T, typename U, bool is_batch_norm_ex = false>
void* FusedBatchNormOp_Create(TF_OpKernelConstruction* ctx) {
  using FbnActivationMode = functor::FusedBatchNormActivationMode;
  auto* kernel = new FusedBatchNormOp<T, U>;
  OpKernelConstruction context(ctx);
  float epsilon;
  OP_REQUIRES_OK_PTR(&context, context.GetAttr("epsilon", &epsilon));
  kernel->epsilon_ = U(epsilon);
  float exponential_avg_factor;
  OP_REQUIRES_OK_PTR(&context, context.GetAttr("exponential_avg_factor",
                                               &exponential_avg_factor));
  kernel->exponential_avg_factor_ = U(exponential_avg_factor);
  string tensor_format;
  OP_REQUIRES_OK_PTR(&context, context.GetAttr("data_format", &tensor_format));
  OP_REQUIRES_PTR(&context,
                  FormatFromString(tensor_format, &kernel->tensor_format_),
                  errors::InvalidArgument("Invalid data format"));
  OP_REQUIRES_OK_PTR(&context,
                     context.GetAttr("is_training", &kernel->is_training_));

  kernel->depth_ = 0;
  kernel->mean_values_ = nullptr;

  if (!is_batch_norm_ex) {
    kernel->activation_mode_ = FbnActivationMode::kIdentity;
  } else {
    int num_side_inputs;
    OP_REQUIRES_OK_PTR(&context,
                       context.GetAttr("num_side_inputs", &num_side_inputs));
    // Currently FusedBatchNormEx do not support "SideInput"
    OP_REQUIRES_PTR(&context, num_side_inputs == 0,
                    errors::InvalidArgument(
                        "FusedBatchNorm do not support side input now."));

    OP_REQUIRES_OK_PTR(
        &context, ParseActivationMode(&context, &kernel->activation_mode_));
    OP_REQUIRES_PTR(
        &context, kernel->activation_mode_ == FbnActivationMode::kRelu,
        errors::InvalidArgument("FusedBatchNorm only support Relu activation"));
  }
  return kernel;
}

template <typename T, typename U>
void FusedBatchNormOp_Delete(void* kernel) {
  if (kernel) delete static_cast<FusedBatchNormOp<T, U>*>(kernel);
}

template <class T, class U, bool float_one, bool reserved_space>
class VarAdjust {};
template <typename T, typename U, bool reserved_space>
void FusedBatchNormOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  OpKernelContext context(ctx);
  try {
    auto op_kernel = static_cast<FusedBatchNormOp<T, U>*>(kernel);
    auto* dpcpp_stream = context.GetDeviceStream();
    auto onednn_engine = CreateDnnlEngine(context);

    const size_t kSrcIndex = 0;       // index of src input tensor
    const size_t kScaleIndex = 1;     // index of scale tensor
    const size_t kShiftIndex = 2;     // index of shift tensor
    const size_t kMeanIndex = 3;      // index of est_mean tensor
    const size_t kVarianceIndex = 4;  // index of est_variance tensor

    const Tensor& src_tensor = context.input(kSrcIndex);
    const Tensor& scale_tensor = context.input(kScaleIndex);
    const Tensor& shift_tensor = context.input(kShiftIndex);
    const Tensor& est_mean_tensor = context.input(kMeanIndex);
    const Tensor& est_variance_tensor = context.input(kVarianceIndex);

    TensorShape tf_shape_src = src_tensor.shape();
    OP_REQUIRES(&context, src_tensor.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        src_tensor.shape().DebugString()));
    OP_REQUIRES(&context, scale_tensor.dims() == 1,
                errors::InvalidArgument("scale must be 1-dimensional",
                                        scale_tensor.shape().DebugString()));
    OP_REQUIRES(&context, shift_tensor.dims() == 1,
                errors::InvalidArgument("offset must be 1-dimensional",
                                        shift_tensor.shape().DebugString()));
    OP_REQUIRES(&context, est_mean_tensor.dims() == 1,
                errors::InvalidArgument("estimated_mean must be 1-dimensional",
                                        est_mean_tensor.shape().DebugString()));
    OP_REQUIRES(
        &context, est_variance_tensor.dims() == 1,
        errors::InvalidArgument("estimated_variance must be 1-dimensional",
                                est_variance_tensor.shape().DebugString()));

    // Handle the special case: input with 0 elements and 0 batch size.
    Tensor* dst_tensor = nullptr;
    TensorShape workspace_tf_shape;
    if (tf_shape_src.num_elements() == 0) {
      size_t workspace_bytes = 0;
      workspace_tf_shape.AddDim(workspace_bytes);
      HandleEmptyInput<T, reserved_space>(
          &context, tf_shape_src, workspace_tf_shape, scale_tensor.shape(),
          &dst_tensor, dpcpp_stream);
      return;
    }

    op_kernel->depth_ = static_cast<int>(
        GetTensorDim(src_tensor, op_kernel->tensor_format_, 'C'));
    // Index of output tensor.

    // Allocate 5 output TF tensors.
    Tensor* batch_mean_tensor = nullptr;
    Tensor* batch_variance_tensor = nullptr;
    Tensor* saved_mean_tensor = nullptr;
    Tensor* saved_variance_tensor = nullptr;
    Tensor* reserved_space_tensor = nullptr;

    memory::format_tag dnn_fmt;
    MklTensorFormat dnnl_tensor_fmt;

    dnnl_tensor_fmt = TFDataFormatToMklDnnDataFormat(op_kernel->tensor_format_);
    dnn_fmt = MklTensorFormatToMklDnnDataFormat(dnnl_tensor_fmt);

    // Set src memory descriptor.
    memory::dims src_dims = TFShapeToMklDnnDimsInNCHW(
        src_tensor.shape(), op_kernel->tensor_format_);

    auto src_md = memory::desc(src_dims, MklDnnType<T>(), dnn_fmt);
    auto propagation = op_kernel->is_training_ ? prop_kind::forward_training
                                               : prop_kind::forward_scoring;
    auto flag = op_kernel->is_training_
                    ? dnnl::normalization_flags::use_scale_shift
                    : (dnnl::normalization_flags::use_scale_shift |
                       dnnl::normalization_flags::use_global_stats);
    batch_normalization_forward::desc bn_fwd_desc(propagation, src_md,
                                                  op_kernel->epsilon_, flag);
    batch_normalization_forward::primitive_desc bn_fwd_pd(bn_fwd_desc,
                                                          onednn_engine);
    batch_normalization_forward bn_fwd_primitive(bn_fwd_pd);

    T* src_data = static_cast<T*>(const_cast<T*>(src_tensor.flat<T>().data()));

    // Allocate workspace tensor
    // There is acutally no workspace tensor out, so we make a dummy one.
    size_t workspace_bytes = 0;
    AllocateTFOutputs<reserved_space>(
        &context, scale_tensor.shape(), workspace_tf_shape, &batch_mean_tensor,
        &batch_variance_tensor, &saved_mean_tensor, &saved_variance_tensor,
        &reserved_space_tensor);

    if (op_kernel->is_training_) {
      op_kernel->mean_values_ = reinterpret_cast<U*>(
          const_cast<U*>(batch_mean_tensor->flat<U>().data()));
      op_kernel->variance_values_ = reinterpret_cast<U*>(
          const_cast<U*>(batch_variance_tensor->flat<U>().data()));
    } else {
      op_kernel->mean_values_ = reinterpret_cast<U*>(
          const_cast<U*>(est_mean_tensor.flat<U>().data()));
      op_kernel->variance_values_ = reinterpret_cast<U*>(
          const_cast<U*>(est_variance_tensor.flat<U>().data()));
    }

    U* weights_data = static_cast<float*>(
        context.eigen_gpu_device().allocate(2 * op_kernel->depth_ * sizeof(U)));
    const U* scale_tf = scale_tensor.flat<U>().data();
    const U* shift_tf = shift_tensor.flat<U>().data();
    dpcppMemcpyDtoDAsync(weights_data, scale_tf, op_kernel->depth_ * sizeof(U),
                         dpcpp_stream);
    dpcppMemcpyDtoDAsync(weights_data + op_kernel->depth_, shift_tf,
                         op_kernel->depth_ * sizeof(U), dpcpp_stream);
    char* saved_mean_data_tf =
        reinterpret_cast<char*>(saved_mean_tensor->flat<U>().data());
    dpcppMemcpyDtoDAsync(saved_mean_data_tf,
                         reinterpret_cast<char*>(op_kernel->mean_values_),
                         op_kernel->depth_ * sizeof(U), dpcpp_stream);
    char* saved_variance_data_tf =
        reinterpret_cast<char*>(saved_variance_tensor->flat<U>().data());
    dpcppMemcpyDtoDAsync(saved_variance_data_tf,
                         reinterpret_cast<char*>(op_kernel->variance_values_),
                         op_kernel->depth_ * sizeof(U), dpcpp_stream);
    OP_REQUIRES_OK(&context, context.forward_input_or_allocate_output(
                                 {0}, 0, src_tensor.shape(), &dst_tensor));

    U* weights_op_data = weights_data;
    U* mean_op_data = saved_mean_tensor->flat<U>().data();
    U* variance_op_data = saved_variance_tensor->flat<U>().data();
    T* dst_data = dst_tensor->flat<T>().data();

    auto src_mem = CreateDnnlMemory(bn_fwd_pd.src_desc(), onednn_engine,
                                    static_cast<void*>(src_data));
    auto dst_mem = CreateDnnlMemory(bn_fwd_pd.dst_desc(), onednn_engine,
                                    static_cast<void*>(dst_data));
    auto weights_mem = CreateDnnlMemory(bn_fwd_pd.weights_desc(), onednn_engine,
                                        static_cast<void*>(weights_op_data));
    auto mean_memory = CreateDnnlMemory(bn_fwd_pd.mean_desc(), onednn_engine,
                                        static_cast<void*>(mean_op_data));
    auto var_memory = CreateDnnlMemory(bn_fwd_pd.variance_desc(), onednn_engine,
                                       static_cast<void*>(variance_op_data));

    // Execute
    auto onednn_stream = CreateDnnlStream(context, onednn_engine);
    std::unordered_map<int, memory> args = {{DNNL_ARG_SRC, src_mem},
                                            {DNNL_ARG_DST, dst_mem}};
    if ((bool)(flag & dnnl::normalization_flags::use_scale_shift))
      args.insert({DNNL_ARG_SCALE_SHIFT, weights_mem});

    args.insert({DNNL_ARG_MEAN, mean_memory});
    args.insert({DNNL_ARG_VARIANCE, var_memory});
    bn_fwd_primitive.execute(onednn_stream, args);
    float adjust_factor = 1.0;
    if (op_kernel->is_training_) {
      size_t orig_size = src_dims[0] * src_dims[2] * src_dims[3];
      size_t adjust_size = (orig_size > 1) ? (orig_size - 1) : 1;
      adjust_factor = (static_cast<float>(orig_size)) / adjust_size;
    }

    auto mean_data = reinterpret_cast<U*>(saved_mean_data_tf);
    auto variance_data = reinterpret_cast<U*>(saved_variance_data_tf);
    auto batch_mean_data = batch_mean_tensor->flat<U>().data();
    auto batch_variance_data = batch_variance_tensor->flat<U>().data();
    auto est_mean_data = est_mean_tensor.flat<U>().data();
    auto est_variance_data = est_variance_tensor.flat<U>().data();

    auto total_threads =
        dpcpp_stream->get_device()
            .template get_info<cl::sycl::info::device::max_work_group_size>();
    int depth = op_kernel->depth_;
    if (op_kernel->is_training_) {
      if (op_kernel->exponential_avg_factor_ == U(1.0)) {
        dpcpp_stream->submit([&](cl::sycl::handler& cgh) {
          cgh.parallel_for<class VarAdjust<T, U, true, reserved_space>>(cl::sycl::range<1>(total_threads), [=](cl::sycl::item<1> item) {
            auto id = item.get_id(0);
            for (auto k = id; k < depth; k += total_threads) {
              batch_mean_data[k] = mean_data[k];
              batch_variance_data[k] =
                  static_cast<U>(adjust_factor) * variance_data[k];
            }
          });
        });
      } else {
        U one_minus_factor = U(1.0) - op_kernel->exponential_avg_factor_;
        dpcpp_stream->submit([&](cl::sycl::handler& cgh) {
          cgh.parallel_for<class VarAdjust<T, U, false, reserved_space>>(cl::sycl::range<1>(total_threads), [=](cl::sycl::item<1> item) {
            auto id = item.get_id(0);
            for (auto k = id; k < depth; k += total_threads) {
              batch_mean_data[k] =
                  one_minus_factor * est_mean_data[k] +
                  op_kernel->exponential_avg_factor_ * mean_data[k];
              batch_variance_data[k] = one_minus_factor * est_variance_data[k] +
                                       op_kernel->exponential_avg_factor_ *
                                           static_cast<U>(adjust_factor) *
                                           variance_data[k];
            }
          });
        });
      }
    } else {
      dpcppMemcpyDtoDAsync(batch_mean_data, mean_data, depth * sizeof(U),
                           dpcpp_stream);
      dpcppMemcpyDtoDAsync(batch_variance_data, variance_data,
                           depth * sizeof(U), dpcpp_stream);
    }

  } catch (dnnl::error& e) {
    string error_msg = "Status:" + std::to_string(e.status) +
                       ", message: " + string(e.message) + ", in file " +
                       string(__FILE__) + ":" + std::to_string(__LINE__);
    OP_REQUIRES_OK(&context, errors::Aborted("Operation received an exception:",
                                             error_msg));
  }
}

template <typename T>
void RegisterFusedBatchNormOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto builder = TF_NewKernelBuilder(
        "FusedBatchNorm", device_type, &FusedBatchNormOp_Create<T, float>,
        &FusedBatchNormOp_Compute<T, float, false>,
        &FusedBatchNormOp_Delete<T, float>);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering FusedBatchNorm kernel with attribute T";
    TF_RegisterKernelBuilder("FusedBatchNormOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering FusedBatchNorm kernel";
  }
}

template <typename T, typename U>
void RegisterFusedBatchNormV2OpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto builder = TF_NewKernelBuilder(
        "FusedBatchNormV2", device_type, &FusedBatchNormOp_Create<T, U>,
        &FusedBatchNormOp_Compute<T, U, false>, &FusedBatchNormOp_Delete<T, U>);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering FusedBatchNorm kernel with attribute T";
    TF_KernelBuilder_TypeConstraint(
        builder, "U",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<U>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering FusedBatchNorm kernel with attribute U";
    TF_RegisterKernelBuilder("FusedBatchNormV2Op", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering FusedBatchNorm kernel";
  }
}

template <typename T, typename U>
void RegisterFusedBatchNormV3OpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto builder = TF_NewKernelBuilder(
        "FusedBatchNormV3", device_type, &FusedBatchNormOp_Create<T, U>,
        &FusedBatchNormOp_Compute<T, U, true>, &FusedBatchNormOp_Delete<T, U>);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering FusedBatchNorm kernel with attribute T";
    TF_KernelBuilder_TypeConstraint(
        builder, "U",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<U>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering FusedBatchNorm kernel with attribute U";
    TF_RegisterKernelBuilder("FusedBatchNormV3Op", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering FusedBatchNorm kernel";
  }
}

}  // namespace intel_plugin

void RegisterGPUFusedBatchNorm(const char* device_type) {
  intel_plugin::RegisterFusedBatchNormOpKernel<float>(device_type);
  intel_plugin::RegisterFusedBatchNormOpKernel<Eigen::bfloat16>(device_type);
  intel_plugin::RegisterFusedBatchNormOpKernel<Eigen::half>(device_type);
}

void RegisterGPUFusedBatchNormV2(const char* device_type) {
  intel_plugin::RegisterFusedBatchNormV2OpKernel<float, float>(device_type);
  intel_plugin::RegisterFusedBatchNormV2OpKernel<Eigen::bfloat16, float>(
      device_type);
  intel_plugin::RegisterFusedBatchNormV2OpKernel<Eigen::half, float>(
      device_type);
}

void RegisterGPUFusedBatchNormV3(const char* device_type) {
  intel_plugin::RegisterFusedBatchNormV3OpKernel<float, float>(device_type);
  intel_plugin::RegisterFusedBatchNormV3OpKernel<Eigen::bfloat16, float>(
      device_type);
  intel_plugin::RegisterFusedBatchNormV3OpKernel<Eigen::half, float>(
      device_type);
}
