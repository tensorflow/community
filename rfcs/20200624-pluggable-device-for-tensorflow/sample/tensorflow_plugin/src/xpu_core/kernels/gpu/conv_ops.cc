#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow_plugin/src/xpu_core/kernels/gpu/conv_ops.h"
#include "tensorflow_plugin/src/xpu_core/util/errors.h"
#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/op_requires.h"
#include "tensorflow_plugin/src/xpu_core/util/padding.h"
#include "tensorflow_plugin/src/xpu_core/util/tensor_format.h"

namespace intel_plugin {

using dnnl::algorithm;
using dnnl::convolution_forward;
using dnnl::engine;
using dnnl::memory;
using dnnl::primitive;
using dnnl::prop_kind;
using dnnl::stream;

void ConvOpBase_Create(ConvOpBase* kernel, TF_OpKernelConstruction* ctx) {
  auto op_kernel = static_cast<ConvOpBase*>(kernel);
  OpKernelConstruction context(ctx);
  OP_REQUIRES_OK(&context,
                 context.GetAttr("dilations", &op_kernel->dilations_));
  OP_REQUIRES_OK(&context, context.GetAttr("strides", &op_kernel->strides_));
  string padding_str;
  OP_REQUIRES_OK(&context, context.GetAttr("padding", &padding_str));
  if (padding_str == "VALID") {
    op_kernel->padding_ = Padding::VALID;
  } else if (padding_str == "SAME") {
    op_kernel->padding_ = Padding::SAME;
  } else if (padding_str == "EXPLICIT") {
    op_kernel->padding_ = Padding::EXPLICIT;
  } else {
    OP_REQUIRES(
        &context, false,
        errors::InvalidArgument("Unknown padding type: ", op_kernel->padding_));
  }
  if (context.HasAttr("explicit_paddings")) {
    OP_REQUIRES_OK(&context, context.GetAttr("explicit_paddings",
                                             &op_kernel->explicit_paddings_));
  }
  string data_format_string;
  OP_REQUIRES_OK(&context, context.GetAttr("data_format", &data_format_string));
  OP_REQUIRES(&context,
              FormatFromString(data_format_string, &op_kernel->data_format_),
              errors::InvalidArgument("Invalid data format"));
  OP_REQUIRES(&context, op_kernel->dilations_.size() == 4,
              errors::InvalidArgument("Sliding window dilations field must "
                                      "specify 4 dimensions"));
  OP_REQUIRES(&context, op_kernel->strides_.size() == 4,
              errors::InvalidArgument("Sliding window strides field must "
                                      "specify 4 dimensions"));
  const int64 stride_n =
      GetTensorDim(op_kernel->strides_, op_kernel->data_format_, 'N');
  const int64 stride_c =
      GetTensorDim(op_kernel->strides_, op_kernel->data_format_, 'C');
  const int64 stride_h =
      GetTensorDim(op_kernel->strides_, op_kernel->data_format_, 'H');
  const int64 stride_w =
      GetTensorDim(op_kernel->strides_, op_kernel->data_format_, 'W');
  OP_REQUIRES(
      &context, stride_n == 1 && stride_c == 1,
      errors::InvalidArgument("Current implementation does not yet support "
                              "strides in the batch and depth dimensions."));
  OP_REQUIRES(&context, stride_h > 0 && stride_w > 0,
              errors::InvalidArgument(
                  "Row and column strides should be larger than 0."));

  const int64 dilation_n =
      GetTensorDim(op_kernel->dilations_, op_kernel->data_format_, 'N');
  const int64 dilation_c =
      GetTensorDim(op_kernel->dilations_, op_kernel->data_format_, 'C');
  const int64 dilation_h =
      GetTensorDim(op_kernel->dilations_, op_kernel->data_format_, 'H');
  const int64 dilation_w =
      GetTensorDim(op_kernel->dilations_, op_kernel->data_format_, 'W');
  OP_REQUIRES(
      &context, dilation_n == 1 && dilation_c == 1,
      errors::InvalidArgument("Current implementation does not yet support "
                              "dilations in the batch and depth dimensions."));
  OP_REQUIRES(
      &context, dilation_h > 0 && dilation_w > 0,
      errors::InvalidArgument("Dilated rates should be larger than 0."));

  TF_DCHECK_OK(CheckValidPadding(op_kernel->padding_,
                                 op_kernel->explicit_paddings_,
                                 /*num_dims=*/4, op_kernel->data_format_));

  op_kernel->is_depthwise_ = false;
  /* Fused Conv */
  op_kernel->num_args_ = 0;
  op_kernel->fuse_add_ = false;
  op_kernel->fuse_biasadd_ = false;
  op_kernel->fuse_pad_ = false;
  op_kernel->fuse_activation_ = false;
  op_kernel->relu_up_bound_ = 0.0;
  op_kernel->activation_alg_ = algorithm::undef;
  return;
}

template <typename T>
void ConvOpBase_Compute(void* kernel, TF_OpKernelContext* ctx) {
  OpKernelContext context(ctx);
  try {
    auto op_kernel = static_cast<ConvOpBase*>(kernel);
    auto onednn_engine = CreateDnnlEngine(context);

    auto onednn_stream = CreateDnnlStream(context, onednn_engine);

    const Tensor& src_tensor = context.input(kInputIndex_Src);
    const Tensor& filter_tensor = context.input(kInputIndex_Filter);
    TensorShape src_tensor_shape = src_tensor.shape();
    TensorShape filter_tensor_shape = filter_tensor.shape();

    // Memory dimensions
    memory::dims src_dims, filter_dims, pad_left_dims, pad_right_dims,
        dilation_dims, stride_dims, bias_dims;
    memory::dims dst_dims_tf, dst_dims_onednn;
    std::unordered_map<int, memory> fwd_primitives_args;

    ConvUtil conv_util(&context, op_kernel);

    if (op_kernel->fuse_pad_) {
      const Tensor& pad_tensor = context.input(op_kernel->InputIndex_Pad);
      conv_util.InitPadWithFusion(pad_tensor, &pad_left_dims, &pad_right_dims,
                                  false /*quantized_pad_enabled*/);
    }

    conv_util.InitPostOps();
    conv_util.InitFwdDimensions(
        src_tensor_shape, filter_tensor_shape, &src_dims, &filter_dims,
        &stride_dims, &dilation_dims, &dst_dims_tf, &dst_dims_onednn,
        &pad_left_dims, &pad_right_dims, op_kernel->fuse_pad_);

    // OneDNN dilations start from 0.
    for (int i = 0; i < dilation_dims.size(); ++i) {
      --dilation_dims[i];
    }

    MklTensorFormat data_format_onednn =
        TFDataFormatToMklDnnDataFormat(op_kernel->data_format_);
    memory::format_tag data_layout =
        MklTensorFormatToMklDnnDataFormat(data_format_onednn);
    // TODO(leicongl): using any for filter layout
    auto filter_layout = memory::format_tag::hwio;

    memory::desc src_md =
        memory::desc({src_dims}, MklDnnType<T>(), data_layout);
    memory::desc filter_md =
        memory::desc({filter_dims}, MklDnnType<T>(), filter_layout);
    memory::desc dst_md =
        memory::desc({dst_dims_onednn}, MklDnnType<T>(), data_layout);

    ConvFwdDesc fwd_desc =
        ConvFwdDesc(prop_kind::forward, dnnl::algorithm::convolution_direct,
                    src_md, filter_md, dst_md, stride_dims, dilation_dims,
                    pad_left_dims, pad_right_dims);

    if (op_kernel->fuse_biasadd_) {
      const Tensor& bias_tensor = context.input(kInputIndex_Bias);
      TensorShape bias_tensor_shape = bias_tensor.shape();
      conv_util.GetBiasDimension(bias_tensor_shape, &bias_dims);
      auto bias_md =
          memory::desc(bias_dims, MklDnnType<T>(), memory::format_tag::x);
      void* bias_data = const_cast<void*>(
          static_cast<const void*>(bias_tensor.flat<T>().data()));
      auto bias_mem = CreateDnnlMemory(bias_md, onednn_engine, bias_data);
      fwd_primitives_args.insert({DNNL_ARG_BIAS, bias_mem});
      fwd_desc =
          ConvFwdDesc(prop_kind::forward, dnnl::algorithm::convolution_direct,
                      src_md, filter_md, bias_md, dst_md, stride_dims,
                      dilation_dims, pad_left_dims, pad_right_dims);
    }

    ConvFwdPd fwd_pd = ConvFwdPd(fwd_desc, onednn_engine);

    dnnl::primitive_attr post_ops_attr;
    dnnl::post_ops post_ops;
    if (!op_kernel->post_op_params_.empty()) {
      for (auto const& post_op_param : op_kernel->post_op_params_) {
        if (post_op_param.name == "activation") {
          OP_REQUIRES(&context, post_op_param.param.size() == 3,
                      errors::InvalidArgument(
                          "Activation post op require size of 3, got ",
                          post_op_param.param.size()));
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.append_eltwise(op_scale, post_op_param.alg, op_alpha,
                                  op_beta);
        } else if (post_op_param.name == "sum") {
          OP_REQUIRES(
              &context, post_op_param.param.size() == 1,
              errors::InvalidArgument("Sum post op require size of 1, got ",
                                      post_op_param.param.size()));
          float op_scale = post_op_param.param[0];
          post_ops.append_sum(op_scale);
        } else if (post_op_param.name == "output_scale") {
          if (post_op_param.param.size() == 1) {
            post_ops_attr.set_output_scales(0, post_op_param.param);
          } else {
            post_ops_attr.set_output_scales(2, post_op_param.param);
          }
        } else {
          OP_REQUIRES(
              &context,
              (post_op_param.name == "activation") ||
                  (post_op_param.name == "sum") ||
                  (post_op_param.name == "output_scale"),
              errors::InvalidArgument("Unknown post op: ", post_op_param.name));
        }
      }
      post_ops_attr.set_post_ops(post_ops);
      fwd_pd = ConvFwdPd(fwd_desc, post_ops_attr, onednn_engine);
    }

    primitive fwd_primitive = convolution_forward(fwd_pd);

    // output tensor
    Tensor* dst_tensor = nullptr;
    TensorShape dst_tensor_shape = MklDnnDimsToTFShape(dst_dims_tf);
    OP_REQUIRES_OK(&context, context.allocate_output(
                                 static_cast<const int>(kOutputIndex_Dst),
                                 dst_tensor_shape, &dst_tensor));

    const T* src_data =
        static_cast<T*>(const_cast<T*>(src_tensor.flat<T>().data()));
    T* filter_data =
        static_cast<T*>(const_cast<T*>(filter_tensor.flat<T>().data()));
    T* dst_data = static_cast<T*>(const_cast<T*>(dst_tensor->flat<T>().data()));

    auto src_mem =
        CreateDnnlMemory(fwd_pd.src_desc(), onednn_engine,
                         static_cast<void*>(const_cast<T*>(src_data)));
    auto filter_mem =
        CreateDnnlMemory(fwd_pd.weights_desc(), onednn_engine,
                         static_cast<void*>(const_cast<T*>(filter_data)));
    auto dst_mem =
        CreateDnnlMemory(fwd_pd.dst_desc(), onednn_engine,
                         static_cast<void*>(const_cast<T*>(dst_data)));

    // Execute convolution
    fwd_primitives_args.insert({DNNL_ARG_SRC, src_mem});
    fwd_primitives_args.insert({DNNL_ARG_WEIGHTS, filter_mem});
    fwd_primitives_args.insert({DNNL_ARG_DST, dst_mem});

    fwd_primitive.execute(onednn_stream, fwd_primitives_args);
  } catch (dnnl::error& e) {
    string error_msg = "Status: " + std::to_string(e.status) +
                       ", message: " + string(e.message) + ", in file " +
                       string(__FILE__) + ":" + std::to_string(__LINE__);
    OP_REQUIRES_OK(&context, errors::Aborted("Operation received an exception:",
                                             error_msg));
  }
  return;
}

/* Conv2D */
template <typename T>
struct Conv2DOp : public ConvOpBase {};

template <typename T>
void* Conv2DOp_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel = new Conv2DOp<T>;
  ConvOpBase_Create(kernel, ctx);
  return kernel;
}

template <typename T>
void Conv2DOp_Delete(void* kernel) {
  if (kernel != nullptr) {
    delete static_cast<Conv2DOp<T>*>(kernel);
  }
}

/* FusedConv2D*/
struct FusedConvOp : public ConvOpBase {};

template <typename T>
void FusedConvOp_CreateHelper(FusedConvOp* kernel,
                              TF_OpKernelConstruction* ctx) {
  OpKernelConstruction context(ctx);
  auto op_kernel = static_cast<FusedConvOp*>(kernel);
  OP_REQUIRES_OK(&context,
                 context.GetAttr("fused_ops", &op_kernel->fused_ops_));
  OP_REQUIRES_OK(&context, context.GetAttr("num_args", &op_kernel->num_args_));
  OP_REQUIRES(
      &context, !(op_kernel->fused_ops_.empty()),
      errors::InvalidArgument("Fused Conv2D must have at least one fused op."));

  int num_args_expected = 0;
  for (string fused_op_name : op_kernel->fused_ops_) {
    if (fused_op_name == "Add") {
      num_args_expected += 1;
      op_kernel->fuse_add_ = true;
    } else if (fused_op_name == "BiasAdd") {
      num_args_expected += 1;
      op_kernel->fuse_biasadd_ = true;
    } else if (fused_op_name == "Elu") {
      op_kernel->fuse_activation_ = true;
      op_kernel->activation_alg_ = algorithm::eltwise_elu;
      op_kernel->relu_up_bound_ = kEluUpBound;
    } else if (fused_op_name == "Relu") {
      op_kernel->fuse_activation_ = true;
      op_kernel->activation_alg_ = algorithm::eltwise_relu;
      op_kernel->relu_up_bound_ = kReluUpBound;
    } else if (fused_op_name == "Relu6") {
      op_kernel->fuse_activation_ = true;
      op_kernel->activation_alg_ = algorithm::eltwise_bounded_relu;
      op_kernel->relu_up_bound_ = kRelu6UpBound;
    } else {
      OP_REQUIRES(&context, false,
                  errors::Unimplemented(
                      "Fusion is not implemented: [",
                      absl::StrJoin(op_kernel->fused_ops_, ","), "]"));
    }
  }
  OP_REQUIRES(
      &context, num_args_expected == op_kernel->num_args_,
      errors::InvalidArgument("Expect num_args to be ", num_args_expected,
                              " but received ", op_kernel->num_args_));

  if (kPadEnabled) {
    op_kernel->fuse_pad_ = true;
    op_kernel->InputIndex_Pad = op_kernel->fuse_biasadd_ ? 3 : 2;
  }

  return;
}

template <typename T>
void* FusedConvOp_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel = new FusedConvOp;
  ConvOpBase_Create(kernel, ctx);
  FusedConvOp_CreateHelper<T>(kernel, ctx);
  return kernel;
}

template <typename T>
void FusedConvOp_Delete(void* kernel) {
  if (kernel != nullptr) {
    delete static_cast<FusedConvOp*>(kernel);
  }
}

/* Kernel Registration */
template <typename T>
void RegisterConv2DOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  auto* builder =
      TF_NewKernelBuilder("Conv2D", device_type, &Conv2DOp_Create<T>,
                          &ConvOpBase_Compute<T>, &Conv2DOp_Delete<T>);
  TF_KernelBuilder_TypeConstraint(
      builder, "T",
      static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
      status.get());
  CHECK_EQ(TF_OK, TF_GetCode(status.get()))
      << " Error while registering Conv ops kernel with attribute T";
  TF_RegisterKernelBuilder("Conv2D", builder, status.get());
  CHECK_EQ(TF_OK, TF_GetCode(status.get()))
      << " Error while registering Conv kernel";
}

template <typename T>

void RegisterFusedConv2DOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  auto* builder =
      TF_NewKernelBuilder("_FusedConv2D", device_type, &FusedConvOp_Create<T>,
                          &ConvOpBase_Compute<T>, &FusedConvOp_Delete<T>);
  TF_KernelBuilder_TypeConstraint(
      builder, "T",
      static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
      status.get());
  CHECK_EQ(TF_OK, TF_GetCode(status.get()))
      << " Error while registering _FusedConv2D kernel with attribute T";
  TF_RegisterKernelBuilder("_FusedConv2D", builder, status.get());
  CHECK_EQ(TF_OK, TF_GetCode(status.get()))
      << " Error while registering _FusedConv2D kernel";
}

}  // namespace intel_plugin

void RegisterGPUConvOps(const char* device_type) {
  intel_plugin::RegisterConv2DOpKernel<float>(device_type);
  intel_plugin::RegisterConv2DOpKernel<Eigen::bfloat16>(device_type);
  intel_plugin::RegisterConv2DOpKernel<Eigen::half>(device_type);

  intel_plugin::RegisterFusedConv2DOpKernel<float>(device_type);
  intel_plugin::RegisterFusedConv2DOpKernel<Eigen::bfloat16>(device_type);
  intel_plugin::RegisterFusedConv2DOpKernel<Eigen::half>(device_type);
}
