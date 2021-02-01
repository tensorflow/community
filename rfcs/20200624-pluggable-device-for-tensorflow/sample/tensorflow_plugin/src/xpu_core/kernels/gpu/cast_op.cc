#include "tensorflow_plugin/src/xpu_core/kernels/gpu/cast_op.h"
#include "tensorflow_plugin/src/xpu_core/kernels/gpu/cast_op_impl.h"
#include "tensorflow_plugin/src/xpu_core/util/dnnl_util.h"
#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/plugin_tensor.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"

namespace intel_plugin {
extern CastFunctorType GetGpuCastFromBool(DataType);
extern CastFunctorType GetGpuCastFromUint8(DataType);
extern CastFunctorType GetGpuCastFromUint16(DataType);
extern CastFunctorType GetGpuCastFromInt8(DataType);
extern CastFunctorType GetGpuCastFromUint32(DataType);
extern CastFunctorType GetGpuCastFromUint64(DataType);
extern CastFunctorType GetGpuCastFromInt16(DataType);
extern CastFunctorType GetGpuCastFromInt32(DataType);
extern CastFunctorType GetGpuCastFromInt64(DataType);
extern CastFunctorType GetGpuCastFromHalf(DataType);
extern CastFunctorType GetGpuCastFromFloat(DataType);
extern CastFunctorType GetGpuCastFromBfloat(DataType);

typedef Eigen::GpuDevice GPUDevice;
CAST_FUNCTORS(GPUDevice);

#define DEFINE(O, I) template struct CastFunctor<GPUDevice, O, I>

#define DEFINE_ALL_FROM(in_type) \
  DEFINE(in_type, bool);         \
  DEFINE(in_type, uint8);        \
  DEFINE(in_type, uint16);       \
  DEFINE(in_type, uint32);       \
  DEFINE(in_type, uint64);       \
  DEFINE(in_type, int8);         \
  DEFINE(in_type, int16);        \
  DEFINE(in_type, int32);        \
  DEFINE(in_type, int64);        \
  DEFINE(in_type, Eigen::half);  \
  DEFINE(in_type, float);

DEFINE_ALL_FROM(bool);
DEFINE_ALL_FROM(uint8);
DEFINE_ALL_FROM(uint16);
DEFINE_ALL_FROM(uint32);
DEFINE_ALL_FROM(uint64);
DEFINE_ALL_FROM(int8);
DEFINE_ALL_FROM(int16);
DEFINE_ALL_FROM(int32);
DEFINE_ALL_FROM(int64);
DEFINE(float, Eigen::bfloat16);

#define DEFINE_ALL_TO_FLOAT(out_type) \
  DEFINE(out_type, bool);             \
  DEFINE(out_type, uint8);            \
  DEFINE(out_type, uint16);           \
  DEFINE(out_type, uint32);           \
  DEFINE(out_type, uint64);           \
  DEFINE(out_type, int8);             \
  DEFINE(out_type, int16);            \
  DEFINE(out_type, int32);            \
  DEFINE(out_type, int64);            \
  DEFINE(out_type, Eigen::half);      \
  DEFINE(out_type, float);

#define DEFINE_ALL_TO_HALF(out_type) \
  DEFINE(out_type, bool);            \
  DEFINE(out_type, uint8);           \
  DEFINE(out_type, uint16);          \
  DEFINE(out_type, uint32);          \
  DEFINE(out_type, uint64);          \
  DEFINE(out_type, int8);            \
  DEFINE(out_type, int16);           \
  DEFINE(out_type, int32);           \
  DEFINE(out_type, int64);           \
  DEFINE(out_type, Eigen::half)

DEFINE_ALL_TO_HALF(Eigen::half);
DEFINE_ALL_TO_HALF(Eigen::bfloat16);
DEFINE_ALL_TO_FLOAT(float);

#undef DEFINE_ALL_TO_FLOAT
#undef DEFINE_ALL_TO_HALF
#undef DEFINE_ALL_FROM
#undef DEFINE

Status GpuCastOp::Unimplemented() {
  return errors::Unimplemented("Cast ", DataTypeString(external_src_dtype),
                               " to ", DataTypeString(external_dst_dtype),
                               " is not supported");
}

Status GpuCastOp::Prepare() {
  // TODO(schen2): Support quantize types.
  if (external_src_dtype != src_dtype || external_dst_dtype != dst_dtype) {
    Unimplemented();
  } else if (external_src_dtype == external_dst_dtype) {
    cast_work = nullptr;  // Identity
    return Status::OK();
  }
  if (src_dtype == DT_BOOL) {
    cast_work = GetGpuCastFromBool(dst_dtype);
  } else if (src_dtype == DT_UINT8) {
    cast_work = GetGpuCastFromUint8(dst_dtype);
  } else if (src_dtype == DT_UINT16) {
    cast_work = GetGpuCastFromUint16(dst_dtype);
  } else if (src_dtype == DT_UINT32) {
    cast_work = GetGpuCastFromUint32(dst_dtype);
  } else if (src_dtype == DT_UINT64) {
    cast_work = GetGpuCastFromUint64(dst_dtype);
  } else if (src_dtype == DT_INT8) {
    cast_work = GetGpuCastFromInt8(dst_dtype);
  } else if (src_dtype == DT_INT16) {
    cast_work = GetGpuCastFromInt16(dst_dtype);
  } else if (src_dtype == DT_INT32) {
    cast_work = GetGpuCastFromInt32(dst_dtype);
  } else if (src_dtype == DT_INT64) {
    cast_work = GetGpuCastFromInt64(dst_dtype);
  } else if (src_dtype == DT_HALF) {
    cast_work = GetGpuCastFromHalf(dst_dtype);
  } else if (src_dtype == DT_FLOAT) {
    cast_work = GetGpuCastFromFloat(dst_dtype);
  } else if (src_dtype == DT_BFLOAT16) {
    cast_work = GetGpuCastFromBfloat(dst_dtype);
  }
  return cast_work == nullptr ? Unimplemented() : Status::OK();
}

void* GpuCastOp_Create(TF_OpKernelConstruction* ctx) {
  OpKernelConstruction context(ctx);
  auto* kernel = new GpuCastOp;
  OP_REQUIRES_OK_PTR(&context,
                     context.GetAttr("SrcT", &kernel->external_src_dtype));
  OP_REQUIRES_OK_PTR(&context,
                     context.GetAttr("DstT", &kernel->external_dst_dtype));
  OP_REQUIRES_OK_PTR(&context,
                     context.GetAttr("Truncate", &kernel->use_truncation));

  // Quantized data types use the same underlying format as their non quantized
  // version so we use the non quantized implementation for casting.
  if (kernel->external_dst_dtype == DT_QUINT8) {
    kernel->dst_dtype = DT_UINT8;
  } else if (kernel->external_dst_dtype == DT_QINT8) {
    kernel->dst_dtype = DT_INT8;
  } else if (kernel->external_dst_dtype == DT_QINT32) {
    kernel->dst_dtype = DT_INT32;
  } else if (kernel->external_dst_dtype == DT_QINT16) {
    kernel->dst_dtype = DT_INT16;
  } else if (kernel->external_dst_dtype == DT_QUINT16) {
    kernel->dst_dtype = DT_UINT16;
  } else {
    kernel->dst_dtype = kernel->external_dst_dtype;
  }

  if (kernel->external_src_dtype == DT_QUINT8) {
    kernel->src_dtype = DT_UINT8;
  } else if (kernel->external_src_dtype == DT_QINT8) {
    kernel->src_dtype = DT_INT8;
  } else if (kernel->external_src_dtype == DT_QINT32) {
    kernel->src_dtype = DT_INT32;
  } else if (kernel->external_src_dtype == DT_QINT16) {
    kernel->src_dtype = DT_INT16;
  } else if (kernel->external_src_dtype == DT_QUINT16) {
    kernel->src_dtype = DT_UINT16;
  } else {
    kernel->src_dtype = kernel->external_src_dtype;
  }

  OP_REQUIRES_OK_PTR(&context, kernel->Prepare());
  return kernel;
}

void GpuCastOp_Delete(void* kernel) {
  if (kernel) {
    delete static_cast<GpuCastOp*>(kernel);
  }
}

void GpuCastOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  OpKernelContext context(ctx);
  auto opKernel = static_cast<GpuCastOp*>(kernel);
  const Tensor& inp = context.input(0);
  if (opKernel->cast_work == nullptr) {
    context.set_output(0, inp);
  } else {
    Tensor* out = nullptr;
    OP_REQUIRES_OK(&context, context.allocate_output(0, inp.shape(), &out));
    if (inp.NumElements() > 0) {
      opKernel->cast_work(context, inp, out, opKernel->use_truncation);
    }
  }
}

template <typename srctype, typename dsttype>
void RegisterGpuCastOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  auto* builder = TF_NewKernelBuilder("Cast", device_type, &GpuCastOp_Create,
                                      &GpuCastOp_Compute, &GpuCastOp_Delete);
  TF_KernelBuilder_TypeConstraint(
      builder, "SrcT",
      static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<srctype>::v()),
      status.get());
  CHECK_EQ(TF_OK, TF_GetCode(status.get()))
      << " Error while registering Cast kernel with attribute SrcT";
  TF_KernelBuilder_TypeConstraint(
      builder, "DstT",
      static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<dsttype>::v()),
      status.get());
  CHECK_EQ(TF_OK, TF_GetCode(status.get()))
      << " Error while registering Cast kernel with attribute DstT";
  TF_RegisterKernelBuilder("Cast", builder, status.get());
  CHECK_EQ(TF_OK, TF_GetCode(status.get()))
      << " Error while registering Cast kernel";
}

}  // namespace intel_plugin

void RegisterGPUCast(const char* device_type) {
  REGISTER_GPU_CAST_OP_KERNEL(bool, device_type);
  REGISTER_GPU_CAST_OP_KERNEL(intel_plugin::uint8, device_type);
  REGISTER_GPU_CAST_OP_KERNEL(intel_plugin::uint16, device_type);
  REGISTER_GPU_CAST_OP_KERNEL(intel_plugin::uint32, device_type);
  REGISTER_GPU_CAST_OP_KERNEL(intel_plugin::uint64, device_type);
  REGISTER_GPU_CAST_OP_KERNEL(intel_plugin::int8, device_type);
  REGISTER_GPU_CAST_OP_KERNEL(intel_plugin::int16, device_type);
  REGISTER_GPU_CAST_OP_KERNEL(intel_plugin::int32, device_type);
  REGISTER_GPU_CAST_OP_KERNEL(intel_plugin::int64, device_type);
  REGISTER_GPU_CAST_OP_KERNEL(Eigen::half, device_type);
  REGISTER_GPU_CAST_OP_KERNEL(float, device_type);
  intel_plugin::RegisterGpuCastOpKernel<float, Eigen::bfloat16>(device_type);
  intel_plugin::RegisterGpuCastOpKernel<Eigen::bfloat16, float>(device_type);
}
