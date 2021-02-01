#include "tensorflow_plugin/src/xpu_core/kernels/gpu/cwise_ops_common.h"

namespace intel_plugin {
void SetUnimplementedError(OpKernelContext* context) {
  context->SetStatus(errors::Unimplemented(
      "Broadcast between ", context->input(0).shape().DebugString(), " and ",
      context->input(1).shape().DebugString(), " is not supported yet."));
}

void SetComputeError(OpKernelContext* context, const string& op) {
  // For speed, errors during compute are caught only via boolean flag, with no
  // associated information.  This is sufficient for now, since the only binary
  // ops that have compute errors are integer division and mod, and the only
  // error they produce is zero division.
  if ((op == "Div" || op == "Mod" || op == "FloorMod" || op == "FloorDiv") &&
      DataTypeIsInteger(context->input_dtype(0))) {
    context->CtxFailure(errors::InvalidArgument("Integer division by zero"));
  } else if ((op == "Pow") && DataTypeIsInteger(context->input_dtype(0)) &&
             DataTypeIsSigned(context->input_dtype(1))) {
    context->CtxFailure(errors::InvalidArgument(
        "Integers to negative integer powers are not allowed"));
  } else {
    context->CtxFailure(
        errors::Internal("Unexpected error in binary operator "
                         "(only integer div and mod should have errors)"));
  }
}

BinaryOpState::BinaryOpState(OpKernelContext* ctx, BinaryOp* kernel)
    : in0(ctx->input(0)),
      in1(ctx->input(1)),
      bcast(BCast::FromShape(in0.shape()), BCast::FromShape(in1.shape())) {
  if (!bcast.IsValid()) {
    if (kernel->has_attr && !kernel->incompatible_shape_error) {
      const string& op = kernel->op;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
      result = (op == "NotEqual");
      return;
    }

    ctx->SetStatus(errors::InvalidArgument(
        "Incompatible shapes: ", in0.shape().DebugString(), " vs. ",
        in1.shape().DebugString()));
    return;
  }

  const TensorShape output_shape = BCast::ToShape(bcast.output_shape());
  out_num_elements = output_shape.num_elements();
  in0_num_elements = in0.NumElements();
  in1_num_elements = in1.NumElements();
  OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                          {0, 1}, 0, output_shape, &out));

  ndims = static_cast<int>(bcast.x_reshape().size());
}

void UnaryOp_Delete(void* kernel) {
  if (kernel) delete static_cast<UnaryOp*>(kernel);
}

void* BinaryOp_Create(TF_OpKernelConstruction* ctx) {
  BinaryOp* kernel = new BinaryOp;
  OpKernelConstruction context(ctx);
  kernel->op = context.OpName();
  kernel->has_attr = context.HasAttr("incompatible_shape_error");
  kernel->incompatible_shape_error = false;
  if (kernel->has_attr) {
    context.GetAttr("incompatible_shape_error",
                    &(kernel->incompatible_shape_error));
  }
  return kernel;
}

void BinaryOp_Delete(void* kernel) {
  if (kernel) delete static_cast<BinaryOp*>(kernel);
}

void* SimpleBinaryOp_Create(TF_OpKernelConstruction* ctx) {
  SimpleBinaryOp* kernel = new SimpleBinaryOp;
  return kernel;
}

void SimpleBinaryOp_Delete(void* kernel) {
  if (kernel) delete static_cast<SimpleBinaryOp*>(kernel);
}

void* DnnBinaryOp_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel = new DnnBinaryOp;
  return kernel;
}

void DnnBinaryOp_Delete(void* kernel) {
  if (kernel) delete static_cast<DnnBinaryOp*>(kernel);
}

// Expand dimension size to `max_dim`, the new dimensions will be added to
// left: 4 -- > 1x4.
// It follows Numpy broadcast rule:
// http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
void DnnBinaryOp_ExpandDim(dnnl::memory::dims& dims, int max_dim) {
  if (dims.size() >= max_dim) return;

  int start_offset = max_dim - dims.size();
  std::vector<int> expanded(max_dim, 1);
  for (int i = 0; i < dims.size(); i++) {
    expanded[start_offset + i] = dims[i];
  }

  dims.resize(max_dim);
  for (int i = 0; i < max_dim; i++) {
    dims[i] = expanded[i];
  }
}

bool DnnBinaryOp_UnsupportShape(const TensorShape& shape0,
                                const TensorShape& shape1) {
  // Bi-bcast like 8x1 * 1x4 isn't supported in oneDNN. Compare output
  // shape(8x4) with input shapes, and fall back to Eigen if output has more
  // elements than all inputs.
  int64 dst_elements = 1;
  TensorShape l = shape0.dims() > shape1.dims() ? shape0 : shape1;
  TensorShape s = shape0.dims() > shape1.dims() ? shape1 : shape0;
  int gap = l.dims() - s.dims();
  for (int i = 0; i < gap; ++i) dst_elements *= l.dim_size(i);
  for (int i = 0; i < s.dims(); ++i)
    dst_elements *= std::max(s.dim_size(i), l.dim_size(i + gap));

  if (dst_elements > shape0.num_elements() &&
      dst_elements > shape1.num_elements())
    return true;

  // Eigen will fill specific shape to output when **the** input shape is 0,
  // oneDNN does not handle this case.
  if (shape0.num_elements() == 0 || shape1.num_elements() == 0) return true;

  // Currently oneDnn can only support up to 5 dimensions.
  if (shape0.dims() > 5 || shape1.dims() > 5) return true;

  return false;
}
}  // namespace intel_plugin
