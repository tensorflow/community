
#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_REDUCTION_OPS_COMMON_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_REDUCTION_OPS_COMMON_H_

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"

#include "tensorflow_plugin/src/xpu_core/kernels/gpu/reduction_ops.h"
#include "tensorflow_plugin/src/xpu_core/kernels/gpu/transpose_functor.h"
#include "tensorflow_plugin/src/xpu_core/util/allocator.h"
#include "tensorflow_plugin/src/xpu_core/util/gtl/inlined_vector.h"
#include "tensorflow_plugin/src/xpu_core/util/logging.h"
#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/op_requires.h"
#include "tensorflow_plugin/src/xpu_core/util/plugin_tensor.h"
#include "tensorflow_plugin/src/xpu_core/util/status.h"
#include "tensorflow_plugin/src/xpu_core/util/tensor_shape.h"
#include "tensorflow_plugin/src/xpu_core/util/tensor_types.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace intel_plugin {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device>
struct Constants {
  // Derive Index type. int (32-bit) or long (64-bit) depending on the
  // compile-time configuration. "float" here is not relevant.
  typedef TTypes<float>::Tensor::Index Index;
  Eigen::array<Index, 1> kZero;
  Eigen::array<Index, 1> kOne;
  Eigen::array<Index, 2> kZeroTwo;

  Constants() {
    kZero[0] = 0;
    kOne[0] = 1;
    kZeroTwo[0] = 0;
    kZeroTwo[1] = 2;
  }
};

class ReductionHelper {
 public:
  ReductionHelper() : reduce_first_axis_(false) {}

  Status Simplify(const Tensor& data, const Tensor& axis, const bool keep_dims);

  // We need to do roughly:
  //   tmp_out = allocate(out_reshape())
  //   tmp_out.reshape(out_reshape) = data.reshape(data_reshape).reduce(axes)
  //   out = tmp_out.reshape(out_shape)

  // The reduction result must be allocated with this shape.
  TensorShape out_reshape() const;

  // The final output shape must be allocated with this shape.
  TensorShape out_shape() const;

  // The reduction is on a reshaped tensor of this rank.
  int ndims() const { return data_reshape_.size(); }

  // True if need to reduce the 0-th dimension.
  bool reduce_first_axis() const { return reduce_first_axis_; }

  // The output is reshaped.
  template <typename T, int N>
  typename TTypes<T, N>::Tensor out(Tensor* out) {
    return out->shaped<T, N>(out_reshape_);
  }

  // The input is reshaped.
  template <typename T, int N>
  typename TTypes<T, N>::ConstTensor in(const Tensor& data) {
    return data.shaped<T, N>(data_reshape_);
  }

  // Shape of shuffled input
  TensorShape data_reshape() const {
    TensorShape shape;
    for (auto s : data_reshape_) shape.AddDim(s);
    return shape;
  }

  // Shape with all reduction dimensions at the end
  TensorShape shuffled_shape();

  // Permutation of reduced dims needed to put reduction dimensions at the end
  gtl::InlinedVector<int32, 8> permutation();

 private:
  bool reduce_first_axis_;  // True if need to reduce the 0-th dimension.
  gtl::InlinedVector<int64, 4> data_reshape_;  // Reshape data before reduction.
  gtl::InlinedVector<int64, 4> out_shape_;     // The final output shape.
  gtl::InlinedVector<int64, 4> out_reshape_;   // Reshape output for reduction.
};

struct ReductionOp {
  // True if the number of dimensions should be maintained.
  bool keep_dims_;
};

template <typename T, typename Tidx>
void* ReductionOp_Create(TF_OpKernelConstruction* construction) {
  ReductionOp* kernel = new ReductionOp;
  // TODO(Maozhou): same as UnaryOp_Create()
  //  const DataType dt = DataTypeToEnum<T>::v();
  //  const DataType pt = DataTypeToEnum<Tidx>::v();
  //  OP_REQUIRES_OK(&ctx, ctx.MatchSignature({dt, pt}, {dt}));
  OpKernelConstruction ctx(construction);
  OP_REQUIRES_OK_PTR(&ctx, ctx.GetAttr("keep_dims", &kernel->keep_dims_));
  return kernel;
}

template <typename Dummy = void>
void ReductionOp_Delete(void* kernel) {
  if (kernel) {
    delete static_cast<ReductionOp*>(kernel);
  }
}

template <typename T, typename Reducer>
void ReductionOp_Compute(void* kernel, TF_OpKernelContext* context) {
  OpKernelContext ctx(context);
  auto reduction_op = static_cast<ReductionOp*>(kernel);

  const Tensor& data = ctx.input(0);
  const Tensor& axes = ctx.input(1);
  VLOG(1) << "data shape: " << data.shape().DebugString();

  ReductionHelper helper;
  OP_REQUIRES_OK(&ctx, helper.Simplify(data, axes, reduction_op->keep_dims_));
  CHECK_GE(helper.ndims(), 0);

  if (helper.ndims() == 0 ||
      (helper.ndims() == 1 && !helper.reduce_first_axis())) {
    // Special case. Reduces nothing.  It is unclear why this is
    // necessary, but tests fail without it.  Look into why this
    // case occurs.
    Tensor out;
    if (!out.CopyFrom(data, helper.out_shape()).ok()) {
      ctx.SetStatus(errors::Internal("Error during reduction copy."));
    }
    ctx.set_output(0, out);
    return;
  }

  // A temporary tensor whose size matches the size of the reduced
  // output.
  Tensor tmp_out;
  OP_REQUIRES_OK(&ctx, ctx.allocate_temp(ctx.expected_output_dtype(0),
                                         helper.out_reshape(), &tmp_out));

  typedef functor::ReduceFunctor<Reducer> Functor;
  Constants<GPUDevice> constants;
  Reducer reducer;

  if (tmp_out.NumElements() == 0) {
    // Nothing to do, fall through to final reshaping.
  } else if (data.NumElements() == 0) {
    // Degenerate reduction where the input is empty but the output is
    // nonempty (thus tmp_out.NumElements() > 0), and we must fill the output
    // with identity elements.  Example: tf.reduce_sum(tf.zeros((0, 3)), [0]).
    // Eigen sometimes crashes in this case, so we do it manually.
    Functor::FillIdentity(ctx.eigen_gpu_device(), tmp_out.flat<T>(), reducer);
  } else if ((helper.ndims() == 1) && helper.reduce_first_axis()) {
    // Reduce to a scalar.
    Functor::Reduce(&ctx, helper.out<T, 0>(&tmp_out), helper.in<T, 1>(data),
                    constants.kZero, reducer);
  } else if ((helper.ndims() == 2) && helper.reduce_first_axis()) {
    // Can be viewed as a reduction of a matrix along 1st dimension.
    Functor::Reduce(&ctx, helper.out<T, 1>(&tmp_out), helper.in<T, 2>(data),
                    constants.kZero, reducer);
  } else if ((helper.ndims() == 2) && !helper.reduce_first_axis()) {
    // Can be viewed as a reduction of a matrix along 2nd dimension.
    Functor::Reduce(&ctx, helper.out<T, 1>(&tmp_out), helper.in<T, 2>(data),
                    constants.kOne, reducer);
  } else if ((helper.ndims() == 3) && helper.reduce_first_axis()) {
    // Can be viewed as a reduction of a 3D tensor along 1st and 3rd
    // dimensions.
    Functor::Reduce(&ctx, helper.out<T, 1>(&tmp_out), helper.in<T, 3>(data),
                    constants.kZeroTwo, reducer);
  } else if ((helper.ndims() == 3) && !helper.reduce_first_axis()) {
    // Can be viewed as a reduction of a 3D tensor along 2nd dimension.
    Functor::Reduce(&ctx, helper.out<T, 2>(&tmp_out), helper.in<T, 3>(data),
                    constants.kOne, reducer);
  } else {
    // If we don't hit one of the cases above, transpose the data so that
    // all reduced dimensions are last and reuse the 2-D -> 1-D case.
    Tensor data_reshaped;
    CHECK(data_reshaped.CopyFrom(data, helper.data_reshape()).ok());
    Tensor shuffled;
    OP_REQUIRES_OK(&ctx, ctx.allocate_temp(DataTypeToEnum<T>::value,
                                           helper.shuffled_shape(), &shuffled));
    OP_REQUIRES_OK(
        &ctx, DoTranspose<T, false>(&ctx, data_reshaped, helper.permutation(),
                                    &shuffled));
    const int64 unreduced = tmp_out.NumElements();
    const int64 reduced = shuffled.NumElements() / unreduced;
    const Tensor& const_shuffled = shuffled;
    Functor::Reduce(&ctx, tmp_out.flat<T>(),
                    const_shuffled.shaped<T, 2>({unreduced, reduced}),
                    constants.kOne, reducer);
  }

  // Set the real output using the contents of the reduction but the
  // real expected output shape.  The number of elements should
  // match between the two shapes.
  Tensor out;
  if (!out.CopyFrom(tmp_out, helper.out_shape()).ok()) {
    ctx.SetStatus(errors::Internal("Error during reduction copy."));
  }
  ctx.set_output(0, out);
}
}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_REDUCTION_OPS_COMMON_H_
