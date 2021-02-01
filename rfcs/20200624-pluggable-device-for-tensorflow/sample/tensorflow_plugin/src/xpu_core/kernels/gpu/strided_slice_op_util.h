#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_STRIDED_SLICE_OP_UTIL_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_STRIDED_SLICE_OP_UTIL_H_

#include "tensorflow_plugin/src/xpu_core/util/gtl/inlined_vector.h"
#include "tensorflow_plugin/src/xpu_core/util/plugin_tensor.h"
#include "tensorflow_plugin/src/xpu_core/util/tensor_shape.h"

namespace intel_plugin {

// Runs validation on the strided slice op parameters.
//
// Is a separate translation unit from the kernel so that:
// 1. The op's shape function can use it.
// 2. The code size is reduced vs templating this on the kernel's type.
//
// Note that when input_shape is not fully specified, only <final_shape> and
// <processing_shape> are valid; <is_identity>, <is_simple_slice> and other
// output parameters will not be accurate.
//
// If <begin_tensor> or <end_tensor> are nullptr, <begin> and <end> will not be
// valid. In this case, <slice_dim0> and <is_identity> will be true only if a
// determination can be made based on the information given. A best effort is
// made to set <processing_shape> and <final_shape> based on <input_shape>, but
// some dimensions of <processing_shape> and/or <final_shape> may be unknown
// (-1). Any validation that can be done without complete information is
// performed.
Status ValidateStridedSliceOp(
    const Tensor* begin_tensor, const Tensor* end_tensor,
    const Tensor& strides_tensor, const PartialTensorShape& input_shape,
    int32 begin_mask_spec, int32 end_mask_spec, const int32 ellipsis_mask,
    int32 new_axis_mask, int32 shrink_axis_mask,
    PartialTensorShape* processing_shape, PartialTensorShape* final_shape,
    bool* is_identity, bool* is_simple_slice, bool* slice_dim0,
    gtl::InlinedVector<int64, 4>* begin, gtl::InlinedVector<int64, 4>* end,
    gtl::InlinedVector<int64, 4>* strides);

// Same as above, but the outputs are TensorShape, not PartialTensorShape
Status ValidateStridedSliceOp(
    const Tensor* begin_tensor, const Tensor* end_tensor,
    const Tensor& strides_tensor, const PartialTensorShape& input_shape,
    int32 begin_mask_spec, int32 end_mask_spec, const int32 ellipsis_mask,
    int32 new_axis_mask, int32 shrink_axis_mask, TensorShape* processing_shape,
    TensorShape* final_shape, bool* is_identity, bool* is_simple_slice,
    bool* slice_dim0, gtl::InlinedVector<int64, 4>* begin,
    gtl::InlinedVector<int64, 4>* end, gtl::InlinedVector<int64, 4>* strides);

}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_STRIDED_SLICE_OP_UTIL_H_