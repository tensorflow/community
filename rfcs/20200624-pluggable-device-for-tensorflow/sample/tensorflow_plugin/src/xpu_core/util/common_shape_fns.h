#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_COMMON_SHAPE_FNS_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_COMMON_SHAPE_FNS_H_

#include "tensorflow_plugin/src/xpu_core/util/padding.h"
#include "tensorflow_plugin/src/xpu_core/util/status.h"

namespace intel_plugin {

// Returns the same output dimensions as in GetWindowedOutputSize, but returns
// verbose padding dimensions (before/after), and EXPLICIT padding is supported.
// When padding_type is EXPLICIT, *padding_before and *padding_after must
// already point to initialized integers with the padding amounts. Otherwise,
// *padding_before and *padding_after are set by this function, and any
// excess padding (caused by an odd padding size value) is added to the
// 'padding_after' dimension.
Status GetWindowedOutputSizeVerbose(int64 input_size, int64 filter_size,
                                    int64 stride, Padding padding_type,
                                    int64* output_size, int64* padding_before,
                                    int64* padding_after);

// The V2 version computes the same outputs with arbitrary dilation_rate. For
// detailed equations, refer to the comments for GetWindowedOutputSizeV2().
Status GetWindowedOutputSizeVerboseV2(int64 input_size, int64 filter_size,
                                      int64 dilation_rate, int64 stride,
                                      Padding padding_type, int64* output_size,
                                      int64* padding_before,
                                      int64* padding_after);

}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_COMMON_SHAPE_FNS_H_
