#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_PADDING_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_PADDING_H_

#include "tensorflow_plugin/src/xpu_core/util/status.h"
#include "tensorflow_plugin/src/xpu_core/util/tensor_format.h"

namespace intel_plugin {
// Padding: the padding we apply to the input tensor along the rows and columns
// dimensions. This is usually used to make sure that the spatial dimensions do
// not shrink when we progress with convolutions. Three types of padding are
// supported:
//   VALID: No padding is carried out.
//   SAME: The pad value is computed so that the output will have the same
//         dimensions as the input.
//   EXPLICIT: The user specifies the pad values in the explicit_paddings
//             attribute.
// The padded area is typically zero-filled. For pooling ops, the padded area is
// instead ignored. For max pool, this is equivalent to padding with -infinity.
enum Padding {
  VALID = 1,     // No padding.
  SAME = 2,      // Input and output layers have the same size.
  EXPLICIT = 3,  // Padding is explicitly specified
};

// Return the string containing the list of valid padding types, that can be
// used as an Attr() in REGISTER_OP.
std::string GetPaddingAttrString();

// Like GetPaddingAttrString(), but also includes EXPLICIT.
std::string GetPaddingAttrStringWithExplicit();

std::string GetExplicitPaddingsAttrString();

// Sets padding value based on the given string padding value.
Status GetPaddingFromString(StringPiece str_value, Padding* value);

}  // end namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_PADDING_H_
