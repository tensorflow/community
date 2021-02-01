#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_REGISTER_TYPES_H
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_REGISTER_TYPES_H
// This file is used by cuda code and must remain compilable by nvcc.

#include "tensorflow_plugin/src/xpu_core/util/numeric_types.h"
//#include "tensorflow/core/framework/resource_handle.h"
//#include "tensorflow/core/framework/variant.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"

// Two sets of macros:
// - TF_CALL_float, TF_CALL_double, etc. which call the given macro with
//   the type name as the only parameter - except on platforms for which
//   the type should not be included.
// - Macros to apply another macro to lists of supported types. These also call
//   into TF_CALL_float, TF_CALL_double, etc. so they filter by target platform
//   as well.
// If you change the lists of types, please also update the list in types.cc.
//
// See example uses of these macros in core/ops.
//
//
// Each of these TF_CALL_XXX_TYPES(m) macros invokes the macro "m" multiple
// times by passing each invocation a data type supported by TensorFlow.
//
// The different variations pass different subsets of the types.
// TF_CALL_ALL_TYPES(m) applied "m" to all types supported by TensorFlow.
// The set of types depends on the compilation platform.
//.
// This can be used to register a different template instantiation of
// an OpKernel for different signatures, e.g.:
/*
   #define REGISTER_PARTITION(type)                                      \
     REGISTER_KERNEL_BUILDER(                                            \
         Name("Partition").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
         PartitionOp<type>);
   TF_CALL_ALL_TYPES(REGISTER_PARTITION)
   #undef REGISTER_PARTITION
*/

#if !defined(IS_MOBILE_PLATFORM) || defined(SUPPORT_SELECTIVE_REGISTRATION) || \
    defined(ANDROID_TEGRA)

// All types are supported, so all macros are invoked.
//
// Note: macros are defined in same order as types in types.proto, for
// readability.
#define TF_CALL_float(m) m(float)
#define TF_CALL_double(m) m(double)
#define TF_CALL_int32(m) m(::intel_plugin::int32)
#define TF_CALL_uint32(m) m(::intel_plugin::uint32)
#define TF_CALL_uint8(m) m(::intel_plugin::uint8)
#define TF_CALL_int16(m) m(::intel_plugin::int16)

#define TF_CALL_int8(m) m(::intel_plugin::int8)
#define TF_CALL_string(m) m(::intel_plugin::tstring)
#define TF_CALL_tstring(m) m(::intel_plugin::tstring)
#define TF_CALL_resource(m)   // m(::intel_plugin::ResourceHandle)
#define TF_CALL_variant(m)    // m(::intel_plugin::Variant)
#define TF_CALL_complex64(m)  // m(::intel_plugin::complex64)
#define TF_CALL_int64(m) m(::intel_plugin::int64)
#define TF_CALL_uint64(m) m(::intel_plugin::uint64)
#define TF_CALL_bool(m) m(bool)

#define TF_CALL_qint8(m)   // m(::intel_plugin::qint8)
#define TF_CALL_quint8(m)  // m(::intel_plugin::quint8)
#define TF_CALL_qint32(m)  // m(::intel_plugin::qint32)
#define TF_CALL_bfloat16(m) m(Eigen::bfloat16)
#define TF_CALL_qint16(m)  // m(::intel_plugin::qint16)

#define TF_CALL_quint16(m)  // m(::intel_plugin::quint16)
#define TF_CALL_uint16(m) m(::intel_plugin::uint16)
#define TF_CALL_complex128(m)  // m(::intel_plugin::complex128)
#define TF_CALL_half(m) m(Eigen::half)

#elif defined(__ANDROID_TYPES_FULL__)

// Only string, half, float, int32, int64, bool, and quantized types
// supported.
#define TF_CALL_float(m) m(float)
#define TF_CALL_double(m)
#define TF_CALL_int32(m) m(::intel_plugin::int32)
#define TF_CALL_uint32(m)
#define TF_CALL_uint8(m)
#define TF_CALL_int16(m)

#define TF_CALL_int8(m)
#define TF_CALL_string(m) m(::intel_plugin::tstring)
#define TF_CALL_tstring(m) m(::intel_plugin::tstring)
#define TF_CALL_resource(m)
#define TF_CALL_variant(m)
#define TF_CALL_complex64(m)
#define TF_CALL_int64(m) m(::intel_plugin::int64)
#define TF_CALL_uint64(m)
#define TF_CALL_bool(m) m(bool)

#define TF_CALL_qint8(m)   // m(::intel_plugin::qint8)
#define TF_CALL_quint8(m)  // m(::intel_plugin::quint8)
#define TF_CALL_qint32(m)  // m(::intel_plugin::qint32)
#define TF_CALL_bfloat16(m)
#define TF_CALL_qint16(m)  // m(::intel_plugin::qint16)

#define TF_CALL_quint16(m)  // m(::intel_plugin::quint16)
#define TF_CALL_uint16(m)
#define TF_CALL_complex128(m)
#define TF_CALL_half(m) m(Eigen::half)

#else  // defined(IS_MOBILE_PLATFORM) && !defined(__ANDROID_TYPES_FULL__)

// Only float, int32, and bool are supported.
#define TF_CALL_float(m) m(float)
#define TF_CALL_double(m)
#define TF_CALL_int32(m) m(::intel_plugin::int32)
#define TF_CALL_uint32(m)
#define TF_CALL_uint8(m)
#define TF_CALL_int16(m)

#define TF_CALL_int8(m)
#define TF_CALL_string(m)
#define TF_CALL_tstring(m)
#define TF_CALL_resource(m)
#define TF_CALL_variant(m)
#define TF_CALL_complex64(m)
#define TF_CALL_int64(m)
#define TF_CALL_uint64(m)
#define TF_CALL_bool(m) m(bool)

#define TF_CALL_qint8(m)
#define TF_CALL_quint8(m)
#define TF_CALL_qint32(m)
#define TF_CALL_bfloat16(m)
#define TF_CALL_qint16(m)

#define TF_CALL_quint16(m)
#define TF_CALL_uint16(m)
#define TF_CALL_complex128(m)
#define TF_CALL_half(m)

#endif  // defined(IS_MOBILE_PLATFORM)  - end of TF_CALL_type defines

// Defines for sets of types.
#define TF_CALL_INTEGRAL_TYPES(m)                                       \
  TF_CALL_uint64(m) TF_CALL_int64(m) TF_CALL_uint32(m) TF_CALL_int32(m) \
      TF_CALL_uint16(m) TF_CALL_int16(m) TF_CALL_uint8(m) TF_CALL_int8(m)

#define TF_CALL_FLOAT_TYPES(m) \
  TF_CALL_half(m) TF_CALL_bfloat16(m) TF_CALL_float(m) TF_CALL_double(m)

#define TF_CALL_REAL_NUMBER_TYPES(m) \
  TF_CALL_INTEGRAL_TYPES(m)          \
  TF_CALL_FLOAT_TYPES(m)

#define TF_CALL_REAL_NUMBER_TYPES_NO_BFLOAT16(m) \
  TF_CALL_INTEGRAL_TYPES(m) TF_CALL_half(m) TF_CALL_float(m) TF_CALL_double(m)

#define TF_CALL_REAL_NUMBER_TYPES_NO_INT32(m)                                \
  TF_CALL_half(m) TF_CALL_bfloat16(m) TF_CALL_float(m) TF_CALL_double(m)     \
      TF_CALL_uint64(m) TF_CALL_int64(m) TF_CALL_uint32(m) TF_CALL_uint16(m) \
          TF_CALL_int16(m) TF_CALL_uint8(m) TF_CALL_int8(m)

#define TF_CALL_COMPLEX_TYPES(m) TF_CALL_complex64(m) TF_CALL_complex128(m)

// Call "m" for all number types, including complex types
#define TF_CALL_NUMBER_TYPES(m) \
  TF_CALL_REAL_NUMBER_TYPES(m) TF_CALL_COMPLEX_TYPES(m)

#define TF_CALL_NUMBER_TYPES_NO_INT32(m) \
  TF_CALL_REAL_NUMBER_TYPES_NO_INT32(m) TF_CALL_COMPLEX_TYPES(m)

#define TF_CALL_POD_TYPES(m) TF_CALL_NUMBER_TYPES(m) TF_CALL_bool(m)

// Call "m" on all types.
#define TF_CALL_ALL_TYPES(m) \
  TF_CALL_POD_TYPES(m) TF_CALL_tstring(m) TF_CALL_resource(m) TF_CALL_variant(m)

// Call "m" on POD and string types.
#define TF_CALL_POD_STRING_TYPES(m) TF_CALL_POD_TYPES(m) TF_CALL_tstring(m)

// Call "m" on all number types supported on GPU.
#define TF_CALL_GPU_NUMBER_TYPES(m) \
  TF_CALL_half(m) TF_CALL_float(m) TF_CALL_double(m)

// Call "m" on all types supported on GPU.
#define TF_CALL_GPU_ALL_TYPES(m) \
  TF_CALL_GPU_NUMBER_TYPES(m) TF_CALL_COMPLEX_TYPES(m) TF_CALL_bool(m)

#define TF_CALL_GPU_NUMBER_TYPES_NO_HALF(m) TF_CALL_float(m) TF_CALL_double(m)

// Call "m" on all quantized types.
// TODO(cwhipkey): include TF_CALL_qint16(m) TF_CALL_quint16(m)
#define TF_CALL_QUANTIZED_TYPES(m) \
  TF_CALL_qint8(m) TF_CALL_quint8(m) TF_CALL_qint32(m)

// Types used for save and restore ops.
#define TF_CALL_SAVE_RESTORE_TYPES(m)      \
  TF_CALL_REAL_NUMBER_TYPES_NO_BFLOAT16(m) \
  TF_CALL_COMPLEX_TYPES(m)                 \
  TF_CALL_QUANTIZED_TYPES(m) TF_CALL_bool(m) TF_CALL_tstring(m)

#ifdef TENSORFLOW_SYCL_NO_DOUBLE
#define TF_CALL_SYCL_double(m)
#else  // TENSORFLOW_SYCL_NO_DOUBLE
#define TF_CALL_SYCL_double(m) TF_CALL_double(m)
#endif  // TENSORFLOW_SYCL_NO_DOUBLE

#ifdef __ANDROID_TYPES_SLIM__
#define TF_CALL_SYCL_NUMBER_TYPES(m) TF_CALL_float(m)
#else  // __ANDROID_TYPES_SLIM__
#define TF_CALL_SYCL_NUMBER_TYPES(m) TF_CALL_float(m) TF_CALL_SYCL_double(m)
#endif  // __ANDROID_TYPES_SLIM__

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_REGISTER_TYPES_H
