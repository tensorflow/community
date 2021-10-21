#ifndef TENSORFLOW_PLUGIN_SRC_UTILS_NUMERIC_TYPES_H_
#define TENSORFLOW_PLUGIN_SRC_UTILS_NUMERIC_TYPES_H_

#include "tensorflow_plugin/src/utils/tstring.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <complex>
// Disable clang-format to prevent 'FixedPoint' header from being included
// before 'Tensor' header on which it depends.
// clang-format off
#include "third_party/eigen3/unsupported/Eigen/CXX11/FixedPoint"
// clang-format on

namespace demo_plugin {

// Single precision complex.
typedef std::complex<float> complex64;
// Double precision complex.
typedef std::complex<double> complex128;

// We use Eigen's QInt implementations for our quantized int types.
typedef Eigen::QInt8 qint8;
typedef Eigen::QUInt8 quint8;
typedef Eigen::QInt32 qint32;
typedef Eigen::QInt16 qint16;
typedef Eigen::QUInt16 quint16;

} // namespace demo_plugin

static inline Eigen::bfloat16 FloatToBFloat16(float float_val) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  return *reinterpret_cast<Eigen::bfloat16 *>(
      reinterpret_cast<uint16_t *>(&float_val));
#else
  return *reinterpret_cast<Eigen::bfloat16 *>(
      &(reinterpret_cast<uint16_t *>(&float_val)[1]));
#endif
}

namespace Eigen {
template <>
struct NumTraits<demo_plugin::tstring>
    : GenericNumTraits<demo_plugin::tstring> {
  enum {
    RequireInitialization = 1,
    ReadCost = HugeCost,
    AddCost = HugeCost,
    MulCost = HugeCost
  };

  static inline int digits10() { return 0; }

private:
  static inline demo_plugin::tstring epsilon();
  static inline demo_plugin::tstring dummy_precision();
  static inline demo_plugin::tstring lowest();
  static inline demo_plugin::tstring highest();
  static inline demo_plugin::tstring infinity();
  static inline demo_plugin::tstring quiet_NaN();
};

} // namespace Eigen

#if defined(_MSC_VER) && !defined(__clang__)
namespace std {
template <> struct hash<Eigen::half> {
  std::size_t operator()(const Eigen::half &a) const {
    return static_cast<std::size_t>(a.x);
  }
};

template <> struct hash<Eigen::bfloat16> {
  std::size_t operator()(const Eigen::bfloat16 &a) const {
    return hash<float>()(static_cast<float>(a));
  }
};
} // namespace std
#endif // _MSC_VER

#endif // TENSORFLOW_PLUGIN_SRC_UTILS_NUMERIC_TYPES_H_
