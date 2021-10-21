#ifndef TENSORFLOW_PLUGIN_SRC_UTILS_GTL_MAP_TRAITS_H_
#define TENSORFLOW_PLUGIN_SRC_UTILS_GTL_MAP_TRAITS_H_

#include <utility>

// Traits classes for performing uniform lookup on different map value types.
//
// The access is computed as follows:
//
//   1. If T has a `first` or `second` field, use them.
//   2. Otherwise if it has `key()` or `value()` methods, use them.
//   3. Otherwise the program is ill-formed.
namespace demo_plugin {
namespace gtl {
namespace subtle {
namespace internal_map_traits {
struct Rank1 {};
struct Rank0 : Rank1 {};

template <class V>
auto GetKey(V&& v, Rank0) -> decltype((std::forward<V>(v).first)) {
  return std::forward<V>(v).first;
}
template <class V>
auto GetKey(V&& v, Rank1) -> decltype(std::forward<V>(v).key()) {
  return std::forward<V>(v).key();
}

template <class V>
auto GetMapped(V&& v, Rank0) -> decltype((std::forward<V>(v).second)) {
  return std::forward<V>(v).second;
}
template <class V>
auto GetMapped(V&& v, Rank1) -> decltype(std::forward<V>(v).value()) {
  return std::forward<V>(v).value();
}

}  // namespace internal_map_traits

// Accesses the `key_type` from a `value_type`.
template <typename V>
auto GetKey(V&& v)
    -> decltype(internal_map_traits::GetKey(std::forward<V>(v),
                                            internal_map_traits::Rank0())) {
  return internal_map_traits::GetKey(std::forward<V>(v),
                                     internal_map_traits::Rank0());
}

// Accesses the `mapped_type` from a `value_type`.
template <typename V>
auto GetMapped(V&& v)
    -> decltype(internal_map_traits::GetMapped(std::forward<V>(v),
                                               internal_map_traits::Rank0())) {
  return internal_map_traits::GetMapped(std::forward<V>(v),
                                        internal_map_traits::Rank0());
}

}  // namespace subtle
}  // namespace gtl
}  // namespace demo_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_UTILS_GTL_MAP_TRAITS_H_
