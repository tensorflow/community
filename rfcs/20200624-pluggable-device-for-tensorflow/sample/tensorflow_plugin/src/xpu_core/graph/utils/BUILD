load("//tensorflow_plugin:intel_plugin.bzl", "tf_copts")
load(
    "//tensorflow_plugin/src/xpu_core/util:build_config.bzl",
    "tf_protobuf_deps",
)

package(
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "tf_buffer",
    srcs = ["tf_buffer.cc"],
    hdrs = [
        "tf_buffer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin/src/xpu_core/util:common_utils",
        "@local_config_tf//:tf_header_lib",
    ] + tf_protobuf_deps(),
)

cc_library(
    name = "grappler_item",
    srcs = ["grappler_item.cc"],
    hdrs = [
        "grappler_item.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin/src/xpu_core/util:common_utils",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "graph_properties",
    srcs = ["graph_properties.cc"],
    hdrs = [
        "graph_properties.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":grappler_item",
        "//tensorflow_plugin/src/xpu_core/util:common_utils",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "function",
    srcs = ["function.cc"],
    hdrs = [
        "function.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":tf_buffer",
        "//tensorflow_plugin/src/xpu_core/util:common_utils",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "graph_view_internal",
    hdrs = ["graph_view_internal.h"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "graph_view",
    srcs = ["graph_view.cc"],
    hdrs = ["graph_view.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_view_internal",
        ":op_types",
    ],
)

cc_library(
    name = "op_types",
    srcs = ["op_types.cc"],
    hdrs = ["op_types.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
    ],
    alwayslink = True,
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = True,
)

cc_library(
    name = "node_type_attr_map",
    srcs = ["node_type_attr_map.cc"],
    hdrs = [
        "node_type_attr_map.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":function",
        ":utils",
        "//tensorflow_plugin/src/xpu_core/util:common_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@local_config_tf//:tf_header_lib",
    ],
)
