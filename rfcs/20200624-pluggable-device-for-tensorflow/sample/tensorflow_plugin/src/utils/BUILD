load("@org_tensorflow_plugin//tensorflow_plugin:build_config.bzl", "cc_proto")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_proto(
    name = "types",
    src = "types.proto",
)

cc_proto(
    name = "tensor_shape",
    src = "tensor_shape.proto",
)

cc_proto(
    name = "versions",
    src = "versions.proto",
)

cc_proto(
    name = "cost_graph",
    src = "cost_graph.proto",
    deps = [
        ":tensor_shape_proto",
        ":types_proto",
    ],
)

cc_proto(
    name = "resource_handle",
    src = "resource_handle.proto",
    deps = [
        ":tensor_shape_proto",
        ":types_proto",
    ],
)

cc_proto(
    name = "tensor",
    src = "tensor.proto",
    deps = [
        ":resource_handle_proto",
    ],
)

cc_proto(
    name = "attr_value",
    src = "attr_value.proto",
    deps = [
        ":tensor_proto",
    ],
)

cc_proto(
    name = "node_def",
    src = "node_def.proto",
    deps = [
        ":attr_value_proto",
    ],
)

cc_proto(
    name = "op_def",
    src = "op_def.proto",
    deps = [
        ":attr_value_proto",
    ],
)

cc_proto(
    name = "function",
    src = "function.proto",
    deps = [
        ":node_def_proto",
        ":op_def_proto",
    ],
)

cc_proto(
    name = "graph",
    src = "graph.proto",
    deps = [
        ":function_proto",
        ":node_def_proto",
        ":versions_proto",
    ],
)

cc_proto(
    name = "device_properties",
    src = "device_properties.proto",
)

cc_proto(
    name = "op_performance_data",
    src = "op_performance_data.proto",
    deps = [
        ":attr_value_proto",
        ":device_properties_proto",
    ],
)

cc_proto(
    name = "api_def",
    src = "api_def.proto",
    deps = [
        ":attr_value_proto",
    ],
)

cc_proto(
    name = "xplane",
    src = "xplane.proto",
)



cc_library(
    name = "protos_all",
    visibility = ["//visibility:public"],
    deps = [
        ":api_def_proto",
        ":graph_proto",
        ":op_performance_data_proto",
        ":xplane_proto",
    ],
)

cc_library(
  name = "platform",
  hdrs = ["platform.h"],
)


cc_library(
  name = "prefetch",
  hdrs = ["prefetch.h"],
  visibility = ["//visibility:public"],
  deps = [":platform"],
)

cc_library(
  name = "tstring",
  hdrs = [
     "ctstring.h",
     "tstring.h",
  ],
  deps = [
    "@com_google_absl//absl/strings",
    "@local_config_tf//:tf_header_lib",
  ],
)

cc_library(
  name = "types",
  hdrs = ["types.h"],
  visibility = ["//visibility:public"], 
  deps = [
    ":platform",
    ":tstring"
  ],
)


cc_library(
    name = "logging",
    srcs = ["logging.cc"],
    visibility = ["//visibility:public"],
    hdrs = [
       "env_time.h",
       "integral_types.h",
       "logging.h",
       "macros.h",
       #"numeric_types.h",
       "stringpiece.h",
       "types.h",
       "protobuf.h"
    ],
    deps = [
        ":platform",
        ":tstring",
        "//tensorflow_plugin/src/utils/gtl:gtl_libs",
        #"//third_party/eigen3",
        ":protos_all",
    ],
)
