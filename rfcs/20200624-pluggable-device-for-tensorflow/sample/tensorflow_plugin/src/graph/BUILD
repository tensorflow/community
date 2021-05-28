load(
    "//tensorflow_plugin:build_config.bzl",
    "tf_protobuf_deps",
)

cc_library(
    name = "tf_buffer",
    srcs = ["tf_buffer.cc"],
    hdrs = [
        "tf_buffer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@local_config_tf//:tf_header_lib",
    ] + tf_protobuf_deps(),
)

cc_library(
    name = "plugin_optimizer",
    srcs = ["plugin_optimizer.cc"],
    hdrs = ["plugin_optimizer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin/src/utils:protos_all",
        # "//tensorflow_plugin/src/utils:types_proto",
        "tf_buffer",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = True,
)
