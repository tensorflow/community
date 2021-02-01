package(
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "xpu_util",
    srcs = ["xpu_util.cc"],
    hdrs = ["xpu_util.h"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    alwayslink = True,
)

cc_library(
    name = "xpu_device",
    srcs = ["xpu_device.cc"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "xpu_util",
        "@local_config_tf//:tf_header_lib",
    ] + select({
        "@local_config_dpcpp//dpcpp:using_dpcpp": ["//tensorflow_plugin/src/xpu_core/device/gpu:gpu_device_impl"],
        "//conditions:default": [],
    }),
    alwayslink = True,
)

cc_library(
    name = "xpu_kernel",
    srcs = ["xpu_kernel.cc"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "xpu_util",
        "//tensorflow_plugin/src/xpu_core/kernels/onednn:onednn_kernel_impl",
        "//tensorflow_plugin/src/xpu_core/ops:op_impl",
        "@local_config_tf//:tf_header_lib",
    ] + select({
        "@local_config_dpcpp//dpcpp:using_dpcpp": [
            "//tensorflow_plugin/src/xpu_core/kernels/gpu:gpu_kernel_impl",
        ],
        "//conditions:default": [],
    }),
    alwayslink = True,
)

cc_library(
    name = "xpu_graph",
    srcs = ["xpu_graph.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "xpu_util",
        "//tensorflow_plugin/src/xpu_core/graph:gpu_optimizer",
        "//tensorflow_plugin/src/xpu_core/graph:xpu_optimizer",
    ],
    alwayslink = True,
)
