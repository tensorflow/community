load("//third_party/onednn:build_defs.bzl", "onednn_deps")

package(
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "gpu_device_impl",
    srcs = ["gpu_device_plugin.cc"],
    hdrs = [
        "gpu_device_plugin.h",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin/src/xpu_core/runtime/gpu:dpcpp_runtime_imp",
        "@local_config_dpcpp//dpcpp:dpcpp_headers",
        "@local_config_tf//:tf_header_lib",
    ] + onednn_deps(),
    alwayslink = True,
)

cc_library(
    name = "eigen_stream_device",
    srcs = ["eigen_stream_device.cc"],
    hdrs = ["eigen_stream_device.h"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin/src/xpu_core/util/gtl:gtl_libs",
        "//third_party/eigen3",
        "@eigen_archive//:eigen",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = True,
)

exports_files(
    srcs = [
        "gpu_device_plugin.h",
    ],
    visibility = ["//visibility:public"],
)
