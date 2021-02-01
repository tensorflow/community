load("//tensorflow_plugin:intel_plugin.bzl", "tf_copts")
load("//third_party/onednn:build_defs.bzl", "onednn_deps")

package(
    licenses = ["notice"],  # Apache 2.0)
)

ONEDNN_KERNELS = [
]

cc_library(
    name = "onednn_kernel_impl",
    srcs = ["onednn_kernel_init.cc"],
    hdrs = [
        "onednn_kernel_init.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
    ] + onednn_deps() + ONEDNN_KERNELS,
    alwayslink = True,
)
