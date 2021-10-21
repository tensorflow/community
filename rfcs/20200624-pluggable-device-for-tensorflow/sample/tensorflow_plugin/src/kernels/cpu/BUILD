load("//tensorflow_plugin:demo_plugin.bzl", "tf_copts")

package(
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "relu_op",
    srcs = ["relu_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
        "@com_google_absl//absl/container:inlined_vector",
    ],
    alwayslink = True,
)

cc_library(
    name = "conv2d_op",
    srcs = ["conv_ops_using_gemm.cc"],
    hdrs = ["gemm_functors.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
        "@com_google_absl//absl/container:inlined_vector",
        "@eigen_archive//:eigen",
        "//third_party/eigen3"
    ],
    alwayslink = True,
)

CPU_KERNELS = [
    ":relu_op",
    ":conv2d_op",
]

cc_library(
    name = "cpu_kernel_impl",
    srcs = ["cpu_kernel_init.cc"],
    hdrs = [
        "cpu_kernel_init.h",
        "//tensorflow_plugin/src/device/cpu:cpu_device_plugin.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
    ] + CPU_KERNELS,
    alwayslink = True,
)
