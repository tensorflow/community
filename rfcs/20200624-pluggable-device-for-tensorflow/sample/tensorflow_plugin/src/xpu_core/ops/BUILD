load("//tensorflow_plugin:intel_plugin.bzl", "tf_copts")
load("//third_party/onednn:build_defs.bzl", "onednn_deps")

package(
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "gelu",
    srcs = ["gelu.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":shape_inference_fns",
        "//tensorflow_plugin:core",
    ],
    alwayslink = True,
)

OPS = [
    ":gelu",
]

cc_library(
    name = "shape_inference_fns",
    srcs = ["shape_inference_fns.cc"],
    hdrs = [
        "shape_inference_fns.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
    ],
)

cc_library(
    name = "op_impl",
    srcs = ["op_init.cc"],
    hdrs = [
        "op_init.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
    ] + OPS + onednn_deps(),
    alwayslink = True,
)
