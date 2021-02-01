package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "common_util_hdrs",
    srcs = glob(["*.h"]),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "random",
    srcs = ["random.cc"],
    hdrs = ["random.h"],
    deps = [
        ":common_utils",
    ],
)

cc_library(
    name = "common_utils",
    srcs = glob([
        "*.cc",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    linkstatic = 1,
    deps = [
        "//tensorflow_plugin/src/xpu_core/util/gtl:gtl_libs",
        "//third_party/eigen3",
        "@double_conversion//:double-conversion",
        "@eigen_archive//:eigen",
        "@local_config_tf//:protos_all",
        "@local_config_tf//:tf_header_lib",
        "@nsync//:nsync_cpp",
    ] + select({
        "@local_config_dpcpp//dpcpp:using_dpcpp": [
            "//tensorflow_plugin/src/xpu_core/device/gpu:eigen_stream_device",
            "//tensorflow_plugin/src/xpu_core/device/gpu:gpu_device_impl",
        ],
        "//conditions:default": [],
    }),
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
        ":tstring",
    ],
)
