cc_binary(
    name = "libintel_plugin.so",
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin/src/xpu_core:xpu_device",
        "//tensorflow_plugin/src/xpu_core:xpu_graph",
        "//tensorflow_plugin/src/xpu_core:xpu_kernel",
        "@local_config_tf//:_pywrap_tensorflow_internal",
    ],
)

config_setting(
    name = "linux_x86_64",
    values = {"cpu": "k8"},
    visibility = ["//visibility:public"],
)

cc_library(
    name = "core",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin/src/xpu_core/lib/random:philox",
        "//tensorflow_plugin/src/xpu_core/lib/random:philox_random",
        "//tensorflow_plugin/src/xpu_core/util:common_utils",
        "//third_party/eigen3",
        "@local_config_tf//:tf_header_lib",
    ] + select({
        "@local_config_dpcpp//dpcpp:using_dpcpp": [
            "//tensorflow_plugin/src/xpu_core/device/gpu:eigen_stream_device",
            "//tensorflow_plugin/src/xpu_core/device/gpu:gpu_device_impl",
        ],
        "//conditions:default": [],
    }),
    alwayslink = True,
)
