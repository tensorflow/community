cc_binary(
    name = "libdemo_plugin.so",
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin/src:plugin_device",
        "//tensorflow_plugin/src:plugin_graph",
        "//tensorflow_plugin/src:plugin_kernel",
        "//tensorflow_plugin/src:plugin_profiler",
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
        "//tensorflow_plugin/src/device/cpu:cpu_device_impl",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = True,
)
