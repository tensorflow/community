package(
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "plugin_device",
    srcs = ["plugin_device.cc"],
    hdrs = ["plugin_device.h"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin/src/device/cpu:cpu_device_impl",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = True,
)

cc_library(
    name = "plugin_graph",
    srcs = ["plugin_graph.cc"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":plugin_device",
        "//tensorflow_plugin/src/graph:plugin_optimizer",
        "//tensorflow_plugin/src/kernels/cpu:cpu_kernel_impl",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = True,
)

cc_library(
    name = "plugin_kernel",
    srcs = ["plugin_kernel.cc"],
    hdrs = ["plugin_kernel.h"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":plugin_device",
        "//tensorflow_plugin/src/kernels/cpu:cpu_kernel_impl",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = True,
)


cc_library(
    name = "plugin_profiler",
    srcs = ["plugin_profiler.cc"],
    hdrs = ["plugin_device.h"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin/src/profiler/cpu:demo_profiler",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = True,
)



