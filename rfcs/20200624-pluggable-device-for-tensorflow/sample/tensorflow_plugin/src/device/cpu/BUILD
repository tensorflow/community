package(
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "cpu_device_impl",
    srcs = ["cpu_device_plugin.cc"],
    hdrs = [
        "cpu_device_plugin.h",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "@local_config_gcc//:framework_lib",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = True,
)

exports_files(
    srcs = [
        "cpu_device_plugin.h",
    ],
    visibility = ["//visibility:public"],
)
