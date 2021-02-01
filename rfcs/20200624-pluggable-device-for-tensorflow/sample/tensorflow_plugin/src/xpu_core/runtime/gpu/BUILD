load("@local_config_dpcpp//dpcpp:build_defs.bzl", "if_dpcpp")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "dpcpp_runtime_imp",
    srcs = ["dpcpp_runtime.cc"],
    #hdrs = ["dpcpp_runtime.h",],
    deps = [
        ":eigen_runtime_impl",
        "@com_google_absl//absl/synchronization",
        "@eigen_archive//:eigen",
        "@local_config_dpcpp//dpcpp:dpcpp_headers",
    ],
    alwayslink = True,
)

cc_library(
    name = "eigen_runtime_impl",
    srcs = ["eigen_dpcpp_runtime.cc"],
    deps = [
        "@com_google_absl//absl/synchronization",
        "@eigen_archive//:eigen",
        "@local_config_dpcpp//dpcpp:dpcpp_headers",
    ],
    alwayslink = True,
)
