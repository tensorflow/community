package(default_visibility = ["//visibility:public"])

load(":platform.bzl", "dpcpp_library_path")
load("@local_config_dpcpp//dpcpp:build_defs.bzl", "if_dpcpp")

config_setting(
    name = "using_dpcpp",
    values = {
        "define": "using_dpcpp=true",
    },
)

cc_library(
    name = "dpcpp_headers",
    hdrs = glob([
        "runtime/dpcpp_runtime.h",
    ]),
    includes = [
        ".",
        "include",
    ],
)
