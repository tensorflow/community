package(
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "gtl_libs",
    srcs = glob([
        "*.cc",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin/src/xpu_core/util:prefetch",
        "//tensorflow_plugin/src/xpu_core/util:types",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)
