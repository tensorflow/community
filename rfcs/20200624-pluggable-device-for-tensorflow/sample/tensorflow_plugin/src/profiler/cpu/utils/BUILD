package(default_visibility = ["//visibility:public"])



cc_library(
    name = "time_utils_impl",
    srcs = [
        "time_utils.cc",
        "time_utils.h",
    ],
    deps = [
        "@com_google_absl//absl/time",
        "//tensorflow_plugin/src/utils:logging",
	"//third_party/eigen3",
    ],
    alwayslink = True,
)

cc_library(
    name = "xplane_builder",
    srcs = ["xplane_builder.cc"],
    hdrs = ["xplane_builder.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":time_utils",
        ":timespan",
        "//tensorflow_plugin/src/utils:protos_all",
        "//tensorflow_plugin/src/utils:logging",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)


cc_library(
    name = "xplane_visitor",
    srcs = ["xplane_visitor.cc"],
    hdrs = ["xplane_visitor.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":time_utils",
        ":timespan",
        "//tensorflow_plugin/src/utils:protos_all",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
) 

cc_library(
    name = "time_utils",
    hdrs = ["time_utils.h"],
    deps = [
      ":time_utils_impl",
    ],
)


cc_library(
    name = "trace_utils",
    hdrs = ["trace_utils.h"],
    deps = [
    ],
)


cc_library(
    name = "timespan",
    hdrs = ["timespan.h"],
    deps = [
        ":time_utils",
        "@com_google_absl//absl/strings",
    ],
)


cc_library(
    name = "xplane_utils",
    srcs = ["xplane_utils.cc"],
    hdrs = ["xplane_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":time_utils",
        ":timespan",
        ":trace_utils",
        ":xplane_builder",
	":xplane_visitor",
        "//tensorflow_plugin/src/utils:protos_all",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
	"//third_party/eigen3",
    ],
)

