package(default_visibility = ["//visibility:public"])

cc_library(
  name = "demo_profiler",
  srcs = ["demo_profiler.cc"],
  hdrs = ["demo_profiler.h"],
  linkstatic = 1,
  visibility = ["//visibility:public"],
  deps = [
    "@local_config_tf//:tf_header_lib",
    "//tensorflow_plugin/src/utils:protos_all",
    "//tensorflow_plugin/src/profiler/cpu/utils:xplane_utils",
  ],
  alwayslink = True,
)

