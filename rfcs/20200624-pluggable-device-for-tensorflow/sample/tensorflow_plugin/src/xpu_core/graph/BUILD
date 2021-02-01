package(
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "remapper",
    srcs = ["remapper.cc"],
    hdrs = ["remapper.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin/src/xpu_core/graph/utils:graph_properties",
        "//tensorflow_plugin/src/xpu_core/graph/utils:graph_view",
        "//tensorflow_plugin/src/xpu_core/graph/utils:grappler_item",
        "//tensorflow_plugin/src/xpu_core/graph/utils:tf_buffer",
    ],
    alwayslink = True,
)

cc_library(
    name = "onednn_layout",
    srcs = ["onednn_layout.cc"],
    hdrs = ["onednn_layout.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin/src/xpu_core/graph/utils:function",
        "//tensorflow_plugin/src/xpu_core/graph/utils:graph_properties",
        "//tensorflow_plugin/src/xpu_core/graph/utils:graph_view",
        "//tensorflow_plugin/src/xpu_core/graph/utils:grappler_item",
        "//tensorflow_plugin/src/xpu_core/graph/utils:node_type_attr_map",
    ],
    alwayslink = True,
)

cc_library(
    name = "gpu_optimizer",
    srcs = ["gpu_optimizer.cc"],
    hdrs = ["gpu_optimizer.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":onednn_layout",
        ":remapper",
    ],
    alwayslink = True,
)

cc_library(
    name = "xpu_optimizer",
    srcs = ["xpu_optimizer.cc"],
    hdrs = ["xpu_optimizer.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":onednn_layout",
        ":remapper",
    ],
    alwayslink = True,
)
