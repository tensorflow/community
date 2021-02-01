load("//tensorflow_plugin:intel_plugin.bzl", "tf_copts")
load("//third_party/onednn:build_defs.bzl", "onednn_deps")

package(
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "aggregate_ops",
    srcs = ["aggregate_ops.cc"],
    hdrs = ["aggregate_ops.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
    ],
    alwayslink = True,
)

cc_library(
    name = "argmax_op",
    srcs = ["argmax_op.cc"],
    hdrs = ["argmax_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
    ],
    alwayslink = True,
)

cc_library(
    name = "bias_op",
    srcs = ["bias_op.cc"],
    hdrs = ["bias_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
    ],
    alwayslink = True,
)

cc_library(
    name = "broadcast_to_op",
    srcs = ["broadcast_to_op.cc"],
    hdrs = [
        "broadcast_to_op.h",
        "fill_functor.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
    ],
    alwayslink = True,
)

cc_library(
    name = "cast_op",
    srcs = [
        "cast_op.cc",
        "cast_op_impl.cc",
    ],
    hdrs = [
        "cast_op.h",
        "cast_op_impl.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
    ],
    alwayslink = True,
)

cc_library(
    name = "concat_v2_op",
    srcs = ["concat_op.cc"],
    hdrs = ["concat_lib.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
    ],
    alwayslink = True,
)

cc_library(
    name = "constant_ops",
    srcs = ["constant_ops.cc"],
    hdrs = ["fill_functor.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":fill_functor",
        "//tensorflow_plugin:core",
    ],
    alwayslink = True,
)

cc_library(
    name = "conv_ops",
    srcs = ["conv_ops.cc"],
    hdrs = ["conv_ops.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
    ],
    alwayslink = True,
)

cc_library(
    name = "data_format_ops",
    srcs = ["data_format_ops.cc"],
    hdrs = ["data_format_ops.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
    ],
    alwayslink = True,
)

cc_library(
    name = "fill_functor",
    srcs = ["fill_functor.cc"],
    hdrs = ["fill_functor.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
    ],
    alwayslink = True,
)

cc_library(
    name = "fused_batch_norm_op",
    srcs = ["fused_batch_norm_op.cc"],
    hdrs = ["fused_batch_norm_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
    ],
    alwayslink = True,
)

cc_library(
    name = "gather_op",
    srcs = ["gather_op.cc"],
    hdrs = ["gather_functor.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
    ],
    alwayslink = True,
)

cc_library(
    name = "l2loss_op",
    srcs = ["l2loss_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
    ],
    alwayslink = True,
)

cc_library(
    name = "matmul_op",
    srcs = ["matmul_op.cc"],
    hdrs = ["matmul_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [":fill_functor"],
    alwayslink = True,
)

cc_library(
    name = "reduction_ops_common",
    srcs = ["reduction_ops_common.cc"],
    hdrs = [
        "reduction_ops.h",
        "reduction_ops_common.h",
        "transpose_functor.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = ["//tensorflow_plugin:core"],
    alwayslink = True,
)

cc_library(
    name = "mean_op",
    srcs = [":reduction_ops_mean.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":reduction_ops_common",
    ],
    alwayslink = True,
)

cc_library(
    name = "no_op",
    srcs = ["no_op.cc"],
    hdrs = ["no_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
    ],
    alwayslink = True,
)

cc_library(
    name = "one_hot_op",
    srcs = ["one_hot_op.cc"],
    hdrs = ["one_hot_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
    ],
    alwayslink = True,
)

cc_library(
    name = "pack_op",
    srcs = [
        "concat_lib.cc",
        "pack_op.cc",
    ],
    hdrs = [
        "concat_lib.h",
        "gpu_device_array.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
    ],
    alwayslink = True,
)

cc_library(
    name = "pad_op",
    srcs = ["pad_op.cc"],
    hdrs = ["pad_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
    ],
    alwayslink = True,
)

cc_library(
    name = "pooling_ops",
    srcs = [
        "avgpooling_op.cc",
        "maxpooling_op.cc",
        "pooling_ops_common.cc",
    ],
    hdrs = [
        "pooling_ops_common.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    deps = [
        "//tensorflow_plugin:core",
    ],
    alwayslink = True,
)

cc_library(
    name = "random_op",
    srcs = [
        "random_op.cc",
        "random_op_gpu.cc",
    ],
    hdrs = [
        "random_op_gpu.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
        "//tensorflow_plugin/src/xpu_core/lib/random:guarded_philox_random",
    ],
    alwayslink = True,
)

cc_library(
    name = "relu_op",
    srcs = ["relu_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
    ],
    alwayslink = True,
)

cc_library(
    name = "select_op",
    srcs = ["select_op.cc"],
    hdrs = ["cwise_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
    ],
    alwayslink = True,
)

cc_library(
    name = "slice_op",
    srcs = ["slice_op.cc"],
    hdrs = ["slice_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
    ],
    alwayslink = True,
)

cc_library(
    name = "snapshot_op",
    srcs = ["snapshot_op.cc"],
    hdrs = ["snapshot_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
    ],
    alwayslink = True,
)

cc_library(
    name = "softmax_op",
    srcs = ["softmax_op.cc"],
    hdrs = ["softmax_op_functor.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
    ],
    alwayslink = True,
)

cc_library(
    name = "strided_slice_op",
    srcs = [
        "strided_slice_op.cc",
        "strided_slice_op_util.cc",
    ],
    hdrs = [
        "slice_op.h",
        "strided_slice_op.h",
        "strided_slice_op_impl.h",
        "strided_slice_op_util.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
    ],
    alwayslink = True,
)

cc_library(
    name = "sum_op",
    srcs = [":reduction_ops_sum.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":reduction_ops_common",
    ],
    alwayslink = True,
)

cc_library(
    name = "transpose_op",
    srcs = ["transpose_op.cc"],
    hdrs = ["transpose_functor.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
    ],
    alwayslink = True,
)

cc_library(
    name = "segment_reduction_ops",
    srcs = ["segment_reduction_ops.cc"],
    hdrs = ["segment_reduction_ops.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
    ],
    alwayslink = True,
)

cc_library(
    name = "cwise_lib",
    srcs = ["cwise_ops_common.cc"],
    hdrs = [
        "cwise_ops.h",
        "cwise_ops_common.h",
        "cwise_ops_gradients.h",
        "fill_functor.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
    ],
)

cc_library(
    name = "unary_and_binary_kernels",
    srcs = ["unary_and_binary_ops"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":cwise_lib",
        "//tensorflow_plugin:core",
    ],
    alwayslink = True,
)

filegroup(
    name = "unary_and_binary_ops",
    srcs = [
        "cwise_op_add.cc",
        "cwise_op_div.cc",
        "cwise_op_equal_to.cc",
        "cwise_op_erf.cc",
        "cwise_op_exp.cc",
        "cwise_op_floor_div.cc",
        "cwise_op_greater.cc",
        "cwise_op_greater_equal.cc",
        "cwise_op_isfinite.cc",
        "cwise_op_less.cc",
        "cwise_op_less_equal.cc",
        "cwise_op_logical_and.cc",
        "cwise_op_logical_not.cc",
        "cwise_op_maximum.cc",
        "cwise_op_minimum.cc",
        "cwise_op_mul.cc",
        "cwise_op_neg.cc",
        "cwise_op_pow.cc",
        "cwise_op_reciprocal.cc",
        "cwise_op_rsqrt.cc",
        "cwise_op_sqrt.cc",
        "cwise_op_square.cc",
        "cwise_op_squared_difference.cc",
        "cwise_op_sub.cc",
        "cwise_op_tanh.cc",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "unpack_op",
    srcs = [
        "split_lib.cc",
        "unpack_op.cc",
    ],
    hdrs = ["split_lib.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
    ],
    alwayslink = True,
)

GPU_KERNELS = [
    ":aggregate_ops",
    ":argmax_op",
    ":broadcast_to_op",
    ":cast_op",
    ":concat_v2_op",
    ":bias_op",
    ":constant_ops",
    ":conv_ops",
    ":data_format_ops",
    ":fused_batch_norm_op",
    ":gather_op",
    ":l2loss_op",
    ":matmul_op",
    ":mean_op",
    ":no_op",
    ":one_hot_op",
    ":pack_op",
    ":pad_op",
    ":pooling_ops",
    ":random_op",
    ":relu_op",
    ":segment_reduction_ops",
    ":select_op",
    ":slice_op",
    ":snapshot_op",
    ":softmax_op",
    ":strided_slice_op",
    ":sum_op",
    ":transpose_op",
    ":unary_and_binary_kernels",
    ":unpack_op",
]

cc_library(
    name = "gpu_kernel_impl",
    srcs = ["gpu_kernel_init.cc"],
    hdrs = [
        "gpu_kernel_init.h",
        "//tensorflow_plugin/src/xpu_core/device/gpu:gpu_device_plugin.h",
        "//tensorflow_plugin/src/xpu_core/util:padding.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin:core",
    ] + GPU_KERNELS + onednn_deps(),
    alwayslink = True,
)
