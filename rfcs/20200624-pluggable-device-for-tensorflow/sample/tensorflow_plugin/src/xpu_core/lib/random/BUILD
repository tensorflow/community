package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "exact_uniform_int",
    hdrs = ["exact_uniform_int.h"],
)

cc_library(
    name = "philox",
    srcs = [
        "distribution_sampler.cc",
        "random_distributions.cc",
        "simple_philox.cc",
    ],
    hdrs = [
        "distribution_sampler.h",
        "random_distributions.h",
        "simple_philox.h",
    ],
    deps = [
        ":exact_uniform_int",
        ":philox_random",
        "//tensorflow_plugin/src/xpu_core/util:common_utils",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
)

cc_library(
    name = "philox_random",
    hdrs = ["philox_random.h"],
    deps = [
        "//tensorflow_plugin/src/xpu_core/util:common_utils",
    ],
)

cc_library(
    name = "guarded_philox_random",
    srcs = ["guarded_philox_random.cc"],
    hdrs = [
        "guarded_philox_random.h",
        "philox_random.h",
    ],
    deps = [
        ":random",
        "//tensorflow_plugin/src/xpu_core/util:common_utils",
    ],
)

cc_library(
    name = "random",
    hdrs = ["random.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//tensorflow_plugin/src/xpu_core/util:random",
    ],
)

cc_library(
    name = "weighted_picker",
    srcs = ["weighted_picker.cc"],
    hdrs = ["weighted_picker.h"],
    deps = [
        ":philox",
        "//tensorflow_plugin/src/xpu_core/util:common_utils",
    ],
    alwayslink = 1,
)
