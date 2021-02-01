load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

exports_files(["LICENSE"])

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
)
