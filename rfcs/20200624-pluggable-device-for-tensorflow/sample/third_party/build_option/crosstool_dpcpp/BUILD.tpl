# This file is expanded from a template by cuda_configure.bzl
# Update cuda_configure.bzl#verify_build_defines when adding new variables.

load(":cc_toolchain_config.bzl", "cc_toolchain_config")

licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

toolchain(
    name = "toolchain-linux-x86_64",
    exec_compatible_with = [
        "@bazel_tools//platforms:linux",
        "@bazel_tools//platforms:x86_64",
    ],
    target_compatible_with = [
        "@bazel_tools//platforms:linux",
        "@bazel_tools//platforms:x86_64",
    ],
    toolchain = ":cc-compiler-local",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)

cc_toolchain_suite(
    name = "toolchain",
    toolchains = {
        "local|compiler": ":cc-compiler-local",
        "k8": ":cc-compiler-local",
    },
)


cc_toolchain(
    name = "cc-compiler-local",
    all_files = ":empty",
    compiler_files = ":empty",
    #cpu = "local",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,                                                   
    toolchain_identifier = "local_linux",
    toolchain_config = ":cc-compiler-local-config",
)

cc_toolchain_config(
    name = "cc-compiler-local-config",
    cpu = "local",
    #compiler = "compiler",
    builtin_include_directories = [%{cxx_builtin_include_directories}],
   #extra_no_canonical_prefixes_flags = ["%{extra_no_canonical_prefixes_flags}"],
    host_compiler_path = "%{host_compiler_path}",
    dpcpp_compiler_root = "%{dpcpp_compiler_root}",
    host_compiler_prefix = "%{host_compiler_prefix}",
    #host_compiler_warnings = ["{host_compiler_warnings}"],
    host_unfiltered_compile_flags = ["%{unfiltered_compile_flags}"],
#linker_bin_path = "%{linker_bin_path}",
)

filegroup(
    name = "empty",
    srcs = [],
)

filegroup(
    name = "windows_msvc_wrapper_files",
    srcs = glob(["windows/msvc_*"]),
)
