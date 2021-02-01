"""cc_toolchain_config rule for configuring CUDA toolchains on Linux, Mac, and Windows."""

load(
    "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "action_config",
    "env_entry",
    "env_set",
    "feature",
    "feature_set",
    "flag_group",
    "flag_set",
    "tool",
    "tool_path",
    "variable_with_value",
)
load(
    "@bazel_tools//tools/build_defs/cc:action_names.bzl",
    "ASSEMBLE_ACTION_NAME",
    "CC_FLAGS_MAKE_VARIABLE_ACTION_NAME",
    "CLIF_MATCH_ACTION_NAME",
    "CPP_COMPILE_ACTION_NAME",
    "CPP_HEADER_PARSING_ACTION_NAME",
    "CPP_LINK_DYNAMIC_LIBRARY_ACTION_NAME",
    "CPP_LINK_EXECUTABLE_ACTION_NAME",
    "CPP_LINK_NODEPS_DYNAMIC_LIBRARY_ACTION_NAME",
    "CPP_LINK_STATIC_LIBRARY_ACTION_NAME",
    "CPP_MODULE_CODEGEN_ACTION_NAME",
    "CPP_MODULE_COMPILE_ACTION_NAME",
    "C_COMPILE_ACTION_NAME",
    "LINKSTAMP_COMPILE_ACTION_NAME",
    "LTO_BACKEND_ACTION_NAME",
    "LTO_INDEXING_ACTION_NAME",
    "OBJCPP_COMPILE_ACTION_NAME",
    "OBJCPP_EXECUTABLE_ACTION_NAME",
    "OBJC_ARCHIVE_ACTION_NAME",
    "OBJC_COMPILE_ACTION_NAME",
    "OBJC_EXECUTABLE_ACTION_NAME",
    "OBJC_FULLY_LINK_ACTION_NAME",
    "PREPROCESS_ASSEMBLE_ACTION_NAME",
    "STRIP_ACTION_NAME",
)

ACTION_NAMES = struct(
    c_compile = C_COMPILE_ACTION_NAME,
    cpp_compile = CPP_COMPILE_ACTION_NAME,
    linkstamp_compile = LINKSTAMP_COMPILE_ACTION_NAME,
    cc_flags_make_variable = CC_FLAGS_MAKE_VARIABLE_ACTION_NAME,
    cpp_module_codegen = CPP_MODULE_CODEGEN_ACTION_NAME,
    cpp_header_parsing = CPP_HEADER_PARSING_ACTION_NAME,
    cpp_module_compile = CPP_MODULE_COMPILE_ACTION_NAME,
    assemble = ASSEMBLE_ACTION_NAME,
    preprocess_assemble = PREPROCESS_ASSEMBLE_ACTION_NAME,
    lto_indexing = LTO_INDEXING_ACTION_NAME,
    lto_backend = LTO_BACKEND_ACTION_NAME,
    cpp_link_executable = CPP_LINK_EXECUTABLE_ACTION_NAME,
    cpp_link_dynamic_library = CPP_LINK_DYNAMIC_LIBRARY_ACTION_NAME,
    cpp_link_nodeps_dynamic_library = CPP_LINK_NODEPS_DYNAMIC_LIBRARY_ACTION_NAME,
    cpp_link_static_library = CPP_LINK_STATIC_LIBRARY_ACTION_NAME,
    strip = STRIP_ACTION_NAME,
    objc_archive = OBJC_ARCHIVE_ACTION_NAME,
    objc_compile = OBJC_COMPILE_ACTION_NAME,
    objc_executable = OBJC_EXECUTABLE_ACTION_NAME,
    objc_fully_link = OBJC_FULLY_LINK_ACTION_NAME,
    objcpp_compile = OBJCPP_COMPILE_ACTION_NAME,
    objcpp_executable = OBJCPP_EXECUTABLE_ACTION_NAME,
    clif_match = CLIF_MATCH_ACTION_NAME,
    objcopy_embed_data = "objcopy_embed_data",
    ld_embed_data = "ld_embed_data",
)

def _impl(ctx):
    toolchain_identifier = "local_linux"

    host_system_name = "local"

    target_system_name = "local"

    target_cpu = "local"
    if (ctx.attr.cpu == "local"):
        target_libc = "local"
    if (ctx.attr.cpu == "local"):
        compiler = "compiler"
    abi_version = "local"

    abi_libc_version = "local"

    cc_target_os = None

    builtin_sysroot = None

    all_link_actions = [
        ACTION_NAMES.cpp_link_executable,
        ACTION_NAMES.cpp_link_dynamic_library,
        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
    ]

    if (ctx.attr.cpu == "local"):
        action_configs = []
    else:
        fail("Unreachable")

    pic_feature = feature(
        name = "pic",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [
                    flag_group(flags = ["-fPIC"], expand_if_available = "pic"),
                    flag_group(
                        flags = ["-fPIE"],
                        expand_if_not_available = "pic",
                    ),
                ],
            ),
        ],
    )

    preprocessor_defines_feature = feature(
        name = "preprocessor_defines",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["/D%{preprocessor_defines}"],
                        iterate_over = "preprocessor_defines",
                    ),
                ],
            ),
        ],
    )

    generate_pdb_file_feature = feature(
        name = "generate_pdb_file",
        requires = [
            feature_set(features = ["dbg"]),
            feature_set(features = ["fastbuild"]),
        ],
    )

    linkstamps_feature = feature(
        name = "linkstamps",
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = ["%{linkstamp_paths}"],
                        iterate_over = "linkstamp_paths",
                        expand_if_available = "linkstamp_paths",
                    ),
                ],
            ),
        ],
    )

    unfiltered_compile_flags_feature = feature(
        name = "unfiltered_compile_flags",
        flag_sets = ([
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                ],
                flag_groups = [
                    flag_group(
                        flags = ctx.attr.host_unfiltered_compile_flags,
                    ),
                ],
            ),
        ] if ctx.attr.host_unfiltered_compile_flags else []),
    )

    determinism_feature = feature(
        name = "determinism",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-Wno-builtin-macro-redefined",
                            "-D__DATE__=\"redacted\"",
                            "-D__TIMESTAMP__=\"redacted\"",
                            "-D__TIME__=\"redacted\"",
                            "-no-canonical-prefixes",
                        ],
                    ),
                ],
            ),
        ],
    )

    nologo_feature = feature(
        name = "nologo",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.cpp_link_static_library,
                ],
                flag_groups = [flag_group(flags = ["/nologo"])],
            ),
        ],
    )

    supports_pic_feature = feature(name = "supports_pic", enabled = True)

    output_execpath_flags_feature = feature(
        name = "output_execpath_flags",
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = ["/OUT:%{output_execpath}"],
                        expand_if_available = "output_execpath",
                    ),
                ],
            ),
        ],
    )

    if (ctx.attr.cpu == "local"):
        hardening_feature = feature(
            name = "hardening",
            flag_sets = [
                flag_set(
                    actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-U_FORTIFY_SOURCE",
                                "-D_FORTIFY_SOURCE=1",
                                "-fstack-protector",
                            ],
                        ),
                    ],
                ),
                flag_set(
                    actions = [
                        ACTION_NAMES.cpp_link_dynamic_library,
                        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ],
                    flag_groups = [flag_group(flags = ["-Wl,-z,relro,-z,now"])],
                ),
                flag_set(
                    actions = [ACTION_NAMES.cpp_link_executable],
                    flag_groups = [flag_group(flags = ["-pie", "-Wl,-z,relro,-z,now"])],
                ),
            ],
        )
    else:
        hardening_feature = None

    supports_dynamic_linker_feature = feature(name = "supports_dynamic_linker", enabled = True)

    warnings_feature = feature(
        name = "warnings",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [
                    flag_group(
                        flags = ["-Wall"] + ctx.attr.host_compiler_warnings,
                    ),
                ],
            ),
        ],
    )

    if (ctx.attr.cpu == "local"):
        dbg_feature = feature(
            name = "dbg",
            flag_sets = [
                flag_set(
                    actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                    flag_groups = [flag_group(flags = ["-g"])],
                ),
            ],
            implies = ["common"],
        )
    else:
        dbg_feature = None

    undefined_dynamic_feature = feature(
        name = "undefined-dynamic",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.cpp_link_executable,
                ],
                flag_groups = [flag_group(flags = ["-undefined", "dynamic_lookup"])],
            ),
        ],
    )

    parse_showincludes_feature = feature(
        name = "parse_showincludes",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_header_parsing,
                ],
                flag_groups = [flag_group(flags = ["/showIncludes"])],
            ),
        ],
    )

    linker_param_file_feature = feature(
        name = "linker_param_file",
        flag_sets = [
            flag_set(
                actions = all_link_actions +
                          [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [
                    flag_group(
                        flags = ["@%{linker_param_file}"],
                        expand_if_available = "linker_param_file",
                    ),
                ],
            ),
        ],
    )

    supports_interface_shared_libraries_feature = feature(
        name = "supports_interface_shared_libraries",
        enabled = True,
    )

    disable_assertions_feature = feature(
        name = "disable-assertions",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-DNDEBUG"])],
            ),
        ],
    )

    if (ctx.attr.cpu == "local"):
        fastbuild_feature = feature(name = "fastbuild", implies = ["common"])
    else:
        fastbuild_feature = None

    user_compile_flags_feature = feature(
        name = "user_compile_flags",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["%{user_compile_flags}"],
                        iterate_over = "user_compile_flags",
                        expand_if_available = "user_compile_flags",
                    ),
                ],
            ),
        ],
    )

    compiler_input_flags_feature = feature(
        name = "compiler_input_flags",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-c", "%{source_file}"],
                        expand_if_available = "source_file",
                    ),
                ],
            ),
        ],
    )

    no_legacy_features_feature = feature(name = "no_legacy_features")

    linker_bin_path_feature = feature(
        name = "linker-bin-path",
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [flag_group(flags = ["-B" + ctx.attr.linker_bin_path])],
            ),
        ],
    )

    if (ctx.attr.cpu == "local"):
        opt_feature = feature(
            name = "opt",
            flag_sets = [
                flag_set(
                    actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                    flag_groups = [
                        flag_group(
                            flags = ["-g0", "-O0", "-ffunction-sections", "-fdata-sections"],
                        ),
                    ],
                ),
                flag_set(
                    actions = [
                        ACTION_NAMES.cpp_link_dynamic_library,
                        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                        ACTION_NAMES.cpp_link_executable,
                    ],
                    flag_groups = [flag_group(flags = ["-Wl,--gc-sections"])],
                ),
            ],
            implies = ["common", "disable-assertions"],
        )
    else:
        opt_feature = None

    frame_pointer_feature = feature(
        name = "frame-pointer",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-fno-omit-frame-pointer"])],
            ),
        ],
    )

    build_id_feature = feature(
        name = "build-id",
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = ["-Wl,--build-id=md5", "-Wl,--hash-style=gnu"],
                    ),
                ],
            ),
        ],
    )

    sysroot_feature = feature(
        name = "sysroot",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["--sysroot=%{sysroot}"],
                        iterate_over = "sysroot",
                        expand_if_available = "sysroot",
                    ),
                ],
            ),
        ],
    )

    if (ctx.attr.cpu == "local"):
        stdlib_feature = feature(
            name = "stdlib",
            flag_sets = [
                flag_set(
                    actions = all_link_actions,
                    flag_groups = [flag_group(flags = ["-lstdc++", "-B/usr/bin"])],
                ),
            ],
        )
    else:
        stdlib_feature = None

    no_stripping_feature = feature(name = "no_stripping")

    alwayslink_feature = feature(
        name = "alwayslink",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                ],
                flag_groups = [flag_group(flags = [
                    "-Wl,-no-as-needed",
                    "-Wl,-rpath=%{DPCPP_ROOT_DIR}%/lib/",
                    "-Wl,-rpath=%{TF_SHARED_LIBRARY_DIR}%/python",
                    "-Wl,-rpath=%{PYTHON_LINK_PATH}%",
                    "-lsycl",
                    "-L%{TF_SHARED_LIBRARY_DIR}%/python/",
                    "-L%{PYTHON_LINK_PATH}%",
                    "-l%{PYTHON_LINK_LIB_NAME}%",
                    "-rdynamic",
                    "-fsycl",
                    "-fsycl-unnamed-lambda",
                    "-sycl-std=2017",
                    "-fPIC",
                    "-no-canonical-prefixes",
                    "-Wl,-no-as-needed",
                    "-Wl,-z,relro,-z,now",
                    "-Wl,--build-id=md5",
                    "-Wl,--hash-style=gnu",
                    "-fsycl-device-code-split=per_source",
                ])],
            ),
        ],
    )

    input_param_flags_feature = feature(
        name = "input_param_flags",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["/IMPLIB:%{interface_library_output_path}"],
                        expand_if_available = "interface_library_output_path",
                    ),
                ],
            ),
            flag_set(
                actions = all_link_actions +
                          [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [
                    flag_group(
                        iterate_over = "libraries_to_link",
                        flag_groups = [
                            flag_group(
                                iterate_over = "libraries_to_link.object_files",
                                flag_groups = [flag_group(flags = ["%{libraries_to_link.object_files}"])],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "object_file_group",
                                ),
                            ),
                            flag_group(
                                flag_groups = [flag_group(flags = ["%{libraries_to_link.name}"])],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "object_file",
                                ),
                            ),
                            flag_group(
                                flag_groups = [flag_group(flags = ["%{libraries_to_link.name}"])],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "interface_library",
                                ),
                            ),
                            flag_group(
                                flag_groups = [
                                    flag_group(
                                        flags = ["%{libraries_to_link.name}"],
                                        expand_if_false = "libraries_to_link.is_whole_archive",
                                    ),
                                    flag_group(
                                        flags = ["/WHOLEARCHIVE:%{libraries_to_link.name}"],
                                        expand_if_true = "libraries_to_link.is_whole_archive",
                                    ),
                                ],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "static_library",
                                ),
                            ),
                        ],
                        expand_if_available = "libraries_to_link",
                    ),
                ],
            ),
        ],
    )

    if (ctx.attr.cpu == "local"):
        no_canonical_prefixes_feature = feature(
            name = "no-canonical-prefixes",
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.cpp_link_executable,
                        ACTION_NAMES.cpp_link_dynamic_library,
                        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ],
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-no-canonical-prefixes",
                            ] + ctx.attr.extra_no_canonical_prefixes_flags,
                        ),
                    ],
                ),
            ],
        )
    else:
        no_canonical_prefixes_feature = None

    has_configured_linker_path_feature = feature(name = "has_configured_linker_path")

    copy_dynamic_libraries_to_binary_feature = feature(name = "copy_dynamic_libraries_to_binary")

    cpp11_feature = feature(
        name = "c++11",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-std=c++11"])],
            ),
        ],
    )

    dpcpp_compiler_inc_feature = feature(
        name = "dpcpp_inc",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                ],
                flag_groups = [
                    flag_group(flags = ["-isystem"]),
                    flag_group(flags = ["%{DPCPP_ROOT_DIR}%/include/sycl"]),
                    flag_group(flags = ["-isystem"]),
                    flag_group(flags = ["%{DPCPP_ROOT_DIR}%/lib/clang/11.0.0/include"]),
                    flag_group(flags = ["-isystem"]),
                    flag_group(flags = ["%{DPCPP_ROOT_DIR}%/lib/clang/12.0.0/include"]),
                    flag_group(flags = ["-iquote"]),
                    flag_group(flags = ["%{DPCPP_RUNTIME_INC}%"]),
                    flag_group(flags = ["-isystem"]),
                    flag_group(flags = ["external/com_google_protobuf/src"]),
                ],
            ),
        ],
    )

    dpcpp_compiler_feature = feature(
        name = "dpcpp_feature",
        flag_sets = [
            flag_set(
                actions = [
                    #ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                ],
                flag_groups = [
                    flag_group(flags = [
                        "-std=c++11",
                        "-fsycl",
                        "-fsycl-unnamed-lambda",
                        "-sycl-std=2017",
                        "-fPIC",
                        "-Xclang",
                        "-fdenormal-fp-math=preserve-sign",
                        "-cl-fp32-correctly-rounded-divide-sqrt",
                        "-Xclang",
                        "-cl-mad-enable",
                        "-DTENSORFLOW_USE_DPCPP=1",
                        "-DTENSORFLOW_USE_DPCPP_BUILD=1",
                        "-DTENSORFLOW_USE_DPCPP_USM=1",
                        "-DEIGEN_USE_DPCPP=1",
                        "-DEIGEN_USE_GPU=1",
                        "-DEIGEN_USE_DPCPP_BUILD=1",
                        "-DEIGEN_USE_DPCPP_USM=1",
                        "-DDNNL_USE_DPCPP_USM=1",
                        "-DDNNL_WITH_LEVEL_ZERO=1",
                        "-DNGEN_NO_OP_NAMES=1",
                        "-DNGEN_CPP11=1",
                        "-DNGEN_NEO_INTERFACE=1",
                        "-DEIGEN_HAS_C99_MATH=1",
                        "-DEIGEN_HAS_CXX11_MATH=1",
                        "-DEIGEN_DONT_VECTORIZE_SYCL=1",
                        "-Wno-unused-variable",
                        "-Wno-unused-const-variable",
                        "-Wno-deprecated-declarations",
                    ]),
                ],
            ),
        ],
    )

    if (ctx.attr.cpu == "local"):
        common_feature = feature(
            name = "common",
            implies = [
                "stdlib",
                "dpcpp_inc",
                "dpcpp_feature",
                "c++11",
                "determinism",
                "alwayslink",
                "hardening",
                "warnings",
                "frame-pointer",
                "build-id",
                "no-canonical-prefixes",
                "linker-bin-path",
            ],
        )
    if (ctx.attr.cpu == "local"):
        features = [
            cpp11_feature,
            dpcpp_compiler_inc_feature,
            dpcpp_compiler_feature,
            stdlib_feature,
            determinism_feature,
            alwayslink_feature,
            pic_feature,
            hardening_feature,
            warnings_feature,
            frame_pointer_feature,
            build_id_feature,
            no_canonical_prefixes_feature,
            disable_assertions_feature,
            linker_bin_path_feature,
            common_feature,
            opt_feature,
            fastbuild_feature,
            dbg_feature,
            supports_dynamic_linker_feature,
            supports_pic_feature,
        ]
    sys_inc = [
        "/usr/local/gcc/",
        "/usr/lib",
        "/usr/lib64",
        "/usr/local/include",
        "/usr/include",
        "%{DPCPP_ROOT_DIR}%/lib/clang/11.0.0/include",
        "%{DPCPP_ROOT_DIR}%/lib/clang/12.0.0/include",
        "%{DPCPP_RUNTIME_INC}%",
    ]
    include_directories = ctx.attr.builtin_include_directories + sys_inc

    if (ctx.attr.cpu == "local"):
        tool_paths = [
            tool_path(name = "gcc", path = ctx.attr.dpcpp_compiler_root + "/bin/clang"),
            tool_path(name = "g++", path = ctx.attr.dpcpp_compiler_root + "/bin/clang++"),
            tool_path(name = "ar", path = ctx.attr.host_compiler_prefix + "/ar"),
            tool_path(name = "compat-ld", path = ctx.attr.host_compiler_prefix + "/ld"),
            tool_path(name = "cpp", path = ctx.attr.host_compiler_prefix + "/cpp"),
            tool_path(name = "dwp", path = ctx.attr.host_compiler_prefix + "/dwp"),
            tool_path(name = "gcov", path = ctx.attr.host_compiler_prefix + "/gcov"),
            tool_path(name = "ld", path = ctx.attr.host_compiler_prefix + "/ld"),
            tool_path(name = "nm", path = ctx.attr.host_compiler_prefix + "/nm"),
            tool_path(name = "objcopy", path = ctx.attr.host_compiler_prefix + "/objcopy"),
            tool_path(name = "objdump", path = ctx.attr.host_compiler_prefix + "/objdump"),
            tool_path(name = "strip", path = ctx.attr.host_compiler_prefix + "/strip"),
        ]
    out = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.write(out, "Fake executable")
    return [
        cc_common.create_cc_toolchain_config_info(
            ctx = ctx,
            features = features,
            action_configs = action_configs,
            artifact_name_patterns = [],
            cxx_builtin_include_directories = include_directories,
            toolchain_identifier = toolchain_identifier,
            host_system_name = host_system_name,
            target_system_name = target_system_name,
            target_cpu = target_cpu,
            target_libc = target_libc,
            compiler = compiler,
            abi_version = abi_version,
            abi_libc_version = abi_libc_version,
            tool_paths = tool_paths,
            make_variables = [],
            builtin_sysroot = builtin_sysroot,
            cc_target_os = cc_target_os,
        ),
        DefaultInfo(
            executable = out,
        ),
    ]

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {
        "cpu": attr.string(mandatory = True, values = ["local"]),
        "builtin_include_directories": attr.string_list(),
        "extra_no_canonical_prefixes_flags": attr.string_list(),
        "dpcpp_compiler_root": attr.string(),
        "host_compiler_path": attr.string(),
        "host_compiler_prefix": attr.string(),
        "host_compiler_warnings": attr.string_list(),
        "host_unfiltered_compile_flags": attr.string_list(),
        "linker_bin_path": attr.string(),
        "msvc_cl_path": attr.string(default = "msvc_not_used"),
        "msvc_env_include": attr.string(default = "msvc_not_used"),
        "msvc_env_lib": attr.string(default = "msvc_not_used"),
        "msvc_env_path": attr.string(default = "msvc_not_used"),
        "msvc_env_tmp": attr.string(default = "msvc_not_used"),
        "msvc_lib_path": attr.string(default = "msvc_not_used"),
        "msvc_link_path": attr.string(default = "msvc_not_used"),
        "msvc_ml_path": attr.string(default = "msvc_not_used"),
    },
    provides = [CcToolchainConfigInfo],
    executable = True,
)
