load("//third_party:repo.bzl", "tf_http_archive", "third_party_http_archive")
load("//third_party/build_option:dpcpp_configure.bzl", "dpcpp_configure")
load("//third_party/systemlibs:syslibs_configure.bzl", "syslibs_configure")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def clean_dep(dep):
    return str(Label(dep))

def intel_plugin_workspace(path_prefix = "", tf_repo_name = ""):
    """All external dependencies for TF builds"""
    dpcpp_configure(name = "local_config_dpcpp")
    syslibs_configure(name = "local_config_syslibs")

    http_archive(
        name = "bazel_toolchains",
        sha256 = "109a99384f9d08f9e75136d218ebaebc68cc810c56897aea2224c57932052d30",
        strip_prefix = "bazel-toolchains-94d31935a2c94fe7e7c7379a0f3393e181928ff7",
        urls = [
            "http://mirror.tensorflow.org/github.com/bazelbuild/bazel-toolchains/archive/94d31935a2c94fe7e7c7379a0f3393e181928ff7.tar.gz",
            "https://github.com/bazelbuild/bazel-toolchains/archive/94d31935a2c94fe7e7c7379a0f3393e181928ff7.tar.gz",
        ],
    )

    tf_http_archive(
        name = "onednn_cpu",
        build_file = clean_dep("//third_party/onednn:onednn_cpu.BUILD"),
        sha256 = "5369f7b2f0b52b40890da50c0632c3a5d1082d98325d0f2bff125d19d0dcaa1d",
        strip_prefix = "oneDNN-1.6.4",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/oneapi-src/oneDNN/archive/v1.6.4.tar.gz",
            "https://github.com/oneapi-src/oneDNN/archive/v1.6.4.tar.gz",
        ],
    )

    third_party_http_archive(
        name = "com_google_absl",
        build_file = clean_dep("//third_party:com_google_absl.BUILD"),
        sha256 = "56cd3fbbbd94468a5fff58f5df2b6f9de7a0272870c61f6ca05b869934f4802a",
        strip_prefix = "abseil-cpp-daf381e8535a1f1f1b8a75966a74e7cca63dee89",
        urls = [
            "http://mirror.tensorflow.org/github.com/abseil/abseil-cpp/archive/daf381e8535a1f1f1b8a75966a74e7cca63dee89.tar.gz",
            "https://github.com/abseil/abseil-cpp/archive/daf381e8535a1f1f1b8a75966a74e7cca63dee89.tar.gz",
        ],
    )

    tf_http_archive(
        name = "double_conversion",
        build_file = clean_dep("//third_party:double_conversion.BUILD"),
        sha256 = "2f7fbffac0d98d201ad0586f686034371a6d152ca67508ab611adc2386ad30de",
        strip_prefix = "double-conversion-3992066a95b823efc8ccc1baf82a1cfc73f6e9b8",
        system_build_file = clean_dep("//third_party/systemlibs:double_conversion.BUILD"),
        urls = [
            "http://mirror.tensorflow.org/github.com/google/double-conversion/archive/3992066a95b823efc8ccc1baf82a1cfc73f6e9b8.zip",
            "https://github.com/google/double-conversion/archive/3992066a95b823efc8ccc1baf82a1cfc73f6e9b8.zip",
        ],
    )

    tf_http_archive(
        name = "zlib",
        build_file = clean_dep("//third_party:zlib.BUILD"),
        sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
        strip_prefix = "zlib-1.2.11",
        system_build_file = clean_dep("//third_party/systemlibs:zlib.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/zlib.net/zlib-1.2.11.tar.gz",
            "https://zlib.net/zlib-1.2.11.tar.gz",
        ],
    )

    tf_http_archive(
        name = "com_google_protobuf",
        patch_file = clean_dep("//third_party/protobuf:protobuf.patch"),
        #build_file = clean_dep("//third_party/systemlibs:protobuf.BUILD"),
        sha256 = "cfcba2df10feec52a84208693937c17a4b5df7775e1635c1e3baffc487b24c9b",
        strip_prefix = "protobuf-3.9.2",
        system_build_file = clean_dep("//third_party/systemlibs:protobuf.BUILD"),
        system_link_files = {
            "//third_party/systemlibs:protobuf.bzl": "protobuf.bzl",
        },
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/protocolbuffers/protobuf/archive/v3.9.2.zip",
            "https://github.com/protocolbuffers/protobuf/archive/v3.9.2.zip",
        ],
    )

    tf_http_archive(
        name = "nsync",
        sha256 = "caf32e6b3d478b78cff6c2ba009c3400f8251f646804bcb65465666a9cea93c4",
        strip_prefix = "nsync-1.22.0",
        system_build_file = clean_dep("//third_party/systemlibs:nsync.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/nsync/archive/1.22.0.tar.gz",
            "https://github.com/google/nsync/archive/1.22.0.tar.gz",
        ],
    )

    tf_http_archive(
        name = "onednn_gpu",
        build_file = clean_dep("//third_party/onednn:onednn_gpu.BUILD"),
        sha256 = "d8b79e2c772dd5fb6729d1ae262792a805bc14e1546ff70f1f4f8624e3f3bff4",
        strip_prefix = "oneDNN-7bd73fb5297425e3d38a80a49559f55fdbad2d8c",
        urls = [
            "http://mirror.tensorflow.org/github.com/oneapi-src/oneDNN/archive/7bd73fb5297425e3d38a80a49559f55fdbad2d8c.tar.gz",
            "https://github.com/oneapi-src/oneDNN/archive/7bd73fb5297425e3d38a80a49559f55fdbad2d8c.tar.gz",
        ],
    )

    tf_http_archive(
        name = "onednn_ats",
        build_file = clean_dep("//third_party/onednn:onednn_ats.BUILD"),
        sha256 = "951334bd0e362f618d72f9fc229ef199eb3807c7a466080c8990e17059fd269e",
        strip_prefix = "mkl-dnn/",
        urls = [
            "http://mirror.tensorflow.org/file://PWD" + "/third_party/onednn/mkl-dnn-ats.tar.gz",
            "file://PWD" + "//third_party/onednn/mkl-dnn-ats.tar.gz",
        ],
    )

    tf_http_archive(
        name = "eigen_archive",
        build_file = clean_dep("//third_party:eigen.BUILD"),
        # generate from Eigen commit id: 221e43077a99b76c508674f04a2e824fcd5c0158
        #sha256 = "5c8e7323275bdbff5b8583e0f0817cfd75bafbb4af522f998af95f6009b67fd4",
        sha256 = "aba063504663d52cbdaa0534ca9066617bd6ac1bced6fe8142e5326e8b403aad",
        strip_prefix = "eigen-tf2.4_master_for_plugin",
        urls = [
            "http://mirror.tensorflow.org/file://PWD" + "/third_party/gpus/dpcpp/eigen.tar.gz",
            "file://PWD" + "//third_party/eigen3/eigen.tar.gz",
        ],
    )

    tf_http_archive(
        name = "swig",
        build_file = clean_dep("//third_party:swig.BUILD"),
        sha256 = "58a475dbbd4a4d7075e5fe86d4e54c9edde39847cdb96a3053d87cb64a23a453",
        strip_prefix = "swig-3.0.8",
        system_build_file = clean_dep("//third_party/systemlibs:swig.BUILD"),
        urls = [
            "http://mirror.tensorflow.org/ufpr.dl.sourceforge.net/project/swig/swig/swig-3.0.8/swig-3.0.8.tar.gz",
            "http://ufpr.dl.sourceforge.net/project/swig/swig/swig-3.0.8/swig-3.0.8.tar.gz",
            "http://pilotfiber.dl.sourceforge.net/project/swig/swig/swig-3.0.8/swig-3.0.8.tar.gz",
        ],
    )
