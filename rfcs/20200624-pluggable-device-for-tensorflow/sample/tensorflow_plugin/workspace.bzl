load("//third_party:repo.bzl", "tf_http_archive", "third_party_http_archive")
load("//third_party/build_option:gcc_configure.bzl", "gcc_configure")
load("//third_party/systemlibs:syslibs_configure.bzl", "syslibs_configure")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def clean_dep(dep):
    return str(Label(dep))

def demo_plugin_workspace(path_prefix = "", tf_repo_name = ""):
    """All external dependencies for TF builds"""
    gcc_configure(name = "local_config_gcc")
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
        name = "eigen_archive",
        build_file = clean_dep("//third_party:eigen.BUILD"),
        sha256 = "df23a89e4cdfa7de2d81ee28190bd194413e47ff177c94076f845b32d7280344",  # SHARED_EIGEN_SHA
        strip_prefix = "eigen-5dc2fbabeee17fe023c38756ebde0c1d56472913",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/gitlab.com/libeigen/eigen/-/archive/5dc2fbabeee17fe023c38756ebde0c1d56472913/eigen-5dc2fbabeee17fe023c38756ebde0c1d56472913.tar.gz",
            "https://gitlab.com/libeigen/eigen/-/archive/5dc2fbabeee17fe023c38756ebde0c1d56472913/eigen-5dc2fbabeee17fe023c38756ebde0c1d56472913.tar.gz",
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
