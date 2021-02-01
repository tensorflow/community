_TF_SHARED_LIBRARY_DIR = "TF_SHARED_LIBRARY_DIR"

def _cpu_autoconf_imp(repository_ctx):
    tf_lib_path = repository_ctx.os.environ[_TF_SHARED_LIBRARY_DIR]
    repository_ctx.symlink(tf_lib_path, "proper")
    repository_ctx.file("BUILD", """
cc_import(
  name = "framework_lib",
  shared_library = "proper/libtensorflow_framework.so.2",
  # interface_library = "libtensorflow_framework.so",
  # system_provided = 1,
  visibility = ['//visibility:public'],
)
""")

gcc_configure = repository_rule(
    implementation = _cpu_autoconf_imp,
    local = True,
)
