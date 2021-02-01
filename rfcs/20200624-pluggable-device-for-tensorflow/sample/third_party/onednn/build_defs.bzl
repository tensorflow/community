def onednn_deps():
    """Shorthand for select() to pull in the correct set of MKL library deps.

    Can pull in MKL-ML, MKL-DNN, both, or neither depending on config settings.

    Returns:
      a select evaluating to a list of library dependencies, suitable for
      inclusion in the deps attribute of rules.
    """
    return select({
        str(Label("//third_party/build_option/dpcpp:build_with_dpcpp")): ["@onednn_gpu"],
        str(Label("//third_party/build_option/dpcpp:build_with_dpcpp_ats")): ["@onednn_ats"],
        "//conditions:default": ["@onednn_cpu"],
    })
