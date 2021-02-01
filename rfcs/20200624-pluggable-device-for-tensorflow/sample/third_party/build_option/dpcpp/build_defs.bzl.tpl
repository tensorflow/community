# Macros for building DPCPP code.
def if_dpcpp(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with DPCPP.

    Returns a select statement which evaluates to if_true if we're building
    with DPCPP enabled.  Otherwise, the select statement evaluates to if_false.

    """
    return select({
        "@local_config_dpcpp//dpcpp:using_dpcpp": if_true,
        "//conditions:default": if_false,
    })

def dpcpp_is_configured():
    """Returns true if DPCPP was enabled during the configure process."""
    return %{dpcpp_is_configured}

def dpcpp_build_is_configured():
    """Returns true if DPCPP compiler was enabled during the configure process."""
    return %{dpcpp_build_is_configured}

def if_dpcpp_is_configured(x):
    """Tests if the DPCPP was enabled during the configure process.

    Unlike if_dpcpp(), this does not require that we are building with
    --config=dpcpp. Used to allow non-DPCPP code to depend on DPCPP libraries.
    """
    if dpcpp_is_configured():
      return select({"//conditions:default": x})
    return select({"//conditions:default": []})

def if_dpcpp_build_is_configured(x, y):
    if dpcpp_build_is_configured():
      return x
    return y
