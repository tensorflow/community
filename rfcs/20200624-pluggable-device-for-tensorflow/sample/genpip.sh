#!/bin/bash
PACKAGE_PATH=$PWD
bazel-bin/tensorflow_plugin/tools/pip_package/build_pip_package $PACKAGE_PATH

