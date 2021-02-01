#!/bin/bash
bazel build  -c opt //tensorflow_plugin/tools/pip_package:build_pip_package --verbose_failures
