#!/usr/bin/env bash

set -e

function is_absolute {
  [[ "$1" = /* ]] || [[ "$1" =~ ^[a-zA-Z]:[/\\].* ]]
}

function real_path() {
  is_absolute "$1" && echo "$1" || echo "$PWD/${1#./}"
}

function cp_external() {
  local src_dir=$1
  local dest_dir=$2

  pushd .
  cd "$src_dir"
  for f in `find . ! -type d ! -name '*.py' ! -path '*local_config_syslibs*' ! -path '*org_tensorflow_tensorflow-plugin*'`; do
    mkdir -p "${dest_dir}/$(dirname ${f})"
    cp "${f}" "${dest_dir}/$(dirname ${f})/"
  done
  popd
}

PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"
function is_windows() {
  if [[ "${PLATFORM}" =~ (cygwin|mingw32|mingw64|msys)_nt* ]]; then
    true
  else
    false
  fi
}

function prepare_src() {
  if [ $# -lt 1 ] ; then
    echo "No destination dir provided"
    exit 1
  fi

  TMPDIR="$1"
  mkdir -p "$TMPDIR"
  EXTERNAL_INCLUDES="${TMPDIR}/tensorflow-plugins/include/external"

  echo $(date) : "=== Preparing sources in dir: ${TMPDIR}"

  if [ ! -d bazel-bin/tensorflow_plugin ]; then
    echo "Could not find bazel-bin.  Did you run from the root of the build tree?"
    exit 1
  fi

  RUNFILES=bazel-bin/tensorflow_plugin/tools/pip_package/build_pip_package.runfiles/org_tensorflow_plugin
  if [ -d bazel-bin/tensorflow_plugin/tools/pip_package/build_pip_package.runfiles/org_tensorflow_plugin/external ]; then
    # Old-style runfiles structure (--legacy_external_runfiles).
    cp -R \
      bazel-bin/tensorflow_plugin/tools/pip_package/build_pip_package.runfiles/org_tensorflow_plugin/tensorflow_plugin \
      "${TMPDIR}"
    mkdir -p ${EXTERNAL_INCLUDES}
    if [ -d bazel-tensorflow-plugin/external/com_google_absl ]; then
      cp -R bazel-tensorflow-plugin/external/com_google_absl "${EXTERNAL_INCLUDES}"
    fi
    if [ -d bazel-tensorflow-plugin/external/eigen_archive ]; then
      cp -R bazel-tensorflow-plugin/external/eigen_archive "${EXTERNAL_INCLUDES}"
    fi
    # Copy MKL libs over so they can be loaded at runtime
    so_lib_dir=$(ls $RUNFILES | grep solib) || true
    if [ -n "${so_lib_dir}" ]; then
      mkl_so_dir=$(ls ${RUNFILES}/${so_lib_dir} | grep mkl) || true
      plugin_so_dir=$(ls ${RUNFILES}/${so_lib_dir} | grep plugin) || true
      if [ -n "${mkl_so_dir}" ]; then
        mkdir "${TMPDIR}/${so_lib_dir}"
        cp -R ${RUNFILES}/${so_lib_dir}/${mkl_so_dir} "${TMPDIR}/${so_lib_dir}"
      fi
      if [ -n "${plugin_so_dir}" ]; then
        #mkdir "${TMPDIR}/${so_lib_dir}"
        cp -R -d ${RUNFILES}/${so_lib_dir}/${plugin_so_dir} "${TMPDIR}/tensorflow-plugins"
      fi
    fi
  else
    # New-style runfiles structure (--nolegacy_external_runfiles).
    cp -R \
      bazel-bin/tensorflow_plugin/tools/pip_package/build_pip_package.runfiles/org_tensorflow_plugin/plugin \
      "${TMPDIR}"
    cp_external \
      bazel-bin/tensorflow_plugin/tools/pip_package/build_pip_package.runfiles \
      "${EXTERNAL_INCLUDES}"
    # Copy MKL libs over so they can be loaded at runtime
    so_lib_dir=$(ls $RUNFILES | grep solib) || true
    if [ -n "${so_lib_dir}" ]; then
      mkl_so_dir=$(ls ${RUNFILES}/${so_lib_dir} | grep mkl) || true
      if [ -n "${mkl_so_dir}" ]; then
        mkdir "${TMPDIR}/${so_lib_dir}"
        cp -R ${RUNFILES}/${so_lib_dir}/${mkl_so_dir} "${TMPDIR}/${so_lib_dir}"
      fi
    fi
  fi
  

  cp tensorflow_plugin/tools/pip_package/MANIFEST.in ${TMPDIR}
  cp tensorflow_plugin/tools/pip_package/README ${TMPDIR}
  cp tensorflow_plugin/tools/pip_package/setup.py ${TMPDIR}
  # my_plugin_dir should be the same with _MY_PLUGIN_PATH in setup.py
  mkdir -p ${TMPDIR}/my_plugin_dir
  cp -r tensorflow_plugin/python/ ${TMPDIR}/my_plugin_dir
  touch ${TMPDIR}/my_plugin_dir/__init__.py
  if [ -d ${TMPDIR}/tensorflow_plugin ] ; then
    mv ${TMPDIR}/tensorflow_plugin/* ${TMPDIR}/tensorflow-plugins
  fi

}

function build_wheel() {
  if [ $# -lt 2 ] ; then
    echo "No src and dest dir provided"
    exit 1
  fi

  TMPDIR="$1"
  DEST="$2"
  PKG_NAME_FLAG="$3"

  # Before we leave the top-level directory, make sure we know how to
  # call python.
  if [[ -e tools/python_bin_path.sh ]]; then
    source tools/python_bin_path.sh
  fi

  pushd ${TMPDIR} > /dev/null
  rm -f MANIFEST
  echo $(date) : "=== Building wheel"
  "${PYTHON_BIN_PATH:-python}" setup.py bdist_wheel ${PKG_NAME_FLAG} >/dev/null
  mkdir -p ${DEST}
  cp dist/* ${DEST}
  popd > /dev/null
  echo $(date) : "=== Output wheel file is in: ${DEST}"
}

function usage() {
  echo "Usage:"
  echo "$0 [--src srcdir] [--dst dstdir] [options]"
  echo "$0 dstdir [options]"
  echo ""
  echo "    --src                 prepare sources in srcdir"
  echo "                              will use temporary dir if not specified"
  echo ""
  echo "    --dst                 build wheel in dstdir"
  echo "                              if dstdir is not set do not build, only prepare sources"
  echo ""
  exit 1
}

function main() {
  PKG_NAME_FLAG=""
  PROJECT_NAME=""
  GPU_BUILD=0
  NIGHTLY_BUILD=0
  SRCDIR=""
  DSTDIR=""
  CLEANSRC=1
  while true; do
    if [[ "$1" == "--help" ]]; then
      usage
      exit 1
    elif [[ "$1" == "--project_name" ]]; then
      shift
      if [[ -z "$1" ]]; then
        break
      fi
      PROJECT_NAME="$1"
    elif [[ "$1" == "--src" ]]; then
      shift
      SRCDIR="$(real_path $1)"
      CLEANSRC=0
    elif [[ "$1" == "--dst" ]]; then
      shift
      DSTDIR="$(real_path $1)"
    else
      DSTDIR="$(real_path $1)"
    fi
    shift

    if [[ -z "$1" ]]; then
      break
    fi
  done

  if [[ -z "$DSTDIR" ]] && [[ -z "$SRCDIR" ]]; then
    echo "No destination dir provided"
    usage
    exit 1
  fi

  if [[ -z "$SRCDIR" ]]; then
    # make temp srcdir if none set
    SRCDIR="$(mktemp -d -t tmp.XXXXXXXXXX)"
  fi

  prepare_src "$SRCDIR"

  if [[ -z "$DSTDIR" ]]; then
      # only want to prepare sources
      exit
  fi

  build_wheel "$SRCDIR" "$DSTDIR" "$PKG_NAME_FLAG"

  if [[ $CLEANSRC -ne 0 ]]; then
    rm -rf "${TMPDIR}"
  fi
}

main "$@"
