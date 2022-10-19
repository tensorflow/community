# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""configure script to get build parameters from user."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import errno
import os
import platform
import re
import subprocess
import sys

# pylint: disable=g-import-not-at-top
try:
  from shutil import which
except ImportError:
  from distutils.spawn import find_executable as which
# pylint: enable=g-import-not-at-top


_DEFAULT_GCC_TOOLCHAIN_PATH = ''
_DEFAULT_GCC_TOOLCHAIN_TARGET = ''

_DEFAULT_PROMPT_ASK_ATTEMPTS = 10

_TF_BAZELRC_FILENAME = '.tf_plugin_configure.bazelrc'
_TF_WORKSPACE_ROOT = ''
_TF_BAZELRC = ''
_TF_CURRENT_BAZEL_VERSION = None

NCCL_LIB_PATHS = [
    'lib64/', 'lib/powerpc64le-linux-gnu/', 'lib/x86_64-linux-gnu/', ''
]


class UserInputError(Exception):
  pass


def is_windows():
  return platform.system() == 'Windows'


def is_linux():
  return platform.system() == 'Linux'


def is_macos():
  return platform.system() == 'Darwin'


def is_ppc64le():
  return platform.machine() == 'ppc64le'


def is_cygwin():
  return platform.system().startswith('CYGWIN_NT')


def get_input(question):
  try:
    try:
      answer = raw_input(question)
    except NameError:
      answer = input(question)  # pylint: disable=bad-builtin
  except EOFError:
    answer = ''
  return answer


def symlink_force(target, link_name):
  """Force symlink, equivalent of 'ln -sf'.

  Args:
    target: items to link to.
    link_name: name of the link.
  """
  try:
    os.symlink(target, link_name)
  except OSError as e:
    if e.errno == errno.EEXIST:
      os.remove(link_name)
      os.symlink(target, link_name)
    else:
      raise e


def sed_in_place(filename, old, new):
  """Replace old string with new string in file.

  Args:
    filename: string for filename.
    old: string to replace.
    new: new string to replace to.
  """
  with open(filename, 'r') as f:
    filedata = f.read()
  newdata = filedata.replace(old, new)
  with open(filename, 'w') as f:
    f.write(newdata)


def write_to_bazelrc(line):
  with open(_TF_BAZELRC, 'a') as f:
    f.write(line + '\n')


def write_action_env_to_bazelrc(var_name, var):
  write_to_bazelrc('build --action_env %s="%s"' % (var_name, str(var)))


def run_shell(cmd, allow_non_zero=False):
  if allow_non_zero:
    try:
      output = subprocess.check_output(cmd)
    except subprocess.CalledProcessError as e:
      output = e.output
  else:
    output = subprocess.check_output(cmd)
  return output.decode('UTF-8').strip()


def cygpath(path):
  """Convert path from posix to windows."""
  return os.path.abspath(path).replace('\\', '/')


def get_python_path(environ_cp, python_bin_path):
  """Get the python site package paths."""
  python_paths = []
  if environ_cp.get('PYTHONPATH'):
    python_paths = environ_cp.get('PYTHONPATH').split(':')
  try:
    library_paths = run_shell([
        python_bin_path, '-c',
        'import site; print("\\n".join(site.getsitepackages()))'
    ]).split('\n')
  except subprocess.CalledProcessError:
    library_paths = [
        run_shell([
            python_bin_path, '-c',
            'from distutils.sysconfig import get_python_lib;'
            'print(get_python_lib())'
        ])
    ]

  all_paths = set(python_paths + library_paths)

  paths = []
  for path in all_paths:
    if os.path.isdir(path):
      paths.append(path)
  return paths


def get_python_major_version(python_bin_path):
  """Get the python major version."""
  return run_shell([python_bin_path, '-c', 'import sys; print(sys.version[0])'])


def setup_python(environ_cp):
  """Setup python related env variables."""
  # Get PYTHON_BIN_PATH, default is the current running python.
  default_python_bin_path = sys.executable
  ask_python_bin_path = ('Please specify the location of python. [Default is '
                         '%s]: ') % default_python_bin_path
  while True:
    python_bin_path = get_from_env_or_user_or_default(environ_cp,
                                                      'PYTHON_BIN_PATH',
                                                      ask_python_bin_path,
                                                      default_python_bin_path)
    # Check if the path is valid
    if os.path.isfile(python_bin_path) and os.access(python_bin_path, os.X_OK):
      break
    elif not os.path.exists(python_bin_path):
      print('Invalid python path: %s cannot be found.' % python_bin_path)
    else:
      print('%s is not executable.  Is it the python binary?' % python_bin_path)
    environ_cp['PYTHON_BIN_PATH'] = ''

  # Convert python path to Windows style before checking lib and version
  if is_windows() or is_cygwin():
    python_bin_path = cygpath(python_bin_path)

  # Get PYTHON_LIB_PATH
  python_lib_path = environ_cp.get('PYTHON_LIB_PATH')
  if not python_lib_path:
    python_lib_paths = get_python_path(environ_cp, python_bin_path)
    if environ_cp.get('USE_DEFAULT_PYTHON_LIB_PATH') == '1':
      python_lib_path = python_lib_paths[0]
    else:
      print('Found possible Python library paths:\n  %s' %
            '\n  '.join(python_lib_paths))
      default_python_lib_path = python_lib_paths[0]
      python_lib_path = get_input(
          'Please input the desired Python library path to use.  '
          'Default is [%s]\n' % python_lib_paths[0])
      if not python_lib_path:
        python_lib_path = default_python_lib_path
    environ_cp['PYTHON_LIB_PATH'] = python_lib_path

  _ = get_python_major_version(python_bin_path)

  # Convert python path to Windows style before writing into bazel.rc
  if is_windows() or is_cygwin():
    python_lib_path = cygpath(python_lib_path)

  # Set-up env variables used by python_configure.bzl
  write_action_env_to_bazelrc('PYTHON_BIN_PATH', python_bin_path)
  write_action_env_to_bazelrc('PYTHON_LIB_PATH', python_lib_path)
  write_to_bazelrc('build --python_path=\"%s"' % python_bin_path)
  environ_cp['PYTHON_BIN_PATH'] = python_bin_path

  # If choosen python_lib_path is from a path specified in the PYTHONPATH
  # variable, need to tell bazel to include PYTHONPATH
  if environ_cp.get('PYTHONPATH'):
    python_paths = environ_cp.get('PYTHONPATH').split(':')
    if python_lib_path in python_paths:
      write_action_env_to_bazelrc('PYTHONPATH', environ_cp.get('PYTHONPATH'))

  # Write tools/python_bin_path.sh
  with open(
      os.path.join(_TF_WORKSPACE_ROOT, 'tensorflow_plugin', 'tools', 'python_bin_path.sh'),
      'w') as f:
    f.write('export PYTHON_BIN_PATH="%s"' % python_bin_path)

def get_python_lib_name(environ_cp):
    python_bin_path = environ_cp['PYTHON_BIN_PATH']
    path_list = python_bin_path.split(os.sep)[:-2]
    path_list.append('lib')
    py_lib_path = os.sep.join(path_list)
    for _, _, files in os.walk(py_lib_path):
        for name in files:
            if str(name).startswith('libpython') and str(name).endswith('.so'):
                # strip libxxx.so to get xxx
                return str(name).strip()[3:-3]


def get_python_link_path(environ_cp):
    # TODO(quintin): we need to link libpythonx.y.so for _pywrap_tensorflow_internal.so
    # once google change CAPI symbols into libtensorflow.so, we don't need this
    python_bin_path = environ_cp['PYTHON_BIN_PATH']
    path_list = python_bin_path.split(os.sep)[:-2]
    path_list.append('lib')
    py_lib_path = os.sep.join(path_list)
    return py_lib_path

def create_build_configuration(environ_cp):

    tf_header_dir = environ_cp['PYTHON_LIB_PATH'] + "/tensorflow/include"
    tf_shared_lib_dir = environ_cp['PYTHON_LIB_PATH'] + "/tensorflow/"

    write_action_env_to_bazelrc("TF_HEADER_DIR", tf_header_dir)
    write_action_env_to_bazelrc("TF_SHARED_LIBRARY_DIR", tf_shared_lib_dir)
    write_action_env_to_bazelrc("TF_CXX11_ABI_FLAG", 1)
    write_action_env_to_bazelrc("PYTHON_LINK_LIB_NAME", get_python_lib_name(environ_cp))
    write_action_env_to_bazelrc("PYTHON_LINK_PATH", get_python_link_path(environ_cp))


def reset_tf_configure_bazelrc():
  """Reset file that contains customized config settings."""
  open(_TF_BAZELRC, 'w').close()


def cleanup_makefile():
  """Delete any leftover BUILD files from the Makefile build.

  These files could interfere with Bazel parsing.
  """
  makefile_download_dir = os.path.join(_TF_WORKSPACE_ROOT, 'tensorflow',
                                       'contrib', 'makefile', 'downloads')
  if os.path.isdir(makefile_download_dir):
    for root, _, filenames in os.walk(makefile_download_dir):
      for f in filenames:
        if f.endswith('BUILD'):
          os.remove(os.path.join(root, f))


def get_var(environ_cp,
            var_name,
            query_item,
            enabled_by_default,
            question=None,
            yes_reply=None,
            no_reply=None):
  """Get boolean input from user.

  If var_name is not set in env, ask user to enable query_item or not. If the
  response is empty, use the default.

  Args:
    environ_cp: copy of the os.environ.
    var_name: string for name of environment variable, e.g. "TF_NEED_CUDA".
    query_item: string for feature related to the variable, e.g. "CUDA for
      Nvidia GPUs".
    enabled_by_default: boolean for default behavior.
    question: optional string for how to ask for user input.
    yes_reply: optional string for reply when feature is enabled.
    no_reply: optional string for reply when feature is disabled.

  Returns:
    boolean value of the variable.

  Raises:
    UserInputError: if an environment variable is set, but it cannot be
      interpreted as a boolean indicator, assume that the user has made a
      scripting error, and will continue to provide invalid input.
      Raise the error to avoid infinitely looping.
  """
  if not question:
    question = 'Do you wish to build TensorFlow plug-in with %s support?' % query_item
  if not yes_reply:
    yes_reply = '%s support will be enabled for TensorFlow plug-in.' % query_item
  if not no_reply:
    no_reply = 'No %s' % yes_reply

  yes_reply += '\n'
  no_reply += '\n'

  if enabled_by_default:
    question += ' [Y/n]: '
  else:
    question += ' [y/N]: '

  var = environ_cp.get(var_name)
  if var is not None:
    var_content = var.strip().lower()
    true_strings = ('1', 't', 'true', 'y', 'yes')
    false_strings = ('0', 'f', 'false', 'n', 'no')
    if var_content in true_strings:
      var = True
    elif var_content in false_strings:
      var = False
    else:
      raise UserInputError(
          'Environment variable %s must be set as a boolean indicator.\n'
          'The following are accepted as TRUE : %s.\n'
          'The following are accepted as FALSE: %s.\n'
          'Current value is %s.' %
          (var_name, ', '.join(true_strings), ', '.join(false_strings), var))

  while var is None:
    user_input_origin = get_input(question)
    user_input = user_input_origin.strip().lower()
    if user_input == 'y':
      print(yes_reply)
      var = True
    elif user_input == 'n':
      print(no_reply)
      var = False
    elif not user_input:
      if enabled_by_default:
        print(yes_reply)
        var = True
      else:
        print(no_reply)
        var = False
    else:
      print('Invalid selection: %s' % user_input_origin)
  return var


def set_build_var(environ_cp,
                  var_name,
                  query_item,
                  option_name,
                  enabled_by_default,
                  bazel_config_name=None):
  """Set if query_item will be enabled for the build.

  Ask user if query_item will be enabled. Default is used if no input is given.
  Set subprocess environment variable and write to .bazelrc if enabled.

  Args:
    environ_cp: copy of the os.environ.
    var_name: string for name of environment variable, e.g. "TF_NEED_CUDA".
    query_item: string for feature related to the variable, e.g. "CUDA for
      Nvidia GPUs".
    option_name: string for option to define in .bazelrc.
    enabled_by_default: boolean for default behavior.
    bazel_config_name: Name for Bazel --config argument to enable build feature.
  """

  var = str(int(get_var(environ_cp, var_name, query_item, enabled_by_default)))
  environ_cp[var_name] = var
  if var == '1':
    write_to_bazelrc('build:%s --define %s=true' %
                     (bazel_config_name, option_name))
    write_to_bazelrc('build --config=%s' % bazel_config_name)
  elif bazel_config_name is not None:
    # TODO(mikecase): Migrate all users of configure.py to use --config Bazel
    # options and not to set build configs through environment variables.
    write_to_bazelrc('build:%s --define %s=true' %
                     (bazel_config_name, option_name))


def set_action_env_var(environ_cp,
                       var_name,
                       query_item,
                       enabled_by_default,
                       question=None,
                       yes_reply=None,
                       no_reply=None):
  """Set boolean action_env variable.

  Ask user if query_item will be enabled. Default is used if no input is given.
  Set environment variable and write to .bazelrc.

  Args:
    environ_cp: copy of the os.environ.
    var_name: string for name of environment variable, e.g. "TF_NEED_CUDA".
    query_item: string for feature related to the variable, e.g. "CUDA for
      Nvidia GPUs".
    enabled_by_default: boolean for default behavior.
    question: optional string for how to ask for user input.
    yes_reply: optional string for reply when feature is enabled.
    no_reply: optional string for reply when feature is disabled.
  """
  var = int(
      get_var(environ_cp, var_name, query_item, enabled_by_default, question,
              yes_reply, no_reply))

  write_action_env_to_bazelrc(var_name, var)
  environ_cp[var_name] = str(var)


def convert_version_to_int(version):
  """Convert a version number to a integer that can be used to compare.

  Version strings of the form X.YZ and X.Y.Z-xxxxx are supported. The
  'xxxxx' part, for instance 'homebrew' on OS/X, is ignored.

  Args:
    version: a version to be converted

  Returns:
    An integer if converted successfully, otherwise return None.
  """
  version = version.split('-')[0]
  version_segments = version.split('.')
  # Treat "0.24" as "0.24.0"
  if len(version_segments) == 2:
    version_segments.append('0')
  for seg in version_segments:
    if not seg.isdigit():
      return None

  version_str = ''.join(['%03d' % int(seg) for seg in version_segments])
  return int(version_str)


def check_bazel_version(min_version, max_version):
  """Check installed bazel version is between min_version and max_version.

  Args:
    min_version: string for minimum bazel version (must exist!).
    max_version: string for maximum bazel version (must exist!).

  Returns:
    The bazel version detected.
  """
  if which('bazel') is None:
    print('Cannot find bazel. Please install bazel.')
    sys.exit(0)
  curr_version = run_shell(
      ['bazel', '--batch', '--bazelrc=/dev/null', 'version'])

  for line in curr_version.split('\n'):
    if 'Build label: ' in line:
      curr_version = line.split('Build label: ')[1]
      break

  min_version_int = convert_version_to_int(min_version)
  curr_version_int = convert_version_to_int(curr_version)
  max_version_int = convert_version_to_int(max_version)

  # Check if current bazel version can be detected properly.
  if not curr_version_int:
    print('WARNING: current bazel installation is not a release version.')
    print('Make sure you are running at least bazel %s' % min_version)
    return curr_version

  print('You have bazel %s installed.' % curr_version)

  if curr_version_int < min_version_int:
    print('Please upgrade your bazel installation to version %s or higher to '
          'build TensorFlow!' % min_version)
    sys.exit(1)
  if (curr_version_int > max_version_int and
      'TF_IGNORE_MAX_BAZEL_VERSION' not in os.environ):
    print('Please downgrade your bazel installation to version %s or lower to '
          'build TensorFlow! To downgrade: download the installer for the old '
          'version (from https://github.com/bazelbuild/bazel/releases) then '
          'run the installer.' % max_version)
    sys.exit(1)
  return curr_version


def set_cc_opt_flags(environ_cp):
  """Set up architecture-dependent optimization flags.

  Also append CC optimization flags to bazel.rc..

  Args:
    environ_cp: copy of the os.environ.
  """
  if is_ppc64le():
    # gcc on ppc64le does not support -march, use mcpu instead
    default_cc_opt_flags = '-mcpu=native'
  elif is_windows():
    default_cc_opt_flags = '/arch:AVX'
  else:
    default_cc_opt_flags = '-march=native -Wno-sign-compare'
  question = ('Please specify optimization flags to use during compilation when'
              ' bazel option "--config=opt" is specified [Default is %s]: '
             ) % default_cc_opt_flags
  cc_opt_flags = get_from_env_or_user_or_default(environ_cp, 'CC_OPT_FLAGS',
                                                 question, default_cc_opt_flags)
  for opt in cc_opt_flags.split():
    write_to_bazelrc('build:opt --copt=%s' % opt)
  # It should be safe on the same build host.
  if not is_ppc64le() and not is_windows():
    write_to_bazelrc('build:opt --host_copt=-march=native')
  write_to_bazelrc('build:opt --define with_default_optimizations=true')



def set_tf_download_clang(environ_cp):
  """Set TF_DOWNLOAD_CLANG action_env."""
  question = 'Do you wish to download a fresh release of clang? (Experimental)'
  yes_reply = 'Clang will be downloaded and used to compile tensorflow.'
  no_reply = 'Clang will not be downloaded.'
  set_action_env_var(
      environ_cp,
      'TF_DOWNLOAD_CLANG',
      None,
      False,
      question=question,
      yes_reply=yes_reply,
      no_reply=no_reply)


def get_from_env_or_user_or_default(environ_cp, var_name, ask_for_var,
                                    var_default):
  """Get var_name either from env, or user or default.

  If var_name has been set as environment variable, use the preset value, else
  ask for user input. If no input is provided, the default is used.

  Args:
    environ_cp: copy of the os.environ.
    var_name: string for name of environment variable, e.g. "TF_NEED_CUDA".
    ask_for_var: string for how to ask for user input.
    var_default: default value string.

  Returns:
    string value for var_name
  """
  var = environ_cp.get(var_name)
  if not var:
    var = get_input(ask_for_var)
    print('\n')
  if not var:
    var = var_default
  return var


def prompt_loop_or_load_from_env(environ_cp,
                                 var_name,
                                 var_default,
                                 ask_for_var,
                                 check_success,
                                 error_msg,
                                 suppress_default_error=False,
                                 n_ask_attempts=_DEFAULT_PROMPT_ASK_ATTEMPTS):
  """Loop over user prompts for an ENV param until receiving a valid response.

  For the env param var_name, read from the environment or verify user input
  until receiving valid input. When done, set var_name in the environ_cp to its
  new value.

  Args:
    environ_cp: (Dict) copy of the os.environ.
    var_name: (String) string for name of environment variable, e.g. "TF_MYVAR".
    var_default: (String) default value string.
    ask_for_var: (String) string for how to ask for user input.
    check_success: (Function) function that takes one argument and returns a
      boolean. Should return True if the value provided is considered valid. May
      contain a complex error message if error_msg does not provide enough
      information. In that case, set suppress_default_error to True.
    error_msg: (String) String with one and only one '%s'. Formatted with each
      invalid response upon check_success(input) failure.
    suppress_default_error: (Bool) Suppress the above error message in favor of
      one from the check_success function.
    n_ask_attempts: (Integer) Number of times to query for valid input before
      raising an error and quitting.

  Returns:
    [String] The value of var_name after querying for input.

  Raises:
    UserInputError: if a query has been attempted n_ask_attempts times without
      success, assume that the user has made a scripting error, and will
      continue to provide invalid input. Raise the error to avoid infinitely
      looping.
  """
  default = environ_cp.get(var_name) or var_default
  full_query = '%s [Default is %s]: ' % (
      ask_for_var,
      default,
  )

  for _ in range(n_ask_attempts):
    val = get_from_env_or_user_or_default(environ_cp, var_name, full_query,
                                          default)
    if check_success(val):
      break
    if not suppress_default_error:
      print(error_msg % val)
    environ_cp[var_name] = ''
  else:
    raise UserInputError('Invalid %s setting was provided %d times in a row. '
                         'Assuming to be a scripting mistake.' %
                         (var_name, n_ask_attempts))

  environ_cp[var_name] = val
  return val


def create_android_ndk_rule(environ_cp):
  """Set ANDROID_NDK_HOME and write Android NDK WORKSPACE rule."""
  if is_windows() or is_cygwin():
    default_ndk_path = cygpath('%s/Android/Sdk/ndk-bundle' %
                               environ_cp['APPDATA'])
  elif is_macos():
    default_ndk_path = '%s/library/Android/Sdk/ndk-bundle' % environ_cp['HOME']
  else:
    default_ndk_path = '%s/Android/Sdk/ndk-bundle' % environ_cp['HOME']

  def valid_ndk_path(path):
    return (os.path.exists(path) and
            os.path.exists(os.path.join(path, 'source.properties')))

  android_ndk_home_path = prompt_loop_or_load_from_env(
      environ_cp,
      var_name='ANDROID_NDK_HOME',
      var_default=default_ndk_path,
      ask_for_var='Please specify the home path of the Android NDK to use.',
      check_success=valid_ndk_path,
      error_msg=('The path %s or its child file "source.properties" '
                 'does not exist.'))
  write_action_env_to_bazelrc('ANDROID_NDK_HOME', android_ndk_home_path)
  write_action_env_to_bazelrc(
      'ANDROID_NDK_API_LEVEL',
      get_ndk_api_level(environ_cp, android_ndk_home_path))


def create_android_sdk_rule(environ_cp):
  """Set Android variables and write Android SDK WORKSPACE rule."""
  if is_windows() or is_cygwin():
    default_sdk_path = cygpath('%s/Android/Sdk' % environ_cp['APPDATA'])
  elif is_macos():
    default_sdk_path = '%s/library/Android/Sdk' % environ_cp['HOME']
  else:
    default_sdk_path = '%s/Android/Sdk' % environ_cp['HOME']

  def valid_sdk_path(path):
    return (os.path.exists(path) and
            os.path.exists(os.path.join(path, 'platforms')) and
            os.path.exists(os.path.join(path, 'build-tools')))

  android_sdk_home_path = prompt_loop_or_load_from_env(
      environ_cp,
      var_name='ANDROID_SDK_HOME',
      var_default=default_sdk_path,
      ask_for_var='Please specify the home path of the Android SDK to use.',
      check_success=valid_sdk_path,
      error_msg=('Either %s does not exist, or it does not contain the '
                 'subdirectories "platforms" and "build-tools".'))

  platforms = os.path.join(android_sdk_home_path, 'platforms')
  api_levels = sorted(os.listdir(platforms))
  api_levels = [x.replace('android-', '') for x in api_levels]

  def valid_api_level(api_level):
    return os.path.exists(
        os.path.join(android_sdk_home_path, 'platforms',
                     'android-' + api_level))

  android_api_level = prompt_loop_or_load_from_env(
      environ_cp,
      var_name='ANDROID_API_LEVEL',
      var_default=api_levels[-1],
      ask_for_var=('Please specify the Android SDK API level to use. '
                   '[Available levels: %s]') % api_levels,
      check_success=valid_api_level,
      error_msg='Android-%s is not present in the SDK path.')

  build_tools = os.path.join(android_sdk_home_path, 'build-tools')
  versions = sorted(os.listdir(build_tools))

  def valid_build_tools(version):
    return os.path.exists(
        os.path.join(android_sdk_home_path, 'build-tools', version))

  android_build_tools_version = prompt_loop_or_load_from_env(
      environ_cp,
      var_name='ANDROID_BUILD_TOOLS_VERSION',
      var_default=versions[-1],
      ask_for_var=('Please specify an Android build tools version to use. '
                   '[Available versions: %s]') % versions,
      check_success=valid_build_tools,
      error_msg=('The selected SDK does not have build-tools version %s '
                 'available.'))

  write_action_env_to_bazelrc('ANDROID_BUILD_TOOLS_VERSION',
                              android_build_tools_version)
  write_action_env_to_bazelrc('ANDROID_SDK_API_LEVEL', android_api_level)
  write_action_env_to_bazelrc('ANDROID_SDK_HOME', android_sdk_home_path)


def get_ndk_api_level(environ_cp, android_ndk_home_path):
  """Gets the appropriate NDK API level to use for the provided Android NDK path."""

  # First check to see if we're using a blessed version of the NDK.
  properties_path = '%s/source.properties' % android_ndk_home_path
  if is_windows() or is_cygwin():
    properties_path = cygpath(properties_path)
  with open(properties_path, 'r') as f:
    filedata = f.read()

  revision = re.search(r'Pkg.Revision = (\d+)', filedata)
  if revision:
    ndk_version = revision.group(1)
  else:
    raise Exception('Unable to parse NDK revision.')
  if int(ndk_version) not in _SUPPORTED_ANDROID_NDK_VERSIONS:
    print('WARNING: The NDK version in %s is %s, which is not '
          'supported by Bazel (officially supported versions: %s). Please use '
          'another version. Compiling Android targets may result in confusing '
          'errors.\n' % (android_ndk_home_path, ndk_version,
                         _SUPPORTED_ANDROID_NDK_VERSIONS))

  # Now grab the NDK API level to use. Note that this is different from the
  # SDK API level, as the NDK API level is effectively the *min* target SDK
  # version.
  platforms = os.path.join(android_ndk_home_path, 'platforms')
  api_levels = sorted(os.listdir(platforms))
  api_levels = [
      x.replace('android-', '') for x in api_levels if 'android-' in x
  ]

  def valid_api_level(api_level):
    return os.path.exists(
        os.path.join(android_ndk_home_path, 'platforms',
                     'android-' + api_level))

  android_ndk_api_level = prompt_loop_or_load_from_env(
      environ_cp,
      var_name='ANDROID_NDK_API_LEVEL',
      var_default='18',  # 18 is required for GPU acceleration.
      ask_for_var=('Please specify the (min) Android NDK API level to use. '
                   '[Available levels: %s]') % api_levels,
      check_success=valid_api_level,
      error_msg='Android-%s is not present in the NDK path.')

  return android_ndk_api_level


def set_gcc_host_compiler_path(environ_cp):
  """Set GCC_HOST_COMPILER_PATH."""
  default_gcc_host_compiler_path = which('gcc')
  if os.path.islink(default_gcc_host_compiler_path):
    # os.readlink is only available in linux
    default_gcc_host_compiler_path = os.path.realpath(default_gcc_host_compiler_path)

  gcc_host_compiler_path = prompt_loop_or_load_from_env(
      environ_cp,
      var_name='GCC_HOST_COMPILER_PATH',
      var_default=default_gcc_host_compiler_path,
      ask_for_var='Please specify which gcc should be used by nvcc as the host compiler.',
      check_success=os.path.exists,
      error_msg='Invalid gcc path. %s cannot be found.',
  )

  write_action_env_to_bazelrc('GCC_HOST_COMPILER_PATH', gcc_host_compiler_path)


def reformat_version_sequence(version_str, sequence_count):
  """Reformat the version string to have the given number of sequences.

  For example:
  Given (7, 2) -> 7.0
        (7.0.1, 2) -> 7.0
        (5, 1) -> 5
        (5.0.3.2, 1) -> 5

  Args:
      version_str: String, the version string.
      sequence_count: int, an integer.

  Returns:
      string, reformatted version string.
  """
  v = version_str.split('.')
  if len(v) < sequence_count:
    v = v + (['0'] * (sequence_count - len(v)))

  return '.'.join(v[:sequence_count])


def set_host_cxx_compiler(environ_cp):
  """Set HOST_CXX_COMPILER."""
  default_cxx_host_compiler = which('g++') or ''

  host_cxx_compiler = prompt_loop_or_load_from_env(
      environ_cp,
      var_name='HOST_CXX_COMPILER',
      var_default=default_cxx_host_compiler,
      ask_for_var=('Please specify which C++ compiler should be used as the '
                   'host C++ compiler.'),
      check_success=os.path.exists,
      error_msg='Invalid C++ compiler path. %s cannot be found.',
  )

  write_action_env_to_bazelrc('HOST_CXX_COMPILER', host_cxx_compiler)


def set_host_c_compiler(environ_cp):
  """Set HOST_C_COMPILER."""
  default_c_host_compiler = which('gcc') or ''

  host_c_compiler = prompt_loop_or_load_from_env(
      environ_cp,
      var_name='HOST_C_COMPILER',
      var_default=default_c_host_compiler,
      ask_for_var=('Please specify which C compiler should be used as the host '
                   'C compiler.'),
      check_success=os.path.exists,
      error_msg='Invalid C compiler path. %s cannot be found.',
  )

  write_action_env_to_bazelrc('HOST_C_COMPILER', host_c_compiler)


def set_opencl_sdk_root(environ_cp):
  """Set OPENCL SDK ROOT"""

  def toolkit_exists(toolkit_path):
    """Check if a CL header path is valid."""
    if toolkit_path == '':
      return True

    if is_linux():
      cl_header_path = 'opencl/SDK/include/CL/cl.h'
    else:
      cl_header_path = ''

    cl_path_full = os.path.join(toolkit_path, cl_header_path)
    exists = os.path.exists(cl_path_full)
    if not exists:
      print('Invalid OPENCL SDK ROOT path. %s cannot be found' %
            (cl_path_full))
    return exists

  ocl_sdk_root = prompt_loop_or_load_from_env(
      environ_cp,
      var_name='OCL_SDK_ROOT',
      var_default=_DEFAULT_OCL_SDK_ROOT,
      ask_for_var=(
          'Please specify the location of opencl SDK install path '
          'for ocl headers and libOpenCL.so'),
      check_success=toolkit_exists,
      error_msg='Invalid OPENCL SDK ROOT path.',
      suppress_default_error=True)

  write_action_env_to_bazelrc('OCL_SDK_ROOT',
                              ocl_sdk_root)

def set_gcc_toolchain_path(environ_cp):
  """Set GCC_TOOLCHAIN_PATH."""
  def no_check(arg):
      return True

  gcc_toolchain_path = prompt_loop_or_load_from_env(
      environ_cp,
      var_name='GCC_TOOLCHAIN_PATH',
      var_default=_DEFAULT_GCC_TOOLCHAIN_PATH,
      ask_for_var=(
          'Please specify the location of gcc toolchain used by the compiler'),
      check_success=no_check,
      error_msg='Invalid GCC_TOOLCHAIN path.',
      suppress_default_error=True)

  write_action_env_to_bazelrc('GCC_TOOLCHAIN_PATH',
                              gcc_toolchain_path)
  return gcc_toolchain_path

def set_gcc_toolchain_target(environ_cp, gcc_toolchain_path):
  """Set GCC_TOOLCHAIN_TARGET."""
  if gcc_toolchain_path == "":
    return ""

  def toolkit_exists(target):
    """Check if a gcc toolchain-target is valid."""
    if is_linux():
      if target == '':
        gcc_bin_path = 'bin/gcc'
      else:
        gcc_bin_path = 'bin/' + target + '-gcc'
    else:
      gcc_bin_path = ''

    gcc_bin_path_full = os.path.join(gcc_toolchain_path, gcc_bin_path)
    exists = os.path.exists(gcc_bin_path_full)
    if not exists:
      print('Invalid GCC_TOOLCHAIN path and TARGET. %s cannot be found' %
            (gcc_bin_path_full))
    return exists

  gcc_toolchain_target = prompt_loop_or_load_from_env(
      environ_cp,
      var_name='GCC_TOOLCHAIN_TARGET',
      var_default=_DEFAULT_GCC_TOOLCHAIN_TARGET,
      ask_for_var=(
          'Please specify the target of gcc toolchain (e.g. x86_64-pc-linux) '
          'the compiler will use.'),
      check_success=toolkit_exists,
      error_msg='Invalid GCC_TOOLCHAIN_TARGET',
      suppress_default_error=True)

  write_action_env_to_bazelrc('GCC_TOOLCHAIN_TARGET',
                              gcc_toolchain_target)

def set_mpi_home(environ_cp):
  """Set MPI_HOME."""

  default_mpi_home = which('mpirun') or which('mpiexec') or ''
  default_mpi_home = os.path.dirname(os.path.dirname(default_mpi_home))

  def valid_mpi_path(mpi_home):
    exists = (
        os.path.exists(os.path.join(mpi_home, 'include')) and
        (os.path.exists(os.path.join(mpi_home, 'lib')) or
         os.path.exists(os.path.join(mpi_home, 'lib64')) or
         os.path.exists(os.path.join(mpi_home, 'lib32'))))
    if not exists:
      print(
          'Invalid path to the MPI Toolkit. %s or %s or %s or %s cannot be found'
          % (os.path.join(mpi_home, 'include'),
             os.path.exists(os.path.join(mpi_home, 'lib')),
             os.path.exists(os.path.join(mpi_home, 'lib64')),
             os.path.exists(os.path.join(mpi_home, 'lib32'))))
    return exists

  _ = prompt_loop_or_load_from_env(
      environ_cp,
      var_name='MPI_HOME',
      var_default=default_mpi_home,
      ask_for_var='Please specify the MPI toolkit folder.',
      check_success=valid_mpi_path,
      error_msg='',
      suppress_default_error=True)


def set_other_mpi_vars(environ_cp):
  """Set other MPI related variables."""
  # Link the MPI header files
  mpi_home = environ_cp.get('MPI_HOME')
  symlink_force('%s/include/mpi.h' % mpi_home, 'third_party/mpi/mpi.h')

  # Determine if we use OpenMPI or MVAPICH, these require different header files
  # to be included here to make bazel dependency checker happy
  if os.path.exists(os.path.join(mpi_home, 'include/mpi_portable_platform.h')):
    symlink_force(
        os.path.join(mpi_home, 'include/mpi_portable_platform.h'),
        'third_party/mpi/mpi_portable_platform.h')
    # TODO(gunan): avoid editing files in configure
    sed_in_place('third_party/mpi/mpi.bzl', 'MPI_LIB_IS_OPENMPI=False',
                 'MPI_LIB_IS_OPENMPI=True')
  else:
    # MVAPICH / MPICH
    symlink_force(
        os.path.join(mpi_home, 'include/mpio.h'), 'third_party/mpi/mpio.h')
    symlink_force(
        os.path.join(mpi_home, 'include/mpicxx.h'), 'third_party/mpi/mpicxx.h')
    # TODO(gunan): avoid editing files in configure
    sed_in_place('third_party/mpi/mpi.bzl', 'MPI_LIB_IS_OPENMPI=True',
                 'MPI_LIB_IS_OPENMPI=False')

  if os.path.exists(os.path.join(mpi_home, 'lib/libmpi.so')):
    symlink_force(
        os.path.join(mpi_home, 'lib/libmpi.so'), 'third_party/mpi/libmpi.so')
  elif os.path.exists(os.path.join(mpi_home, 'lib64/libmpi.so')):
    symlink_force(
        os.path.join(mpi_home, 'lib64/libmpi.so'), 'third_party/mpi/libmpi.so')
  elif os.path.exists(os.path.join(mpi_home, 'lib32/libmpi.so')):
    symlink_force(
        os.path.join(mpi_home, 'lib32/libmpi.so'), 'third_party/mpi/libmpi.so')

  else:
    raise ValueError(
        'Cannot find the MPI library file in %s/lib or %s/lib64 or %s/lib32' %
        (mpi_home, mpi_home, mpi_home))


def set_system_libs_flag(environ_cp):
  syslibs = environ_cp.get('TF_SYSTEM_LIBS', '')
  if syslibs:
    if ',' in syslibs:
      syslibs = ','.join(sorted(syslibs.split(',')))
    else:
      syslibs = ','.join(sorted(syslibs.split()))
    write_action_env_to_bazelrc('TF_SYSTEM_LIBS', syslibs)

  if 'PREFIX' in environ_cp:
    write_to_bazelrc('build --define=PREFIX=%s' % environ_cp['PREFIX'])
  if 'LIBDIR' in environ_cp:
    write_to_bazelrc('build --define=LIBDIR=%s' % environ_cp['LIBDIR'])
  if 'INCLUDEDIR' in environ_cp:
    write_to_bazelrc('build --define=INCLUDEDIR=%s' % environ_cp['INCLUDEDIR'])


def set_windows_build_flags(environ_cp):
  """Set Windows specific build options."""
  # The non-monolithic build is not supported yet
  write_to_bazelrc('build --config monolithic')
  # Suppress warning messages
  write_to_bazelrc('build --copt=-w --host_copt=-w')
  # Fix winsock2.h conflicts
  write_to_bazelrc(
      'build --copt=-DWIN32_LEAN_AND_MEAN --host_copt=-DWIN32_LEAN_AND_MEAN '
      '--copt=-DNOGDI --host_copt=-DNOGDI')
  # Output more verbose information when something goes wrong
  write_to_bazelrc('build --verbose_failures')
  # The host and target platforms are the same in Windows build. So we don't
  # have to distinct them. This avoids building the same targets twice.
  write_to_bazelrc('build --distinct_host_configuration=false')

  if get_var(
      environ_cp, 'TF_OVERRIDE_EIGEN_STRONG_INLINE', 'Eigen strong inline',
      True, ('Would you like to override eigen strong inline for some C++ '
             'compilation to reduce the compilation time?'),
      'Eigen strong inline overridden.', 'Not overriding eigen strong inline, '
      'some compilations could take more than 20 mins.'):
    # Due to a known MSVC compiler issue
    # https://github.com/tensorflow/tensorflow/issues/10521
    # Overriding eigen strong inline speeds up the compiling of
    # conv_grad_ops_3d.cc and conv_ops_3d.cc by 20 minutes,
    # but this also hurts the performance. Let users decide what they want.
    write_to_bazelrc('build --define=override_eigen_strong_inline=true')


def config_info_line(name, help_text):
  """Helper function to print formatted help text for Bazel config options."""
  print('\t--config=%-12s\t# %s' % (name, help_text))


def main():
  global _TF_WORKSPACE_ROOT
  global _TF_BAZELRC
  global _TF_CURRENT_BAZEL_VERSION

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--workspace',
      type=str,
      default=os.path.abspath(os.path.dirname(__file__)),
      help='The absolute path to your active Bazel workspace.')
  args = parser.parse_args()

  _TF_WORKSPACE_ROOT = args.workspace
  _TF_BAZELRC = os.path.join(_TF_WORKSPACE_ROOT, _TF_BAZELRC_FILENAME)

  # Make a copy of os.environ to be clear when functions and getting and setting
  # environment variables.
  environ_cp = dict(os.environ)

  current_bazel_version = check_bazel_version('3.1.0', '3.7.0')
  _TF_CURRENT_BAZEL_VERSION = convert_version_to_int(current_bazel_version)

  reset_tf_configure_bazelrc()

  cleanup_makefile()
  setup_python(environ_cp)
  create_build_configuration(environ_cp)

  if is_windows():
    environ_cp['TF_DOWNLOAD_CLANG'] = '0'
    environ_cp['TF_NEED_MPI'] = '0'

  # The numpy package on ppc64le uses OpenBLAS which has multi-threading
  # issues that lead to incorrect answers.  Set OMP_NUM_THREADS=1 at
  # runtime to allow the Tensorflow testcases which compare numpy
  # results to Tensorflow results to succeed.
  if is_ppc64le():
    write_action_env_to_bazelrc('OMP_NUM_THREADS', 1)


  set_build_var(environ_cp, 'TF_NEED_MPI', 'MPI', 'with_mpi_support', False)
  if environ_cp.get('TF_NEED_MPI') == '1':
    set_mpi_home(environ_cp)
    set_other_mpi_vars(environ_cp)

  set_cc_opt_flags(environ_cp)
  set_system_libs_flag(environ_cp)
  if is_windows():
    set_windows_build_flags(environ_cp)

if __name__ == '__main__':
  main()
