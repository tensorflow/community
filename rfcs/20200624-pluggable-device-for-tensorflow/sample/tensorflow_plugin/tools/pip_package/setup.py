'''
setup.py file to build wheel for tensorflow_plugin
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fnmatch
import os
import re
import sys

from setuptools import Command
from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install as InstallCommandBase
from setuptools.dist import Distribution

DOCLINES = __doc__.split('\n')

# This version string is semver compatible, but incompatible with pip.
# For pip, we will remove all '-' characters from this string, and use the
# result for pip.
# Also update tensorflow/demo_plugin.bzl and
# tensorflow/core/public/version.h
_VERSION = '0.0.1'
# this path can't be modified.
_PLUGIN_LIB_PATH = 'tensorflow-plugins'
_MY_PLUGIN_PATH = 'my_plugin_dir'

REQUIRED_PACKAGES = [
    'tensorflow >= 2.5.0',
]

if sys.byteorder == 'little':
  # grpcio does not build correctly on big-endian machines due to lack of
  # BoringSSL support.
  # See https://github.com/tensorflow/tensorflow/issues/17882.
  REQUIRED_PACKAGES.append('grpcio >= 1.8.6')

# The wheel package name, change it as your requirements
project_name = 'my_tensorflow_plugin_package'

# python3 requires wheel 0.26
if sys.version_info.major == 3:
  REQUIRED_PACKAGES.append('wheel >= 0.26')
else:
  REQUIRED_PACKAGES.append('wheel')
  # mock comes with unittest.mock for python3, need to install for python2
  REQUIRED_PACKAGES.append('mock >= 2.0.0')

# weakref.finalize and enum were introduced in Python 3.4
if sys.version_info < (3, 4):
  REQUIRED_PACKAGES.append('backports.weakref >= 1.0rc1')
  REQUIRED_PACKAGES.append('enum34 >= 1.1.6')

# pylint: disable=line-too-long
CONSOLE_SCRIPTS = [
#    'freeze_graph = tensorflow.python.tools.freeze_graph:run_main',
#    'toco_from_protos = tensorflow.lite.toco.python.toco_from_protos:main',
#    'tflite_convert = tensorflow.lite.python.tflite_convert:main',
#    'toco = tensorflow.lite.python.tflite_convert:main',
#    'saved_model_cli = tensorflow.python.tools.saved_model_cli:main',
#    # We need to keep the TensorBoard command, even though the console script
#    # is now declared by the tensorboard pip package. If we remove the
#    # TensorBoard command, pip will inappropriately remove it during install,
#    # even though the command is not removed, just moved to a different wheel.
#    'tensorboard = tensorboard.main:run_main',
#    'tf_upgrade_v2 = tensorflow.tools.compatibility.tf_upgrade_v2_main:main',
]
# pylint: enable=line-too-long

TEST_PACKAGES = [
    'scipy >= 0.15.1',
]


class BinaryDistribution(Distribution):

  def has_ext_modules(self):
    return True


class InstallCommand(InstallCommandBase):
  """Override the dir where the headers go."""

  def finalize_options(self):
    ret = InstallCommandBase.finalize_options(self)
    self.install_headers = os.path.join(self.install_purelib,
                                        'tensorflow-plugins', 'include')
    return ret


class InstallHeaders(Command):
  """Override how headers are copied.

  The install_headers that comes with setuptools copies all files to
  the same directory. But we need the files to be in a specific directory
  hierarchy for -I <include_dir> to work correctly.
  """
  description = 'install C/C++ header files'

  user_options = [('install-dir=', 'd',
                   'directory to install header files to'),
                  ('force', 'f',
                   'force installation (overwrite existing files)'),
                 ]

  boolean_options = ['force']

  def initialize_options(self):
    self.install_dir = None
    self.force = 0
    self.outfiles = []

  def finalize_options(self):
    self.set_undefined_options('install',
                               ('install_headers', 'install_dir'),
                               ('force', 'force'))

  def mkdir_and_copy_file(self, header):
    install_dir = os.path.join(self.install_dir, os.path.dirname(header))
    # Get rid of some extra intervening directories so we can have fewer
    # directories for -I
    install_dir = re.sub('/google/protobuf_archive/src', '', install_dir)

    # Copy external code headers into tensorflow/include.
    # A symlink would do, but the wheel file that gets created ignores
    # symlink within the directory hierarchy.
    # NOTE(keveman): Figure out how to customize bdist_wheel package so
    # we can do the symlink.
    external_header_locations = [
        'tensorflow-plugins/include/external/eigen_archive/',
        'tensorflow-plugins/include/external/com_google_absl/',
        'tensorflow-plugins/include/external/com_google_protobuf',
    ]
    for location in external_header_locations:
      if location in install_dir:
        extra_dir = install_dir.replace(location, '')
        if not os.path.exists(extra_dir):
          self.mkpath(extra_dir)
        self.copy_file(header, extra_dir)

    if not os.path.exists(install_dir):
      self.mkpath(install_dir)
    return self.copy_file(header, install_dir)

  def run(self):
    hdrs = self.distribution.headers
    if not hdrs:
      return

    self.mkpath(self.install_dir)
    for header in hdrs:
      (out, _) = self.mkdir_and_copy_file(header)
      self.outfiles.append(out)

  def get_inputs(self):
    return self.distribution.headers or []

  def get_outputs(self):
    return self.outfiles


def find_files(pattern, root):
  """Return all the files matching pattern below root dir."""
  for dirpath, _, files in os.walk(root):
    for filename in fnmatch.filter(files, pattern):
      yield os.path.join(dirpath, filename)


so_lib_paths = [
    i for i in os.listdir('.')
    if os.path.isdir(i) and fnmatch.fnmatch(i, '_solib_*')
]

print(os.listdir('.'))
matches = []
for path in so_lib_paths:
  matches.extend(
      ['../' + x for x in find_files('*', path) if '.py' not in x]
  )

if os.name == 'nt':
  EXTENSION_NAME = 'libdemo_plugin.pyd'
else:
  EXTENSION_NAME = 'libdemo_plugin.so'

headers = (
    list(find_files('*.h', 'tensorflow-plugins/c_api/c')) +
    list(find_files('*.h', 'tensorflow-plugins/c_api/src')))

setup(
    name=project_name,
    version=_VERSION.replace('-', ''),
    description=DOCLINES[0],
    long_description='\n'.join(DOCLINES[2:]),
    url='https://github.com/tensorflow',
    download_url='https://github.com/tensorflow',
    author='Tensorflow',
    author_email='packages@tensorflow.org',
    # Contained modules and scripts.
    packages= [_PLUGIN_LIB_PATH, _MY_PLUGIN_PATH],
    entry_points={
        'console_scripts': CONSOLE_SCRIPTS,
    },
    headers=headers,
    install_requires=REQUIRED_PACKAGES,
    tests_require=REQUIRED_PACKAGES + TEST_PACKAGES,
    package_data={
        _PLUGIN_LIB_PATH: [
            '*.so'
        ],
        _MY_PLUGIN_PATH: [
            '*', '*/*'
        ]
    },
    zip_safe=False,
    distclass=BinaryDistribution,
    cmdclass={
        'install_headers': InstallHeaders,
        'install': InstallCommand,
    },
    # PyPI package information.
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='Apache 2.0',
    keywords='tensorflow tensor machine learning plugin',
)
