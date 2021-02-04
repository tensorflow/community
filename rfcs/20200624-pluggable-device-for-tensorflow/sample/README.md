# TensorFlow Plugin demo
This sample is a simple demo shows how to implement, build, install and run a TensorFlow plugin.

## Supported OS
* Linux

## Prerequisites

* [Bazel](https://docs.bazel.build/versions/master/install-ubuntu.html) (version 3.1 and above)
* Git (version 1.8 and above)
* Python (version 3.6 and above)

## Build and Run
This demo depends on the folllowing ongoing PRs:
* [\[PluggableDevice\]] PluggableDevice mechanism implementation](https://github.com/tensorflow/tensorflow/pull/45784)

* [\[PluggableDevice\] Add device alias support] (https://github.com/tensorflow/tensorflow/pull/43611) (gpu only)

### Linux
1. Run the following command to install `tf-nightly`.
```
$ pip install tf-nightly
```
2. In the plug-in `sample` code folder, configure build options:
```
$ ./configure 

Please specify the location of python. [Default is /home/guizili/miniconda2/envs/sycl3.6/bin/python]: 


Found possible Python library paths:
  /home/guizili/miniconda2/envs/sycl3.6/lib/python3.6/site-packages
Please input the desired Python library path to use.  Default is [/home/guizili/miniconda2/envs/sycl3.6/lib/python3.6/site-packages]

Do you wish to build TensorFlow plug-in with MPI support? [y/N]: 
No MPI support will be enabled for TensorFlow plug-in.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native -Wno-sign-compare]: 
```

3. Built the plug-in with
```
$ bazel build  -c opt //tensorflow_plugin/tools/pip_package:build_pip_package --verbose_failures
```
4. Then generate a python wheel and install it.
```
$ bazel-bin/tensorflow_plugin/tools/pip_package/build_pip_package .
$ pip install tensorflow_plugins-0.0.1-cp36-cp36m-linux_x86_64.whl
```
5. Now We can run the TensorFlow with plug-in device enabled.
```
$ python
>>> import tensorflow as tf
>>> tf.config.list_physical_devices()
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), PhysicalDevice(name='/physical_device:MY_DEVICE:0', device_type='MY_DEVICE')]
```
*  Relu case:
$ python relu.py
* Conv + Relu case:
$ python conv_relu.py
