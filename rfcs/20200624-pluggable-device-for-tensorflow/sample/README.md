# TensorFlow Plugin demo
This sample is a simple demo shows how to implement, build, install and run a TensorFlow plugin.

## Supported OS
* Linux

## Prerequisites

* [Bazel](https://docs.bazel.build/versions/master/install-ubuntu.html) (version 3.1 and above)
* Git (version 1.8 and above)
* Python (version 3.6 and above)

## Build and Run

### Linux
1. Run the following command to install the latest `tensorflow`.
```
$ pip install tensorflow
```
2. In the plug-in `sample` code folder, configure the build options:
```
$ ./configure 

Please specify the location of python. [Default is /home/test/miniconda2/envs/sycl3.6/bin/python]: 


Found possible Python library paths:
  /home/test/miniconda2/envs/sycl3.6/lib/python3.6/site-packages
Please input the desired Python library path to use.  Default is [/home/test/miniconda2/envs/sycl3.6/lib/python3.6/site-packages]

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
5. Now we can run the TensorFlow with plug-in device enabled.
```
$ python
>>> import tensorflow as tf
>>> tf.config.list_physical_devices()
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), PhysicalDevice(name='/physical_device:MY_DEVICE:0', device_type='MY_DEVICE')]
```
*  Relu case:
```
$ python relu.py
random_normal/RandomStandardNormal: (RandomStandardNormal): /job:localhost/replica:0/task:0/device:CPU:0
2021-10-21 12:48:20.714819: I tensorflow/core/common_runtime/placer.cc:114] random_normal/RandomStandardNormal: (RandomStandardNormal): /job:localhost/replica:0/task:0/device:CPU:0
random_normal/mul: (Mul): /job:localhost/replica:0/task:0/device:CPU:0
2021-10-21 12:48:20.714864: I tensorflow/core/common_runtime/placer.cc:114] random_normal/mul: (Mul): /job:localhost/replica:0/task:0/device:CPU:0
random_normal: (AddV2): /job:localhost/replica:0/task:0/device:CPU:0
2021-10-21 12:48:20.714903: I tensorflow/core/common_runtime/placer.cc:114] random_normal: (AddV2): /job:localhost/replica:0/task:0/device:CPU:0
Relu: (Relu): /job:localhost/replica:0/task:0/device:MY_DEVICE:0
2021-10-21 12:48:20.714937: I tensorflow/core/common_runtime/placer.cc:114] Relu: (Relu): /job:localhost/replica:0/task:0/device:MY_DEVICE:0
random_normal/shape: (Const): /job:localhost/replica:0/task:0/device:CPU:0
2021-10-21 12:48:20.714968: I tensorflow/core/common_runtime/placer.cc:114] random_normal/shape: (Const): /job:localhost/replica:0/task:0/device:CPU:0
random_normal/mean: (Const): /job:localhost/replica:0/task:0/device:CPU:0
2021-10-21 12:48:20.714997: I tensorflow/core/common_runtime/placer.cc:114] random_normal/mean: (Const): /job:localhost/replica:0/task:0/device:CPU:0
random_normal/stddev: (Const): /job:localhost/replica:0/task:0/device:CPU:0
2021-10-21 12:48:20.715022: I tensorflow/core/common_runtime/placer.cc:114] random_normal/stddev: (Const): /job:localhost/replica:0/task:0/device:CPU:0
[2.9109507 0.        0.        0.        0.        0.        0.
 0.        0.        1.316411 ]

```
* Conv + Relu case:
```
$ python conv_relu.py
2021-10-21 12:53:36.389514: I tensorflow/core/common_runtime/placer.cc:114] random_normal_3/mul: (Mul): /job:localhost/replica:0/task:0/device:CPU:0
random_normal_3: (AddV2): /job:localhost/replica:0/task:0/device:CPU:0
2021-10-21 12:53:36.389537: I tensorflow/core/common_runtime/placer.cc:114] random_normal_3: (AddV2): /job:localhost/replica:0/task:0/device:CPU:0
Relu: (Relu): /job:localhost/replica:0/task:0/device:MY_DEVICE:0
2021-10-21 12:53:36.389565: I tensorflow/core/common_runtime/placer.cc:114] Relu: (Relu): /job:localhost/replica:0/task:0/device:MY_DEVICE:0
Conv2D: (Conv2D): /job:localhost/replica:0/task:0/device:MY_DEVICE:0
2021-10-21 12:53:36.389592: I tensorflow/core/common_runtime/placer.cc:114] Conv2D: (Conv2D): /job:localhost/replica:0/task:0/device:MY_DEVICE:0
Relu_1: (Relu): /job:localhost/replica:0/task:0/device:CPU:0
2021-10-21 12:53:36.389617: I tensorflow/core/common_runtime/placer.cc:114] Relu_1: (Relu): /job:localhost/replica:0/task:0/device:CPU:0
Conv2D_1: (Conv2D): /job:localhost/replica:0/task:0/device:CPU:0
2021-10-21 12:53:36.389641: I tensorflow/core/common_runtime/placer.cc:114] Conv2D_1: (Conv2D): /job:localhost/replica:0/task:0/device:CPU:0
```
* Profiler case:
```
$python test_profiler.py
```
<div>
<img src=profiler_result.png>
</div>

