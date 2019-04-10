# TensorFlow On Demand

| Status        | Accepted      |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Gunhan Gulsoy (gunan@google.com), Hye Soo Yang (hyey@google.com) |
| **Sponsor**   | Gunhan Gulsoy (gunan@google.com)                |
| **Collaborators** | Christoph Goern (goern@redhat.com), Subin Modeel (smodeel@redhat.com)         |
| **Updated**   | 2019-02-25                                           |

## Objective

This document proposes a system to build optimized binaries in the cloud and deliver the artifacts to users of TensorFlow. With this system, we aim to achieve the following:

*   Hide the complexity of the build system of TF from beginner and intermediate users.
*   Improve the out-of-the-box performance for users.
*   Build on the strength of TensorFlow's community partners.

## Motivation

### Overview

TF build is difficult and the friction present during building phase is apparent in our user experiences. For a successful build, users needs to be aware of requirements and configurations that they might not be too familiar with. This can be extremely challenging for beginner and intermediate users.

Currently, TF python release artifacts are hosted on PyPI as a single source of TF binaries for download. However, PyPI is quite limited in capability; it is unable to recognize and/or deploy on user machines.

Historically, this has been a problem when trying to satisfy the following requirements for our artifacts:

*   **Portability**: The artifacts should be able to run on as many platforms as possible.
*   **Performance**: The artifacts should run as fast as possible.
*   **Size**: The artifacts should be as small as possible in size.

To achieve the above, we propose a system that takes inputs from the user, build artifacts based on the inputs and send the output back to the user. We are proposing two user endpoints to the system:

1.  Command line tool (python)
1.  Web interface

An overview of the system:

![drawing](https://docs.google.com/drawings/d/11jAVBtR4nV4bkDVW1WrHhLXdzX2ZouOPwsztW2hZQz4/export/png)

A system flow chart summarizing the ordering of events for all use cases of the system:

![drawing](https://docs.google.com/drawings/d/1lSSHaYktst8MNTqF860cPlkfbgc3Ft_t_8PWzH0VkW4/export/png)


### Collaboration with Red Hat

Red Hat already built a system to build and deliver TF on the cloud. Reference artifacts released by Red Hat are listed below:

*   [tensorflow-build-s2i ](https://github.com/thoth-station/tensorflow-build-s2i)
*   [Tensorflow-release-job](https://github.com/thoth-station/tensorflow-release-job)
*   [tensorflow-serving-build](https://github.com/thoth-station/tensorflow-serving-build)

We propose collaborating with the Red Hat team and leveraging the team's open-source software as much as possible. The server side of the project, in particular, can have concentrated involvement from the team given the backend GKE cluster that already has a lot of functionality built and available. (Please refer to the "Server" block in the diagram above for details on the server-side event flow.) Based on Additional support and functionality will be built around and on top of it to match what TensorFlow currently supports and should support moving forward.

## Design Proposal

### Front-end

**Web UI (download.tensorflow.org)**

We propose a simple web interface that allows users to manually enter necessary system specs and desired build options for requesting a custom TF binary. The UI will be straightforward and easy-to-use. A sample mock up UI is shown below. (Please refer to [PyTorch](https://pytorch.org) download UI as an additional reference.)

![drawing](https://docs.google.com/drawings/d/1Krze2no7zjfqe7nldOm-ArECOGXVVjgtk0VMXCabzkw/export/png)

Once all fields are filled in, the system backend will check if the corresponding binary has already been built and present in cache. If it exists, then it will provide the user a URL to the binary for download. If it does not exist, then it will ask the user for an email address to receive a link for downloading the newly built binary.

Before sending out the binary, the system will first check whether the binary is supported (or unsupported) by going through what is being tested in CI. It will then inform the users accordingly.

**Command Line Tool (TF Downloader Binary)**

We propose a simple binary which will detect and fill out most of the inputs that the Web UI requires. It will then send the request to the backend to build. A sample execution (not final) is shown below:

```
> python tfdownloader.py
Downloader will now detect your system:
	- Detecting CPU…….  Found Intel Core i7-8700K
	- Detecting GPU……. Found NVIDIA GTX 980
	- Detecting CUDA…… Found 9.2
	- Detecting cuDNN….. Found 7.4
	- Detecting Distribution……. Found Ubuntu 18.04

Requesting TF build with options:
    CPU options: -mavx2
    GPU enabled: yes
    CUDA version: 9.2
    cuDNN version: 7.4
    CUDA compute capability: 5.2
    GCC version: 7.3
```


It is possible that the system this script is run on may be incompatible for running TF. Such incapabilities and what the script is able to do can be categorized into two failure modes:



*   *Hard Constraint Failures*
    *   32-bit OS
    *   GPU too old (Cuda Compute Capability older than 3.0)
*   *Soft Constraint Failures*
    *   CUDA version too old
    *   cuDNN missing
    *   Other runtime libraries missing or too old (libstdc++)
    *   Unsupported python version found

In hard constraint failure cases, the tool will point users to cloud options. In soft constraint failure cases, the tool will offer alternative build configurations (CPU only) or request users to install the missing software.

We propose distributing this binary through PyPI. However, please note that binary distribution method is still under discussion.

### Back-end

**TF Builder Service**

TF builder service will require a proto with build options. This proto will define all options the system recognizes to build TF. Once the request is received, the system will check the cache to see if we already have such a package built. If yes, then it will send back the URL for downloading this package. A sample proto (not final) is shown below:

```
proto build_options {
   string version = 0;
   enum CpuOptions {
       SSE3 = 0;
       SSE4 = 1;
       SSE4_1 = 2;
       SSE4_A = 3;
       SSE4_2 = 4;
       AVX = 5;
       AVX2 = 6;
       AVX512F = 7;
   }
   repeated CpuOptions cpu_options = 1;

   // CUDA options
   bool cuda_enabled = 2;
   enum NvidiaGPUGeneration {
       KEPLER = 0;
       MAXWELL = 1;
       PASCAL = 2;
       VOLTA = 3;
       TURING = 4;
   }
   NvidiaGPUGeneration gpu_generation = 3;
   string CUDA_version = 4;
   string cuDNN version = 5;
……
   // Free formed string of options to append to bazel.
   string extra_options = 100;
}
```

If no such package exists in cache, then the system will execute the following commands with the appropriate flags and environment variables to newly build it.

```
git checkout <tag>
yes "" | ./configure
bazel build tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package
```

Once the build is complete, the package will be stored in the cache before sending out a link to the user for download.

**Package Cache**

We will design the package cache as a simple GCS bucket supported by a simple database. The database can have the following schema: 

```
PackageCacheDataStore {
   // This is the primary key to our datastore.
   // As build_options is such a complex data type,
   // this will be the the build_options fed through a hash function
   string build_options_key;

   // The path to the artifact. Will look like:
   // gs://tensorflow-artifacts/on-demand/<hash>/packagename.whl
   string artifact_location;

   // Maybe, maybe not
   date last_built;
   date last_downloaded;
   int download_count;
}
```

### Other Details

**Bug Filing**

With the complicated build system TF currently has, there is undoubtedly going to be issues when building TF. In such cases, we would like the system to go through a process in which it will:

1.  Prepare full reproduction instructions.
1.  File the bug to most relevant teams.
    *  e.g. Bugs related to the build system will be filed to appropriate TF teams while server-side bugs will be filed to the Red Hat's support team. (This is just an example and is not final.)
1.  While the bug is being filed, unblock the user by recommending alternate download options.
