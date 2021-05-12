# Modular TensorFlow profiler C API
| Status        | Proposed                                             |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [262](https://github.com/tensorflow/community/pull/262)|
| **Author(s)** | Zhoulong Jiang (zhoulong.jiang@intel.com), Yiqiang Li (yiqiang.li@intel.com), Eric Lin (eric.lin@intel.com), Jianhui Li (jian.hui.li@intel.com) |
| **Sponsor**   | Situ Yi (yisitu@google.com)                          |
| **Updated**   | 2021-04-27                                           |

## **Objective**

Provide general profiler C API to allow new device profilers to modularly connect to the current TensorFlow runtime.

This RFC is based on the Modular TensorFlow  [RFC](https://github.com/tensorflow/community/pull/77), which aims at extending the TensorFlow design to plug in capabilities like adding a new device profiler.

## **Motivation**

Performance is a key consideration of successful ML research and production solutions. TensorFlow profiler provides a set of good tools to help user better understand the hardware resource consumption(time and memory) of various TensorFlow operations(ops) as well as perforamnce bottlenecks of TensorFlow models. [PluggableDevice](https://github.com/tensorflow/community/pull/262) provides a mechanism which allows third-party devices to be modularly integrated to current TensorFlow runtime. However, current TensorFlow does not provide a stable application binary infterface(ABI) for registering a new profiler in a modular way. We propose a C API wrapper of methods in [ProfilerInterface](https://github.com/tensorflow/tensorflow/blob/0a3773ed7b4c1fc60486dddfacbe9a5cbf2b2bdd/tensorflow/core/profiler/lib/profiler_interface.h#L33) as an ABI-stable way to register a custom profiler. The pluggable profiler discovery and initialization are transparent to end users. As long as the profiler plugin libraries follow the design described in this RFC, it can be plugged to TensorFlow proper and register a new profiler into ProfilerFactory.

## **User Benefit**

This RFC provides a plugin infrastructure for extending third-party device profilers.

## **Design Proposal**

## Design Overview

This RFC is intended to provide a set of C APIs for plugin writers to implement and register their own pluggable profilers. To make C APIs protable, we propose serialized `XSpace` and `RunMetadata` as the object to pass between proper and plugin. When proper invokes `CollectData`, plugin serializes `XSpace` and `RunMetadata` to the big enough buffers provided by proper, then proper deserializes them back to `XSpace` and `RunMetadata` and finally generates a trace view and a set of summaries based on these collected data.

- Xspace:
<div align=center>
<img src=20210427-pluggable-profiler-for-tensorflow/Xspace.png>
</div>

- RunMetadata:
<div align=center>
<img src=20210427-pluggable-profiler-for-tensorflow/RunMetadata.png>
</div>

To achieve the goal, this RFC extends the TensorFlow profiler class hierachy to add a new profiler named `PluggableProfiler` which is built on top of a set of C APIs, all plugin writers who want to integrate their own device profilers to  current TensorFlow runtime only need to implement Profiler C APIs(shown as diagram of Architecture overview).
<div align=center>
<img src=20210427-pluggable-profiler-for-tensorflow/Architecture.png>
</div>

## Versioning Strategy and Stability
* **Profiler C API**
    Version strategy of Profiler C API follows Semantic Versioning 2.0.0 ([semver](http://semver.org/)). Each release version has a format `MAJOR.MINOR.PATCH`, as outlined in [TensorFlow version compatibility](https://www.tensorflow.org/guide/versions#semantic_versioning_20). Struct size is used to track compatibility. More details can be found in [StreamExecutor C API Versioning Strategy RFC](https://github.com/tensorflow/community/blob/master/rfcs/20200612-stream-executor-c-api/C_API_versioning_strategy.md)
* **XPlane and RunMetadata**
   The compatibility of `XPlane` and `RunMetadata` between plugin and proper follows the same compatibility [rules](https://developers.google.com/protocol-buffers/docs/cpptutorial?hl=en#extending-a-protocol-buffer) and [guarantees](https://developers.google.com/protocol-buffers/docs/proto3?hl=en#updating) as protobuf library.

## Usage Overview

The table below summarizes all structures defined and the functionality they involve.
| Action | Function call(s) | Populated by Core TensorFlow | Populated by plug-in |
| :----- | :-------------- | :--------------------------- | :------------------- |
| Register profiler | `TF_InitProfiler` | `TF_ProfilerRegistrationParams` | `TP_Profiler`, `TP_ProfilerFns` |
| start profiling | `TP_ProfilerFns::start` | None | None |
| stop profiling | `TP_ProfilerFns::stop` | None | None |
| collect Xspace | `TP_ProfilerFns::collect_data_xspace` | None | None |
| collect RunMetadata | `TP_ProfilerFns::collect_data_run_metadata` | None | None |

