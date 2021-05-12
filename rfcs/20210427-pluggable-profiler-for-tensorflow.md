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

This RFC is intended to provide a set of C APIs for plugin writers to implement and register their own pluggable profilers. To make C APIs protable, we propose serialized `XSpace` and `RunMetadata` as the object to pass between proper and plugin. When proper calls `CollectData`, plugin serializes XSpace and RunMetadata to big enough buffers provided by proper, then proper deserializes them back to `XSpace` and `RunMetadata` and finally generates a trace view and set of summaries based on these data.

- Xspace:
<div align=center>
<img src=20210427-pluggable-profiler-for-tensorflow/Xspace.png>
</div>

- RunMetadata:
<div align=center>
<img src=src=20210427-pluggable-profiler-for-tensorflow/RunMetadata.png>
<div>
