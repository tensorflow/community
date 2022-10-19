/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.                                                                                                                                                                   

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_PLUGIN_SRC_UTILS_PLATFORM_DEFINE_H_
#define TENSORFLOW_PLUGIN_SRC_UTILS_PLATFORM_DEFINE_H_

#define PLATFORM_POSIX

// Look for both gcc/clang and Visual Studio macros indicating we're compiling
// for an x86 device.
#if defined(__x86_64__) || defined(__amd64__) || defined(_M_IX86) ||           \
    defined(_M_X64)
#define PLATFORM_IS_X86
#endif

#endif // TENSORFLOW_PLUGIN_SRC_UTILS_PLATFORM_DEFINE_H_
