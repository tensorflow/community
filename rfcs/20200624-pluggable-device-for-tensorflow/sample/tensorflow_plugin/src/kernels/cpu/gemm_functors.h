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


#ifndef TENSORFLOW_PLUGIN_SRC_KERNEL_CPU_H_
#define TENSORFLOW_PLUGIN_SRC_KERNEL_CPU_H_

//#define EIGEN_USE_THREADS

#include <map>
#include <string.h>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

// FixedPoint header must be included after Tensor.
// clang-format off
#include "third_party/eigen3/unsupported/Eigen/CXX11/FixedPoint"
// clang-format on

template <class T1, class T2, class T3> class FastGemmFunctor {
public:
  void operator()(size_t m, size_t n, size_t k, const T1 *a, size_t lda,
                  const T2 *b, size_t ldb, T3 *c, size_t ldc) {
    Eigen::array<size_t, 2> dim_a = {{m, k}};
    Eigen::array<size_t, 2> dim_b = {{k, n}};
    Eigen::array<size_t, 2> dim_c = {{m, n}};
    Eigen::TensorMap<
        Eigen::Tensor<const T1, 2, Eigen::RowMajor, Eigen::DenseIndex>,
        Eigen::Aligned>
        a_matrix(a, dim_a);
    Eigen::TensorMap<
        Eigen::Tensor<const T2, 2, Eigen::RowMajor, Eigen::DenseIndex>,
        Eigen::Aligned>
        b_matrix(b, dim_b);
    Eigen::TensorMap<Eigen::Tensor<T3, 2, Eigen::RowMajor, Eigen::DenseIndex>,
                     Eigen::Aligned>
        c_matrix(c, dim_c);

    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
    dim_pair[0].first = 1;
    dim_pair[0].second = 0;
    Eigen::ThreadPool tp(8);
    Eigen::ThreadPoolDevice thread_pool_device(&tp, 8);
    c_matrix.device(thread_pool_device) = a_matrix.contract(b_matrix, dim_pair);
  }
};

#endif // TENSORFLOW_PLUGIN_SRC_KERNEL_CPU_H_
