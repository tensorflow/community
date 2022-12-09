# [RFC] Enable Stochastic Rounding for Quantized Training in Tensorflow

Status        | Proposed
:------------ | :----------------------------------------
**RFC #**     | 436
**Author(s)** | elfieguo@google.com, chiachenc@google.com
**Sponsor**   | cantonios@google.com, lidon@google.com
**Updated**   | 2022-11-06

## Objective

[Stochastic rounding](https://en.wikipedia.org/wiki/Rounding#Stochastic_rounding) is a rounding method that rounds a value to a nearest value with a probability dependent on the proximity, which gives an unbiased result on average. It has been proven\[[1](https://arxiv.org/pdf/1502.02551.pdf)\]\[[2](https://arxiv.org/pdf/1812.08011.pdf)\]\[[3](https://arxiv.org/pdf/1804.05267.pdf)\]\[[4](https://www.eecs.harvard.edu/htk/static/files/2022-hpca-zhang-mcdanel-kung.pdf)\] that such a rounding method is critical to accuracy when it comes to low-precision quantized training, but there is no such functionality in TF, and users must implement a stochastic rounding kernel in various ways. This document details an implementation of  stochastic rounding in TF that facilitates the use of stochastic rounding in quantized training. Below are the requirements: 

*   [P0] Enable stochastic rounding for float32/float64 to int8 on TPU and CPU
    and GPU.
*   [P0] Enable stochastic rounding for bf16 to int8 on TPU and CPU.
*   [P2] Enable stochastic rounding for float32/float64 to lower precision
    floating point types on TPU and CPU and GPU.

## Motivation

Many companies have invested in low-precision quantized training because of its
efficient acceleration and reduced memory usage, including Google’s TPU,
Microsoft’s Project Brainwave, Tesla’s Dojo, Amazon’s Trianium etc. However,
low-precision training can also lead to many issues like model divergence or
quality compromises. Many studies have proven that stochastic rounding has
substantial improvements in reducing rounding errors in low precision formats
and helps resolve issues like divergence and quality loss. Many chips like
Google TPU, Amazon Trainium, Tesla Dojo and Graphcore have implemented
stochastic rounding on their hardwares. Due to the crucial role stochastic
rounding plays in low precision training, also in order to utilize many
hardwares’ accelerations, we propose a stochastic rounding API primarily used
for quantized training in Tensorflow.

## Background

There have been many rounding schemes proposed and studied for different
applications (IEEE 754, 2019), such as round-to-nearest, round-to-zero,
round-to-negative-infinity, and round-to-positive-infinity, each of them having
different round-off errors. When a large sequence of computations is
implemented, round-off errors may be accumulated and magnified, leading to
severe failures. Stochastic rounding differs from the standard deterministic
modes, in that instead of always rounding to the nearest number, the decision
about which way to round is non-deterministic and the probability of rounding up
is proportional to the residual (value of the trailing bits that do not fit into
the destination format). Different from deterministic rounding, stochastic
rounding is an unbiased rounding mechanism. For example, when x = 3.7, it has a
30% chance to be rounded to 3 and 70% chance to be rounded to 4. Thus the
expected value of the rounding result would be 

$$ 
E(x) = 3 \times 0.3 + 4 \times 0.7 = 3.7. 
$$

Let $x ∈ R$, and let $\delta$ be the rounding precision. Then the rounded
value $sr(x)$ of $x$ using stochastic rounding is defined as

$$
sr(x) = \begin{cases} \lfloor{x}\rfloor,& \text{with probability }
p_1(x)=1-\frac{x-\lfloor{x}\rfloor}{\delta}\\
\lfloor{x}\rfloor + \delta, &
\text{with probability } p_2(x)=\frac{x-\lfloor{x}\rfloor}{\delta}\\ \end{cases}
$$

Where $\delta$ indicates the greatest representable floating/fixed-point
number less than or equal to x.

## User Benefit

Users can be benefited from following aspects:

-   Ready-to-use API for performing accurate stochastic rounding on many
    different floating point formats.
-   Performance speedups on Google TPUs.

## Design Proposal

During quantization, users replace high-precision floating-point numbers with
low-precision fixed-point numbers or lower-bit floats to reduce the
computational cost of neural network training or inference.

For example, below is a code snippet that represents a part of process to
quantize a tensor from floating-points to int8, which is then used for matrix
multiplications:

```
inputs = tf.clip_by_value(inputs, -128.0, 127.0)
inputs = tf.math.round(inputs) 
inputs = tf.cast(inputs, dtype=tf.int8)
// 'weights' tensor has been converted to int8 format beforehand.
output = tf.einsum(inputs, weights)
```

For quantizing bigger floats to smaller floats, users need to do casting like
below:

```
inputs = tf.cast(inputs, dtype=tf.bfloat16)
// 'weights' tensor has been converted to bfloat16 format beforehand.
output = tf.einsum(inputs, weights)
```

There are two essentials to achieve stochastic rounding:

-   A random value generated from uniform distributions to emulate
    stochasticity.
-   Perform deterministic rounding on the operand with the rounding direction
    determined by the given random value.

Although stochastic rounding is probabilistic, the probability can be explicitly
controlled by choosing an appropriate random number generator. Users might want
to reproduce the rounding results for debugging or behavioral enforcement
purposes. This usually requires a reproducible pseudo random number generator,
whose output is deterministic by a given state. In tensorflow, this generator is
exposed via the tf.random.generator API. Since the purpose of quantized training
is to reduce the bits used in calculations, changing the data type after
rounding is a necessity. Given these considerations, an API is ultimately
proposed like below:

```
tf.stochastic_cast(input, dtype, seed, alg=’auto_select’)
```

<table>
 <tr><td><strong>Args</strong></td><td><strong>Definition</strong></td></tr>                                      
 <tr><td>input</td><td>The original tensor to be casted.</td></tr>                              
 <tr><td>dtype</td><td>Desired type after casting.</td></tr>       
 <tr><td>seed</td><td>A required shape[2] tensor, the seed for the RNG. Recommended to use different seeds for each forward pass to eliminate potential bias introduced by PRNG. </td></tr>                              
 <tr><td>alg</td><td> The RNG algorithm used to generate the random numbers. Valid choices are "philox" for the Philox algorithm, "threefry" for the ThreeFry algorithm, and "auto_select" (default) for the system to automatically select an algorithm based on the device type. </td></tr>
</table>

Returns |                                                                 |
------- | ---------------------------------------------------------------
Tensor  | A tensor whose values are casted, with the same shape as input.

The API returns a tensor that contains the rounded values in specified data type
with the same shape as the operand.

The API asks users to provide random seed and algorithm so that the random
number sequence can be reproduced and users  can replay the rounding results.

This option encapsulates the random number generation aiming at simplifying API
calls. Also, doing it this way, we can potentially avoid OOM issues introduced
by storing and moving large random tensors by optimizing the backend
implementations. Otherwise, the OOM issue could be inevitable when users need to
generate random tensors and put them into the API.

However, as more use cases emerge, there might be occasions where custom random
number generators other than the default need to be provided. Or there might be
experimental random number generation algorithms to be passed in. Under such
circumstances, encapsulation might complicate future developments. However, up
till now we haven’t received any user requests regarding random number
generation algorithms or custom generators.

### Alternatives Considered

Similar to the proposal, but this option doesn't encapsulate random number generation
inside of the API call and asks users to provide random tensors.

```
tf.stochastic_cast(input, randoms, dtype)
```
                                     
<table>
 <tr><td><strong>Args</strong></td><td><strong>Definition</strong></td></tr>
 <tr><td>input</td><td>The original tensor to be casted.</td></tr>
 <tr><td>dtype</td><td>Desired type after casting.</td></tr>
 <tr><td>randoms</td><td>Random tensor for determining the rounding direction. If the random number is less than the fractional part, the  result will be rounded up. This is required to be unsigned integers whose bit width is the same as the operand. The shape should be the same as input.</td></tr>
</table>


Returns |                                                                 |
------- | ---------------------------------------------------------------
Tensor  | A tensor whose values are casted, with the same shape as input.

This API asks users to provide random numbers so that users can control the
rounding results as long as they can reproduce the sequence of random numbers,
which can be achieved via tf.random.generator API. Depending on the use case,
users can also achieve round-to-zero or round-away-from-zero by setting the
random number to 0 or 0xFFFF (depending on the operand bit width).

With rounding and downcasting encapsulated in one API, the op can be emulated in
the backend with higher efficiency.

There is a disadvantage of this implementation that users need to materialize
large tensors before calling this API. Once they are
materialized, calling this API will require storing and passing the large tensor
around, causing OOM issues. When facing this OOM issue, users will need to potentially break down
the tensor in order to bypass the memory limit.

Another way that sticks to the current rounding API design:

```
tf.stochastic_round(input, to_precision, seed, alg=’auto_select’)
```

<table>
 <tr><td><strong>Args</strong></td><td><strong>Definition</strong></td></tr>
 <tr><td>input</td><td>The original tensor to be rounded.</td></tr>
 <tr><td>to_precision</td><td>Desired precision after rounding.</td></tr>
 <tr><td>seed</td><td>A required shape[2] tensor, the seed for the RNG. Recommended to use different seeds for each forward pass to eliminate potential bias introduced by PRNG.</td></tr>
 <tr><td>alg</td><td>The RNG algorithm used to generate the random numbers. Valid choices are "philox" for the Philox algorithm, "threefry" for the ThreeFry algorithm, and "auto_select" (default) for the system to automatically select an algorithm based on the device type.</td></tr>
</table>

Returns |                                                                  |
------- | ----------------------------------------------------------------
Tensor  | A tensor whose values are rounded, with the same shape as input.

`tf.stochastic_round` will round the input to the desired precision, but the
output will still keep the original data type. For example, in the code snippet
below, both the input and output of `tf.stochastic_round` will be bfloat16.

```
  // inputs are in bfloat16
 inputs = tf.clip_by_value(inputs, -128.0, 127.0)
 inputs = tf.stochastic_round(inputs, dtype=tf.bf8, seed=[1,2])
 inputs = tf.cast(inputs, dtype=tf.bf8)
 // 'weights' tensor has been converted to int8 format beforehand.
 output = tf.einsum(inputs, weights)
```

This option keeps consistency with the current usage of tf.math.round for
quantization. It also provides more flexibility in terms of how users want to
treat their data type going forward, thus not limiting the usage of stochastic
rounding only to quantization. Worth mentioning that we currently don’t have a
use case for this.

However, there are also cons of this option:

-   It might lead to efficiency degradation if the compiler does not fuse the
    two ops: `tf.stochastic_round` and `tf.cast`. As for hardwares that have
    stochastic_rounding capability, the data will be rounded and cast in one
    instruction, but in order to keep the original input data type, there will
    be an extra step for up-casting the data. In addition to that, `tf.cast`
    currently does an extra round-to-even before truncation for float casting,
    without fusing the two API, this round-to-even is unnecessary.
-   Extra work will be introduced for fusing two APIs in compilers. On some
    paths (hand-written CPU/GPU) where each operation is executed separately by
    the TensorFlow executor, and has a precompiled kernel implementation that
    the executor dispatches to, fusing the two operations would be hard.
-   Users need to call two APIs for executing stochastically downcast for their
    data, which might not be intuitive to new users.

### Software Emulation

Software emulation is required on hardwares that don't have built-in support for stochastic rounding. This includes CPU, GPU and some TPU versions. To keep consistent with the existing hardware implementations, the emulated algorithms are demonstrated below.
Given a random value, below is the algorithm for rounding floats to integers:

```
function stochastic_rounding_f32_int32(x, random):
  if X > INT32_MAX:
    return INT32_MAX
  else if X < INT32_MIN:
    return INT32_MIN

  integral = floor(abs(x))
  fractional = abs(x) - integral
  should_round_up = random < (fractional * UINT32_MAX)
  result = should_round_up? (integral + 1) : integral
  return is_negative(x): -result : result
```

Algorithm for rounding to small floats:

```
function stochastic_rounding_f32_bf8(x, random):
  if X > BF8_MAX:
    return BF8_MAX
  else if X < BF8_MIN:
    return BF8_MIN

  fraction = x & (1 << kF32FractionBits) - 1
  round_to_zero = truncate_to_bf8(x)
  round_to_infinity = next_away_from_zero(x)
  truncation_mask = (1 << (kF32FractionBits - kBf8FractionBits)) - 1
  truncation_residue = fraction & truncation_mask
  masked_random = (random >> kBf8FractionBits) & truncation_mask;
  return (masked_random >= truncation_residue) ? round_to_zero : round_to_inf;
```

### Performance Implications

We expect a longer latency on hardwares that require software emulations,
compared to round-to-nearest-even. Random number generation will contribute to a
significant portion of the latency. There will be unit tests for benchmarking
end-to-end tensorflow API running on CPU/GPU/XLA. Besides these, there will also
be unit tests for hand-written kernel CPU/GPU implementations, and tests for XLA
on CPU/GPU/TPU.

### Engineering Impact

There will be no change to binary size, startup time, and build time. Test times
might vary slightly due to different computation times, but they should be
negligible. The Google performance team will maintain the code.

### Platforms and Environments

The API is temporarily not supported on embedded systems due to lack of use
cases.

### Best Practices

No changes in best practices.

### Tutorials and Examples

Users will call this API as in the snippet:

```
// inputs are in bfloat16
inputs = tf.clip_by_value(inputs, -128.0, 127.0)
inputs = tf.stochastic_cast(inputs, randoms, tf.dtype.int8, random_seed=[1,2], random_algo=”Philox”)
// 'weights' tensor has been converted to int8 format beforehand.
output = tf.einsum(inputs, weights)
```

### Compatibility

-   This is no backwards & forwards compatibility issue since we’re proposing a
    new API.
-   Interactions with other parts of the TensorFlow ecosystem:

    -   TFLite: This proposal doesn’t affect TFLite
    -   Distribution strategies: The process of stochastic rounding relies on
        random number generators. Stochastic rounding will support distribution
        strategies so long as random number generators also support it.
    -   GPU/TPU: The op will work on CPU GPU and TPU.
    -   SavedModel: The op will be serialized to a SavedModel. Models from
        SavedModel will be able to use the op.

