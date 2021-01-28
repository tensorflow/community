# TensorFlow Lite for Microcontrollers Port of 16x8 Quantized Operators

| Status        | Proposed                                             |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [NNN](https://github.com/tensorflow/community/pull/NNN) (update when you have community PR #)|
| **Author(s)** | Daniel Situnayake (me@example.org)                   |
| **Sponsor**   | Pete Warden (petewarden@google.com)                  |
| **Updated**   | 2021-01-27                                           |

## Objective

TensorFlow Lite has kernel implementations that support 8 bit quantized weights
but use 16 bit activations. We wish to port these implementations to TensorFlow
Lite for Microcontrollers. The increased precision available for activations can
improve performance for some quantized models.

Arm have agreed to support the initiative by adding the necessary 16x8 APIs to
CMSIS-NN and porting the CMSIS-NN kernels.

### Goals
- Port a subset of 16x8 reference kernels from TensorFlow Lite to TensorFlow Lite Micro
- Avoid increasing default code size of TensorFlow Lite Micro
- Lay the groundwork for creating a CMSIS-NN port of the 16x8 kernels

### Non-goals
- Port every single operator to 16x8; we only plan to port a subset of those with existing reference implementations

## Motivation

Some networks that suffer unacceptable degradation when quantized with 8 bit weights
and 8 bit activations perform adequately when quantized with 8 bit weights and 16
bit activations. The [TensorFlow Lite documentation](https://www.tensorflow.org/lite/performance/post_training_integer_quant_16x8) states the following:

> [16x8 quantization] mode can improve accuracy of the quantized model significantly, when activations are sensitive to the quantization, while still achieving almost 3-4x reduction in model size. Moreover, this fully quantized model can be consumed by integer-only hardware accelerators.

Edge Impulse, a company that deploys TensorFlow Lite for Microcontrollers as part of its embedded
machine learning pipeline, has gathered feedback from customers with production models for which 8 bit
quantization results in unacceptable degradation but for whom 16x8 is fine.

While 16x8 quantization is well supported within TensorFlow Lite, it is not currently supported
within TensorFlow Lite for Microcontrollers. Porting the TensorFlow Lite reference kernels is
relatively straightforward and will improve adoption of TensorFlow Lite for Microcontrollers with users
for whom degradation is too severe with full 8 bit quantization.

## User Benefit

The headline would be "16x8 kernels improve accuracy for quantized models on microcontrollers without
increasing model size".

Users would benefit in the following ways:

- Improved accuracy for quantized models without increasing model size (in exchange for additional
  runtime memory usage)
- Improved performance under certain conditions (for example, 16x8 CMSIS-NN kernels will run faster)
  than 8 bit kernels since less unpacking is required)

## Design Proposal

This is the meat of the document, where you explain your proposal. If you have
multiple alternatives, be sure to use sub-sections for better separation of the
idea, and list pros/cons to each approach. If there are alternatives that you
have eliminated, you should also list those here, and explain why you believe
your chosen approach is superior.

Make sure you’ve thought through and addressed the following sections. If a section is not relevant to your specific proposal, please explain why, e.g. your RFC addresses a convention or process, not an API.


We propose that the 16x8 kernels are ported from the TensorFlow Lite reference kernels to
TensorFlow Lite for Microcontrollers following the process in the [Porting TensorFlow Lite Ops to Micro](https://docs.google.com/document/d/1KLJTPWm4TUKB9YyIqFJl9VCP0ZMJDt_P8RNpRmwqMxw/edit#heading=h.5x0d5h95i329)
guide.

We wish to ensure that the following kernels are compatible with 16x8 mode:

- Conv2D
- MaxPool2D
- DepthwiseConv2D
- FullyConnected
- Relu
- Relu6
- Tanh
- Softmax
- Pad
- Reshape
- Pack
- Unpack
- Add
- Mul


Adding the 16x8 kernels directly to TFLM alongside the existing kernels would increase the default code size by an unacceptable amount. Instead, we will make use of the kernel registration API currently under development by the TFLM team. The use of this is demonstrated in the
[Keyword benchmark code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/benchmarks/keyword_benchmark.cc#L56).
By doing this, the end user can decide which kernels and dependencies they want to include (e.g. 8 bit, 16x8,
or float32).

This means that kernels not currently using this registration API will need to be refactored to use it. Currently only **FullyConnected** uses the API.

The following associated tasks will be required to support this work:

- Build or port unit tests for the new kernels
- Prove that code memory is not impacted by running benchmarks before and after the port

### Alternatives Considered
* Make sure to discuss the relative merits of alternatives to your proposal.

### Performance Implications
* Do you expect any (speed / memory)? How will you confirm?
* There should be microbenchmarks. Are there?
* There should be end-to-end tests and benchmarks. If there are not (since this is still a design), how will you track that these will be created?

### Dependencies
* Dependencies: does this proposal add any new dependencies to TensorFlow?
* Dependent projects: are there other areas of TensorFlow or things that use TensorFlow (TFX/pipelines, TensorBoard, etc.) that this affects? How have you identified these dependencies and are you sure they are complete? If there are dependencies, how are you managing those changes?

### Engineering Impact
* Do you expect changes to binary size / startup time / build time / test times?
* Who will maintain this code? Is this code in its own buildable unit? Can this code be tested in its own? Is visibility suitably restricted to only a small API surface for others to use?

### Platforms and Environments
* Platforms: does this work on all platforms supported by TensorFlow? If not, why is that ok? Will it work on embedded/mobile? Does it impact automatic code generation or mobile stripping tooling? Will it work with transformation tools?
* Execution environments (Cloud services, accelerator hardware): what impact do you expect and how will you confirm?

### Best Practices
* Does this proposal change best practices for some aspect of using/developing TensorFlow? How will these changes be communicated/enforced?

### Tutorials and Examples
* If design changes existing API or creates new ones, the design owner should create end-to-end examples (ideally, a tutorial) which reflects how new feature will be used. Some things to consider related to the tutorial:
    - The minimum requirements for this are to consider how this would be used in a Keras-based workflow, as well as a non-Keras (low-level) workflow. If either isn’t applicable, explain why.
    - It should show the usage of the new feature in an end to end example (from data reading to serving, if applicable). Many new features have unexpected effects in parts far away from the place of change that can be found by running through an end-to-end example. TFX [Examples](https://github.com/tensorflow/tfx/tree/master/tfx/examples) have historically been good in identifying such unexpected side-effects and are as such one recommended path for testing things end-to-end.
    - This should be written as if it is documentation of the new feature, i.e., consumable by a user, not a TensorFlow developer.
    - The code does not need to work (since the feature is not implemented yet) but the expectation is that the code does work before the feature can be merged.

### Compatibility
* Does the design conform to the backwards & forwards compatibility [requirements](https://www.tensorflow.org/programmers_guide/version_compat)?
* How will this proposal interact with other parts of the TensorFlow Ecosystem?
    - How will it work with TFLite?
    - How will it work with distribution strategies?
    - How will it interact with tf.function?
    - Will this work on GPU/TPU?
    - How will it serialize to a SavedModel?

### User Impact
* What are the user-facing changes? How will this feature be rolled out?

## Detailed Design

This section is optional. Elaborate on details if they’re important to
understanding the design, but would make it hard to read the proposal section
above.

## Questions and Discussion Topics

Seed this with open questions you require feedback on from the RFC process.
