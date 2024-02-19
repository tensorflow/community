# On-device training with TensorFlow Lite

| Status        | Draft                                                                                                        |
| :------------ | :----------------------------------------------------------------------------------------------------------- |
| **Author(s)** | Yu-Cheng Ling (ycling@google.com), Haoliang Zhang (haoliang@google.com), Jaesung Chung (jaesung@google.com)  |
| **Sponsor**   | Jared Duke (jdduke@google.com)                                                                               |
| **Updated**   | 2021-06-04                                                                                                   |

## Introduction

TensorFlow Lite is TensorFlow's solution for on-device machine learning.
Initially it only focused on *inference* use cases. We have increasingly heard
from users regarding the need for on-device *training*. This proposal lays out
the concrete plan & roadmap for supporting training in TensorFlow Lite.

## Goals & Non-Goals

Goals:

*   Describe the developer experience of training with TensorFlow Lite
*   Illustrate a roadmap for implementing training in TensorFlow Lite

Non-goals for this proposal:

*   Training on devices with Model Optimization Toolkit techniques (quantization
    / sparsity) is future work.
*   Support TensorFlow Lite training with legacy model or TensorFlow 1.x
    features (reference variable, v1 control flow...etc)

## Overview

We have already demonstrated basic on-device training capability with the
TensorFlow Lite personalization
[example](https://blog.tensorflow.org/2019/12/example-on-device-model-personalization.html).
Along with existing functionalities, TensorFlow Lite can provide generic and
developer-friendly training features by supporting a number of low-level
features (e.g. gradients, optimizers, variables, TensorList, weight
serialization and multiple signatures).

In the rest of the document, we will start by describing the expected developer
experience, and talk about these low-level features (how it relates to training
and how to support it).

See also TensorFlow Lite's overall
[roadmap](https://www.tensorflow.org/lite/guide/roadmap). We also plan to
release an end to end Colaboratory example to demonstrate the functionalities
(e.g. training a simple model, transfer learning with an existing backbone
model...etc).

## Developer Experience

To use TensorFlow Lite, a developer needs to prepare a TensorFlow model, use the
[converter](https://www.tensorflow.org/lite/convert) to convert it to TensorFlow
Lite model format, and run the model with the TensorFlow Lite runtime on device.
This is true for inference use cases, and a similar flow can be applied to
training too.

Instead of converting a single TensorFlow model or `tf.function` to a TensorFlow
Lite model with a single entry point, we can convert multiple `tf.function`(s)
into a TensorFlow Lite model. To be able to do that, we're extending the
TensorFlow Lite's converter & runtime to handle **multiple signatures**.

The following pseudo code illustrates the high-level flow of preparing a
TensorFlow Lite model, converting it to TensorFlow Lite model and running in
TensorFlow Lite runtime for a training use case. Disclaimer:

*   The pseudo code is simplified for easier explanation and the real code may
    need to be longer (e.g. Technically functions like Keras `model.fit` cannot
    be wrapped in `@tf.function` now, so it takes more lines of code to do the
    same thing)
*   As of this writing, the implementation is a work-in-progress and it isn’t
    yet fully functional. We'd like to submit this RFC before proceeding with
    the full implementation.

**Preparing a TensorFlow Model**. The code constructs a `tf.module` with 4
`tf.functions`:

*   `train` function trains the model with training data.
*   `infer` function invokes the inference.
*   `save` function saves the trainable weights into the file system.
*   `restore` function loads the trainable weights into the file system.


```python
import tensorflow as tf
import numpy as np

class Model(tf.Module):
  def __init__(self):
    self.model = tf.keras.applications.MobileNet(alpha=0.25, dropout=0)
    self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
    self.optimizer = tf.keras.optimizers.SGD()

  @tf.function
  def train(self, inputs, labels):
    self.model.fit(inputs, labels)

  @tf.function
  def infer(self, inputs):
    self.model.predict(inputs)

  @tf.function
  def save(self, checkpoint_path):
    self.model.save_weights(checkpoint_path)

  @tf.function
  def restore(self, checkpoint_path):
    self.model.load_weights(checkpoint_path)

model = Model()
tf.saved_model.save(
    model, "/tmp/model", signatures={
        "train": model.train,
        "infer": model.infer,
        "save": model.save,
        "restore": model.restore,
    })
```

**Converting to TensorFlow Lite format.** Note the `TFLiteConverter` API will be
extended to choose which functions / signatures to convert.

```python
converter = tf.lite.TFLiteConverter.from_saved_model(
    "/tmp/model", signature_keys=["train", "infer", "save", "restore"])
tflite_model = converter.convert()
```

**Executing in TensorFlow Lite runtime**. TensorFlow Lite's Interpreter
capability will be extended to support multiple signatures too. Developers can
choose to invoke restoring, training, saving and inferring signatures
separately. The pseudo code is written in Python, but developers can use
equivalent APIs in Java/ObjC/Swift..etc for mobile development.

```python
# Construct the interpreter and get the signature runners.
interpreter = tf.lite.Interpreter(TFLITE_FILE_PATH)
train = interpreter.get_signature_runner(method_name="train")
infer = interpreter.get_signature_runner(method_name="infer")
save = interpreter.get_signature_runner(method_name="save")
restore = interpreter.get_signature_runner(method_name="restore")

# Restore trainable weights from files
restore("/path/to/checkpoints")
# Run training for a few steps
train(train_input, train_labels)
# Save trainable weights back to checkpoint files files
save("/path/to/checkpoints")
# Developers can also use the trained model to run inference now
prediction = infer(input)
```

## Low-level features required for training

As mentioned, training can be well supported by supporting a number of low-level
features in TensorFlow Lite. The necessity of these features are explained as
follows:

1.  Gradients: To compute the gradient values in the backward pass
2.  Optimizers: To apply the optimization algorithm (like gradient descent,
    Adam...etc) to the trainable weights
3.  Variables: To store & update the trainable weights
4.  TensorList (for RNN training): Required to compute gradient over loops,
    which is used in important model structures like recurrent networks.
5.  Weights serialization: To save and load trainable weights into files
6.  Multiple signatures: Be able to deploy a TensorFlow Lite model with
    different functionalities (training, inference...etc)

By breaking it down to smaller feature requests, it comes with a few nice
properties:

*   The first 5 features can be supported by enhancing TensorFlow Lite's
    [Select TensorFlow Ops](https://www.tensorflow.org/lite/guide/ops_select)
    functionality, since all these features are implemented in terms of
    TensorFlow ops and kernels. We can leverage TensorFlow code to make it work
    first, then optimize later.
*   Most of these features (including optimizing the first 5 features) are
    orthogonal, so we can easily divide & conquer to make progress in parallel.

The following sections briefly talk about each of these features.

### Gradients

In TensorFlow, when developers define the forward pass of the model, the
backward pass (including gradient computation) is generated by the
[automatic differentiation](https://www.tensorflow.org/guide/autodiff) feature.
Sometimes the gradient is consisted of "regular" ops (e.g. the gradient of `Sub`
is `Neg` if there's no broadcasting), and sometimes fused gradient ops are used
(e.g. `ReluGrad`, `Conv2DBackpropInput`, `Conv2DBackpropFilter`).

From the infrastructure's perspective, fused gradient ops are not different from
regular mathematical ops.

For regular ops like `Neg`, we can just implement the TFLite native kernels (or
reusing the existing ones).

For fused gradient ops, there are a few options:

1.  Utilize TensorFlow kernels (e.g. via
    [Select TensorFlow Ops](https://www.tensorflow.org/lite/guide/ops_select))
2.  Decompose it to other TF ops, which are supported by TFLite
3.  Consider to implement fuesd ops/kernels in TFLite

We recommend to start with option 1 to get a really broad use case coverage,
then invest on option 2 & 3 later.

### Optimizers

There are at least 18 fused optimizer ops in TF which access resource variables
directly, like `ResourceApplyRMSProp`
(assuming we're not supporting TF1 reference variables).

Similar to the gradient support problem, the options to support optimizer ops
includes:

1.  Utilize TensorFlow kernels (e.g. via
    [Select TensorFlow Ops](https://www.tensorflow.org/lite/guide/ops_select))
    1.  This requires TFLite to properly support resource types, and enable
        passing resource tensors across delegate boundaries and subgraphs.
    2.  Note this will NOT work with TFLite native variables.
2.  De-compose to other TF ops which are supported by TFLite. This enables
    utilizing TFLite native variables.
3.  Consider to implement fuesd ops/kernels in TFLite natively

Though option 2 or 3 are possible, it is a significant amount of work. We'd
recommend to bootstrap the training feature with option 1, and move to option 2
or 3 later.

### Variables

The option to support variable includes:

1.  Utilize TensorFlow resource variable kernels (e.g. `ReadVariableOp`,
    `AssignVariableOp`)
2.  TFLite native variables: There is ongoing work to support native variables
    in TFLite.

However, since TF resource variables and TFLite native variables cannot be mixed
together, before we move away from TF optimizer kernels (which uses TF resource
variables), we cannot use TFLite native variables.

Our recommendation is bootstrapping the training feature with option 1. We can
invest in option 2 later, along with a solution to move away from using
TensorFlow kernels for optimizers.

### TensorList

`TensorList` is a data structure used in TensorFlow internally. The major use
cases includes:

1.  When `tf.TensorArray` is used in TensorFlow 2.x, `TensorList` is used behind
    the scene.
2.  When there are loops in the forward pass of the model, `TensorList` is
    automatically used in the backward pass to memorize the information when
    training over the loops.

Therefore, TensorList is critical for use cases where loops are used (including
important model structures like unfused RNN). There are at least 19
TensorList ops in TensorFlow.

The option to support TensorList includes:

1.  Utilize TF TensorList kernels (e.g. via
    [Select TensorFlow Ops](https://www.tensorflow.org/lite/guide/ops_select))
2.  Lower it to native TFLite ops (already implemented, but only support a
    subset of functionalities)
3.  Implement native TFLite TensorList kernels, or make an alternative design to
    provide a set of TFLite native kernels to support TF TensorList ops with
    generality.

It's hard to get generality from option 2, and it may take some time to design
and implement option 3. Our recommendation is bootstrapping the project with
option 1, and consider moving to option 2 or 3 later.

### Multiple signatures

Developers may want to ship a model that contains a few different
functionalities: Training, inference, restoring weights, saving weights,
validation, and more. To make the developer experience convenient, this requires
properly supporting multiple signatures in TFLite.

We already introduced the concept of signature into TensorFlow Lite's
C++/Java/Python APIs, but those APIs do not support multiple signatures yet. The
converter support is also working in progress.

### Weights serialization

The typical training flow involves restoring saved weights, performing training,
and saving weights back to files. To make the developer experience smooth,
TensorFlow Lite needs to provide functionalities to handle weights
serialization.

The option includes:

1.  Support TensorFlow's checkpoint format and load / saving functionalities.
    This can be done by reusing TensorFlow's kernels like `SaveV2` and `RestoreV2`.
    via
    [Select TensorFlow Ops](https://www.tensorflow.org/lite/guide/ops_select).
2.  Design and implement TFLite's weight saving format
3.  Consider supporting multiple approaches, so users can choose one format from
    the options above.

Our recommendation:

*   Starting with option 1 first, then look into option 2 or 3 later.
*   Encapsulate the weights saving / restoring logic in TensorFlow functions and
    the corresponding TensorFlow Lite signatures.
*   Use TensorFlow ops or TensorFlow Lite ops to perform variable saving /
    restoring.
