# Tensorflow Model Optimization Training-time compression API

| Status        | Draft       |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | TBD [NNN](https://github.com/tensorflow/community/pull/NNN) (update when you have community PR #)|
| **Author(s)** | Jaehong Kim (kimjaehong@google.com), Alan Chiao (alanchiao@google.com), Jae Yoo (jaeyoo@google.com) |
| **Sponsor**   | TBD (whomever@tensorflow.org)                 |
| **Updated**   | 2020-12-21

## Objective

Build a Keras-base API and set of guidelines that help compression algorithm developer to implement their own model compression algorithm (e.g. [Weight Clustering](https://arxiv.org/abs/1510.00149), [WEST](https://arxiv.org/abs/1811.08417)) and provide a standard way to testing/benchmark and create their own user API for model developers that includes compressed model deployment to TF serving, TFLite, and tf.js.

### Goals
* Enables algorithms that optimize the weights of a model but not the activations, which includes all [traditional lossless compression algorithms](https://en.wikipedia.org/wiki/Lossless_compression#:~:text=Lossless%20compression%20is%20a%20class,reconstructed%20from%20the%20compressed%20data.).
* Enables applying algorithms both during-training and post-training.
* Enables decompressing the weights either before inference or during inference.

### Non-Goals
* Optimize the activations of a model for accelerated inference. (e.g.
  [full-integer quantization](https://www.tensorflow.org/lite/performance/post_training_quantization#full_integer_quantization) changes dtype of activations to integer from float.)
* The algorithms that modify the output shape of a layer. (e.g. variant of structured pruning that reduces some output shape of a layer.)

## Motivation

Today, many compression researchers fork and modify model and layer code directly. For initial training research for a small number of architectures, this would be the simplest thing to do today, given the maximal flexibility on top of existing TF Core and Keras APIs. It’s not too bad since for weight optimization, there are only a few layers to consider (Dense, LSTM, Conv, and
Embedding) for broad model coverage.

With the compression API, algorithm developers can focus the core part of their algorithm. Once they implemented the algorithm, our API and guideline gave them a standard way to test, benchmark and export the model developer APIs for their compression algorithm.

We had a small study for algorithm developer candidates for our compression APIs. It can help us to understand what kinds of requirements are needed to support several compression algorithms and what features are most important. More details are below.

TF MOT already supports several optimization algorithms such as pruning, quantization aware training, and tensor encoding. Also, ARM contributed a weight clustering algorithm. Now we require a common part of these optimization algorithms. For the first step of that, we'd like to start from the compression algorithm (subset of optimization algorithm). because it's much easier than supporting all kinds of optimization algorithms and has a meaningful impact.

## User Benefit

In this design, we'd like to reduce the common engineering cost for the compression algorithm developers.

* Write unit test model coverage test, and benchmark. Provide the comparisons of compression algorithms.
* Deployment compressed model. (TF serving, TFLite, and tf.js)
* Support TF 2.0 Keras features compatibility. (e.g. distributed training.)


## Design Proposal
TBD

### API

```python
class WeightCompressionAlgorithm(metaclass=abc.ABCMeta):
  """Interface for weight compression algorithm that acts on a per-layer basis.

     This allows both options of either decompressing during inference or
     decompressing prior to inference (where compression occurs by applying a
     tool such as zip to the model file).

     This interface is a purely functional one.
  """

  @abc.abstractmethod
  def init_training_weights_repr(
      self, pretrained_weight: tf.Tensor) -> List[WeightRepr]:
    """Create training weight representations for initializing layer variables.

    Args:
      pretrained_weight: tf.Tensor of a pretrained weight of a layer that will
        be compressed eventually.

    Returns:
      A list of `WeightRepr`, a container for arguments to
      `tf.keras.layers.Layer.add_weight`for each tf.Variable to create.
    """

  @abc.abstractmethod
  def training(self, *training_weights: tf.Tensor) -> tf.Tensor:
    """Define a piece of the forward pass during training, which operates on a single compressible weight.
    The default throws an error when training occurs.

    Args:
       *training_weights: tf.Tensors representing any variables used during
         training, for a single compressible weight, in the order returned in
         `init_training_weights_repr`.

    Returns:
       tf.Tensor to set the compressible weight to.
    """

  def compress(self, *training_weights: tf.Tensor) -> List[tf.Tensor]:
    """Define the operations to compress a single weight’s training form after training.

    'Compress' can refer to making the weight more amenable to compression
    or actually compress the weight.

    The default is an identity.

    Args:
      *training_weights: tf.Tensors representing all variables used during
        training, for a single compressible weight, in the order returned in
        `init_training_weights_repr`.

    Returns:
      List of tf.Tensors to set to compressed or more compressible form.
    """

  def decompress(self, *compressed_weights: tf.Tensor) -> tf.Tensor:
    """Define the operations to decompress a single weight’s compressed form during inference.

    The default is an identity.

    Args:
       *compressed_weights: tf.Tensors representing a single weight’s compressed
         form, coming from what’s returned in `compress`.

    Returns:
      A tf.Tensor representing the decompressed `compressed_weights`.
    """

  def get_compressible_weights(
      self, original_layer: tf.keras.layers.Layer) -> List[str]:
    """Define compressible weights for each layer.

    Args:
       original_layer: tf.keras.layers.Layer representing a layer from the
       original model.

    Returns:
       List of attribute names as string representing list of compressible
       weights for the given layer. (e.g. return value ['kernel'] means
       layer.kernel is compressible.)
    """

```


### Alternatives Considered
TBD

### Performance Implications
We’ll provide examples of compression algorithms using the API in this design, and test/benchmark for the algorithm.

### Dependencies
This API is a standalone project that only depends on tensorflow.

### Engineering Impact
TF-MOT team will maintain this API code. For the initial release, we publicize the WeightCompressionAlgorithm class that the algorithm developers have to inherit this class to implement their own compression algorithm, WrapperLayer methods to access original layer, And model clone based default converter functions for model developer to help them implement their own algorithm specific APIs.

### Platforms and Environments
For initial release, we’ve targeted the TF 2.0 Keras model. After compressing the model, the compressed model can deploy to servers as TF model, mobile/embedded environments as TFLite model, and web as tf.js format.

### Best Practices
This is new standalone APIs that doesn’t change any current best practices for Tensorflow. We’ll provide the new best practices for the API.

### Tutorials and Examples
We provide the tutorial for [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition) compression algorithm that shows how we implement the SVD algorithm using TFMOT compression API by colab. This tutorial includes:
* How the algorithm developer implements the SVD algorithm and makes SVD API for model developer.
* How the model developer use the SVD algorithm and deploys their compressed model to TF and TFLite model.

We also want to provide an example of well-known compression algorithms. Here’s algorithm list at least we have to provide:
* [Weight clustering](https://arxiv.org/abs/1510.00149) : Most famous compression algorithm that can be used widely.
* [WEST](https://arxiv.org/abs/1811.08417) : Example for language model area.
* [Pruning](https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras) : Example for scheduling feature.

### Compatibility
This API is compatible with the TF 2.0 Keras model. Because of technical limitations, we only support the Sequential/Functional API Keras model for initial release. (subclass model support API is not a part of the current design.)
Compressed models can be converted to TF model, TFLite model, and tf.js format. Compressed model is also one of the TF 2.0 Keras models. But we need to do additional compatibility engineering work to keep that model compressed after converting. (e.g. prevent constant folding.)

### User Impact
This is a new API for compression algorithm developers. That users can implement their own compression algorithm easier.

## Detailed Design

TBD (optional)

## Questions and Discussion Topics

TBD


