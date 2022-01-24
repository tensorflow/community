# Tensorflow Model Optimization Compression API

| Status        | Accepted       |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [342](https://github.com/tensorflow/community/pull/342) |
| **Author(s)** | Jaehong Kim (kimjaehong@google.com), Alan Chiao (alanchiao@google.com), Jae Yoo (jaeyoo@google.com) |
| **Sponsor**   | Francois Chollet (fchollet@google.com)                 |
| **Updated**   | 2020-12-21

## Objective

Build a Keras-base API and set of guidelines that help compression algorithm developer to implement their own model compression algorithm (e.g. [Weight Clustering](https://arxiv.org/abs/1510.00149), [WEST](https://arxiv.org/abs/1811.08417)) and provide a standard way to testing/benchmark and create their own user API for model developers that includes compressed model deployment to TF serving, TFLite, tf.js, and TF-TRT.

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

We propose the compression algorithm API which helps algorithm developers create model developer APIs for their own compression algorithm.
Our API also provides guidelines for testing and benchmark. For now, we only have guidelines to apply a compression algorithm for simple MNIST vision cases. We'd like to provide an example for tensorflow [official models](https://github.com/tensorflow/models/tree/master/official) in the future.

### Tutorials and Examples
We provide the tutorial for [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition) compression algorithm that shows how we implement the SVD algorithm using TFMOT compression API by colab. This tutorial includes:

#### Algorithm developer side
The algorithm developer implementing the SVD algorithm uses the `WeightCompressor` class. It also includes a custom model developer API for the SVD algorithm.

```python
class SVD(algorithm.WeightCompressor):
  """SVD compression module config."""

  def __init__(self, rank):
    self.rank = rank

  def init_training_weights(
      self, pretrained_weight: tf.Tensor):
    """Init function from pre-trained model case."""
    # Dense Layer
    if len(pretrained_weight.shape) == 2:
      u, sv = tf_svd_factorization_2d(pretrained_weight, self.rank)
    else:
      raise NotImplementedError('Only for dimension=2 is supported.')

    self.add_training_weight(
        name='u',
        shape=u.shape,
        dtype=u.dtype,
        initializer=tf.keras.initializers.Constant(u))
    self.add_training_weight(
        name='sv',
        shape=sv.shape,
        dtype=sv.dtype,
        initializer=tf.keras.initializers.Constant(sv))

  def project_training_weights(self, u: tf.Tensor, sv: tf.Tensor) -> tf.Tensor:
    return tf.matmul(u, sv)

  def get_compressible_weights(
      self, original_layer: tf.keras.layers.Layer) -> List[tf.Variable]:
    if isinstance(original_layer, tf.keras.layers.Dense):
      input_dim = original_layer.kernel.shape[0]
      output_dim = original_layer.kernel.shape[1]
      if input_dim * output_dim > (input_dim + output_dim) * self.rank:
        return [original_layer.kernel]
    return []

  def compress_model(self, model: tf.keras.Model) -> tf.keras.Model:
    """Model developer API for optimizing a model."""

    def _compress_layer(layer):
      # Require layer to be built so that the SVD-factorized weights
      # can be initialized from the weights.
      if not layer.built:
        raise ValueError(
            'Applying SVD currently requires passing in a built model')

      return algorithm.create_layer_for_training(layer, algorithm=self)

    return tf.keras.models.clone_model(
        model, clone_function=_compress_layer)
```

#### Model developer side
1. The model developer uses the SVD algorithm.
```python
compressed_model = SVD(rank=32).compress_model(model)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
compressed_model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

compressed_model.fit(x_train, y_train, epochs=2)
compressed_model.evaluate(x_test, y_test, verbose=2)
```
2. Deploys their compressed model to TFLite model
```python
compressed_model.save('/tmp/model_svd_compressed')

def tflite_convert(saved_model_path, tflite_path):
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
  converted = converter.convert()
  open(tflite_path, 'wb').write(converted)

tflite_convert('/tmp/model_svd_compressed',
               '/tmp/tflite/model_svd_compressed.tflite')
```

We also want to provide an example of well-known compression algorithms. Here’s algorithm list at least we have to provide:
* [Weight clustering](https://arxiv.org/abs/1510.00149) : Most famous compression algorithm that can be used widely.
* [WEST](https://arxiv.org/abs/1811.08417) : Example for language model area.
* [Pruning](https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras) : Example for scheduling feature.

### Weight compression algorithm API

<p align="center">
 <img src=20201221-tfmot-compression-api/class_graph.png />
</p>

This is an API for a layer weight based compression algorithm.

First, we start from a pre-trained model which the model developer has. And then convert the pre-trained model to training phase model for compression fine-tuning training. During the convert to training phase model, We call `init_training_weights` for each tensor that we want to compress which is specified from the `get_compressible_weights` method.

During the training phase, `project_training_weights` method is called for each training step. After fine-tuning training for compression is finished, we convert the training phase model to a compressed model. We only call the `compress_training_weights` function once for each compressible tensor for converting.

Compressed model contains the `decompress_weights` function in the graph. It’s possible to call the `decompress_weights` for each inference step. To improve performance, we’ll cache the decompressed one depending on flags if we have enough space.

```python
class WeightCompressor(metaclass=abc.ABCMeta):
  """Interface for weight compression algorithm that acts on a per-layer basis.

     This allows both options of either decompressing during inference or
     decompressing prior to inference (where compression occurs by applying a
     tool such as zip to the model file).

     This interface is a purely functional one.
  """

  @abc.abstractmethod
  def get_compressible_weights(
      self, original_layer: tf.keras.layers.Layer) -> List[tf.Variable]:
    """Define compressible weights for each layer.

    Args:
       original_layer: tf.keras.layers.Layer representing a layer from the
       original model.

    Returns:
       List of compressible weights for the given layer.
    """

  @abc.abstractmethod
  def init_training_weights(
      self, pretrained_weight: tf.Tensor):
    """Initialize training weights for the compressible weight.

    It calls the `add_training_weight` to add a training weight for a given
    `pretrained_weight`. A `pretrained_weight` can have multiple training
    weights. We initialize the training weights for each compressible
    weight by just calling this function for each.

    Args:
      pretrained_weight: tf.Tensor of a pretrained weight of a layer that will
        be compressed eventually.
    """

  def add_training_weight(
      self, *args, **kwargs):
    """Add a training weight for the compressible weight.

    When this method is called from the `init_training_weights`, this adds
    a training weights for the pretrained_weight that is the input of the
    `init_training_weights`.

    Args:
      *args, **kwargs: args and kwargs for training_model.add_weight.
    """

  @abc.abstractmethod
  def project_training_weights(self, *training_weights: tf.Tensor) -> tf.Tensor:
    """Define a piece of the forward pass during training, which operates on a single compressible weight.
    The default throws an error when training occurs.

    Args:
       *training_weights: tf.Tensors representing any variables used during
         training, for a single compressible weight, in the order returned in
         `init_training_weights`.

    Returns:
       tf.Tensor to set the compressible weight to.
    """

  def update_training_weight(self, training_weight: tf.Tensor, tensor: tf.Tensor):
    """Update a training weight to a given tensor value.

    This method is for the case that training weight should update to specific
    value not from the model optimizer. It'll throws an error if it can't
    find the training weight.

    Args:
      training_weight: tf.Tensor representing a training weight.
      tensor: tf.Tensor representing a value to be assigned to the training weight.
    """

  @abc.abstractmethod
  def compress_training_weights(self, *training_weights: tf.Tensor) -> List[tf.Tensor]:
    """Define the operations to compress a single weight’s training form after training.

    'compress_training_weights' can refer to making the weight more amenable to compression
    or actually compress the weight.

    The default is an identity.

    Args:
      *training_weights: tf.Tensors representing all variables used during
        training, for a single compressible weight, in the order returned in
        `init_training_weights`.

    Returns:
      List of tf.Tensors to set to compressed or more compressible form.
    """

  @abc.abstractmethod
  def decompress_weights(self, *compressed_weights: tf.Tensor) -> tf.Tensor:
    """Define the operations to decompress a single weight’s compressed form during inference.

    The default is an identity.

    Args:
       *compressed_weights: tf.Tensors representing a single weight’s compressed
         form, coming from what’s returned in `compress`.

    Returns:
      A tf.Tensor representing the decompressed `compressed_weights`.
    """
```

### Alternatives Considered

#### Model compression algorithm API
Some compression algorithms require training weights or compressed weights that share the weights across the layer. (e.g. lookup table for weight clustering.)
We decided to support layer variable wise compression algorithm API first, because... :
* Most use cases can be covered by the `WeightCompressor` class API.
* Hard to support a sequential model: That weight across the layer should be placed somewhere outside of the sequential model.

### User Impact
This is a new API for compression algorithm developers. That users can implement their own compression algorithm easier.

#### UX study
We had UX study with user candidates for initial prototype and small colab tutorial. We found what features we have to support, and what makes it hard to understand this API.

Most confusing parts from the study were:
* Method naming is hard to understand

    Changed the method names but still need to discuss for better naming.

* Hard to understand when the overridden methods are called and how many the methods calls

    Added sequential step by step documentation and the diagram for each part of the execution step.

Requested features are in the discussion topic below.

### Performance Implications
We’ll provide examples of compression algorithms using the API in this design, and test/benchmark for the algorithm.

### Dependencies
This API is a standalone project that only depends on tensorflow.

### Engineering Impact
TF-MOT team will maintain this API code. For the initial release, we publicize the `WeightCompressor` class that the algorithm developers have to inherit this class to implement their own compression algorithm, WrapperLayer methods to access original layer, And model clone based default converter functions for model developer to help them implement their own algorithm specific APIs.

### Platforms and Environments
For initial release, we’ve targeted the TF 2.0 Keras model. After compressing the model, the compressed model can deploy to servers as TF model, mobile/embedded environments as TFLite model, and web as tf.js format.

### Best Practices
This is new standalone APIs that doesn’t change any current best practices for Tensorflow. We’ll provide the new best practices for the API.

### Compatibility
This API is compatible with the TF 2.0 Keras model. Because of technical limitations, we only support the Sequential/Functional API Keras model for initial release. (subclass model support API is not a part of the current design.)
Compressed models can be converted to TF model, TFLite model, and tf.js format. Compressed model is also one of the TF 2.0 Keras models. But we need to do additional compatibility engineering work to keep that model compressed after converting. (e.g. prevent constant folding.)

## Detailed Design
This is an API design doc. Engineering details will be determined in the future.
For better explanation of this API, Here's the step-by-step usage documentation below:

### Step-by-step usage documentation of the `WeightCompressor` class methods.

The `WeightCompressor` class has 5 abstract methods. Following explanation shows when these methods are called and used.

#### User facing API Template

We have two steps for user facing API for general cases.
(The SVD case training model is the same as the compressed model, so it only has one step.)

```python
class CustomWeightCompressor(WeightCompressor):
  def optimize_model(self, model: tf.keras.Model) -> tf.keras.Model:
    """Model developer API for optimizing a model."""

    def _optimize_layer(layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
      if not layer.built:
        raise ValueError(
            'Applying compression currently requires passing in a built model')

      return algorithm.create_layer_for_training(
          layer, algorithm=self)

    return tf.keras.models.clone_model(
        model, clone_function=_optimize_layer)


  def compress_model(self, model: tf.keras.Model) -> tf.keras.Model:
    """Model developer API for optimizing a model."""

    def _compress_layer(layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
      if not layer.built:
        raise ValueError(
            'Applying compression currently requires passing in a built model')

      return algorithm.create_layer_for_inference(
          layer, algorithm=self)

    return tf.keras.models.clone_model(
        model, clone_function=_compress_layer)
```

#### Model developer best practice.

Here's the best practice for general compression algorithm model developer code.

```python
compressor = CustomWeightCompressor()
training_model = compressor.optimize_model(model)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
training_model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

training_model.fit(x_train, y_train, epochs=2)

compressed_model = compressor.compress_model(training_model)
compressed_model.evaluate(x_test, y_test, verbose=2)
```

#### Weight compression algorithm API methods usage

Now we'll explain when each method is called and how many that method called for the model developer code before.

1. `get_compressible_weights`
<p align="center">
  <img src=20201221-tfmot-compression-api/get_compressible_weights.png />
</p>

```python
training_model = compressor.optimize_model(model)
```

 `get_compressible_weights` is called when we want to get a list of variables that we will apply compression.
When we try to compress the pre-trained model, we just call this method for each layer in the pre-trained model. The number of the method calling is (# of layers).

2. `init_training_weights`
<p align="center">
  <img src=20201221-tfmot-compression-api/init_training_weights.png />
</p>

```python
training_model = compressor.optimize_model(model)
```

 `init_training_weights` is called when we initialize the cloned training model from the pre-trained model. `optimize_training` method basically clones the model to create a training model for compression, wrapping compressible layers by the training wrapper to create training weights. The number of the method calling is (# of compressible weights).

3. `project_training_weights`
<p align="center">
  <img src=20201221-tfmot-compression-api/project_training_weights.png />
</p>

```python
training_model.fit(x_train, y_train, epochs=2)
```

 `project_training_weights` is called when the training model for the compression algorithm is training. Usually this method function is a part of the training model. It recovers the original weight from the training weights, and should be differentiable. This method enables you to use the original graph to compute the model output, but train the training weights of the training model. For each training step, this method is called for every compressible weight. The number of the method calling is (# of compressible weights) * (training steps).

4. `compress_training_weights`
<p align="center">
  <img src=20201221-tfmot-compression-api/compress_training_weights.png />
</p>

```python
compressed_model = compressor.compress_model(training_model)
```

 `compress_training_weights` is called when we convert the training model to the compressed model. The number of the method calling is (# of compressible weights).

5. `decompress_weights`
<p align="center">
  <img src=20201221-tfmot-compression-api/decompress_weights.png />
</p>

```python
compressed_model.evaluate(x_test, y_test, verbose=2)
```

 `decompress_weights` is called when we do inference on a compressed model. Usually this method function is a part of a compressed model. This method decompresses the weight that can be used on the original graph for each compressible weight. Basically the number of this method called is (# of compressible weights) * (# of inference). To improve performance, the output value of this method can be cached.

## Questions and Discussion Topics

### Support gradient based compression algorithm.

Currently this apis compresses the weights only using the weight value. but some lossy compression algorithm like pruning requires gradient value to determine which weight will be pruned.

### Custom additional custom loss for the compression model.

Some compression algorithms require additional custom loss (e.g. entropy loss for the weights) which is not used in the original training script for pre-trained models. Algorithm developers can add their custom loss on user facing API functions. They can access the training weights of the model using a wrapper layer. To make. sure this code is working, we have to preserve weights order of get_weights method for the wrapper layer. They can call loss function using that training weights on the function. but still it's not possible to add loss that uses training weights and activation together.
Note that every trainable variable that they want to train should be in training weights.

### Error message & Debugging tools.

It's not easy to find the bug there. Usually we get tensorflow bug messages with huge stack traces. We have to provide some bug messages for this API layer.
