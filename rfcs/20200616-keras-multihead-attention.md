# RFC: Multihead Attention and EinsumDense on Keras

| Status        | Proposed         |
| :------------ | :------------------------------------------------------ |
| **RFC #**     | [260](https://github.com/tensorflow/community/pull/260) |
| **Author(s)** | Hongkun Yu (hongkuny@google.com), Mark Omernick (momernick@google.com) |
| **Sponsor**   | Francois Chollet (fchollet@google.com)                  |
| **Updated**   | 2020-06-16                                              |

## Objective

Introduce the MultiHeadAttention layer and EinsumDense layer to tf.keras.

## Motivation

MultiHeadAttention is very popular and has become standard for deep learning
libraries. We propose to contribute a flexible well-defined implementation
inside Keras absorbing common best practices from reference libraries.

## User Benefit

We can standardize the implementation of Transformer layers and use the best
practice. We offer a rich set of functionalities to different use cases, e.g.
different project spaces, outputing multi-head attention scores for analysis,
etc. We also modularize computations to make the MultiHeadAttention layer
extensible to variants.

## Design Proposal

### Key Features

*   Returns multi-headed attention scores, which is commonly useful for
    attention visualization and analysis.
*   Supports query (Q), key (K), value (V) tensors as individual inputs and
    supports projecting Q, K, V to different dimensions.
*   Final outputs projects to user specified dimensions.
*   Using tf.einsum to express high-dimensional computation and adopts
    [tf.keras.layers.experimental.EinsumDense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/EinsumDense)
    layer.
*   Supports high-dimension attention when target and source are 2D, 3D, etc.

### Code Examples

*   How to write a TransformerBlock for an encoder.

```python
class TransformerBlock(tf.keras.layers.Layer):
  def __init__(self, embed_dim, num_heads, ff_dim):
    super(TransformerBlock, self).__init__()
    self.att = attention.MultiHeadAttention(embed_dim, num_heads)
    self.ffn = tf.keras.Sequential(
        [tf.keras.layers.Dense(ff_dim, activation="relu"),
         tf.keras.layers.Dense(embed_dim),]
    )
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

  def call(self, inputs, attention_mask=None):
    attn_output = self.att([inputs, inputs], attention_mask=attention_mask)
    out1 = self.layernorm1(inputs + attn_output)
    ffn_output = self.ffn(out1)
    return self.layernorm2(out1 + ffn_output)
```

*   Use attention mask to avoid performing attention on padding token indices.

```python
test_layer = TransformerBlock(
    embed_dim=2,
    num_heads=2,
    ff_dim=4)
query = np.array([[[0.1, 0.2], [0.0, 0.0]]])
mask = np.array([[[1, 0], [1, 0]]], dtype='bool')
output = test_layer(query, mask)
```

*   Inside a Transformer decoder, we often want to output the cross-attention
    scores to analyze how the target sequence attend to the source sequence. We
    are able to visualize the alignment according to attention scores.

```python
test_layer = MultiHeadAttention(
    num_heads=2, key_size=2, return_attention_scores=True)
target = np.array([[[0.1, 0.2], [0.0, 0.0]]])
source = np.array([[[0.1, 0.2], [3.0, 1.0]]])
output, scores = test_layer(query=target, value=source)
scores = tf.math.reduce_sum(scores, axis=1) # shape = (1, 2, 2)
```

*   Attention beyound sequences. Taking 2D, 3D target and source.

```python
query_shape = [2, 3, 4, 4]  # batch, target, target, embedding.
value_shape = [2, 3, 2, 4]  # batch, source, source, embedding.
mask_shape = [2, 3, 4, 3, 2]
query = 10 * np.random.random_sample(query_shape)
value = 10 * np.random.random_sample(value_shape)
mask_data = np.random.randint(2, size=mask_shape).astype("bool")
output = test_layer(query=query, value=value, attention_mask=mask_data)
```

### Interface

```python
class MultiHeadAttention(tf.keras.layers.Layer):
  """MultiHeadAttention layer.

  This is an implementation of multi-headed attention based on "Attention
  is all you Need". If `query`, `key,` `value` are the same, then
  this is self-attention. Each timestep in `query` attends to the
  corresponding sequence in `key`, and returns a fixed-width vector.

  This layer first projects `query`, `key` and `value`. These are
  (effectively) a list of tensors of length `num_attention_heads`, where the
  corresponding shapes are [batch_size, <query dimensions>, key_size],
  [batch_size, <key/value dimensions>, key_size],
  [batch_size, <key/value dimensions>, value_size].

  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor.

  Finally, the result tensor with the last dimension as value_size can take an
  linear projection and return.

  Examples:

  Performs 1D cross-attention over two sequence inputs with an attention mask.
  Returns the additional attention weights over heads.

  >>> layer = MultiHeadAttention(num_heads=2, key_size=2,
  ...                            return_attention_scores=True)
  >>> target = tf.keras.Input(shape=[8, 16])
  >>> source = tf.keras.Input(shape=[4, 16])
  >>> mask_tensor = tf.keras.Input(shape=[8, 4])
  >>> output_tensor, weights = layer(query=target, value=source
  ...                                attention_mask=mask_tensor)
  >>> print(output_tensor.shape), print(weights.shape)
  (None, 8, 16)  (None, 2, 8, 4)

  Performs 2D self-attention over a 5D input tensor on axes 2 and 3.

  >>> layer = MultiHeadAttention(num_heads=2, key_size=2, attention_axes=(2, 3))
  >>> input_tensor = tf.keras.Input(shape=[5, 3, 4, 16])
  >>> output_tensor = layer(query=input_tensor, value=input_tensor)
  >>> print(output_tensor.shape)
  (None, 5, 3, 4, 16)

  Arguments:
    num_heads: Number of attention heads.
    key_size: Size of each attention head for query and key.
    value_size:  Size of each attention head for value.
    dropout: Dropout probability for a Dropout layer on attention_scores.
    use_bias: Boolean, whether the dense layers use bias vectors/matrices.
    output_shape: The expected shape of an output tensor, besides the batch and
      sequence dims. If not specified, projects back to the key feature dim.
    attention_axes: axes over which the attention is applied. `None` means
      attention over all axes, but batch, heads, and features.
    return_attention_scores: bool, if `True`, returns the multi-head
      attention scores as an additional output argument.
    kernel_initializer: Initializer for dense layer kernels.
    bias_initializer: Initializer for dense layer biases.
    kernel_regularizer: Regularizer for dense layer kernels.
    bias_regularizer: Regularizer for dense layer biases.
    activity_regularizer: Regularizer for dense layer activity.
    kernel_constraint: Constraint for dense layer kernels.
    bias_constraint: Constraint for dense layer kernels.
  """

  def call(self, query, value, key=None, attention_mask=None):
    """Implements the forward pass.

    Size glossary:
      * Number of heads (H): the number of attention heads.
      * Value size (V): the size of each value embedding per head.
      * Key size (K): the size of each key embedding per head. Equally, the size
          of each query embedding per head. Typically K <= V.
      * Batch dimensions (B).
      * Query (target) attention axes shape (T).
      * Value (source) attention axes shape (S), the rank must match the target.

    Args:
      query: Query `Tensor` of shape `[B, T, dim]`.
      value: Value `Tensor` of shape `[B, S, dim]`.
      key: Optional key `Tensor` of shape `[B, S, dim]`. If not given, will
          use `value` for both `key` and `value`, which is the most common case.
      attention_mask: a boolean mask of shape `[B, T, S]`, that prevents
        attention to certain positions.

    Returns:
      attention_output: The result of the computation, of shape [B, T, E],
        where `T` is for target sequence shapes and `E` is the query input last
        dimension if `output_shape` is `None`. Otherwise, the multi-head outputs
        are project to the shape specified by `output_shape`.
      attention_scores: [Optional] multi-head attention coeffients over
        attention axes.
    """
```

### Auxiliary Layers and Changes

*   EinsumDense layer

We use `tf.einsum` to implement a dense layer can perform einsum calculations of
arbitrary dimensionality. This example shows how to instantiate a layer that
applies the same dense operation to every element in a sequence. Here, the
'output_shape' has two values (since there are two non-batch dimensions in the
output); the first dimension in the output_shape is `None`, because the sequence
dimension `b` has an unknown shape.

```python
layer = EinsumDense("abc,cd->abd", output_shape=(None, 64), bias_axes="d")
input_tensor = tf.keras.Input(shape=[32, 128])
output_tensor = layer(input_tensor) # output shape is (None, 32, 64)
```

*   Masked Softmax

Inside the attention computation, we need to mask logits before softmax and it
has become a common treatment in many applications. We propose to add an
optional `mask` argument to `tf.nn.softmax`. The downstream keras `Softmax`
layer will also take an optional `mask` tensor. This `mask` tensor should have
the same rank as the input tensor and mask elements on the axis which will
perform softmax.

Inside `MultiHeadAttention` keras layer, we will use the keras `Softmax` layer
with mask and adjust attention mask shape to match the inputs. The dimension
expension logic and multi-axes softmax will be handled locally in
`MultiHeadAttention` layer.

*   Keras Dense Attention

[tf.keras.layers.Attention](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Attention)
layer call method takes an optional argument, `mask`, which requires two
tensors, `q_mask` and `v_mask`. They are following keras framework requirements
with (batch_size, target_length) and (batch_size, source_length) as shapes. This
limits the flexibility of masking and `MultiHeadAttention` layer generalize the
attention mask to be (batch dims, target dims, source dims). To be consistent,
we would like to introduce an optional argument `attention_mask` for
`tf.keras.layers.Attention`. In the reduced case of `tf.keras.layers.Attention`,
the shape is (batch_size, target_length, source_length). Whenever
`attention_mask` is specified, the `mask` argument is OK to be skipped.

*   TFA `MultiHeadAttention` Deprecation and Re-mapping

[MultiHeadAttention](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/layers/multihead_attention.py)
has been released. The proposed `MultiHeadAttention` has similar `__init__`
arguments and `call` interface, where the minor differences are argument names
and the attention `mask` shape. We expect the new `MultiHeadAttention` keras
layer will cover the functionalities. Once the implementation are merged as
experimental layers, we will work with TF Addons team to design the deprecation
and re-mapping procedure.

### Alternatives Considered

We examined multi-head attention layer implemented in various libraries. There
are a few features that we do not include inside this keras layer and we feel it
is better to subclass the `MultiHeadAttention` layer to fulfill the needs.

*   Attention caching for decoding. Implemented in
    [Flax](https://github.com/google/flax/blob/master/flax/nn/attention.py#L301).
    The caching is a special treatment for inference and we noticied that
    different treatments are required for dynamic or static shape programs.
    Thus, subclassing as a
    [CachedAttention](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/attention.py)
    layer is the solution inside the model garden.
*   [MultiHeadAttention](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/layers/multihead_attention.py)
    keras layer is also implemented in TF-Addons. The design in this doc covers
    the features in TF-addons implementation but generalizes to more use cases.

### Performance Implications

*   We will add microbenchmarks following the common practices of keras layers.
*   We have end-to-end integration/regression tests for models using this layer,
    e.g. BERT.

### Dependencies

No dependencies.

### Engineering Impact

*   The keras layer can be tested inside the package.
*   TensorFlow team will maintain the code.

### Platforms and Environments

*   Work for all platforms and environments

### Best Practices

*   No change for Tensorflow best practices.

### Tutorials and Examples

*   Code examples can be found inside Tensorflow Model Garden. For example, an
    encoder
    [Transformer](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/transformer.py).

*   2D attention example in the
    [unit test](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/attention_test.py#L135).

### Compatibility

*   This is a new layer without compatibility concerns.
*   The proposal works with TFLite, distribution strategy, tf.function, GPU/TPU
    and serializable to SavedModel. These are tested inside TensorFlow Model
    Garden applications.

### User Impacteisum

*   We will first introduce the layer as
    `tf.keras.layers.experimental.MultiHeadAttention` and
    `tf.keras.layers.experimental.EinsumDense`. When the APIs are stable and
    functionalities are fully verified, the next step is to graduate as core
    keras layers by removing `experimental` scope.

## Detailed Design

The layer has been implemented as the
[MultiHeadAttention](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/attention.py#L116)
inside TensorFlow Model Garden.

First, as we rely on `tf.einsum` to define projections and attention
computation, we need to figure out the einsum notation of each computation.
Furthermore, to make the layer generalize to high-dimension cases, i.e. there
are more than one batch dimensions and attention softmax can be performed on
multiple axes, we need to track the batch axes and attention axes inside einsum
notations. We use a vector of chars and use two local methods to generate einsum
notations for projections and attentions.

Second, the layer by default implements the most common dot-product attention.
There are various ways to implement the attention computation, so we modulize it
as two methods `build_attention` and `compute_attention`. Thus, users will be
able to just override them to get a new keras layer with a novel attention
method. For example, we implemented
[TalkingHeadAttention](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/talking_heads_attention.py)
introduced by ["Talking-Heads Attention "](https://arxiv.org/abs/2003.02436)
paper. Using the keras Attention layer as another example, since it supports the
basic single-head case 1-D attention, we can use it inside `_build_attention`
and `_compute_attention`.

## Questions and Discussion Topics

-   cuDNN has the
    [multi-head attention](https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnMultiHeadAttnForward)
    function. How do we incorporate it? A: we modularize the attention
    computation components in order to support new low-level functions without
    changing this layer interface. The cuDNN function supports the classic
    dot-product attention with classic input dimensions. We will be able to use
    it once TensorFlow add an op to use it.
