# Attention for Dense networks on Keras

| Status        | Proposed                                                                   |
:-------------- |:-------------------------------------------------------------------------- |
| **Author(s)** | Georgios Roumpos (roumposg@google.com)                                     |
| **Sponsors**  | Karmel Allison (karmel@google.com), Francois Chollet (fchollet@google.com) |
| **Updated**   | 2019-01-15                                                                 |

## Objective and Motivation

Recently people have had success using the Attention mechanism in dense layers,
e.g. CNN+Attention or Transformer networks. Some examples are the
["Attention is all you need"](https://arxiv.org/abs/1706.03762) paper, and
models in
[semantic text similarity](https://ai.googleblog.com/2018/05/advances-in-semantic-textual-similarity.html).
`tf.keras.layers` is the recommended way to build models in tensorflow, but it
does not have a layer for attention that works with CNN/DNN networks. We would
like to contribute this capability.

Keras is an API spec that can be implemented across different languages and
backends, and `tf.keras` is a particular implementation of that spec. This
document contains code examples for `tensorflow`, but the same API should work
everywhere.

### Recurrent Neural Networks

Although not the primary focus of this proposal, the Attention layers in this
proposal work with some configurations of RNN networks.
Namely, when users create
[tf.keras.layers.LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
with `return_sequences=True`, the rest works the same way as CNN.

Unfortunately, this technique does not cover sequence-to-sequence RNN models.
In these models, the value is the states of encoder, and the query is the input
of the decoder. The decoder needs to slide its input based on the timesteps, and
feed them one by one. So, the output of the attention layer at timestep T
affects the output at T+1.

## Previous Work

[tf.contrib.seq2seq.LuongAttention](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/LuongAttention)
and
[tf.contrib.seq2seq.BahdanauAttention](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BahdanauAttention)
are implementations of dot-product (Luong) and additive (Bahdanau) Attention
respectively for RNN in Tensorflow. This proposal is based on this
implementation, but works with CNN/Dense networks.

There is an implementation of Attention as a `tf.layers.Layer` subclass under
https://github.com/tensorflow/models/tree/master/official/transformer,
specifically in
https://github.com/tensorflow/models/blob/master/official/transformer/model/attention_layer.py.
That file implements dot-product attention proposed in this file, and also
supports multi-head. Our proposal is to expose such a method in
`tf.keras.layers`. In addition, our proposal creates variables inside the
`build()` method, rather than the `call()` method.

There is ongoing work to add Attention in Keras, namely
https://github.com/keras-team/keras/pull/11421. That proposal addresses
Attention mechanism for RNN networks only. I cannot see a way to make it work
for CNN/Dense networks, which are the motivation for our proposal.

https://github.com/keras-team/keras/issues/9263 contains an example of a Keras
Layer that implements a CNN+Attention network. That example merges CNN and
Attention into the same class, whereas our proposal is modular. In the
Examples section, we present an example of how to build a
CNN+Attention model.

https://github.com/keras-team/keras/issues/7341 is a request to add an Attention
Layer. Our proposal will resolve that request.

https://github.com/keras-team/keras/issues/7803 is a request for a Multi-Head
Attention Layer. Multi-Head Attention is not covered in this proposal, but can
be implemented as a follow-up, as discussed in the
[Multi-Head Attention](#multi-head-attention) section.

There are a few more issues that request Attention for RNN. They are covered
either by https://github.com/keras-team/keras/pull/11421 or our proposal:

*   https://github.com/keras-team/keras/issues/5738
*   https://github.com/keras-team/keras/issues/4962
*   https://github.com/keras-team/keras/issues/2525

## Design Proposal

We propose to implement the following common attention layers:

*   `Attention`: Basic dot-product attention, a.k.a. Luong-style attention.
    Follows
    [tf.contrib.seq2seq.LuongAttention](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/LuongAttention).
    This attention has two forms.
    *   The first is standard dot-product attention, as described in: Minh-Thang
        Luong, Hieu Pham, Christopher D. Manning. "Effective Approaches to
        Attention-based Neural Machine Translation." EMNLP 2015.
        https://arxiv.org/abs/1508.04025.
    *   The second is the scaled form inspired partly by the normalized form of
        additive (Bahdanau-style) attention. To enable the second form,
        construct the object with parameter `scale=True`.
*   `AdditiveAttention`: Additive attention, a.k.a. Bahdanau-style attention.
    Follows
    [tf.contrib.seq2seq.BahdanauAttention](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BahdanauAttention).
    This attention has two forms.
    *   The first is additive attention, as described in: Dzmitry Bahdanau,
        Kyunghyun Cho, Yoshua Bengio. "Neural Machine Translation by Jointly
        Learning to Align and Translate." ICLR 2015.
        https://arxiv.org/abs/1409.0473.
    *   The second is the normalized form. This form is inspired by the weight
        normalization article: Tim Salimans, Diederik P. Kingma. "Weight
        Normalization: A Simple Reparameterization to Accelerate Training of
        Deep Neural Networks." https://arxiv.org/abs/1602.07868. To enable the
        second form, construct the object with parameter `normalize=True`.

## Detailed Design

According to the general definition of attention (see
https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1184/lectures/lecture11.pdf),
"Given a set of vector values, and a vector query, attention is a technique to
compute a weighted sum of the values, dependent on the query."

There are four input tensors:

*   `query` of shape `[batch_size, Tq, dim]`
*   `value` of shape `[batch_size, Tv, dim]`
*   `query_mask` (optional) of shape `[batch_size, Tq]`. Boolean tensor,
    typically calculated from the query length tensor. Used to mask the output
    tensor. This is similar to the `mask` argument of
    [tf.keras.backend.rnn](https://www.tensorflow.org/api_docs/python/tf/keras/backend/rnn).
*   `value_mask` (optional) of shape `[batch_size, Tv]`. Boolean tensor,
    typically calculated from the value length tensor. It is used to mask
    `value` elements beyond this length so they do not contribute to the result.

The output is of shape `[batch_size, Tq, dim]`.

Following the pattern of other Keras layers, we pass the list `[query, value]`
as `inputs` and we pass the list `[query_mask, value_mask]` as the `mask`
argument. Namely, the interface for `Attention` will be as follows:

```python
class Attention(tf.keras.layers.Layer):
  """Basic dot-product attention layer, a.k.a. Luong-style attention.

  The calculation follows the steps:
  1. Calculate scores with shape `[batch_size, Tq, Tv]` as a query-value
     dot product: `scores = tf.matmul(query, value, transpose_b=True)`.
  2. Use scores to calculate a distribution with shape
     `[batch_size, Tq, Tv]`: `distribution = tf.nn.softmax(scores)`.
  3. Use `distribution` to create a linear combination of `value` with
     shape `batch_size, Tq, dim]`:
     `return tf.matmul(distribution, value)`.

  Args:
    scale: If `True`, will create a scalar variable to scale the attention
      scores.
  """

  def __init__(
      self,
      scale=False,
      **kwargs):

  def build(self, input_shape):
    """Creates scale variable if scale==True."""

  def call(self, inputs, mask=None):
    """Applies basic dot-product attention.

    Args:
      inputs: List of the following tensors:
        * query: Query `Tensor` of shape `[batch_size, Tq, dim]`.
        * value: Value `Tensor` of shape `[batch_size, Tv, dim]`.
      mask: List of the following tensors:
        * query_mask: A boolean mask `Tensor` of shape `[batch_size, Tq]`.
          If given, the output will be zero at the positions where
          `mask==False`.
        * value_mask: A boolean mask `Tensor` of shape `[batch_size, Tv]`.
          If given, will apply the mask such that values at positions where
          `mask==False` do not contribute to the result.
    Returns:
      Attention outputs of shape `[batch_size, Tq, dim]`.
    """
```

Similarly, the interface for `AdditiveAttention` will be:

```python
class AdditiveAttention(tf.keras.layers.Layer):
  """Additive attention layer, a.k.a. Bahdanau-style attention.

  The calculation follows the steps:
  1. Reshape `query` and `value` into shapes `[batch_size, Tq, 1, dim]`
     and `[batch_size, 1, Tv, dim]` respectively.
  2. Calculate scores with shape `[batch_size, Tq, Tv]` as a non-linear
     sum: `scores = tf.reduce_sum(tf.tanh(query + value), axis=-1)`
  3. Use scores to calculate a distribution with shape
     `[batch_size, Tq, Tv]`: `distribution = tf.nn.softmax(scores)`.
  4. Use `distribution` to create a linear combination of `value` with
     shape `batch_size, Tq, dim]`:
     `return tf.matmul(distribution, value)`.

  Args:
    normalize: If True, will create scale and bias variables to normalize
    scores.
  """

  def __init__(
      self,
      normalize=False,
      **kwargs):

  def build(self, input_shape):
    """Creates variables."""

  def call(self, inputs, mask=None):
    """Applies additive attention.

    Args:
      inputs: List of the following tensors:
        * query: Query `Tensor` of shape `[batch_size, Tq, dim]`.
        * value: Value `Tensor` of shape `[batch_size, Tv, dim]`.
      mask: List of the following tensors:
        * query_mask: A boolean mask `Tensor` of shape `[batch_size, Tq]`.
          If given, the output will be zero at the positions where
          `mask==False`.
        * value_mask: A boolean mask `Tensor` of shape `[batch_size, Tv]`.
          If given, will apply the mask such that values at positions where
          `mask==False` do not contribute to the result.
    Returns:
      Attention outputs of shape `[batch_size, Tq, dim]`.
    """
```

The implementations for both Attention layers can be in the same file. They can
reuse a private method with the following signature:

```python
def _apply_attention_scores(scores, value, value_mask=None):
  """Applies attention scores to the given value tensor.

  Args:
    scores: Scores tensor of shape `[batch_size, Tq, Tv]`.
    value: Value tensor of shape `[batch_size, Tv, dim]`.
    value_mask: A boolean mask `Tensor` of shape `[batch_size, Tv]`.
      If given, will apply the mask such that values at positions where
      `mask==False` do not contribute to the result.

  Returns:
    Tensor of shape `[batch_size, Tq, dim]`.
  """
```

Implementations of other Attention mechanisms can reuse this method, as well.
So, that method can be made public. An alternative using inheritance is
discussed in the "Base Attention Class" section. The
rest of the code is specific to each Attention mechanism.

Although not the primary focus of this proposal, the Attention layers work with
RNN networks, such as
[tf.keras.layers.LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM):
When creating the LSTM, users need to set `return_sequences=True`, and the rest
works the same way as CNN. It is unclear whether this method suffices to create
the most common RNN+Attention models.

We will first work on the implementation for Tensorflow.

### Self-Attention

The Self-Attention variant can be implemented by passing the same tensor to both
`query` and `value`.

### Multi-Head Attention

This is an Attention variant proposed in
["Attention is all you need"](https://arxiv.org/abs/1706.03762). This variant is
not covered in our proposal. But could be implemented as an additional feature,
e.g. by adding a `num_heads` argument that defaults to 1. The implementation
will split the `query` and `value` tensors into `num_heads` tensors, calculate
attention for each pair, then stack the results. This transformation can be
implemented as a private method that is reused by all attention layers.

### Transformer

Transformer is a DNN+Attention network proposed in
["Attention is all you need"](https://arxiv.org/abs/1706.03762). There is an
implementation of it under
https://github.com/tensorflow/models/tree/master/official/transformer, which
uses a custom Attention implementation. Our proposal will simplify the
Transformer network constructions, because users can reuse the Attention layers,
rather than writing custom ones.

### Position Representations

DNN+Attention networks do not model relative or absolute position information in
their structure. Instead, position information is modeled as an additional term
in the model output. The proposed techniques can be implemented as an additional
feature in the Attention API.

*   https://arxiv.org/abs/1803.02155 describes how relative position
    representation can be added to dot-product attention.
*   https://arxiv.org/abs/1503.08895 and https://arxiv.org/abs/1706.03762 show
    how absolute position information can be added as a deterministic function
    of position.

### 2D and 3D

Attention is typically used in 1D sequences, such as text. It is conceivable
that people may try to use it with 2D or 3D sequences, such as with the outputs
of `Conv2D` or `Conv3D` layers. To make this work, users can follow the example
code:

```python
query_orig_shape = tf.shape(query)
query = tf.reshape(query, [batch_size, -1, dim])
value = tf.reshape(value, [batch_size, -1, dim])
attention = tf.keras.layers.Attention()([query, value])
attention = tf.reshape(attention, query_orig_shape)
```

Alternatively, we could add the above reshapes inside the `Attention`
implementation, so that 2D and 3D sequences can be supported out of the box. But
given that this is a rare use case, we will not support it in the first version.

## Examples

Here is an example of a `tf.estimator` `model_fn`. It creates a CNN+Attention
model for query and value sequence features:

```python
def model_fn_with_attention(features, labels, mode):
  """Model function that uses Attention."""
  # Prepare the sequence embeddings for the query and value features.
  query_column = tf.contrib.feature_column.\
    sequence_categorical_column_with_vocabulary_file('query', vocabulary_file)
  value_column = tf.contrib.feature_column.\
    sequence_categorical_column_with_vocabulary_file('value', vocabulary_file)
  query_embedding_column, value_embedding_column = (
      tf.feature_column.shared_embedding_columns(
          [query_column, value_column], dimension=50))
  # Query embeddings with shape [batch_size, Tq, embedding_dim], where Tq is the
  # maximum sequence length for this batch.
  # Query length with shape [batch_size] and values in the range [0, Tq).
  query_embeddings, query_length = (
      tf.contrib.feature_column.sequence_input_layer(
          features, [query_embedding_column]))
  # Value embeddings with shape [batch_size, Tv, embedding_dim] and value length
  # with shape [batch_size].
  value_embeddings, value_length = (
      tf.contrib.feature_column.sequence_input_layer(
          features, [value_embedding_column]))

  # CNN layer.
  cnn_layer = tf.keras.layers.Conv1D(
      filters=100,
      kernel_size=4,
      # Use 'same' padding so outputs have the same shape as inputs.
      padding='same')
  # Query encoding of shape [batch_size, Tq, filters].
  query_seq_encoding = cnn_layer(query_embeddings)
  # Value encoding of shape [batch_size, Tv, filters].
  value_seq_encoding = cnn_layer(value_embeddings)

  # Query-value attention of shape [batch_size, Tq, filters].
  query_value_attention_seq = tf.keras.layers.Attention()(
      [query_seq_encoding, value_seq_encoding],
      mask=[_sequence_mask(query_seq_encoding),
            _sequence_mask(value_seq_encoding)])

  # Reduce over the sequence axis to produce encodings of shape
  # [batch_size, filters].
  query_encoding = tf.keras.layers.GlobalAveragePooling1D()(
      query_seq_encoding)
  query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
      query_value_attention_seq)

  # Concatenate query and document encodings to produce a DNN input layer.
  input_layer = tf.keras.layers.Concatenate()(
      [query_encoding, query_value_attention])

  # Add DNN layers, and use a head to return EstimatorSpec.
  # Follow the code in tf.estimator.DNNClassifier.
  # …

def _sequence_mask(t)
  """Creates a boolean mask for tensor t."""
  return tf.sequence_mask(t, maxlen=tf.shape(t)[-2])
```

There is ongoing work to implement `sequence_input_layer` as a Keras layer.
After this work is completed, all the model above can be written as a succession
of Keras layers. In particular, the input layer will be created as:

```python
query_input_layer = tf.feature_column.SequenceFeatures([query_embedding_column])
query_embeddings, query_length = query_input_layer(features)
value_input_layer = tf.feature_column.SequenceFeatures([value_embedding_column])
value_embeddings, value_length = value_input_layer(features)
```

Here is the same example using Keras. For simplicity, we skip `query_mask` and
`value_mask`, which can be created based on the sequence length.

```python
# Variable-length int sequences.
query_input = keras.Input(shape=(None,), dtype='int32')
value_input = keras.Input(shape=(None,), dtype='int32')

# Embedding lookup.
token_embedding = keras.layers.Embedding(max_tokens, dimension)
# Query embeddings of shape [batch_size, Tq, dimension].
query_embeddings = token_embedding(query_input)
# Value embeddings of shape [batch_size, Tv, dimension].
value_embeddings = token_embedding(query_input)

# CNN layer.
cnn_layer = keras.layers.Conv1D(
    filters=100,
    kernel_size=4,
    # Use 'same' padding so outputs have the same shape as inputs.
    padding='same')
# Query encoding of shape [batch_size, Tq, filters].
query_seq_encoding = cnn_layer(query_embeddings)
# Value encoding of shape [batch_size, Tv, filters].
value_seq_encoding = cnn_layer(value_embeddings)

# Query-value attention of shape [batch_size, Tq, filters].
query_value_attention_seq = keras.layers.Attention()(
    [query_seq_encoding, value_seq_encoding])

# Reduce over the sequence axis to produce encodings of shape
# [batch_size, filters].
query_encoding = keras.layers.GlobalAveragePooling1D()(
    query_seq_encoding)
query_value_attention = keras.layers.GlobalAveragePooling1D()(
    query_value_attention_seq)

# Concatenate query and document encodings to produce a DNN input layer.
input_layer = keras.layers.Concatenate()(
    [query_encoding, query_value_attention])

# Add DNN layers, and create Model.
# ...
```

## Alternatives Considered

### Base Attention Class

We could have a base attention class that implements the
`apply_attention_scores()` method so that subclasses could reuse that method.
The base class could be as follows:

```python
class BaseAttention(tf.keras.layers.Layer):
  """Base Attention class.

  Implementations of attention mechanisms should inherit from this class, and
  reuse the `apply_attention_scores()` method.
  """

  def __init__(self, **kwargs):
    super(BaseAttention, self).__init__(**kwargs)

  def apply_attention_scores(self, scores, value, value_mask=None):
    """Applies attention scores to the given value tensor.

    Args:
      scores: Scores tensor of shape `[batch_size, Tq, Tv]`.
      value: Value tensor of shape `[batch_size, Tv, dim]`.
      value_mask: A boolean mask `Tensor` of shape `[batch_size, Tv]`.
        If given, will apply the mask such that values at positions where
        `mask==False` do not contribute to the result.

    Returns:
      Tensor of shape `[batch_size, Tq, dim]`.
    """
```

Pros:

*   Inheritance is used extensively in Keras. This alternative follows that
    pattern.
*   When external users inherit from `BaseAttention`, they can freely reuse the
    `apply_attention_scores()` method.
*   When new common methods are added, such as `split_heads` and `combine_heads`
    for multi-headed attention, they can be added to this class.

Cons:

*   Inheritance hierarchies in python hinder troubleshooting. Because there is
    no compile-time linking, users need to perform regular-expression searches
    across multiple files to discover which method is called.

## Questions and Discussion Topics

*   The examples in this doc are in Tensorflow. Will the API work in other
    languages and backends?
*   What other implementations do we need for other languages/backends?
*   What is the best interface for RNN? This proposal works for some basic
    cases, but https://github.com/keras-team/keras/pull/11421 proposes a more
    specialized interface. Perhaps we need both?
*   What other arguments should we expose? E.g. Attention distribution
    (probabilities) is calculated from attention scores using `softmax`. Maybe
    we can expose a `distribution_fn`, of `probability_fn` argument that
    defaults to `softmax`.
*   We use terminology from
    https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1184/lectures/lecture11.pdf.
    Namely the inputs are `query` and `value`. Is this the accepted terminology?
*   Are there any other common variants of Attention we should implement?
