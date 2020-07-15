# Keras categorical inputs

| Status        | Implemented (https://github.com/tensorflow/community/pull/209) |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Zhenyu Tan (tanzheny@google.com), Francois Chollet (fchollet@google.com)|
| **Sponsor**   | Karmel Allison (karmel@google.com), Martin Wicke (wicke@google.com) |
| **Updated**   | 2019-02-22                                           |

## Objective

This document proposes 5 new Keras preprocessing layers (KPL) (`StringLookup`, `CategoryCrossing`, `CategoryEncoding`, `Hashing`, `IntegerLookup`) and allow users to:
* Perform basic feature engineering for categorical inputs
* Replace feature columns and `tf.keras.layers.DenseFeatures` with proposed layers
* Introduce sparse inputs that work with Keras linear models and other layers that support sparsity

Other proposed layers for replacement of feature columns such as `tf.feature_column.bucketized_column` and `tf.feature_column.numeric_column` has been discussed [here](https://github.com/keras-team/governance/blob/master/rfcs/20190502-preprocessing-layers.md).

The proposed layers should support ragged tensors.

## Motivation

Specifically, by introducing the 5 layers, we aim to address these pain points:
* Users have to define both feature columns and Keras Inputs for the model, resulting in code duplication and deviation from DRY (Do not repeat yourself) principle. See this [Github issue](https://github.com/tensorflow/tensorflow/issues/27416).
* Users with large dimension categorical inputs will incur large memory footprint and computation cost, if wrapped with indicator column through `tf.keras.layers.DenseFeatures`.
* Currently there is no way to correctly feed Keras linear model or dense layer with multivalent categorical inputs or weighted categorical inputs, or shared embedding inputs.
* Feature columns offer black-box implementations, mix feature engineering with trainable objects, and lead to
  unintended coding pattern.

## User Benefit

We expect to get rid of the user painpoints once migrating off feature columns.

## Example Workflows

Two example workflows are presented below. These workflows can be found at this [colab](https://colab.sandbox.google.com/drive/1cEJhSYLcc2MKH7itwcDvue4PfvrLN-OR).

### Workflow 1 -- Official guide on how to replace feature columns with KPL

Refer to [tf.feature_column](https://www.tensorflow.org/api_docs/python/tf/feature_column) for a complete list of feature columns.

1. Replacing `tf.feature_column.categorical_column_with_hash_bucket` with `Hashing`
from
```python
tf.feature_column.categorical_column_with_hash_bucket(key, hash_bucket_size)
```
to
```python
keras_input = tf.keras.Input(shape=(1,), name=key, dtype=dtype)
hashed_input = tf.keras.experimental.preprocessing.Hashing(num_bins=hash_bucket_size)(keras_input)
```

Note the hashed output from KPL will be different than the hashed output from feature column, given how seed is choosen. `Hashing` also supports customized `salt`.

2. `tf.feature_column.categorical_column_with_identity`
This feature column is merely for having identical inputs and outputs except mapping out-of-range value into `default_value`, thus can easily be done at data cleaning stage,
not be part of feature engineering, and hence dropped in this proposal.

3. Replacing `tf.feature_column.categorical_column_with_vocabulary_file` and `tf.feature_column.categorical_column_with_vocabulary_list` with `StringLookup` or `IntegerLookup`
for string inputs,
from
```python
tf.feature_column.categorical_column_with_vocabulary_file(key, vocabulary_file, vocabulary_size, tf.dtypes.string, default_value, num_oov_buckets)
```
to
```python
keras_input = tf.keras.Input(shape=(1,), name=key, dtype=tf.dtypes.string)
id_input = tf.keras.experimental.preprocessing.StringLookup(max_tokens=vocabulary_size + num_oov_buckets,
  num_oov_indices=num_oov_buckets, mask_token=None, vocabulary=vocabulary_file)(keras_input)
```

Similarly, from
```python
tf.feature_column.categorical_column_with_vocabulary_list(key, vocabulary_list, tf.dtypes.string, default_value, num_oov_buckets)
```
to
```python
keras_input = tf.keras.Input(shape=(1,), name=key, dtype=tf.dtypes.string)
id_input = tf.keras.experimental.preprocessing.StringLookup(max_tokens=len(vocabulary_list) + num_oov_buckets, num_oov_indices=num_oov_buckets,
  mask_token=None, vocabulary=vocabulary_list)(keras_input)
```


Note that `default_value` is mutually exclusive with `num_oov_buckets`, in the case of `num_oov_buckets=0` and `default_value=-1`, simply set `num_oov_indices=0`. We do not support
any values other than `default_value=-1`.

Note the out-of-range values for `StringLookup` is prepended, i.e., [0,..., num_oov_tokens) for out-of-range values, whereas for `categorical_colulmn_with_vocabulary_file` is
appended, i.e., [vocabulary_size, vocabulary_size + num_oov_tokens) for out-of-range values. The former can give you more flexibility when reloading and adding vocab.

for integer inputs,
from
```python
tf.feature_column.categorical_column_with_vocabulary_file(key, vocabulary_file, vocabulary_size, tf.dtypes.int64, default_value, num_oov_buckets)
```
to
```python
keras_input = tf.keras.Input(shape=(1,), name=key, dtype=tf.dtypes.int64)
id_input = tf.keras.experimental.preprocessing.IntegerLookup(max_values=vocabulary_size + num_oov_buckets, num_oov_indices=num_oov_buckets, mask_value=None, vocabulary=vocabulary_file)(keras_input)
```

Similarly, from
```python
tf.feature_column.categorical_column_with_vocabulary_list(key, vocabulary_list, tf.dtypes.int64, default_value, num_oov_buckets)
```
to
```python
keras_input = tf.keras.Input(shape=(1,), name=key, dtype=tf.dtypes.int64)
id_input = tf.keras.experimental.preprocessing.IntegerLookup(max_values=len(vocabulary_list) + num_oov_buckets, num_oov_indices=num_oov_buckets, mask_value=None, vocabulary=vocabulary_list)(keras_input)
```


4. Replacing `tf.feature_column.crossed_column` with `CategoryCrossing` or `Hashing`
from
```python
tf.feature_column.crossed_column(keys, hash_bucket_size, hash_key)
```
to
```python
keras_inputs = []
for key in keys:
  keras_inputs.append(tf.keras.Input(shape=(1,), name=key, dtype=tf.dtypes.string))
hashed_input = tf.keras.layers.experimental.preprocessing.Hashing(num_bins=hash_bucket_size)(keras_inputs)
```

Note when `hash_bucket_size=0`, no hashing is performed, in this case it should be replaced with:
```python
keras_inputs = []
for key in keys:
  keras_inputs.append(tf.keras.Input(shape=(1,), name=key, dtype=tf.dtypes.string))
crossed_input = tf.keras.layers.experimental.preprocessing.CategoryCrossing()(keras_inputs)
```

5. Replacing `tf.feature_column.embedding_column` with `tf.keras.layers.Embedding`
Note that `combiner=sum` can be replaced with `tf.reduce_sum` and `combiner=mean` with `tf.reduce_mean` after
the embedding output. `sqrtn` can also be implemented using tf operations. For example:
```python
categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(key, vocabulary_list)
tf.feature_column.embedding_column(categorical_column, dimension=dimension, combiner="sum", initializer=initializer,
  max_norm=max_norm)
```
can be replaced with:
```python
categorical_input = tf.keras.Input(name=key, dtype=tf.string)
id_input = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=vocabulary_list)(categorical_input)
embedding_input = tf.keras.layers.Embedding(input_dim=len(vocabulary_list), output_dim=dimension,
  embeddings_initializer=initializer, embeddings_constraint=tf.keras.constraints.MaxNorm(max_norm))(id_input)
embedding_input = tf.reduce_sum(embedding_input, axis=-2)
```

6. Replacing `tf.feature_column.indicator_column` with `CategoryEncoding`
from
```python
categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(key, vocabulary_list)
tf.feature_column.indicator_column(categorical_column)
```
to
```python
categorical_input = tf.keras.Input(name=key, dtype=tf.string)
id_input = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=vocabulary_list)(categorical_input)
encoded_input = tf.keras.layers.experimental.preprocessing.CateogoryEncoding(
  max_tokens=categorical_column.num_buckets, output_mode="count", sparse=True)(id_input)
```

Note that `CategoryEncoding` supports one-hot through `output_mode="binary"` as well. This is a much more
efficient approach than `tf.one_hot` + `tf.reduce_sum(axis=-2)` to reduce the multivalent categorical inputs.

Note that by specifing `sparse` flag, the output can be either a `tf.Tensor` or `tf.SparseTensor`.

7. Replacing `tf.feature_column.weighted_categorical_column` with `CategoryEncoding`
from
```python
categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(key, vocabulary_list)
tf.feature_column.weighted_categorical_column(categorical_column, weight_feature_key)
```
to
```python
categorical_input = tf.keras.Input(name=key, dtype=tf.string)
lookup_output = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=vocabulary_list)(categorical_input)
weight_input = tf.keras.Input(shape=(1,), dtype=tf.float32, name=weight_feature_key)
weighted_output = tf.keras.layers.experimental.preprocessing.CategoryEncoding(
  max_tokens=categorical_column.num_buckets)(lookup_output, weight_input)
```

8. Replacing `tf.feature_column.shared_embeddings` with a single `tf.keras.layers.Embedding`
Similar to 5, but with multiple categorical inputs:
from
```python
watched_video_id = tf.feature_column.categorical_column_with_vocabulary_list('watched_video_id', video_vocab_list)
impression_video_id = tf.feature_column.categorical_column_with_vocabulary_list('impression_video_id', video_vocab_list)
tf.feature_column.shared_embeddings([watched_video_id, impression_video_id], dimension)
```
to
```python
watched_video_input = tf.keras.Input(shape=(1,), name='watched_video_id', dtype=tf.int64)
impression_video_input = tf.keras.Input(shape=(1,), name='impression_video_id', dtype=tf.int64)
embed_layer = tf.keras.layers.Embedding(input_dim=len(video_vocab_list), output_dim=dimension)
embedded_watched_video_input = embed_layer(watched_video_input)
embedded_impression_video_input = embed_layer(impression_video_input)
```

9. Replacing `tf.estimator.LinearXXX` with `CategoryEncoding` and `tf.keras.experimental.LinearModel`
LinearClassifier or LinearRegressor treats categorical columns by multi-hot, this can be replaced by encoding layer and Keras linear model, see Workflow 2 for details.

10. Replacing `tf.feature_column.numeric_column` and `tf.feature_column.sequence_numeric_column` with `tf.keras.Input` and `Normalization`
`tf.keras.layers.experimental.preprocessing.Normalization` with `set_weights` on mean and standard deviation.

11. Replacing `tf.feature_column.sequence_categorical_xxx`
Replacing `tf.feature_column.sequence_categorical_xxx` is similar to `tf.feature_column.categorical_xxx` except `tf.keras.Input` should take time dimension into
`input_shape` as well.

12. Replacing `tf.feature_column.bucketized_column` with `Discretization`
from
```python
source_column = tf.feature_column.numeric_column(key)
tf.feature_column.bucketized_column(source_column, boundaries)
```
to
```python
keras_input = tf.keras.Input(shape=(1,), name=key, dtype=tf.float32)
bucketized_input = tf.keras.experimental.preprocessing.Discretization(bins=boundaries)(keras_input)
```


### Workflow 2 -- Complete Example

This example gives an equivalent code snippet to canned `LinearEstimator` [tutorial](https://www.tensorflow.org/tutorials/estimator/linear) on the Titanic dataset:

Refer to this [colab](https://colab.sandbox.google.com/drive/1cEJhSYLcc2MKH7itwcDvue4PfvrLN-OR) to reproduce.

```python
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
y_train = dftrain.pop('survived')

STRING_CATEGORICAL_COLUMNS = ['sex', 'class', 'deck', 'embark_town', 'alone']
INT_CATEGORICAL_COLUMNS = ['n_siblings_spouses', 'parch']
NUMERIC_COLUMNS = ['age', 'fare']

keras_inputs = {}
keras_preproc_inputs = []
for key in STRING_CATEGORICAL_COLUMNS:
  keras_input = tf.keras.Input(shape=(1,), dtype=tf.string, name=key)
  keras_inputs[key] = keras_input
  vocab = dftrain[key].unique()
  keras_preproc_input = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=vocab, num_oov_indices=0, mask_token=None, name='lookup' + key)(keras_input)
  keras_preproc_input = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=len(vocab), output_mode='count', sparse=True, name='encode' + key)(keras_preproc_input)
  keras_preproc_inputs.append(keras_preproc_input)

for key in INT_CATEGORICAL_COLUMNS:
  keras_input = tf.keras.Input(shape=(1,), dtype=tf.int64, name=key)
  keras_inputs[key] = keras_input
  vocab = dftrain[key].unique()
  keras_preproc_input = tf.keras.layers.experimental.preprocessing.IntegerLookup(vocabulary=vocab, num_oov_indices=0, mask_value=None, name='lookup' + key)(keras_input)
  keras_preproc_input = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=len(vocab), output_mode='count', sparse=True, name='encode' + key)(keras_preproc_input)
  keras_preproc_inputs.append(keras_preproc_input)

for key in NUMERIC_COLUMNS:
  keras_input = tf.keras.Input(shape=(1,), dtype=tf.float32, name=key)
  keras_inputs[key] = keras_input
  keras_preproc_inputs.append(keras_preproc_input)

age_x_sex = tf.keras.layers.experimental.preprocessing.CategoryCrossing(name='age_x_sex_crossing')([keras_inputs['age'], keras_inputs['sex']])
age_x_sex = tf.keras.layers.experimental.preprocessing.Hashing(num_bins=100, name='age_x_sex_hashing')(age_x_sex)
keras_output_age_x_sex = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=100, output_mode='count', sparse=True, name='age_x_sex_encoding')(age_x_sex)
keras_preproc_inputs.append(keras_output_age_x_sex)


linear_model = tf.keras.experimental.LinearModel(units=1, kernel_initializer='zeros', activation='sigmoid')
linear_logits = linear_model(keras_preproc_inputs)
sorted_keras_inputs = tuple(keras_inputs[key] for key in sorted(keras_inputs.keys()))
model = tf.keras.Model(sorted_keras_inputs, linear_logits)

model.compile('ftrl', 'binary_crossentropy', metrics=['accuracy'])

df_dataset = tf.data.Dataset.from_tensor_slices((dict(dftrain), y_train))
def encode_map(features, labels):
  encoded_features = tuple(tf.expand_dims(features[key], axis=1) for key in sorted(features.keys()))
  return (encoded_features, labels)
encoded_dataset = df_dataset.batch(32).map(encode_map)

model.fit(encoded_dataset)
```

## Design Proposal

```python
`tf.keras.layers.StringLookup`
StringLookup(PreprocessingLayer):
"""This layer transforms categorical inputs to index space.
   If input is dense/sparse, then output is dense/sparse."""

  def __init__(self, max_tokens=None, num_oov_indices=1, mask_token="",
               oov_token="[UNK]", vocabulary=None, encoding=None,
               invert=False, name=None, **kwargs):
    """Constructs a IndexLookup layer.

    Args:
      max_tokens: The maximum size of the vocabulary for this layer. If None,
              there is no cap on the size of the vocabulary. Note that this vocabulary
              includes the OOV and mask tokens, so the effective number of tokens is
              (max_tokens - num_oov_indices - (1 if mask_token else 0))
      num_oov_indices: The number of out-of-vocabulary tokens to use; defaults to
              1. If this value is more than 1, OOV inputs are hashed to determine their
              OOV value; if this value is 0, passing an OOV input will result in a '-1'
              being returned for that value in the output tensor. (Note that, because
              the value is -1 and not 0, this will allow you to effectively drop OOV
              values from categorical encodings.)
      mask_token: A token that represents masked values, and which is mapped to
              index 0. Defaults to the empty string "". If set to None, no mask term
              will be added and the OOV tokens, if any, will be indexed from
              (0...num_oov_indices) instead of (1...num_oov_indices+1).
      oov_token: The token representing an out-of-vocabulary value. Defaults to
              "[UNK]".
      vocabulary: An optional list of vocabulary terms, or a path to a text file
              containing a vocabulary to load into this layer. The file should contain
              one token per line. If the list or file contains the same token multiple
              times, an error will be thrown.
      encoding: The Python string encoding to use. Defaults to `'utf-8'`.
      invert: If true, this layer will map indices to vocabulary items instead
              of mapping vocabulary items to indices.
      name: Name of the layer.
      **kwargs: Keyword arguments to construct a layer.

    Input shape:
            a string or int tensor of shape `[batch_size, d1, ..., dm]`
    Output shape:
            an int tensor of shape `[batch_size, d1, ..., dm]`

    Example:
      >>> vocab = ["a", "b", "c", "d"]
      >>> data = tf.constant([["a", "c", "d"], ["d", "z", "b"]])
      >>> layer = StringLookup(vocabulary=vocab)
      >>> layer(data)
      <tf.Tensor: shape=(2, 3), dtype=int64, numpy=
      array([[2, 4, 5],
             [5, 1, 3]])>
    """
    pass


`tf.keras.layers.IntegerLookup`
IntegerLookup(PreprocessingLayer):
"""This layer transforms categorical inputs to index space.
   If input is dense/sparse, then output is dense/sparse."""

  def __init__(self, max_values=None, num_oov_indices=1, mask_value=0,
               oov_value=-1, vocabulary=None, invert=False, name=None, **kwargs):
    """Constructs a IndexLookup layer.

    Args:
      max_values: The maximum size of the vocabulary for this layer. If None,
              there is no cap on the size of the vocabulary. Note that this vocabulary
              includes the OOV and mask values, so the effective number of values is
              (max_values - num_oov_values - (1 if mask_token else 0))
      num_oov_indices: The number of out-of-vocabulary values to use; defaults to
              1. If this value is more than 1, OOV inputs are modulated to determine
              their OOV value; if this value is 0, passing an OOV input will result in
              a '-1' being returned for that value in the output tensor. (Note that,
              because the value is -1 and not 0, this will allow you to effectively drop
              OOV values from categorical encodings.)
      mask_value: A value that represents masked inputs, and which is mapped to
              index 0. Defaults to 0. If set to None, no mask term will be added and the
              OOV values, if any, will be indexed from (0...num_oov_values) instead of
              (1...num_oov_values+1).
      oov_value: The value representing an out-of-vocabulary value. Defaults to -1.
      vocabulary: An optional list of values, or a path to a text file containing
              a vocabulary to load into this layer. The file should contain one value
              per line. If the list or file contains the same token multiple times, an
              error will be thrown.
      invert: If true, this layer will map indices to vocabulary items instead
              of mapping vocabulary items to indices.
      name: Name of the layer.
      **kwargs: Keyword arguments to construct a layer.

    Input shape:
            a string or int tensor of shape `[batch_size, d1, ..., dm]`
    Output shape:
            an int tensor of shape `[batch_size, d1, ..., dm]`

    Example:
      >>> vocab = [12, 36, 1138, 42]
      >>> data = tf.constant([[12, 1138, 42], [42, 1000, 36]])
      >>> layer = IntegerLookup(vocabulary=vocab)
      >>> layer(data)
      <tf.Tensor: shape=(2, 3), dtype=int64, numpy=
      array([[2, 4, 5],
             [5, 1, 3]])>
    """
    pass


`tf.keras.layers.CategoryCrossing`
CategoryCrossing(PreprocessingLayer):
"""This layer transforms multiple categorical inputs to categorical outputs
   by Cartesian product, and hash the output if necessary.
   If any of the inputs is sparse, then all outputs will be sparse. Otherwise, all outputs will be dense."""

  def __init__(self, depth=None, separator=None, name=None, **kwargs):
    """Constructs a CategoryCrossing layer.
    Args:
      depth: depth of input crossing. By default None, all inputs are crossed into
            one output. It can also be an int or tuple/list of ints. Passing an
            integer will create combinations of crossed outputs with depth up to that
            integer, i.e., [1, 2, ..., `depth`), and passing a tuple of integers will
            create crossed outputs with depth for the specified values in the tuple,
            i.e., `depth`=(N1, N2) will create all possible crossed outputs with depth
            equal to N1 or N2. Passing `None` means a single crossed output with all
            inputs. For example, with inputs `a`, `b` and `c`, `depth=2` means the
            output will be [a;b;c;cross(a, b);cross(bc);cross(ca)].
      separator: A string added between each input being joined. Defaults to '_X_'.
      name: Name to give to the layer.
      **kwargs: Keyword arguments to construct a layer.

    Input shape: a list of string or int tensors or sparse tensors of shape
            `[batch_size, d1, ..., dm]`

    Output shape: a single string or int tensor or sparse tensor of shape
            `[batch_size, d1, ..., dm]`

    Example: (`depth`=None)
      If the layer receives three inputs:
      `a=[[1], [4]]`, `b=[[2], [5]]`, `c=[[3], [6]]`
      the output will be a string tensor:
      `[[b'1_X_2_X_3'], [b'4_X_5_X_6']]`
    """
    pass

`tf.keras.layers.CategoryEncoding`
CategoryEncoding(PreprocessingLayer):
"""This layer transforms categorical inputs from index space to category space.
   If input is dense/sparse, then output is dense/sparse."""

  def __init__(self, max_tokens=None, output_mode="binary", sparse=False, name=None, **kwargs):
    """Constructs a CategoryEncoding layer.
    Args:
      max_tokens: The maximum size of the vocabulary for this layer. If None,
              there is no cap on the size of the vocabulary.
      output_mode: Specification for the output of the layer.
              Defaults to "binary". Values can be "binary", "count" or "tf-idf",
              configuring the layer as follows:
              "binary": Outputs a single int array per batch, of either vocab_size or
                max_tokens size, containing 1s in all elements where the token mapped
                to that index exists at least once in the batch item.
              "count": As "binary", but the int array contains a count of the number
                of times the token at that index appeared in the batch item.
              "tf-idf": As "binary", but the TF-IDF algorithm is applied to find the
                value in each token slot.
      sparse: Boolean. If true, returns a `SparseTensor` instead of a dense
              `Tensor`. Defaults to `False`.
      name: Name to give to the layer.
     **kwargs: Keyword arguments to construct a layer.

    Input shape: A int tensor of shape `[batch_size, d1, ..., dm-1, dm]`
    Output shape: a float tensor of shape `[batch_size, d1, ..., dm-1, num_categories]`

    Example:
      >>> layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(
      ...           max_tokens=4, output_mode="count")
      >>> layer([[0, 1], [0, 0], [1, 2], [3, 1]])
      <tf.Tensor: shape=(4, 4), dtype=float32, numpy=
        array([[1., 1., 0., 0.],
               [2., 0., 0., 0.],
               [0., 1., 1., 0.],
               [0., 1., 0., 1.]], dtype=float32)>
    """
    pass

`tf.keras.layers.Hashing`
Hashing(PreprocessingLayer):
"""This layer transforms categorical inputs to hashed output.
   If input is dense/sparse, then output is dense/sparse."""
  def __init__(self, num_bins, salt=None, name=None, **kwargs):
    """Constructs a Hashing layer.

    Args:
      num_bins: Number of hash bins.
      salt: A single unsigned integer or None.
              If passed, the hash function used will be SipHash64, with these values
              used as an additional input (known as a "salt" in cryptography).
              These should be non-zero. Defaults to `None` (in that
              case, the FarmHash64 hash function is used). It also supports
              tuple/list of 2 unsigned integer numbers, see reference paper for details.
      name: Name to give to the layer.
      **kwargs: Keyword arguments to construct a layer.

    Input shape: A single or list of string, int32 or int64 `Tensor`,
            `SparseTensor` or `RaggedTensor` of shape `[batch_size, ...,]`

    Output shape: An int64 `Tensor`, `SparseTensor` or `RaggedTensor` of shape
            `[batch_size, ...]`. If any input is `RaggedTensor` then output is
            `RaggedTensor`, otherwise if any input is `SparseTensor` then output is
            `SparseTensor`, otherwise the output is `Tensor`.

    Example:
      >>> layer = tf.keras.layers.experimental.preprocessing.Hashing(num_bins=3)
      >>> inp = [['A'], ['B'], ['C'], ['D'], ['E']]
      >>> layer(inp)
      <tf.Tensor: shape=(5, 1), dtype=int64, numpy=
        array([[1],
               [0],
               [1],
               [1],
               [2]])>
    """
    pass

```

### Alternatives Considered
An alternative is to provide solutions on top of feature columns. This will make user code to be slightly cleaner but far less flexible.

### Performance Implications
End to End benchmark should be same or faster than feature columns implementations.

### Dependencies
This proposal does not add any new dependencies.

### Engineering Impact
These changes will include more layers and thus binary size and build time. It will not impact startup time.
This code can be tested in its own and maintained in its own buildable unit.

### Platforms and Environments
This proposal should work in all platforms and environments.

### Best Practices, Tutorials and Examples
This proposal does not change the best engineering practices.

### Compatibility
No backward compatibility issues.

### User Impact
User facing changes to migrate feature column based Keras modeling to preprocessing layer based Keras modeling, as the example workflow suggests.

## Questions and Meeting Notes
We'd like to gather feedbacks on `IndexLookup`, specifically we propose migrating off from mutually exclusive `num_oov_buckets` and `default_value` and replace with `num_oov_tokens`.
1. Naming for encoding v.s. vectorize: encoding can mean many things, vectorize seems to general. We will go with "CategoryEncoding"
2. "mode" should be "count" or "avg_count", instead of "sum" and "mean".
3. Rename "sparse_combiner" to "mode", which aligns with scikit-learn.
4. Have a 'sparse_out' flag for "CategoryEncoding" layer.
5. Hashing -- we refer to hashing when we mean fingerprinting. Keep using "Hashing" for layer name, but document how it relies on tf.fingerprint, and also provides option for salt.
5. Rename "CategoryLookup" to "IndexLookup"

## Updates on 07/14/20
Mark the RFC as completed, update the layer naming and arguments.
