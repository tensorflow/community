# Generalizing tf.data batching using windowing and reducers 

| Status        | Proposed                                             |
:---------------|:-----------------------------------------------------|
| **Author(s)** | Jiri Simsa (Google)                                  |
| **Sponsor**   | Derek Murray (Google)                                |
| **Updated**   | 2018-07-26                                           |

## Objective 

This proposal addresses the known limitations of the current tf.data batching API:

*   it provides a mechanism for padded batching of sparse tensors
*   it facilitates customization of batching logic (users can now express batching logic as a pure Python function)
*   it enables application of different batching logic on different components


## Motivation

The tf.data API is the de facto standard for creating TensorFlow input pipelines, whose purpose is to extract data from a storage system, transform it, and load it onto an accelerator.

A common transformation performed by TensorFlow input pipelines is batching -- combining multiple tensors into a single tensor of higher dimension, most often to make a minibatch for training. Currently, the core tf.data API for batching consists of [batch](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch) and [padded_batch](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#padded_batch). The former assumes the inputs have the same shape and supports both dense and sparse inputs. The latter supports dynamically shaped inputs, such as you might find in sequential data: it assumes the inputs have the same rank but not necessarily the same shape and can pad differently shaped inputs to a common shape; only dense inputs are supported by padded_batch.

The tf.data batching API has several limitations that has surfaced in various users requests:

*   As already mentioned, the padded_batch transformation does not support sparse tensors inputs ([issue](https://github.com/tensorflow/tensorflow/issues/18302)).
*   The current API is not flexible enough to accept user-provided batching logic (e.g. [issue](https://github.com/tensorflow/tensorflow/issues/20391)).
*   The same batching logic needs to be applied to all components of the input dataset, which is not always desirable (e.g. [issue](https://github.com/tensorflow/tensorflow/issues/20391)). Users can work around this limitation by creating separate datasets to which different batching transformations are applied and them zipping the datasets; however, this can be inefficient, unergonomic, and error prone.


## Proposal

This document proposes leveraging the recently introduced support for _nested_ datasets as inputs to tf.data transformations to perform generalized batching as follows:

1.  A <span style="text-decoration:underline;">window</span> transformation is used to combine consecutive elements of the input into a nested dataset (as opposed to a higher dimensional tensor).
1.  A map transformation is used to, on a per-component basis, apply a suitable <span style="text-decoration:underline;">reducer</span> which transforms the nested dataset to a batched tensor.

The underlined transformations do not exist and are the proposed extensions to the tf.data API.

### Windowing 

Windowing combines elements of a dataset into finite datasets referred to as windows.


```python
def window(window_size):
  """A transformation that creates windows using the input dataset.

  The resulting datasets will contain `window_size` elements (or
  `N % window_size` for the last dataset if `window_size` does not
  divide the number of input elements `N` evenly).

  Args:
    window_size: A `tf.int64` scalar `tf.Tensor`, representing the number
      of elements of the input dataset to combine into a window.

  Returns:
    Dataset: A `Dataset` whose elements are a `Dataset`.
  """
```


### Reducers


#### Example 0: Count Dense Tensors 

To introduce the concept of tf.data to readers unfamiliar with it, we illustrate how it can be used to count the elements of a dataset:


```python
def count(dataset):
  """Counts the elements of a dataset."""

  def init_fn(_):
    return 0

  def reduce_fn(state, value):
    return state + 1

  def finalize_fn(state):
    return state

  count_reducer = tf.data.Reducer(init_fn, reduce_fn, finalize_fn)
  return dataset.reduce(count_reducer)

value = count(tf.data.Dataset.range(10))
with tf.Session() as sess:
  print(sess.run(value)) # produces 10
```


As you can see, a tf.data reducer consists of three functions.

*   An _init_ function that sets up the initial state.
*   A _reduce_ function that defines how to update the intermediate state given the value of the next element.
*   A _finalize_ function that defines how to produce the transform the final state into the output value.

Note that the <span style="text-decoration:underline;">count</span> method can be used to provide the "drop remainder" functionality for the window transformation as follows:


```python
dataset = dataset.window(N).filter(lambda x: tf.equal(count(x), N))
```



#### Example 1: Batch of Dense Tensors

Next, we illustrate how tf.data reducers can be used for batching a dataset of dense tensors.


```python
def batch_dense(dataset):
  """Batches a dataset of dense tensors."""

  if dataset.output_shapes.is_fully_defined():
    shape = dataset.output_shapes
  else:
    first_element = tf.contrib.data.get_single_element(dataset.take(1))
    shape = tf.shape(first_element)

  def batch_init_fn(_):
    batch_shape = tf.concat([[0], shape], 0)
    return gen_array_ops.empty(batch_shape, dtype=dataset.output_types)

  def batch_reduce_fn(state, value):
    return tf.concat([state, [value]], 0)

  def batch_finalize_fn(state):
    return state

  batch_reducer = tf.data.Reducer(batch_init_fn, batch_reduce_fn,
                                  batch_finalize_fn)
  return dataset.reduce(batch_reducer)

batch = batch_dense(tf.data.Dataset.range(5))
with tf.Session() as sess:
  print(sess.run(batch)) # produces [0 1 2 3 4]

```



#### Example 2: Padded Batch of Dense Tensors

Our final tf.data reducer example illustrates how to use it for padded batching a dataset of dense tensors.


```python
def padded_batch_dense(dataset, padded_shape, padding_value):
  """Batches a dataset of dense tensors with padding."""

  padded_shape = tf.cast(
      convert.partial_shape_to_tensor(padded_shape), tf.int32)

  def max_init_fn(_):
    return padded_shape

  def max_reduce_fn(state, value):
    return tf.maximum(state, tf.shape(value))

  def max_finalize_fn(state):
    return state

  # Compute the padded shape.
  max_reducer = tf.contrib.Reducer(max_init_fn, max_reduce_fn, 
                                   max_finalize_fn)
  padded_shape = dataset.reduce(max_reducer)

  def batch_init_fn(_):
    return tf.fill(
        tf.concat([np.array([0], dtype=np.int32), padded_shape], 0),
        tf.constant(padding_value, dtype=dataset.output_types))

  def batch_reduce_fn(state, value):
    return tf.concat([state, [value]], 0)

  def batch_finalize_fn(state):
    return state

  def pad_fn(value):
    shape = tf.shape(value)
    left = tf.zeros_like(shape)
    right = padded_shape - shape
    return tf.pad(value, tf.stack([left, right], 1), 
                  constant_values=padding_value)

  batch_reducer = tf.data.Reducer(batch_init_fn, batch_reduce_fn, 
                                  batch_finalize_fn)
  return dataset.map(pad_fn).reduce(batch_reducer)

padded_batch = padded_batch_dense(
    tf.data.Dataset.from_tensor_slices([[1], [2]]), [2], 0)
with tf.Session() as sess:
  print(sess.run(padded_batch)) # produces [[1 0] [2 0]]
```


Note that the method uses two reducers. The first reducer is used to compute the shape to pad to (as the maximum shape of the input elements) and the second reducer is used to do the actual batching.


### End-to-end Example

Bringing it all together, we now illustrate how to combine the window transformation and reducers to perform generalized tf.data batching:


```python
import tensorflow as tf

def gen():
  yield ('a', [1])
  yield ('b', [2])
  yield ('c', [3])
  yield ('d', [4, 4])

def map_fn(a, b):
  return batch_dense(a), padded_batch_dense(b, [2], 0)
  
dataset = tf.data.Dataset.from_generator(gen, (tf.string, tf.int32))
dataset = dataset.window(2).map(map_fn)
get_next = dataset.make_one_shot_iterator().get_next()

with tf.Session() as sess:
  print(sess.run(get_next)) # produces (['a', 'b'], [[1, 0], [2, 0]])
  print(sess.run(get_next)) # produces (['c', 'd'], [[3, 0], [4, 4]])
```



## API Changes

This design document proposes the following changes to the tf.data API:

*   Adding a `tf.data.Dataset.window` method, which provides the windowing functionality described in this proposal.
*   Promoting the `tf.contrib.data.reduce_dataset()` method to `tf.data.Dataset.reduce()` and the `tf.contrib.data.Reducer` class to `tf.data.Reducer`.
*   Allowing nested datasets as inputs of `map` and `filter`.
*   Adding canned reducers for batching and padded batching of dense and sparse tensors to `tf.contrib.data`.


## Open Questions

*   Any interest in the window transformation supporting parameters for specifying the window shift and stride (similar to tf.contrib.data.sliding_window_batch)? Is there any other type of windowing that people are interested in?
*   Besides batch and padded batch for dense and sparse tensors, what other types of batching should we provide canned reducers for?
