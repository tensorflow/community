# Generalizing tf.data batching using windowing and reducers 

| Status        | Accepted                                             |
:---------------|:-----------------------------------------------------|
| **Author(s)** | Jiri Simsa (Google)                                  |
| **Sponsor**   | Derek Murray (Google)                                |
| **Updated**   | 2018-09-19                                           |

## Objective 

This proposal addresses the known limitations of the current tf.data batching API:

*   it provides a mechanism for padded batching of sparse tensors
*   it facilitates customization of batching logic (users can now express batching logic as a pure Python function)
*   it enables application of different batching logic on different components

## **Motivation**

The tf.data API is the de facto standard for creating TensorFlow input pipelines, whose purpose is to extract data from a storage system, transform it, and load it onto an accelerator.

A common transformation performed by TensorFlow input pipelines is batching -- combining multiple tensors into a single tensor of higher dimension, most often to make a minibatch for training. Currently, the core tf.data API for batching consists of [batch](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch) and [padded_batch](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#padded_batch). The former assumes the inputs have the same shape and supports both dense and sparse inputs. The latter supports dynamically shaped inputs, such as you might find in sequential data: it assumes the inputs have the same rank but not necessarily the same shape and can pad differently shaped inputs to a common shape; only dense inputs are supported by padded_batch.

The tf.data batching API has several limitations that has surfaced in various users requests:

*   As already mentioned, the padded_batch transformation does not support sparse tensors inputs ([issue](https://github.com/tensorflow/tensorflow/issues/18302)).
*   The current API is not flexible enough to accept user-provided batching logic (e.g. [issue](https://github.com/tensorflow/tensorflow/issues/20391)).
*   The same batching logic needs to be applied to all components of the input dataset, which is not always desirable (e.g. [issue](https://github.com/tensorflow/tensorflow/issues/20391)). Users can work around this limitation by creating separate datasets to which different batching transformations are applied and then zipping the datasets; however, this can be inefficient, unergonomic, and error prone.


## Proposal

This document proposes leveraging the recently introduced support for _nested_ datasets as inputs to tf.data transformations to perform generalized batching as follows:



1.  A <span style="text-decoration:underline;">window</span> transformation is used to combine consecutive elements of the input into a nested dataset (as opposed to a higher dimensional tensor).
1.  A map transformation is used to, on a per-component basis, apply a suitable <span style="text-decoration:underline;">reducer</span> which transforms the nested dataset to a batched tensor.

The underlined transformations do not exist and are the proposed extensions to the tf.data API.


### Windowing

Windowing combines elements of a dataset into finite datasets referred to as windows. This is similar to batching, with the main difference being that batching combines elements of dataset into a higher dimensional element, while windowing combines the elements to a dataset.


```python
def window(size, shift=1, stride=1, drop_remainder=True):
  """Combines input elements into a dataset of windows.

  Each window is a dataset itself and contains `size` elements (or
  possibly less if there is not enough input elements to fill the window
  and `drop_remainder` evaluates to false).

  The `stride` argument determines the stride of the input elements,
  and the `shift` argument determines the shift of the window.

  For example:
 - tf.data.range(5).window(3) produces {{0, 1, 2}, {1, 2, 3}, {2, 3, 4}}
 - tf.data.range(5).window(3, 3, 1, False) produces {{0, 1, 2}, {3, 4}}
 - tf.data.range(6).window(3, 1, 2) produces {{0, 2, 4}, {1, 3, 5}}

  Args:
    size: A `tf.int64` scalar `tf.Tensor`, representing the number
      of elements of the input dataset to combine into a window.
    shift: A `tf.int64` scalar `tf.Tensor`, representing the forward
      shift of the sliding window in each iteration.
    stride: A `tf.int64` scalar `tf.Tensor`, representing the stride
      of the input elements in the sliding window.
    drop_remainder: A `tf.bool` scalar `tf.Tensor`, representing whether
      a window should be dropped in case its size is smaller than 
      `window_size`.

  Returns:
    Dataset: A `Dataset` whose elements are a `Dataset`.
  """
```

For example:

*   `tf.data.range(5).window(3)` produces `{{0, 1, 2}, {1, 2, 3}, {2, 3, 4}}`.
*   `tf.data.range(5).window(3, 3, 1, False)` produces `{{0, 1, 2}, {3, 4}}`.
*   `tf.data.range(6).window(3, 1, 2)` produces `{{0, 2, 4}, {1, 3, 5}}`.


### Reducers


#### Example 0: Count Elements

To introduce the concept of tf.data reducers to readers unfamiliar with it, we illustrate how a reducer can be used to count the elements of a dataset:


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


As you can see, a tf.data reducer consists of three functions: 1) an _init()_ function that sets up the initial state, which can be an arbitrary nest of tensor-like objects, 2) a _reduce()_ function that defines how to update the intermediate state given the value of the next element, and 3) a _finalize()_ function that defines how to produce the transform the final state into the output value.

The reducer inputs an entire dataset and reduces it to a single value. This single value is the result of taking the output of init(), calling reduce() successively on every element of the dataset until the dataset is exhausted, and then calling finalize() on the result.


#### Example 1: Batch of Dense Tensors

Next, we illustrate how tf.data reducers can be used to create a batch from a dataset of dense tensors.

```python
def batch_dense(dataset):
  """Batches a dataset of dense tensors."""

  if dataset.output_shapes.is_fully_defined():
    shape = dataset.output_shapes
  else:
    first_element = tf.contrib.data.get_single_element(dataset.take(1))
    shape = tf.shape(first_element)

  def batch_init_fn(_):
    """Return an empty Tensor of the correct shape and type."""
    batch_shape = tf.concat([[0], shape], 0)
    return gen_array_ops.empty(batch_shape, dtype=dataset.output_types)

  def batch_reduce_fn(state, value):
    """Append this value to what we have of the batch so far."""
    return tf.concat([state, [value]], 0)

  def batch_finalize_fn(state):
    """Return the batch tensor as constructed so far."""
    return state

  batch_reducer = tf.data.Reducer(batch_init_fn, batch_reduce_fn,
                                  batch_finalize_fn)
  return dataset.reduce(batch_reducer)

batch = batch_dense(tf.data.Dataset.range(5))
with tf.Session() as sess:
  print(sess.run(batch)) # produces [0 1 2 3 4]

```



#### Example 2: Padded Batch of Dense Tensors

Our next tf.data reducer example illustrates how to use a reducer to create a padded batch from a dataset of dense tensors.

```python
def padded_batch_dense(dataset, padded_shape, padding_value):
  """Batches a dataset of dense tensors with padding."""

  padded_shape = tf.cast(
      convert.partial_shape_to_tensor(padded_shape), tf.int32)

  def init_fn(_):
    return 0, padded_shape

  def reduce_fn(state, value):
    count, shape = state
    return count + 1, tf.maximum(shape, tf.shape(value))

  def finalize_fn(state):
    return state

  # Compute the padded shape and count elements.
  reducer = tf.contrib.Reducer(init_fn, reduce_fn, finalize_fn)
  count, padded_shape = dataset.reduce(reducer)

  def pad_fn(value):
    shape = tf.shape(value)
    left = tf.zeros_like(shape)
    right = padded_shape - shape
    return tf.pad(value, tf.stack([left, right], 1), 
                  constant_values=padding_value)

  return dataset.map(pad_fn).batch(count)

padded_batch = padded_batch_dense(
    tf.data.Dataset.from_tensor_slices([[1], [2]]), [2], 0))
    .make_one_shot_iterator().get_next()
with tf.Session() as sess:
  print(sess.run(padded_batch)) # produces [[1 0] [2 0]]
```



### End-to-end Example

Finally, we illustrate how to use the window transformation to perform generalized tf.data batching:

```python
import tensorflow as tf

def gen():
  yield ('a', [1])
  yield ('b', [2])
  yield ('c', [3])
  yield ('d', [4, 4])

def map_fn(a, b):
  return tf.data.Dataset.zip((a.batch(2), b.padded_batch(2, [2])))
  
dataset = tf.data.Dataset.from_generator(gen, (tf.string, tf.int32))
dataset = dataset.window(2, 2).flat_map(map_fn)
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
*   Adding canned reducers for padded batching of dense and sparse tensors to `tf.contrib.data`, changing implementation of `tf.data.Dataset.padded_batch()` to use these, and marking it as deprecated.

## Summary

This proposal addresses known limitations of the current tf.data batching API:

*   it provides a mechanism for padded batching of sparse tensors
*   it facilitates customization of batching logic (users can now express batching logic as a pure Python function)
*   it enables application of different batching logic on different components


## Discussion Notes

See also notes from [public review](https://github.com/tensorflow/community/pull/5). The following notes were taken in the review committee.

Q: What is the better value added by the new examples?

A: The previous examples were inefficient versions of things that already exist.

Q: The obvious use of the API led to an inefficient implementation (of batching, using tf.concat()). It might be hard to write batching in this API without it being 

A: This API is not meant to be used to implement something that already exists.

Q: Is this not a good API for implementing batching? The structure encourages inefficient implementations.

A: The point was not to illustrate how we do batching efficiently. It's already done.

Q: I thought the point was to show many different ways to do batching.

A: The base case is still an efficient implementation of batch, but we can add other logic around it (e.g. to do different forms of padding, etc.).

Q: What were the biggest questions?

A: Batching efficiency was the biggest one. Some questions about the signature of the newly introduced transformation. One reader commented that the meaning of "window" in other communities (video processing) typically includes some notion of slide/stride. Conclusion was that we will support shift and stride as we already do in `sliding_window_batch()`. Stride = number of elements you skip (i.e. for non-consecutive elements in a window), shift = how much the window shifts between windows.

Q: Is there any significant overhead from elements being datasets (e.g. from extra work in Python)?

A: The amount of computation that you have to do to compute the batch should be the same. There is no additional work in Python.

Q: How do you compile the reduce function to run it in C++?

A: It's a TF function, similar to existing map functions, etc.

Q: Concern about how many times count() is invoked.

A: The example shows how to use it in a filter(), where the count is evaluated in a function context.

Q: Re: runtime efficiency, in the higher dimensional case, would we always make a copy to concatenate?

A: That's what the Dataset.batch() transformation does. The nested dataset elements aren't intended for direct consumption, but to serve as input to other transformations, which e.g. build padded batches, sparse tensors, etc. This proposal lets you mix and match how you treat the different components, as illustrated in the end-to-end example. The goal of the new API isn't to improve efficiency of the existing implementations, but to add support for new kinds of transformation.

Q: What about the parallel proposal for random access datasets? Will count() be an exposed primitive or would you use the efficient random-access count?

A: We would add efficient random-access count for the nested datasets produced by window().

