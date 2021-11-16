# TensorShape Evaluation to Bool

| Status        | Proposed       |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [NNN](https://github.com/tensorflow/community/pull/NNN) (update when you have community PR #)|
| **Author(s)** | Bogdan Alexe (Google), Yu Feng (Google) |
| **Sponsor**   | Rohan Jain (Google)                 |
| **Updated**   | 2021-11-16                                           |

## Objective

Fix inconsistencies in `TensorShape` evaluation to `bool`.

## Motivation
In the current state, a `TensorShape` object evaluates to `True` if the list of dimensions is not `None`, i.e. if the tensor shape is not unspecified.
This is inconsistent with the `numpy` behavior on shapes, as well as with how the list of dimensions in `TensorShape` is evaluated to `bool`.

Moreover, the current behavior has inconsistencies between eager and graph execution modes. A non-scalar tensor with dynamic rank may have a shape that is:
- specified in eager mode, and evaluate to True
- unspecified in graph mode, and evaluate to False, which is not intuitive and can be confusing. 

Example:
```
def fun():
  n = tf.random.poisson((1,), 3, dtype=tf.int32)
  s = tf.random.poisson(n, 9, dtype=tf.int32)
  a = tf.ones(s)
  return bool(a.shape)

fun()              ## True
tf.function(fun)() ## False
```

This change will disallow the evaluation to `bool` on the unspecified shape resulting in the graph execution.

## Design Proposal

With the proposed change, a `TensorShape` will:
- evaluate to `True` if the shape is specified and non-scalar, i.e. if the list of dimensions is not empty
- evaluate to `False` if the shape is specified and represents a scalar, i.e.  if the list of dimensions is empty
- raise an error (`ValueError`) if the shape is unspecified, i.e. if the list of dimensions is undefined.

This will:
- align the `TensorShape` behavior with `numpy`, as well as with the `bool` evaluation on the list of dimensions in the `TensorShape`. 
- explicitly fail by raising errors when conversions to `bool` are attempted on unknown shapes in graph execution mode. (see example above)

An evaluation to `bool` will only succeed on shapes that are specified, and will distinguish between scalar/non-scalar shapes.

| `TensorShape._dims` | Current result | Result after change |
| --- | --- | --- |
| `None` | `False` | Raise `ValueError` |
| `[ ]` (empty list, denotes a scalar) | `True` | `False` |
| `[n1, n2, …]` (non-empty list) | `True` | `True` |

### Performance Implications
* No performance impact is expected, to be confirmed via benchmark results.

### Dependencies
* No new dependencies added.
* This will break users that rely on the existing behavior.

### Engineering Impact
* No expected meaningful changes to binary size / startup time / build time / test times.

### Platforms and Environments
* No expected impact in ability to run on any platform or environment.

### Tutorials and Examples
Users who rely on evaluating a `TensorShape` to a `bool` to check if it has a known number of dimensions will have to change.

**Example 1:**

Before:
``` 
if foo.shape:
  bar()
```
After:
```
if foo.shape.rank is not None:
  bar()
```

**Example 2:**

Before:
```
def foo(bar, shape = None):
  if shape:
    baz()
```
After:
```
def foo(bar, shape = None):
  if shape is not None and 
     shape.rank is not None:
    baz()
```    

### Compatibility
* This is a breaking change: see rollout below.
* Interactions with other parts of the TensorFlow Ecosystem: no expected impact.

### User Impact
* Existing usage that relies on the current behavior of `TensorShape` evaluation to `bool` will be broken.
* Rollout:
  * Add warning in `TensorShape.__bool__`  that behavior is changing in the next release.
  * In the following release, switch behavior in `TensorShape.__bool__` 
