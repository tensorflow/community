# TensorFlow 2.0: Functions, not Sessions.

| Status        | (Proposed / Accepted / Implemented / Obsolete)       |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | ashankar@google.com, joshl@google.com                |
| **Sponsor**   | apassos@google.com                                   |
| **Updated**   | 2018-09-18                                           |

## Objective

This document elaborates on the proposal to make TensorFlow be more "Pythonic" in 2.0. In four bullet points, the proposal is to:



*   Encourage[^1] the encapsulation of graph computation as Python functions \
(where the graph is executed when the function is invoked, instead of via `Session`)
*   Align "state" in the TensorFlow runtime (e.g., resource tensors like those that back `tf.Variable` objects) with state in the Python program (e.g., Python objects corresponding to the runtime state with lifetimes attached to each other).
*   Make it easy to export these encapsulations to a `GraphDef`+Checkpoint and/or `SavedModel`.
*   Enable eager execution by default.
*   Provide a path for incorporating existing code that uses the 1.x APIs to construct TensorFlows graphs as functions in TensorFlow 2.x programs.

This document liberally employs the use of sample code to describe the end-user effect of proposed changes.

(We say "encourage" instead of "require" since removing the Session API from the
Python frontend within a year may be an unrealistic aspiration.  Particularly
given the use in Estimators and the use of MonitoredSession and hooks. The
`Session` API may have to stick around in `tf.compat.v1`).


## Design Proposal

### Basic idea: Python functions as Graphs

Today, the TensorFlow graph defines the union of all computation that the author of the graph may be interested in. The actual computation to execute is defined by the arguments to `tf.Session.run`. Once this subgraph is defined, the runtime can optimize and execute. For example, consider the following:


```python
import tensorflow as tf

x = tf.placeholder(tf.float32)
y = tf.square(x)
z = tf.add(x, y)

sess = tf.Session()

z0 = sess.run([z], feed_dict={x: 2.})        # 6.0
z1 = sess.run([z], feed_dict={x: 2., y: 2.}) # 4.0
```


 \
Though there is one `tf.Graph` object the user is interacting with (`tf.get_default_graph()`), the two `sess.run` calls are executing different programs (indeed the runtime ends up with two separate `Graph` objects in C++, one for each program), equivalent to:


```python
def compute_z0(x):
  return tf.add(x, tf.square(x))

def compute_z1(x, y):
  return tf.add(x,  y)
```


The core proposal of this document is the alignment between computation expressed in Python and the computation executed by the runtime. Instead of defining a graph and then selecting the subgraph to execute at `sess.run()` time, the exact computation of interest is encapsulated in a Python callable. For example, the program above that uses `sess.run()` to compute `z0` and `z1` can be written as:


```python
import tensorflow as tf

@tf.defun
def compute_z1(x, y):
  return tf.add(x, y)

@tf.defun
def compute_z0(x):
  return compute_z1(x, tf.square(x))

z0 = compute_z0(2.)
z1 = compute_z1(2., 2.)
```


Where `tf.defun` is a decorator that "**de**fines a TensorFlow **fun**ction". A "TensorFlow function" defines a computation as a graph of TensorFlow operations, with named arguments and explicit return values. Users define the function they want TensorFlow to "accelerate" as a Python function and integrate it into their Python program like any other Python function call.

Having the Python function correspond to what the runtime will execute reduces conceptual complexity in translating between the two domains. It also affords an opportunity to provide more helpful stacktraces on errors. More advanced features available today (e.g., carving sub-graphs, feeding intermediate values) will still be possible (discussed later), though most users should not need to think in terms of graphs, feeds, and fetches. The constructed graph also provides a natural point for accelerators/acceleration libraries (NVIDIA TensorRT, Google Cloud TPUs etc.) to hook in for rewrites.


### `defun`: A brief specification

`defun` constructs a TensorFlow graph by "tracing" the TensorFlow operations executed by the Python function. Specifically:



*   `defun(f)` is a Python function that returns a Python callable, `C`
*   When the `C` is invoked it:
    1.  Determines an "input signature" \
If one was not explicitly specified by the user (as an argument to `defun`), the signature is computed from the types of the input arguments (including `dtype` and `shape` for `Tensor` arguments)
    1.  If a new input signature is encountered, then it invokes `f` to create a TensorFlow graph, `G`. If the input signature has been seen before, it looks up `G` from a cache keyed by the input signature.
    1.  It executes the graph defined by `G` and feeding each argument as a value of the corresponding `Placeholder` node in the graph.

Changes in input signature result in a new graph being traced. For example:


```python
@tf.defun
def f(x):
  one = tf.constant(1, dtype=x.dtype)
  return tf.add(x, one)

# Traces a graph with int32 operations and executes it
f(tf.constant(1, dtype=tf.int32))
# Traces a graph with float32 operations and executes it.
f(tf.constant(1, dtype=tf.float32))
```



### Referencing state: Variables, tables etc.

A `defun` decorated Python function encapsulates a graph and its execution. The Python function may reference stateful objects (i.e., state backed by `DT_RESOURCE` tensors in the runtime, e.g., `tf.Variable`) by referencing the corresponding Python object, and these will be captured as implicit inputs to the function.

Comparing TensorFlow code today with how we propose it looks in 2.x:


<table>
  <tr>
   <td>TensorFlow 1.x
   </td>
   <td>2.0
   </td>
  </tr>
  <tr>
   <td>



<pre class="prettyprint">W = tf.get_variable(
  "weight", shape=[10, 10])
b = tf.get_variable(
  "bias", shape=[10],
  initializer=tf.zeros_initializer())
c = tf.get_variable(
  "counter", shape=[],
  dtype=tf.int32,
  initializer=tf.zeros_initializer())

x = tf.placeholder(tf.float32)
ctr = c.assign_add(1)
with tf.control_dependencies([ctr]):
  y = tf.matmul(x, W) + b
init = 
  tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)
  print(sess.run(y,
  feed_dict={x: make_input_value()}))
  assert int(sess.run(c)) == 1</pre>


   </td>
   <td>



<pre class="prettyprint">W = tf.Variable(
  tf.glorot_uniform_initializer()(
    (10, 10))
b = tf.Variable(tf.zeros(10))
c = tf.Variable(0)

@tf.defun
def f(x):
  c.assign_add(1)
  return tf.matmul(x, W) + b

print(f(make_input_value())
assert int(c) == 1</pre>


   </td>
  </tr>
</table>


Worthy of note here - in TensorFlow 1.x, the memory underlying the variables `W` and `b` in the runtime lives for the lifetime of the `Session` - unrelated to the lifetime of the Python objects. In 2.x, the lifetime of the Python objects and the runtime state are tied together.


### Program-order semantics / Control dependencies

In TensorFlow graphs today, control dependencies are sometimes needed to ensure correct evaluation order. For example, consider the following:


```python
v = tf.Variable(1.0)
init_op = tf.global_variables_initializer()
assign_op = v.assign(2.0)
read = v.read_value()

with tf.Session() as sess:
  sess.run(init_op)
  val = sess.run(read)
  print(val) # Will print 1.0, the assign is ignored
  val = sess.run([read, assign_op])[0]
  print(val)   # Non-deterministically prints 1.0 or 2.0,
```


The output here is not deterministic, since `val` may evaluate to either 1.0 or 2.0 depending on whether the runtime happened to execute `assign_op` before `y` or not. `tf.control_dependencies` is a mechanism provided to add annotations at graph construction time to influence graph execution. The TensorFlow user, a Python programmer, is thus forced to think about two execution models - TensorFlow graphs and the Python interpreter. To eliminate this cognitive load, `defun` will automatically insert control dependencies to ensure that operations that produce or consume a given `DT_RESOURCE` tensor and operations that are marked stateful (`REGISTER_OP(...).SetIsStateful()`) follow graph construction order. Thus:


```python
v = tf.Variable(1.0)
@tf.defun
def f():
  v.assign(2.0)
  return v.read_value()

print(f()) # Always prints 2.0.
```


A preview of this implemented in `tf.contrib.eager.defun` today (using <code>[AutomaticControlDependencies](https://github.com/tensorflow/tensorflow/blob/2f886d17f1990da418366bd093a09fb01fe5e777/tensorflow/python/eager/function.py#L1800)</code>).


### Functions that create state

In the above code, no `tf.Variable` objects are created inside a `tf.defun` decorated function. This is makes it clear that the code will have the same semantics once wrapped.

Note that if the function naturally creates state only on the first trace, all is well:


```python
v = None

@tf.defun
def f(x):
  global v
  if v is None:
    v = tf.Variable(1.0)
  return tf.cast(x, tf.float32) + v

f(tf.constant(1, dtype=tf.float32)) # Creates the variable, returns 2.0
f(tf.constant(2, dtype=tf.int32))   # Reuses the variable, returns 3.0
```


To support this `defun` imposes some requirements on the decorated function:



1.  State (like `tf.Variable` objects) are only created the first time the function `f` is called. \
How that is accomplished is left up to the implementation of `f`. \
If any variables are created in the first execution of `f`, then `@tf.defun` will trace `f` a second time in order to record the behavior that will be used from then on. No variables may be created during that second trace, or any other trace after that (due to different dtypes, shapes, or non-tensor arguments).
1.  The caller must make sure that any variable referenced by the function still exists whenever the function is evaluated. \
`@tf.defun` itself will keep only weak references to these created variables. Thus, if the referenced state does not exist when the decorated function is invoked, an exception will be raised.

In the future we may want to allow for function local `tf.Variable`s, which are created and destroyed each time the decorated function is invoked.


### API for `defun`

We've introduced a single new symbol: `defun` that consumes a Python function and returns a callable Python object. The precise API of the object needs some iteration, but at a high level it will have methods to:



*   List out all captured state (`tf.Variable` objects, other `DT_RESOURCE` tensors used by the computation and provided as implicit inputs).
*   Access the `tf.Graph` that corresponds to the graph executed by the `__call__` method of the object.
*   Execute the function with custom `RunOptions` and retrieve `RunMetadata`.

Since new graphs are traced when new input signatures are encountered, a `defun` can encapsulate multiple graphs. For example, consider the following snippet:


```python
@tf.defun
def f(x):
  return tf.square(x)

f(int(1))
f(float(1.0))
```


 \
There are two graphs created here - one which corresponds to the `Square` operation applied to `DT_INT32` tensors, and one with the `Square` operation applied to `DT_FLOAT32` tensors. The object returned by `defun` encapsulates multiple graphs (lazily generated based on the type and shape of input arguments), multiplexing between them in `__call__`.

The same holds for the case where arguments are not `Tensor`s, for example:


```python
@tf.defun
def f(x, use_multiply):
  return tf.multiply(x, x) if use_multiply else tf.square(x)

f(2.0, True)
f(2.0, False)
```


will result in 2 graphs being created.

Note that the "type" of `Tensor` inputs to the function also incorporates the shape. For example:


```python
@tf.defun
def f(x): return tf.add(x, 1.)
f([2.0])
f([2.0, 3.0])
f([[2.0]])
```


will result in 3 graphs being created:



1.  One for when the first argument is a `tf.float32` vector with 1 element \
(input signature: `((tf.float32, [1]))`)
1.  One for when the first argument is a `tf.float32` vector with 2 elements \
(input signature: `((tf.float32, [2]]))`)
1.  One for when the first argument is a `tf.float32` 1x1 matrix \
(input signature: `((tf.float32, [1, 1]))`)

Tracing the decorated function to create a new graph on each input shape is a conservative choice (allowing for `f` to create graphs dependent on the shape), which may be unnecessary. Users can explicitly specify an input signature to ensure that the same graph is used for multiple inputs. For example:


```python
@tf.defun(input_signature=((tf.float32, [None]))
def f(x): return tf.add(x, 1.)

f([2.0])      # Returns [3.0]
f([2.0, 3.0]) # Matches the input signature as [None] matches the actual shape [2]
f([[2.0]])    # Raises an error as the arguments don't match the input signature.

# f is backed by a single Graph since the input signature specification allowed
# for the same graph to be used when the input shape is (1,) or (2,).
```


 


### Classes

If a member function of a class does not create variables, it may be decorated with `@tf.defun` and it will work:


```python
class ScalarModel(object):
  def __init__(self):
    self.v = tf.Variable(0)

  @tf.defun
  def increment(self, amount):
    self.v.assign_add(amount)

model1 = ScalarModel()
model1.increment(tf.constant(3))
assert int(model1.v) == 3
model1.increment(tf.constant(4))
assert int(model1.v) == 7
model2 = MyModel()
model2.increment(tf.constant(5))
assert int(model2.v) == 5
```


 \
This works since `increment()` has `self` as a non-tensor argument, and a new trace will be created for each value of `self`. However, if variables are created in a method, we want to allow a new set of variables for every instantiation of `self`. You get this behavior by using `@tf.method`:


```python
class AnyShapeModel(object):
  def __init__(self):
    self.v = None

  @tf.method
  def increment(self, amount):
    if self.v is None:
      self.v = tf.Variable(tf.zeros_like(amount))
    self.v.assign_add(amount)

model1 = AnyShapeModel()
model1.increment(tf.constant(3))
assert int(model1.v) == 3
model1.increment(tf.constant(4))
assert int(model1.v) == 7
model2 = MyModel()
model2.increment(tf.constant([4, 5]))
assert model2.v.numpy() == [4, 5]
```


The semantics here are that each new instance is allowed to create variables in each `@tf.method` once. The simple recommendation would be "always use `@tf.method` on methods, use `@tf.defun` for functions outside of a class". In the above example, if `increment` was decorated with `@tf.defun` instead, then the `model2.increment()` call would raise an exception (as per `defun`s stated behavior of disallowing state creation on anything but the first trace).

In addition, as long as all variable creation/initialization happens while we are tracing, we should be able to support exporting the initialization graph when exporting a `SavedModel` or `MetaGraphDef`.


### Transitioning from 1.x

The definition of `tf.defun` above is careful to check that invoking a decorated Python function would have the same behavior as invoking an undecorated function. This is to guard against it being passed code from TensorFlow v1.x that expects to only be called once (and relies on things like graph collections to track which variables are created), for example:


```python
def f(x, do_add):
  v = tf.Variable(5.0)
  if do_add:
    v.assign_add(x)
  else:
    v.assign_sub(x)
  return v
```


For this case, we use a different API, `tf.compat.v1.wrap_function`, that treats any created variables as static local state:


```python
f_add = tf.compat.v1.wrap_function(f, tf.TensorSpec(tf.float32, ()), True)

assert float(f_add(1.0)) == 6.0
assert float(f_add(1.0)) == 7.0

# Can call tf.compat.v1.wrap_function again to get a new trace, a new set
# of variables, and possibly different non-template arguments.
f_sub = tf.compat.v1.wrap_function(f, tf.TensorSpec(tf.float32, ()), False)

assert float(f_sub(1.0)) == 4.0
assert float(f_sub(1.0)) == 3.0
```


Note these differences from `tf.defun`:



*   Only ever traces `f()` once (per call to `tf.compat.v1.wrap_function`).
*   The complete input tensor signature (via  `tf.TensorSpec` calls) and the values of all non-tensor arguments must be specified when wrapping the function. Note: we may want a `tf.tensor_like(x)` convenience function that returns `tf.TensorSpec(x.dtype, x.shape)`.
*   Will include extra TF v1.x compatibility features like collections, and access v1.x APIs like `tf.compat.v1.get_variable()` 
*   Will not automatically insert control dependencies to maintain program order across stateful operations/state accesses.
*   May only use a function or Python constant to initialize variables, no tensors. This is a technical limitation, required by the fact that we need some way of disentangling the initializers for variables from the other operations from the function.
*   Keeps strong references to variables created in f, weak references to other variables accessed by f. This is to match the v1.x graph behavior that variables have the lifetime of the graph they are created, and can generally be accessed through graph collections. Some common patterns of writing v1.x code don't leave any references to those variables around. Keeping references to those variables extends their lifetime to match that of the object returned by `tf.compat.v1.wrap_function`.
*   Typically won't be used as a decorator. Calling `tf.compat.v1.wrap_function` takes some arguments, traces the function, and creates an object with state. The lifetime of the return value should be tracked explicitly by saving it in a variable. 

Treating state (like `tf.Variable`) as static local does mean that the behavior of a `tf.compat.v1.wrap_function`-decorated Python function differs from that of an undecorated one. In the above example, `f(1.0, True)` will always return 6.0 (as a scalar `Tensor`), while each call to `f_add(1.0)` will return a different value. We propose this separate `tf.compat.v1.wrap_function` endpoint specifically to make it easy to migrate TensorFlow 1.x libraries to the TensorFlow 2.0. The behavior of 2.0 `tf.defun` is restricted to cases where we can say that the behavior will match.

We recognize that code written for TensorFlow 1.x commonly does not encapsulate state in Python objects, instead adding to hidden (graph-)global collections. We will support code that accesses collections inside a `tf.compat.v1.wrap_function`, though those collections will be local to a single trace.

With the `tf.compat.v1.wrap_function` proposed above, most graph construction library functions written against TensorFlow 1.x can be incorporated into TensorFlow 2.x programs.


```python
def f(x):
  W = tf.compat.v1.get_variable(name="weight", shape=[10, 10])
  b = tf.compat.v1.get_variable(name="bias", shape=[10],
                                initializer=tf.zeros_initializer())
  c = tf.Variable(0, dtype=tf.int32, name="counter")
  with tf.control_dependencies([c.assign_add(1)]):
    return tf.matmul(x, W) + b
```



```python
f = tf.compat.v1.wrap_function(f, tf.placeholder(tf.float32, None))
print(f(make_input_value()))
assert len(f.variables) == 3
assert f.variables[0].name == "weight"
```


 \
In this case, the object returned by `tf.compat.v1.wrap_function` owns the state created within `f`, and the `__call__` method on it invokes the corresponding computation.

Long story short, `tf.compat.v1.wrap_function` helps in incorporating graph construction code written against TensorFlow 1.x into TensorFlow 2.x programs. `wrap_function` constructs the same object as a `defun` decorated function, which provides the conceptual equivalent of graph construction and `Session.run`.   


### Serialization: Exporting SavedModel/GraphDefs

So far we've only considered Python programs. One of the key features of TensorFlow is the ability to integrate models created (and possibly trained) in a Python program into an application written in another programming language and/or platform (e.g., servers, mobile phones, self-driving cars). This ability will of course remain, with a smoother path to exporting models.

In TensorFlow 1.x, "saving a model" could mean one of three things:



1.  Saving parameter values, but not the computation: \
A "checkpoint" containing the values of all model parameters. \
(`tf.train.Saver` / `tf.train.Checkpoint`) \
Restoring this model required that the restoring program duplicate the Python code to construct the graph with the same model parameters.
1.  Saving the computation graph, but not the parameter values: \
The computation is represented by a `GraphDef` that can be exported by calls to `tf.Graph.as_graph_def()`, or `tf.train.export_meta_graph()`, and reconstructed by calls to `tf.import_graph_def()` / `tf.train.import_meta_graph()`.  \
Note that the parameter (`tf.Variable`) values are not saved, but their initializers are.
1.  Saving both the computation and the parameter values: \
The two packaged together in a SavedModel. \
At a high level, the SavedModel format packages the `MetaGraphDef`, checkpoint, and a signature (names of input and output tensors). \
(`tf.saved_model.simple_save` / `tf.saved_model.builder.SavedModelBuilder`) \
This is the format preferred for exporting for serving via TensorFlow Serving or to other languages (e.g., `SavedModelBundle.load()` in Java, `LoadSavedModel` in Go)

The objects created by `defun` encapsulate (1) the computation expressed as a `GraphDef`, (2) the state used by it. Thus, these objects are naturally suited for import/export in any of the above formats, using something like the following:


<table>
  <tr>
   <td>TensorFlow 1.x
   </td>
   <td>2.x
   </td>
  </tr>
  <tr>
   <td colspan="2" >Save only the parameters, not the computation
   </td>
  </tr>
  <tr>
   <td>



<pre class="prettyprint">W = tf.get_variable(
  "weights", shape=[10, 10])

# Presumably the train_op is
# a little fancier 
train_op = W.assign_add(1.)
saver = tf.train.Saver()

with tf.Session() as sess:
  sess.run(W.initializer)
  sess.run(train_op)
  saver.save(sess, "/tmp/checkpoint/")

with tf.Session() as sess:
  saver.restore(sess, "/tmp/checkpoint/")
  sess.run(train_op)</pre>


   </td>
   <td>



<pre class="prettyprint">W = tf.Variable(
  tf.glorot_uniform_initializer()(
    (10, 10)))

@tf.defun
def train():
  W.assign_add(1.)

train()
ckpt = tf.train.Checkpoint(W=W)
ckpt.save("/tmp/checkpoint")
ckpt.restore("/tmp/checkpoint")</pre>


   </td>
  </tr>
  <tr>
   <td colspan="2" >Exporting/Importing <code>GraphDefs</code>
   </td>
  </tr>
  <tr>
   <td>



<pre class="prettyprint">W = tf.get_variable("weights", shape=[10, 10])
x = tf.placeholder(
  tf.float32, shape=(None, 10)))
y = tf.matmul(x, W)

graph = tf.get_default_graph()
graph_def =  graph.as_graph_def()
with open("/tmp/graph.pb", "w") as f:
  f.write(
      graph_def.SerializeToString())

tf.reset_default_graph()

graph_def = tf.GraphDef()
with open("/tmp/graph.pbtxt") as f:
  graph_def.ParseFromString(f.read())

tf.import_graph_def(graph_def)</pre>


   </td>
   <td>



<pre class="prettyprint">W = tf.Variable(
  tf.glorot_uniform_initializer()(
    (10, 10)))

@tf.defun
def f(x):
  return tf.matmul(x, W)

# Retrieve the object corresponding to
# a particular input signature:
graph = f.graph_function(
  (tf.float32, (None, 10)).graph
graph_def = graph.as_graph_def()

with open("/tmp/graph.pb", "w") as f:
  f.write(graph_def.SerializeToString())</pre>

 \

   </td>
  </tr>
  <tr>
   <td colspan="2" >Exporting/Importing SavedModels
   </td>
  </tr>
  <tr>
   <td>



<pre class="prettyprint">
def save_model():
  W = tf.get_variable("weights",
                      shape=[10, 10])
  x = tf.placeholder(
    tf.float32, shape=(None, 10))
  y = tf.matmul(x, W)

  with tf.Session() as sess:
    sess.run(
    tf.global_variables_initializer())
    tf.saved_model.simple_save(
      sess,
      "/tmp/model",
      inputs={"x": x},
      outputs={"y": y})

def load_model():
  sess = tf.Session()
  with sess.as_default():
    inputs, outputs =  tf.saved_model.simple_load(sess, "/tmp/model")
  return inputs, outputs, sess</pre>


   </td>
   <td>To be worked on but something along the lines of:



<pre class="prettyprint">
class Model(tf.train.Checkpointable):
  def __init__(self):
    self.W = tf.Variable(...)

  @tf.method
  def f(self, x):
    return tf.matmul(x, self.W)

m = Model()

tf.saved_model.export("/tmp/model", m)

m = tf.saved_model.import("/tmp/model")
</pre>


   </td>
  </tr>
</table>



### Derived/Related Graphs

One reservation expressed by TensorFlow graph/session enthusiasts today is that the ability to write generic analysis/inspection tooling on graphs, precluding the need to understand or modify the Python code that constructed the graph, is important to them. To put it differently, some find it easier to navigate the `GraphDef` program than navigating the Python program. \


This ability will be maintained. `defun`-decorated Python functions have an associated graph, and new functions can be created by specifying the sub-graph of interest. For example:


<table>
  <tr>
   <td>TensorFlow 1.x
   </td>
   <td>TensorFlow 2.x
   </td>
  </tr>
  <tr>
   <td colspan="2" >Carving out a subgraph
   </td>
  </tr>
  <tr>
   <td>



<pre class="prettyprint">def build_graph():
  x = tf.placeholder(tf.float32)
  y = tf.square(x)
  z = tf.square(y)

with tf.Session() as sess:
  build_graph()
  sess.run("Square_1:0",
   feed_dict={"Square:0": 2.0})  # 4.0</pre>


   </td>
   <td>



<pre class="prettyprint">@tf.defun
def f(x):
  return tf.square(tf.square(x))

# tf.Graph corresponding to "x" 
# being a float32 tensor with unknown
# shape
graph = f.graph_function(
  (tf.float32, None)).graph

f2 = tf.NewGraphFunction(
  graph,
  inputs=["Square:0"], 
  outputs=["Square_1:0"])
# The above may optionally take a
# "prune" argument to allow for
# pruning stateful operations in
# `graph` that are not in the path
# from inputs to outputs.
f2(2.0) # 4.0</pre>


   </td>
  </tr>
  <tr>
   <td colspan="2" >Extending a graph
   </td>
  </tr>
  <tr>
   <td>



<pre class="prettyprint">def build_graph():
  x = tf.placeholder(tf.float32)
  y = tf.square(x)
  return y

y = build_graph()
z = tf.square(y)

with tf.Session() as sess:
  # Line below will return 16.0
  sess.run(z, feed_dict={"Placeholder:0": 2.0))</pre>


   </td>
   <td>



<pre class="prettyprint">@tf.defun
def f(x):
  return tf.square(x)

@tf.defun
def g(x):
  return tf.square(f(x))

g(2.0) # 16.0</pre>


   </td>
  </tr>
</table>



### Distributed Execution

At the lowest level of the API, distributed execution continues to work with `tf.device` annotations, where the device name can reference remote devices as well, just like they do today.

The `DistributionStrategy` API, typically aimed at synchronous training will continue to be the method of choice (where the API can be used inside a `defun`). Other APIs such as go/tf-replicator should also be usable.

The author realizes that this section can do with more detail. However, to keep this document more focused, these details will be discussed separately. In particular, usage of `MonitoredSession` and session hooks today needs additional thought.


### `defun`-ing Python control flow

`defun` decorates a graph construction function and transparently recreates graphs if needed. However, this does mean that if the function has data-dependent control flow then though the function will execute fine with eager execution enabled, `defun` decorating it will fail. For example:


```python
def f(x, y):
  if tf.equal(y, 0.0):
    return y
  return x / y

x = tf.constant(2.0)
y = tf.constant(2.0)

f(x, y) # Will be 1.0

df = tf.defun(f)
df(x, y)  # Will raise an error complaining about the data-dependent control flow
```


 \
To fix this, one would have to use the graph construction APIs for control flow (`tf.cond`, `tf.while_loop`):


```python
def f(x, y):
  return tf.cond(tf.equal(y, 0.0), lambda: y, lambda: x/y)

x = tf.constant(2.0)
y = tf.constant(2.0)

f(x, y) # Will be 1.0

df = tf.defun(f)
df(x, y)  # Will be 1.0
```


This situation can be improved with the help of  go/tf-autograph to allow expression of control flow in Python. Whether autograph will be enabled by default or not is still under debate, but the option will be there as a flag on defun.


### Summaries

**Background**: See the updated 2.0 summaries design (go/tf-summaries-2.0) and plan (go/tf-2.0-summaries). Support for TensorFlow 1.x summaries is a non-goal.

The summary writing operations (<code>[tb.summary.scalar](https://www.tensorflow.org/api_docs/python/tf/contrib/summary/scalar)</code>, <code>[tb.summary.image](https://www.tensorflow.org/api_docs/python/tf/contrib/summary/image)</code> etc.) can be naturally placed in the graph by using them in a <code>defun</code>-decorated function. These operations require two "external" inputs - the summary writer resource and the condition, that will be picked up from the context (e.g., <code>[tb.summary.create_file_writer](https://www.tensorflow.org/api_docs/python/tf/contrib/summary/create_file_writer)</code> and <code>[tb.summary.record_summary_every_n_global_steps](https://www.tensorflow.org/api_docs/python/tf/contrib/summary/record_summary_every_n_global_steps)</code>).  When defining the graph, these inputs are converted to placeholders, which are then resolved at function invocation time. Thus, something like this:


```python
writer = tf.contrib.summary.create_file_writer('/tmp/test')
with writer.as_default(), tf.contrib.summary.always_record_summaries():
    f()
with writer.as_default(), tf.contrib.summary.never_record_summaries():
    f()
```


Will write one summary to `writer` whether `f` is defined as:


```python
def f(): 
    tb.summary.scalar("loss", compute_loss()) 
```


Or


```python
f = tf.contrib.eager.defun(f)
```


(NOTE: As of August 2018, this is  not the case, but it will be. See b/112269952).

Note that the runtime is free to prune away the summary writing operations when the function is invoked in a context where there is no summary writer resource or the condition is false.


### What does that have to do with eager execution?

So far this proposal has dealt with the encapsulation of TensorFlow graphs in Python functions with the intention of making it easier to integrate TensorFlow-accelerated computation in Python programs. 

_Additionally_, this proposal suggests enabling eager execution by default in TensorFlow 2.0. Keeping `defun` in mind, this basically means:



*   Inside the context of defining a TensorFlow function (i.e., within a `defun` decorated function) `tf.Tensor` objects created refer to symbolic tensors.
*   Outside this context, `tf.Tensor` objects created are backed by concrete values and TensorFlow API. The underlying memory of the tensor can be backed by any device (i.e., CPU/GPU) and is not restricted to host-memory (like numpy arrays).

See the [docstring for tf.contrib.eager.defun](https://www.tensorflow.org/api_docs/python/tf/contrib/eager/defun) - the evolving playground for the implementation of the proposal in this document. The basic takeaway is that:



*   For users that embrace symbolic tensors and graphs, continue doing so with your code placed inside a `defun` decorated Python function.
*   We believe most users (new ones in particular) will find it more convenient to deal with `Tensor` objects backed by concrete values and then selectively "compiling" portions of their Python program into TensorFlow graphs rather than being exposed to graph metaprogramming in Python upfront. In spirit, this is similar to Swift4TensorFlow with the obvious glaring difference that[ graph program extraction](https://github.com/tensorflow/swift/blob/master/docs/DesignOverview.md#graph-program-extraction) here is manually specified (with the `defun` decoration).

NOTE: In TensorFlow 1.x, eager execution is enabled by <code>[tf.enable_eager_execution()](https://www.tensorflow.org/api_docs/python/tf/enable_eager_execution)</code>. Once invoked, all public API endpoints that consume or produce symbolic <code>Tensor</code> objects begin to produce and consume <code>Tensor</code> objects that are backed by a concrete value. See the "Research and Experimentation" section at [www.tensorflow.org/tutorials](http://www.tensorflow.org/tutorials) for an introduction.


### A few things of note



*   This change **only** applies to the TensorFlow **Python** frontend
    *   [TensorFlow.js](https://js.tensorflow.org/) is already "eager by default".
    *   [Switf4TensorFlow](https://github.com/tensorflow/swift) has [similar design goals](https://github.com/tensorflow/swift/blob/master/docs/DesignOverview.md#swift-for-tensorflow-design-overview), doing away with the define-then-run style of TensorFlow graphs.
    *   Most other language bindings ([Java](https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/package-summary), [C++](https://www.tensorflow.org/api_guides/cc/guide), [Go](https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go), others) are mostly targeting deployment of defined models in applications. While an imperative style might help simplify model development and training in these languages, doing so is explicitly out of scope for TensorFlow 2.0. The notion of graphs and sessions will remain in them, as well as in the stable [C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h).
*   Users of **<code>Estimator</code>** will see no change
    *   Canned Estimators are black boxes that create and train models. Enabling eager execution will have no effect on their usage. This is true today.
    *   The <code>model_fn</code> of a regular (non-canned) <code>Estimator</code> will remain as a graph construction function.
*   <strong>[SavedModel](https://www.tensorflow.org/guide/saved_model#save_and_restore_models)</strong> will continue to be the format encouraged for exporting trained models
    *   Crudely speaking, a SavedModel encapsulates a Graph, a checkpoint of variable values, and some metadata like signature information (names of input and output tensors).
    *   A path will be provided to easily export models in this format (e.g., via <code>tf.keras.Model.save()</code>). There may be instances where converting the Python code to a graph is not trivial, in which case, exporting to a SavedModel (and thus a Graph) will fail.


## Alternatives Considered


### Creating state inside a `defun`

How state (`DT_RESOURCE` tensors)  created inside a `defun` should be handled is actively being debated. Options include:



1.  "Lifting" state out as a static local function variable
1.  Mimic the undecorated code - creating and destroying variables on each call.


#### "Static-local" state

`tf.contrib.eager.defun` today treats state as function-static variables, which allows for code like:


```python
def f(x):
  v = tf.Variable(1, dtype=x.dtype)
  v.assign_add(x)
  return v

df = tf.contrib.eager.defun(f)
# tf.defun(f) proposed in this document will raise an exception on first use
x = tf.constant(1, dtype=tf.float32))
print(df(x))  # 2.0
print(df(x))  # 3.0
```


 \
However, the one major issue with this approach is that it behaves differently from how an undecorated function would:


```python
print(f(1.0), df(1.0))  # 2.0, 2.0
print(f(1.0), df(1.0))  # 2.0, 3.0
```


To be conservative, we propose some restrictions on `defun`, such as:



1.  State is created only once, i.e., `defun` will fail if calling `f` a second time results in new state being created.
1.  `defun` decorated functions can only produce `Tensor` return values.
1.  If you want to convert TF v1.x code like `f` above, you may use `tf.compat.v1.wrap_function` which guarantees it will only trace `f` once.


#### Function-local state

Another option would be to match typical Python functions, where state is created and destroyed during the call to the function. So:


```python
def f(x):
  v = tf.Variable(1.0)
  v.assign_add(x)
  return v

df = tf.defun(f)

assert f(1.0) == df(1.0) # Both will be 2.0
assert f(1.0) == df(1.0) # Still 2.0, since 'v' would be recreated.
```


 \
This seems like an avenue definitely worth pursuing, but requires careful consideration of some additional design points such as escape analysis of return values (e.g. the lifetime of `tf.Variable` objects that are returned from a decorated function).

For now, we propose that `defun` continue with the restricted abilities proposed in this document and a "maintain Python semantics" decorator be investigated independently.


## Questions and Discussion Topics

*   Naming:
    *   `tf.defun` or `tf.function`?
    *   `tf.compat.v1.wrap_function` or `tf.compat.v1.defun` or `tf.compat.v1.function` or `tf.compat.v1.wrap_graph_as_function`?


