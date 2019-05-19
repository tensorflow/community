# Java Tensor NIO

| Status        | Proposed       |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Karl Lessard (karl@kubx.ca) |
| **Updated**   | 2019-05-10                                           |

## Objective

Simplify and improve performances of creating tensors in Java by writing and reading 
directly to/from their native buffers, while preserving their internal format.

## Motivation

Currently, the easiest way to create tensors in Java is by invoking one of the
factory methods exposed by the [`Tensors`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/java/src/main/java/org/tensorflow/Tensors.java)
class. While their signatures are elegant, by accepting concrete Java objects and 
multi-dimensional arrays, they make heavy use of reflection techniques to extract 
the shape and the size of the tensors to allocate. This results in poor performances,
as discussed in [this issue](https://github.com/tensorflow/tensorflow/issues/8244).

Reading tensor data uses a [similar approach](https://github.com/tensorflow/tensorflow/blob/c23fd17c3781b21bd3309faa13fad58472c78e93/tensorflow/java/src/main/java/org/tensorflow/Tensor.java#L449) and faces also performance issues. 
In addition, it requires in some cases that the user allocates a new buffer on the heap
into which the tensor data is copied (see [`writeTo()`](https://github.com/tensorflow/tensorflow/blob/c23fd17c3781b21bd3309faa13fad58472c78e93/tensorflow/java/src/main/java/org/tensorflow/Tensor.java#L483) methods, for example), which is not convenient when dealing
with large tensors.

Now that eager execution environment is (almost) supported by the Java client, it is imperative that the 
I/O operations between the native tensor buffers and the JVM are efficient enough to let the users peek at 
their data and, in some cases, modify it without an important performance hit.

By developing a new set of I/O utility classes, we can allow the user to access directly the tensor data 
buffers while still preventing mistakes that could break their internal format (the main reason why the
tensor buffer is not publicly exposed at this moment). Also, those utilities will help navigating into 
multidimensional arrays flattened into tensor buffers, using indexation features similar to NumPy.

## User Benefit

Users who are actually using factories and read/write methods from `Tensors/Tensor` classes might observe great 
performance improvements after switching to the new set of I/O utilities.

Users executing their operations in an eager environment will also find very useful and efficient 
to access directly the tensor data without the need of copying their buffer.

## Design Proposal

*Note: This design proposal assumes that we run in a Java >= 8 environment, which is not the case with
current client that is set to compile in Java 7 for supporting older Android devices. We need to confirm
with Android team if it is ok now to switch to Java 8.*

### Initializing Tensor Data

Currently, when creating tensors, temporary buffers that contains the initial data are allocated by the user 
and copied to the tensor memory (see [this link](https://github.com/tensorflow/tensorflow/blob/a6003151399ba48d855681ec8e736387960ef06e/tensorflow/java/src/main/java/org/tensorflow/Tensor.java#L187) for example). This data copy and additional memory allocation will be avoided by accessing tensor buffer 
directly at the initialization of the tensor.

Since tensor buffers are not resizable, the size in bytes of a tensor must be known at its creation time. 
This is quite trival for tensors with fixed-length datatype (like any numeric type). For variable-length 
datatypes though (like strings), this represents more a challenge as the values of the tensor elements 
has an impact on the required memory space.

Following factories will be added to the `Tensors` class:
```java
public static Tensor<Float> createFloat(long[] shape, Consumer<FloatOutputBuffer> dataInit);
public static Tensor<Double> createDouble(long[] shape, Consumer<DoubleOutputBuffer> dataInit);
public static Tensor<Integer> createInt(long[] shape, Consumer<IntOutputBuffer> dataInit);
public static Tensor<Long> createLong(long[] shape, Consumer<LongOutputBuffer> dataInit);
public static Tensor<Boolean> createBoolean(long[] shape, Consumer<BooleanOutputBuffer> dataInit);
public static Tensor<UInt8> createUInt8(long[] shape, Consumer<ByteOutputBuffer> dataInit);
public static Tensor<String> createString(long[] shape, Consumer<StringOutputBuffer> dataInit);
```
All methods except `createString` creates an empty `Tensor` first that is then initialized by invoking the 
`dataInit` function (the `*OutputBuffer` interface is described later in this document).

Since the size in bytes of a string tensor cannot be determined before retrieving its data, `createString` will 
collect and storeall the string values in a temporary buffer (or list) before creating and initializing a `Tensor` 
of the right size.

Once created, Tensors are immutable and their data could not be modified anymore.

### Reading Tensor Data

Currently, in order to read a tensor, the user needs to create a temporary buffer into which its data is copied. 
Once again, this data copy and additional memory allocation will be avoided by accessing the tensor buffer 
directly when reading its data.

The following methods will be added to the `Tensor` class:
```java
public FloatInputBuffer floatData();
public DoubleInputBuffer doubleData();
public IntInputBuffer intData();
public LongInputBuffer longData();
public BooleanInputBuffer booleanData();
public ByteInputBuffer uInt8Data();
public StringInputBuffer stringData();
```
It is up to the user to know which of these methods should be called on a tensor of a given type, similar
to the `*Value()` methods of the same class.

### Tensor Input/Output Buffers

There is a specific `*OutputBuffer` and `*InputBuffer` class for each datatype. The reason for not using only
with a generic parameterized interface (e.g. `OutputBuffer<T>`) is mainly because we want to allow the user to work
with primitive Java types, which take less memory-consuming and provide better performances that working exclusively
with their autoboxing wrapper.

These classes mimic those found in the `java.nio` package, with the distinction that output and input operation are 
split into two different interfaces. For simplicity, only the `Double` variant is presented:
```java
class DoubleOutputBuffer {
  DoubleOutputBuffer slice(Object... indices);
  long position();
  void put(double d);
  void put(long index, double d);
  void put(double[] array);
  void put(DoubleStream stream);
  void copyFrom(DoubleBuffer buffer);
}

class DoubleInputBuffer {
  DoubleInputBuffer slice(Object... indices);
  long numElements();
  long position();
  double get();
  double get(long index);
  void get(double[] dst);
  DoubleStream stream();
  void copyTo(DoubleBuffer buffer);
}
```
Here is a summary of what consist each of these methods:
* `slice(Object... indices)`: Returns a partial view of the tensor across all its dimensions. 
  More details on this in the next section
* `put(double d)`, `get()`: Sets/gets the next value in this buffer. The behaviour varies depending on 
  the size of the last dimension of this buffer
  * If size of last dimension is 0 (scalar), it sets/gets the value of this element
  * Else, it sets/gets the value of the current element and position is moved to the next element in the last dimension
* `put(long index, double d)`, `get(long index)`: Sets/gets the value at the given index. Only valid if the size
  of last dimension is greater than 0 (i.e. not a scalar)
* `put(double[] array)`, `put(DoubleStream stream)`, `get(double[] dst)`, `stream()`: Sets/gets all the values of the 
  last dimension of this buffer from/as an array or a stream. Only valid if the size of last dimension is greater 
  than 0 (i.e. not a scalar)
* `copyFrom(DoubleBuffer buffer)`, `copyTo(DoubleBuffer buffer)`: Sets/gets all values of this buffer from/to a standard
  Java NIO buffer.



This is the meat of the document, where you explain your proposal. If you have
multiple alternatives, be sure to use sub-sections for better separation of the
idea, and list pros/cons to each approach. If there are alternatives that you
have eliminated, you should also list those here, and explain why you believe
your chosen approach is superior.

Factors to consider include:

* performance implications
* dependencies
* maintenance
* platforms and environments impacted (e.g. hardware, cloud, other software
  ecosystems)
* [compatibility](https://www.tensorflow.org/programmers_guide/version_compat)
* how will this change impact users, and how will that be managed?

## Detailed Design

This section is optional. Elaborate on details if they’re important to
understanding the design, but would make it hard to read the proposal section
above.

## Questions and Discussion Topics

Seed this with open questions you require feedback on from the RFC process.
