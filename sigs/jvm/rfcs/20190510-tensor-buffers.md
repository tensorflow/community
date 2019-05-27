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
class. While their signatures are elegant, they make heavy use of reflection techniques to extract 
the shape and the size of the tensors to allocate. This results in poor performances,
as discussed in [this issue](https://github.com/tensorflow/tensorflow/issues/8244).

Reading tensor data uses [similar techniques](https://github.com/tensorflow/tensorflow/blob/c23fd17c3781b21bd3309faa13fad58472c78e93/tensorflow/java/src/main/java/org/tensorflow/Tensor.java#L449) and faces also performance issues. It is also possible to copy the data in the tensor to a user-allocated
buffer in order to read (see [`writeTo()`](https://github.com/tensorflow/tensorflow/blob/c23fd17c3781b21bd3309faa13fad58472c78e93/tensorflow/java/src/main/java/org/tensorflow/Tensor.java#L483) methods, for example), which of course is not convenient when dealing with large tensors.

Now that eager execution environment is supported by the Java client, it is imperative that the 
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

### Tensor Data Utils

A new set of utilities will be distributed with TensorFlow to improve read and write operations in a tensor, often represented
as a multidimensional array. At the root of this set is the `[Type]Tensor` interfaces (not to be confused with the existing 
`Tensor<>` class in TF Java, which is in fact just a symbolic handle to a given tensor).

For each tensor datatype supported in Java, a variant of `[Type]Tensor` interface is provided to allow users to work with Java 
primitive types, which tends to be less memory-consuming and provide better performances than their autoboxed equivalent.

For readability, only the `Double` variant of this interface is shown below:
```java
interface DoubleTensor {
  int rank();  // number of dimensions (or rank) of this tensor
  long size(int dimension);  // number of elements in the given dimension
  long totalSize();  // total number of elements in this tensor
  DoubleTensor slice(int... indices);  // returns a slice of this tensor
  DoubleTensor slice(Object... indices);  // returns a slice of this tensor, using various types of indices
  Iterable<DoubleTensor> elements();  // iterates through the elements of the first axis of this tensor
  DoubleIterator scalars();  // iterates through the elements of a rank-1 tensor

  // Read operations
  double get(int... indices);  // get the scalar value of a rank-0 tensor (or a slice of)
  DoubleStream stream(int... indices);  // get values of this tensor (or a slice of) as a stream
  void get(double[] array, int... indices);  // get values of this tensor (or a slice of) into `array`
  void get(DoubleBuffer buffer, int... indices);  // copy values of this tensor (or a slice of) into `buffer`
  void get(DoubleTensor array, int... indices);  // copy values of this tensor (or a slice of) into `tensor`
  void read(OutputStream ostream);  // read elements of this tensor across all dimensions into `ostream` 
  
  // Write operations
  void put(double value, int... indices);  // set the scalar value of this rank-0 tensor (or a slice of)
  void put(DoubleStream stream, int... indices);  // copy elements of `stream` into this tensor
  void put(DoubleBuffer buffer, int... indices);  // copy elements of `buffer` into this tensor
  void put(double[] array, int... indices);  // copy elements of `array` into this tensor
  void put(DoubleTensor tensor, int... indices);  // copy elements of `tensor` into this tensor
  void write(InputStream istream);  // write elements of this tensor across all dimensions from `istream`
}

class DoubleIterator {
  boolean hasMoreElements();  // true if there is more elements
  double get();  // returns the current element and increment position
  void put(double value);  // sets the current element and increment position
  void forEach(DoubleConsumer func);  // consume all remaining elements
  void onEach(DoubleSupplier func);  // supply all remaining elements
}
```
The `[Type]Tensor` interfaces tends to replicate indexation used with arrays in java. For example, to access a value in a 
rank-2 tensor in `y`, `x`:
```java
tensor.get(0, 0);  // returns scalar at y=0, x=0
tensor.put(10.0, 0, 0);  // sets scalar at y=0, x=0
tensor.stream(0);  // returns vector at y=0 as a stream
```
It is also possible to work with slices, which can cut a given tensor in any of its axis by the use 
of special selectors. For example, for a given rank-3 tensor in `z`, `y`, `x`:
```java
tensor.slice(0);  // returns matrix at z=0
tensor.slice(0, 0);  // returns vector at z=0, y=0 (on x axis)
tensor.slice(all(), 0, 0);  // returns vector at y=0, x=0 (on z axis)
tensor.slice(0, all(), 0);  // returns vector at z=0, x=0 (on y axis)
tensor.slice(all(), even());  // returns all (y,x) matrices but only retaining even rows (y)
```
Here's a (not exhaustive) list of special selectors:
* `all()`: matches all elements in the given dimension
* `incl(int i...)`: matches only elements at the given indices
* `excl(int i...)`: matches all elements but those at the given indices
* `range(int start, int end)`: matches all elements whose indices is between `start` and `end`
* `even()`, `odd()`: matches only elements at even/odd indices
* `mod(int m)`: matches only elements whose indices is a multiple of `m``

Finally, the `elements()` and `scalars()` methods simplifies sequential operation on a tensor. For example, 
for a given rank-3 tensor:
```java
double d = 0.0;
for (DoubleTensor vector: tensor.elements()) {
  vector.scalars().onEach(() -> d++);
}
for (DoubleTensor vector: tensor.elements()) {
  vector.scalars().forEach(System.out::println);
}
```
See the next section for some more detailed examples of the usage of such utilities.

### Creating Dense Tensors

Dense tensors are represented in TensorFlow as a multidimensional array serialized in a contiguous memory buffer.

Currently, when creating dense tensors, temporary buffers that contains the initial data are allocated by the user 
and copied to the tensor memory (see [this link](https://github.com/tensorflow/tensorflow/blob/a6003151399ba48d855681ec8e736387960ef06e/tensorflow/java/src/main/java/org/tensorflow/Tensor.java#L187) for example). 

Assuming that the shape of the tensor is predetermined, this data copy and additional memory allocation can be avoided by 
writing the data to the tensor memory directly. This only also only possible for datatypes whose length is fixed. For
variable-length datatypes (like strings), data must be first collected in order to compute the size in bytes of the tensor.

Following factories will be added to the `Tensors` class:
```java
public static Tensor<Float> createFloat(long[] shape, Consumer<FloatTensor> dataInit);
public static Tensor<Double> createDouble(long[] shape, Consumer<DoubleTensor> dataInit);
public static Tensor<Integer> createInt(long[] shape, Consumer<IntTensor> dataInit);
public static Tensor<Long> createLong(long[] shape, Consumer<LongTensor> dataInit);
public static Tensor<Boolean> createBoolean(long[] shape, Consumer<BooleanTensor> dataInit);
public static Tensor<UInt8> createUInt8(long[] shape, Consumer<ByteTensor> dataInit);
public static Tensor<String> createString(long[] shape, int elementLength, byte paddingValue, Consumer<StringTensor> dataInit);

public static Tensor<Float> create(FloatTensor data);
public static Tensor<Double> create(DoubleTensor data);
public static Tensor<Integer> create(IntTensor data);
public static Tensor<Long> create(LongTensor data);
public static Tensor<Boolean> create(BooleanTensor data);
public static Tensor<UInt8> create(ByteTensor data);
public static Tensor<String> create(StringTensor data);
```
The first block of factories create an empty tensor whose memory is then directly mapped to a `[Type]Tensor` by the
`Dense[Type]Tensor` class. This data structure is then passed to the `dataInit` function for initialization. 
Note that for strings, this is only possible if all elements can be padded to the same length.

The last block of factories allows the registration of tensors in TensorFlow from tensor data that has been allocated
and initialized by the user. This is useful when the user does not have all required information to use one factory
method of the previous block, in which case we need to collect all data before knowing what the size of the chunk of
contiguous memory we need to allocate for the dense tensor. This is often the case when working in variable-length datatypes,
like strings. 

An `ArrayList[Type]Tensor` could be an example of a naive implementation of a `[Type]Tensor` that can grow in size.

### Creating Sparse Tensors

A sparse tensor is a collection of 3 dense tensors (indices, values and dense shape). Actually there is no
other way in TF Java to allocate such tensor than allocating individually the 3 tensors and managing the
indices with their values manually.

We can simplify this process by following the same approach as dense tensors and adding those
factories to the `Tensors` class:
```java
public static SparseTensor<Float> createSparseFloat(long[] shape, int numValues, Consumer<FloatTensor> dataInit);
public static SparseTensor<Double> createSparseDouble(long[] shape, int numValues, Consumer<DoubleTensor> dataInit);
public static SparseTensor<Integer> createSparseInt(long[] shape, int numValues, Consumer<IntTensor> dataInit);
public static SparseTensor<Long> createSparseLong(long[] shape, int numValues, Consumer<LongTensor> dataInit);
public static SparseTensor<Boolean> createSparseBoolean(long[] shape, int numValues, Consumer<BooleanTensor> dataInit);
public static SparseTensor<UInt8> createSparseUInt8(long[] shape, int numValues, Consumer<ByteTensor> dataInit);
public static SparseTensor<String> createSparseString(long[] shape, int numValues, int elementLength, int paddingValue, Consumer<StringTensor> dataInit);
```
The same `[Type]Tensor` interfaces can be reused to initialize sparse data. In this case, the backing 
implementation class `Sparse[Type]Tensor` keeps track of elements that are set by writing down their index in a dense tensor 
and their value in another. `numValues` is the number of values actually set in the sparse tensor.

The returned type `SparseTensor<>` just act as a container for the 3 dense tensors that compose a sparse tensor,
where each of them can be retrieved individually to feed operands to an sparse operation like `SparseAdd`.

### Reading Tensor Data

Note that once created, Tensors are immutable and their data could not be modified anymore.

Currently, in order to read a tensor, the user needs to create a temporary buffer into which its data is copied. 
Once again, this data copy and additional memory allocation will be avoided by accessing the tensor buffer 
directly when reading its data.

The following methods will be added to the `Tensor` class:
```java
public FloatTensor floatData();
public DoubleTensor doubleData();
public IntTensor intData();
public LongTensor longData();
public BooleanTensor booleanData();
public ByteTensor uInt8Data();
public StringTensor stringData();
```
It is up to the user to know which of these methods should be called on a tensor of a given type, similar
to the `*Value()` methods of the same class.



- usage of tensors as index
- implementation for user-allocated tensor data (Buffered[Type]Tensor?)
- allow creation of sparse tensor from user-allocated tensor data
- ragged tensors (tensor of other tensors or other ragged tensors)


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

### Example of usage for NdArrays

```java
// Creating tensors and writing data

Tensor<Boolean> scalar = Tensors.createBoolean(new long[0], data -> {
  // Setting scalar value directly
  data.put(true);
});

scalar.rank();  // 0
scalar.size(0);  // error
scalar.totalSize();  // 1

Tensor<Integer> vector = Tensors.createInt(new long[]{4}, data -> {
  // Setting first elements from array and add last element directly
  data.put(new int[]{1, 2, 3}, 0);
  data.put(4, 3); 
});

vector.rank();  // 1
vector.size(0);  // 4
vector.totalSize();  // 4

Tensor<Float> matrix = Tensors.createFloat(new long[]{2, 3}, data -> {
  // Initializing data using iterators
  Iterator<FloatTensor> rows = data.elements();
  rows.put(new float[]{0.0f, 5.0f, 10.0f});  // inits data at the current row (0)
  FloatIterator secondRow = rows.scalars();  // returns a new cursor at the current row (1)
  secondRow.put(15.0f);  // inits each scalar of the second row individually...
  secondRow.put(20.0f);
  secondRow.put(25.0f);
});

matrix.rank();  // 2
matrix.size(0);  // 2
matrix.totalSize();  // 6

Tensor<Float> matrix3d = Tensors.createDouble(new long[]{2, 2, 3}, data -> {
  // Initialize all data from a flat 3d matrix: 
  // {{{10.0, 10.1, 10.2}, {11.0, 11.1, 11.2}}, {{20.0, 20.1, 20.1}, {21.0, 21.1, 21.2}}}
  data.put(DoubleStream.of(10.0, 10.1, 10.2, 11.0, 11.1, 11.2, 20.0, 20.1, 20.2, 21.0, 21.1, 21.2)); 
});

matrix3d.rank();  // 3
matrix3d.size(0);  // 2
matrix3d.totalSize();  // 12

Tensor<String> text = Tensors.createString(new long[]{-1}, data -> {
  // Initializing data from input stream, where `values.txt` contains following modified UTF-8 strings:
  // "in the town", "where I was", "born"
  data.write(new FileInputStream("values.txt"));
});

text.rank();  // 1
text.size(0);  // 3
text.totalSize();  // 3

// Reading data

scalar.get();  // true
vector.get(0);  // 1
matrix.get(0, 1);  // 5.0f
matrix3d.get(1, 1, 1);  // 21.1
text.get(2);  // "born"

IntBuffer buffer = IntBuffer.allocate(vector.numElements());
vector.get(buffer);  // 1, 2, 3, 4
matrix.stream();  // 0.0f, 5.0f, 10.0f, 15.0f, 20.0f, 25.0f

matrix3d.elements().forEach(c -> c.stream());  // [10.0, 10.1, 10.2, 11.0, 11.1, 11.2], 
                                               // [20.0, 20.1, 20.2, 21.0, 21.1, 21.2] 
text.scalars().forEach(System.out::println);  // "In the town", "where I was", "born"

// Working with slices

scalar.slice(0);  // error
vector.slice(0);  // {1} (rank-0)
matrix.slice(1, 1);  // {20.0f} (rank-0)

matrix3d.slice(0, 0);  // {10.0, 10.1} (rank-1)
matrix3d.slice(all(), 0);  // {{10.0, 10.1, 10.2}, {20.0, 20.1, 20.2}} (rank-2)
matrix3d.slice(all(), 0, 0);  // {10.0, 20.0} (rank-1)
matrix3d.slice(all(), 0, incl(0, 2));  // {{10.0, 10.2}, {20.0, 20.2}} (rank-2)
matrix3d.slice(all(), all(), excl(1));  // {{{10.0, 10.2}, {11.0, 11.2}}, {{20.0, 20.2}, {21.0, 21.2}}} (rank-3)

text.slice(tf.constant(1));  // {"where I was"} (rank-0 slice)
```

## Questions and Discussion Topics

Seed this with open questions you require feedback on from the RFC process.
