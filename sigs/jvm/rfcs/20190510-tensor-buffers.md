# Java Tensor NIO

| Status        | Proposed       |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Karl Lessard (karl@kubx.ca) |
| **Updated**   | 2019-05-10                                           |

## Objective

Simplify and improve performances of creating and reading tensor data in Java with non-blocking I/O
access to their native buffers.

## Motivation

Currently, the most common way to create tensors in Java is by invoking one of the
factory methods exposed by the [`Tensors`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/java/src/main/java/org/tensorflow/Tensors.java)
class. While the signature of those methods are pretty elegant, by accepting a simple multidimensional Java array 
as an argument, they make heavy use of reflection techniques to extract the shape and the size of the tensors 
to allocate and result in multiple data copies. This results in poor performances, as discussed in [this issue](https://github.com/tensorflow/tensorflow/issues/8244).

Reading tensor data uses [similar techniques](https://github.com/tensorflow/tensorflow/blob/c23fd17c3781b21bd3309faa13fad58472c78e93/tensorflow/java/src/main/java/org/tensorflow/Tensor.java#L449) and faces the same performance issues. It can also result in multiple data copies, which
is not convenient when dealing with large tensors (e.g. see [`writeTo()`](https://github.com/tensorflow/tensorflow/blob/c23fd17c3781b21bd3309faa13fad58472c78e93/tensorflow/java/src/main/java/org/tensorflow/Tensor.java#L483) methods, for example).

Now that it is possible to run eagerly TensorFlow operations in Java, it is imperative that the 
I/O access to the tensor memory is efficient enough to let the users peek at the data without a significant
performance hit.

By developing a new set of I/O utility classes, we can allow the user to access directly the tensor native
buffers, avoiding data copies, while still preventing mistakes that could break their internal format. Also, 
those utilities can help to improve the manipulation of n-dimension data structures in Java in general.

## User Benefit

Users who are actually using factories and read/write methods from `Tensors/Tensor` classes should notice great 
performance improvements after switching to the new set of I/O utilities.

Users executing their operations in an eager environment will also find it very useful and efficient 
to access directly the tensor data without the need of copying their buffer.

In addition, we will take advantage of those new utilities to help the creation of two other types of tensors in
TensorFlow (sparse and ragged) that is not explicitly supported right now by the Java client.

## Design Proposal

*Note: This design proposal assumes that we run in a Java >= 8 environment, which is not the case with
current client that is configured to compile in Java 7, for supporting older Android devices. We need to confirm
with Android team if it is ok now to switch to Java 8 if the TF Java remains in the main repository.*

### Tensor I/O Utilities

A new utility library (`org.tensorflow:tensorflow-util`) will be distributed with the TensorFlow Java client and
will include a series of interfaces and classes that improve read and write operations in a tensor data structure,
normally represented as a multidimensional arrays.

The <code><i>Type</i>Tensor</code> interfaces are the center of this new set of utilities (should not to be confused 
with the existing `Tensor<>` class in TF Java, which is in fact just a symbolic handle to a tensor allocated by
TensorFlow). For each tensor datatype supported in Java, a <code><i>Type</i>Tensor</code> interface variant is 
provided, allowing users to work with Java primitive types which tends to be less memory-consuming and 
provide better performances than their autoboxed equivalent.

For readability, only the `Double` variant of this interface is shown below:
```java
interface DoubleTensor {
  int rank();  // number of dimensions (or rank) of this tensor
  long size(int dimension);  // number of elements in the given dimension
  long totalSize();  // total number of elements in this tensor
  DoubleTensor slice(int... indices);  // returns a slice of this tensor
  DoubleTensor slice(TensorIndex... indices);  // returns a slice of this tensor, using various types of indices
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
The <code><i>Type</i>Tensor</code> interfaces support normal integer indexation, similar to Java arrays. 

Ex: let `tensor` be a matrix on `(x, y)`
```java
tensor.get(0, 0);  // returns scalar at x=0, y=0 (similar to array[0][0])
tensor.put(10.0, 0, 0);  // sets scalar at x=0, y=0 (similar to array[0][0] = 10.0)
tensor.stream(0);  // returns vector at x=0 as a stream
```
It is also possible to create slices of a tensor, to work with a reduced view of its elements. The first variant 
of `slice()` accept usual integer indices, to slice at a specific element in the tensor. The second variant 
accepts special indices, which offer more flexibility like iterating through the elements of a tensor on any 
of its axis or use values of another tensor as indices.

Here is a non-exhaustive list of special indices that could be possibly created. Each of them are exposed as static
methods in `TensorIndex`, which return an instance of the same class:
* `at(int i)`: match element at index `i`
* `all()`: matches all elements in the given dimension
* `incl(int i...)`: matches only elements at the given indices
* `excl(int i...)`: matches all elements but those at the given indices
* `range(int start, int end)`: matches all elements whose indices is between `start` and `end`
* `even()`, `odd()`: matches only elements at even/odd indices
* `mod(int m)`: matches only elements whose indices is a multiple of `m`
Note that `IntTensor` and `LongTensor` will also implement the `TensorIndex` interface, to allow indexation
using rank-0 or rank-1 tensors.

Ex: let `tensor` be a 3D matrix on `(x, y, z)`
```java
tensor.slice(0);  // returns matrix at x=0
tensor.slice(0, 0);  // returns vector at x=0, y=0 (on z axis)
tensor.slice(all(), at(0), at(0));  // returns vector at y=0, z=0 (on x axis)
tensor.slice(at(0), all(), at(0));  // returns vector at x=0, z=0 (on y axis)
tensor.slice(even());  // returns all (y,z) matrices for all even values of x
tensor.slice(scalar);  // return slice at x=scalar.get()
tensor.slice(vector);  // return slice at x=vector.get(0), y=vector.get(1)
tensor.slice(at(0), vector);  // return slice at x=0, y=vector.get(0), z=vector.get(1)
```
Finally, the `elements()` and `scalars()` methods simplifies sequential operation over the elements of a tensor,
avoiding the user to increment manually an iterator.

Ex: let `tensor` be a 3D matrix
```java
double d = 0.0;
for (DoubleTensor vector: tensor.elements()) {
  vector.scalars().onEach(() -> d++);
}
for (DoubleTensor vector: tensor.elements()) {
  vector.scalars().forEach(System.out::println);
}
tensor.slice(0).put(10.0f).put(20.0f).put(30.0f);
```
See the last section for some more usage examples.

### Creating Dense Tensors

Dense tensors are represented in TensorFlow as a multidimensional array serialized in a contiguous memory buffer.

Currently, when creating dense tensors, temporary buffers that contains the initial data are allocated by the user 
and copied to the tensor memory (see [this link](https://github.com/tensorflow/tensorflow/blob/a6003151399ba48d855681ec8e736387960ef06e/tensorflow/java/src/main/java/org/tensorflow/Tensor.java#L187) for example). 

Assuming that the shape of the tensor is predetermined, this data copy and additional memory allocation can be avoided by 
writing the data to the tensor memory directly. This only also only possible for datatypes whose length is fixed. For
variable-length datatypes (like strings), data must be first collected in order to compute the size in bytes of the tensor.

Following factories will be added to the `Tensors` class:
```java
public static Tensor<Float> denseFloat(long[] shape, Consumer<FloatTensor> dataInit);
public static Tensor<Double> denseDouble(long[] shape, Consumer<DoubleTensor> dataInit);
public static Tensor<Integer> denseInt(long[] shape, Consumer<IntTensor> dataInit);
public static Tensor<Long> denseLong(long[] shape, Consumer<LongTensor> dataInit);
public static Tensor<Boolean> denseBoolean(long[] shape, Consumer<BooleanTensor> dataInit);
public static Tensor<UInt8> denseUInt8(long[] shape, Consumer<ByteTensor> dataInit);
public static Tensor<String> denseString(long[] shape, int elementLength, byte paddingValue, Consumer<StringTensor> dataInit);
public static Tensor<String> denseString(long[] shape, Consumer<StringTensor> dataInit);
```
All factories except the last create an empty tensor whose memory is then directly mapped to a 
<code><i>Type</i>Tensor</code>. This data structure is then passed to the `dataInit` function for 
initialization. Note that for strings, this is only possible if all elements can be padded to the same length.

The last factory allow the creation of tensors of variable-length strings. The `StringTensor` passed in parameter to
`dataInit` first collects all data before allocating a tensor buffer of the right size and initializing its data.

### Creating Sparse Tensors

A sparse tensor is a collection of 3 dense tensors (indices, values and dense shape). Actually there is no
other way in TF Java to allocate such tensor than allocating and manipulating individually the 3 tensors.

We can simplify this process by following the same approach as dense tensors and adding those
factories to the `Tensors` class:
```java
public static SparseTensor<Float> sparseFloat(long[] shape, int numValues, Consumer<FloatTensor> dataInit);
public static SparseTensor<Double> sparseDouble(long[] shape, int numValues, Consumer<DoubleTensor> dataInit);
public static SparseTensor<Integer> sparseInt(long[] shape, int numValues, Consumer<IntTensor> dataInit);
public static SparseTensor<Long> sparseLong(long[] shape, int numValues, Consumer<LongTensor> dataInit);
public static SparseTensor<Boolean> sparseBoolean(long[] shape, int numValues, Consumer<BooleanTensor> dataInit);
public static SparseTensor<UInt8> sparseUInt8(long[] shape, int numValues, Consumer<ByteTensor> dataInit);
public static SparseTensor<String> sparseString(long[] shape, int numValues, int elementLength, int paddingValue, Consumer<StringTensor> dataInit);
public static SparseTensor<String> sparseString(long[] shape, int numValues, Consumer<StringTensor> dataInit);
```
The same <code><i>Type</i>Tensor</code> interfaces can be used to initialize sparse data. In this case, the backing implementation classes keep track of the elements that are set by writing down their 
index in a dense tensor and their value in another. `numValues` is the number of values actually set in the 
sparse tensor.

Like with dense tensors, in case of a tensor of variable-length strings, the data will be first collect before the 
tensor buffers are allocated and initialized.

### Creating Ragged Tensors

A ragged tensor is a tensor that is composed of one or more ragged or dense tensors. Ragged tensors allow
users to work with variable-length elements in any dimension (except of the first). 

To support those tensors as well, following factories can be added to the `Tensors` class:
```java
public static RaggedTensor<Float> raggedFloat(long[] shape, Consumer<FloatTensor> dataInit);
public static RaggedTensor<Double> raggedDouble(long[] shape, Consumer<DoubleTensor> dataInit);
public static RaggedTensor<Integer> raggedInt(long[] shape, Consumer<IntTensor> dataInit);
public static RaggedTensor<Long> raggedLong(long[] shape, Consumer<LongTensor> dataInit);
public static RaggedTensor<Boolean> raggedBoolean(long[] shape, Consumer<BooleanTensor> dataInit);
public static RaggedTensor<UInt8> raggedUInt8(long[] shape, Consumer<ByteTensor> dataInit);
public static RaggedTensor<String> raggedString(long[] shape, Consumer<StringTensor> dataInit);
```
All ragged dimensions in the tensor have a value of `-1` in the `shape` attribute. Since ragged tensors always 
work with variable-length values, data must be first collected before the tensor buffer is allocated 
and initialized. 

It is also important to note that in contrary to other tensors which fail if writing an element out of
the bound of the current shape, ragged dimensions will automatically grow as elements are inserted to the tensor.

### Reading Tensor Data

Once created, Tensors are immutable and their data could not be modified anymore. But right now, to read
data from a tensor, the user needs to create a temporary buffer into which its data is copied. Again, this 
data copy and additional memory allocation can be avoided by accessing the tensor buffer 
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

The returned tensor data should only be used for read operations and all attempt to modify the data will result
in an error.

*Note: an alternative solution would be to split the TypeTensor interfaces and classes in two types: one for 
read-only operations and another inheriting from it for read & write. This gives more control at compile time to 
ensure that users won't attempt to modify a read-only tensor. But it will also increase the size and the 
complexity of the code in the utilities.*

### User-allocated Tensors

This RFC focuses on tensors allocated by the TensorFlow runtime library. But it prepares to solution to accept other
implementations of <code><i>Type</i>Tensor</code> that let the user allocate a tensor outside TensorFlow and 
read/write to its data (e.g. a <code>Array<i>Type</i>Tensor</code> could be backed with a standard Java 
array).

It would be possible then to create a tensor in TensorFlow based on the data from a user-allocated tensor, similar
to the data copy solution already present in the TF Java client. For example:
```java
public static Tensor<Float> create(FloatTensor data);
public static Tensor<Double> create(DoubleTensor data);
public static Tensor<Integer> create(IntTensor data);
public static Tensor<Long> create(LongTensor data);
public static Tensor<Boolean> create(BooleanTensor data);
public static Tensor<UInt8> create(ByteTensor data);
public static Tensor<String> create(StringTensor data);
```
The implementation of such tensors could (and should) be delivered by the utility library as it does not depend 
on any TensorFlow core types.

## Detailed Design

### Suggested class diagram (overview, double only)

![Class Diagram](images/20190510-tensor-data-nio-cd.png)

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
text.scalars().forEach(System.out::println);  // "in the town", "where I was", "born"

// Working with slices

scalar.slice(0);  // error
vector.slice(0);  // {1} (rank-0)
matrix.slice(1, 1);  // {20.0f} (rank-0)

matrix3d.slice(0, 0);  // {10.0, 10.1} (rank-1)
matrix3d.slice(all(), at(0));  // {{10.0, 10.1, 10.2}, {20.0, 20.1, 20.2}} (rank-2)
matrix3d.slice(all(), at(0), at(0));  // {10.0, 20.0} (rank-1)
matrix3d.slice(all(), at(0), incl(0, 2));  // {{10.0, 10.2}, {20.0, 20.2}} (rank-2)
matrix3d.slice(all(), all(), excl(1));  // {{{10.0, 10.2}, {11.0, 11.2}}, {{20.0, 20.2}, {21.0, 21.2}}} (rank-3)

text.slice(tf.constant(1));  // {"where I was"} (rank-0 slice)

// Sparse tensors

SparseTensor<Float> sparseTensor = Tensors.createSparseFloat(new long[]{2, 4}, 3, data -> {
  data.put(10.0f, 0, 0);
  data.put(20.0f, 0, 3);
  data.put(30.0f, 1, 1);
  data.put(40.0f, 2, 1);  // fails, index oob
});

sparseTensor.get(0, 0);  // 10.0f
sparseTensor.get(0, 1);  // 0.0f
sparseTensor.stream();  // [10.0f, 0.0f, 0.0f, 20.0f, 0.0f, 30.0f, 0.0f, 0.0f]

// Ragged tensors

RaggedTensor<Float> raggedTensor = Tensors.createRaggedFloat(new long[]{3, -1}, data -> {
  data.put(10.0f, 0, 0);    
  data.put(20.0f, 0, 1);
  data.put(30.0f, 0, 2); 
  data.put(40.0f, 1, 0);
  data.put(50.0f, 2, 0);
  data.put(60.0f, 2, 1);
});

raggedTensor.get(0, 1);  // 20.0f
raggedTensor.get(1, 0);  // 40.0f
raggedTensor.get(1, 1);  // fails, index oob
raggedTensor.elements().forEach(e -> e.stream());  // [10.0f, 20.0f, 30.0f], [40.0f], [50.0f, 60.0f]
```

## Questions and Discussion Topics

* Should we split the <code><i>Type</i>Tensor</code> into distinct interfaces for read-only and read-write tensors?
* Should we plan now user-allocated tensors or can we live with TensorFlow tensors only for now?
