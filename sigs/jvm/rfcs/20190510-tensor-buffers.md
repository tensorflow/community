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

### Tensor Data Mapping

In TensorFlow, tensors are represented by multidimensional arrays stored in contiguous memory. To ease the task
of accessing data in such structure, following utilities will be provided by TF Java:

* `*Tensor`: An index-based interface mapping a tensor as a multidimensional array. 
* `*TensorCursor`: An sequence-based class to iterate in a `*Tensor`.

There will be a variant of those classes and interfaces for each tensor datatype supported in Java. This will allow 
users to work with Java primitive types, which tends to be less memory-consuming and provide better performances 
than their autoboxed equivalent.

While `*Tensor` classes try to replicate standard ND array in java using indices, `*TensorCursor` classes can be
used to easily iterates in their values. e.g.
```java
// replicates to matrix[x][y]
tensor.get(x, y);  

// replicates
// for (int x = 0; x < matrix.length; ++x) {
//  for (int y = 0; y < matrix[x].length; ++i) {
//    System.out.println(matrix[0][y]);
//  }
//}
tensor.cursor().whileNext(r -> 
  r.cursor().whileNext(System.out::println);
);
```  

For simplicity, only the classes for the `Double` variant are presented below:
```java
interface DoubleTensor {

  int rank();  // number of dimensions (or rank) of this tensor
  long size(int dimension);  // number of elements in the given dimension
  long totalSize();  // total number of elements in this tensor
  DoubleTensorCursor cursor();  // iterates through the elements of this tensor
  DoubleTensor slice(int... indices);  // returns a slice of this tensor across one or more of its dimensions
  DoubleTensor slice(Object... indices);  // returns a slice of this array across one or more of its dimensions, 
                                          // using various types of index

  // Read operations
  double get(int... indices);  // get this rank-0 tensor (or a slice of) as a scalar value
  DoubleStream stream(int... indices);  // get values of this tensor (or a slice of) as a stream
  void get(double[] array, int... indices);  // get values of this tensor (or a slice of) into `array`
  void get(DoubleBuffer buffer, int... indices);  // copy values of this tensor (or a slice of) into `buffer`
  void get(DoubleTensor array, int... indices);  // copy values of this tensor (or a slice of) into `tensor`
  void read(OutputStream ostream);  // read elements of this tensor into `ostream`, up to `totalSize()` 
  
  // Write operations
  void put(double value, int... indices);  // set the scalar value of this rank-0 tensor (or a slice of)
  void put(DoubleStream stream, int... indices);  // copy elements of `stream` into this tensor
  void put(DoubleBuffer buffer, int... indices);  // copy elements of `buffer` into this tensor
  void put(double[] array, int... indices);  // copy elements of `array` into this tensor
  void put(DoubleTensor tensor, int... indices);  // copy elements of `tensor` into this tensor
  void write(InputStream istream);  // write elements of this tensor from `istream`, up to `totalSize()`
}

class DoubleTensorCursor {

  DoubleTensor tensor();  // returns the tensor this cursor is iterating into
  long position();  // position of the current element in the initial sequence
  void position(long value);  // resets position of this cursor
  boolean hasNext();  // true if there is more elements
  DoubleTensor next();  // returns the current element and increment position
  DoubleTensorCursor cursor();  // returns a cursors to the current element and increment position
  void whileNext(Consumer<DoubleTensorCursor> func);
  
  // Read operations
  double get();  // get the scalar value of the current element and increment position
  void get(double[] array);  // copy values of the current element into `array` and increment position
  DoubleStream stream();  // get values of the current element as a stream and increment position
  void get(DoubleBuffer buffer);  // copy values of the current element into `buffer` and increment position
  void get(DoubleTensor tensor);  // copy values of the current element into `tensor` and increment position
  
  // Write operations
  void put(double value);  // set the scalar value of the current element and increment position
  void put(double[] array);  // copy elements of `array` into the current element and increment position
  void put(DoubleStream stream);  // copy elements of `stream` into the current element and increment position
  void put(DoubleBuffer buffer);  // copy elements of `buffer` into the current element and increment position
  void put(DoubleTensor tensor);  // copy elements of `tensor` into the current element and increment position
}
```
See the next section for some detailed examples of usage.

### Creating Dense Tensors

Currently, when creating dense tensors, temporary buffers that contains the initial data are allocated by the user 
and copied to the tensor memory (see [this link](https://github.com/tensorflow/tensorflow/blob/a6003151399ba48d855681ec8e736387960ef06e/tensorflow/java/src/main/java/org/tensorflow/Tensor.java#L187) for example). 

Assuming that the shape of the tensor is predetermined, this data copy and additional memory allocation can be avoided by 
writing the data to the tensor memory directly. This only also only possible for datatypes whose length is fixed. For
variable-length datatypes (like strings), data must be first collected in order to compute the size in bytes of the tensor.

Following factories will be added to the `Tensors` class:
```java
public static Tensor<Float> createFloat(long[] shape, Consumer<FloatNdArray> dataInit);
public static Tensor<Double> createDouble(long[] shape, Consumer<DoubleNdArray> dataInit);
public static Tensor<Integer> createInt(long[] shape, Consumer<IntNdArray> dataInit);
public static Tensor<Long> createLong(long[] shape, Consumer<LongNdArray> dataInit);
public static Tensor<Boolean> createBoolean(long[] shape, Consumer<BooleanNdArray> dataInit);
public static Tensor<UInt8> createUInt8(long[] shape, Consumer<ByteNdArray> dataInit);
public static Tensor<String> createString(long[] shape, int elementLength, byte paddingValue, Consumer<StringNdArray> dataInit);
public static Tensor<String> createString(long[] shape, Consumer<StringNdArray> dataInit);
```
The first n-1 factories create an empty `Tensor` whose memory is then directly mapped to a `NdArray` 
and passed to the `dataInit` function to initialize its data. Note that for strings, this is only possible if all elements 
can be padded to the same length.

The last factory allows the creation of string tensors with variable element length. The array passed to the
`dataInit` method will first buffered all string elements before allocating a tensor buffer of the right
size.

Once created, Tensors are immutable and their data could not be modified anymore.

### Creating Sparse Tensors

A sparse tensor is a collection of 3 dense tensors (indices, values and dense shape). Actually there is no
other way in TF Java to allocate such tensor than allocating individually the 3 tensors and managing the
indices with their values manually.

We can simplify this process by following the same approach as dense tensors and adding those
factories to the `Tensors` class:
```java
public static SparseTensor<Float> createSparseFloat(long[] shape, int numValues, Consumer<FloatNdArray> dataInit);
public static SparseTensor<Double> createSparseDouble(long[] shape, int numValues, Consumer<DoubleNdArray> dataInit);
public static SparseTensor<Integer> createSparseInt(long[] shape, int numValues, Consumer<IntNdArray> dataInit);
public static SparseTensor<Long> createSparseLong(long[] shape, int numValues, Consumer<LongNdArray> dataInit);
public static SparseTensor<Boolean> createSparseBoolean(long[] shape, int numValues, Consumer<BooleanNdArray> dataInit);
public static SparseTensor<UInt8> createSparseUInt8(long[] shape, int numValues, Consumer<ByteNdArray> dataInit);
public static SparseTensor<String> createSparseString(long[] shape, int numValues, int elementLength, int paddingValue, Consumer<StringNdArray> dataInit);

```
The same `*NdArray` interfaces can be reused to write (or read) sparse data. In this case, the backing 
implementation class keeps track of elements that are set by writing down their index in a tensor and their
value in another. `numValues` is the number of values actually set in the sparse tensor.

The returned type `SparseTensor` just act as a container for the 3 tensors that compose a sparse tensor,
where each of them can be retrieved individually to feed operands to an sparse operation like `SparseAdd`.

### Reading Tensor Data

Currently, in order to read a tensor, the user needs to create a temporary buffer into which its data is copied. 
Once again, this data copy and additional memory allocation will be avoided by accessing the tensor buffer 
directly when reading its data.

The following methods will be added to the `Tensor` class:
```java
public FloatNdArray floatData();
public DoubleNdArray doubleData();
public IntNdArray intData();
public LongNdArray longData();
public BooleanNdArray booleanData();
public ByteNdArray uInt8Data();
public StringNdArray stringData();
```
It is up to the user to know which of these methods should be called on a tensor of a given type, similar
to the `*Value()` methods of the same class.



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
  // Initializing data using cursors
  DoubleNdArrayCursor rows = data.cursor();
  rows.put(new int[]{0.0f, 5.0f, 10.0f});  // inits data at the current row (0)
  DoubleNdArrayCursor secondRow = rows.cursor();  // returns a new cursor at the current row (1)
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

matrix3d.cursor().whileNext(c -> c.stream());  // [10.0, 10.1, 10.2, 11.0, 11.1, 11.2], 
                                               // [20.0, 20.1, 20.2, 21.0, 21.1, 21.2] 
text.cursor().whileNext(c -> System.out.println(c.get()));  // "In the town", "where I was", "born"

// Working with slices

scalar.slice(0);  // error
vector.slice(0);  // {1} (rank-0)
matrix.slice(1, 1);  // {20.0f} (rank-0)

matrix3d.slice(0, 0);  // {10.0, 10.1} (rank-1)
matrix3d.slice(all(), 0);  // {{10.0, 10.1, 10.2}, {20.0, 20.1, 20.2}} (rank-2)
matrix3d.slice(all(), 0, 0);  // {10.0, 20.0} (rank-1)
matrix3d.slice(all(), 0, only(0, 2));  // {{10.0, 10.2}, {20.0, 20.2}} (rank-2)
matrix3d.slice(all(), all(), skip(1));  // {{{10.0, 10.2}, {11.0, 11.2}}, {{20.0, 20.2}, {21.0, 21.2}}} (rank-3)

text.slice(tf.constant(1));  // {"where I was"} (rank-0 slice)
```

## Questions and Discussion Topics

Seed this with open questions you require feedback on from the RFC process.
