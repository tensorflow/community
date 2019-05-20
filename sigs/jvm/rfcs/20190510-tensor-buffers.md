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
public static Tensor<Float> createFloat(long[] shape, Consumer<FloatNdArray> dataInit);
public static Tensor<Double> createDouble(long[] shape, Consumer<DoubleNdArray> dataInit);
public static Tensor<Integer> createInt(long[] shape, Consumer<IntNdArray> dataInit);
public static Tensor<Long> createLong(long[] shape, Consumer<LongNdArray> dataInit);
public static Tensor<Boolean> createBoolean(long[] shape, Consumer<BooleanNdArray> dataInit);
public static Tensor<UInt8> createUInt8(long[] shape, Consumer<ByteNdArray> dataInit);
public static Tensor<String> createString(long[] shape, Consumer<StringNdArray> dataInit);
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

### Tensor Input/Output Buffers

There is a specific `*OutputBuffer` and `*NdArray` class for each datatype. This allow the user to
work with primitive Java types, which are less memory-consuming and provide better performances that working 
exclusively with their autoboxed version.

For simplicity, only the `Double` variant is presented:
```java
class DoubleNdArray {

  int rank();  // number of dimensions in this array
  long size(int dimension);  // number of elements in the given dimension
  long totalSize();  // total number of elements in this array
  DoubleNdArray slice(Object... indices);  // returns a slice of this array across one or more dimensions
  DoubleNdArrayIterator iterator();  // iterates through values of this array

  // Read operations
  double get(Object... indices);  // get this rank-0 array (or a slice of) as a scalar value
  DoubleStream stream();  // get values of this array as a stream
  void copyTo(DoubleBuffer buffer);  // copy values of this array into `buffer`
  void copyTo(DoubleNdArray array);  // copy values of this array into `array`
  
  // Write operations
  void put(double value, Object... indices);  // set the scalar value of this rank-0 array (or a slice of)
  void copy(DoubleStream stream);  // copy elements of `stream` into this array
  void copy(DoubleBuffer buffer);  // copy elements of `buffer` into this array
  void copy(double[] array);  // copy elements of `array` into this array
  void copy(DoubleNdArray array);  // copy elements of `array` into this array
}

class DoubleNdArrayIterator {
  boolean hasNext();  // true if there is more elements
  double get();  // return next element value and increment current position
  void put(double value);  // sets next element value and increment current position
}
```
See the next section for detailed examples of usage of these classes.

```java

// Create tensors

Tensor<Boolean> scalar = Tensor.createBoolean(new long[0], data -> {
  // Setting scalar value directly
  data.put(true);
});

Tensor<Integer> vector = Tensor.createInt(new long[]{4}, data -> {
  // Setting first elements from stream
  data.put(IntStream.rangeClosed(1, 3), 0);
  // Setting last element directly
  data.put(4, 3); 
});

Tensor<Float> matrix = Tensor.createFloat(new long[]{2, 3}, data -> {
  // Initializing data with cursors
  DoubleNdArrayCursor rows = data.cursor();
  rows.put(new int[] {0.0f, 5.0f, 10.0f});
  DoubleNdArrayCursor secondRow = rows.cursor();
  secondRow.put(15.0f);
  secondRow.put(20.0f);
  secondRow.put(25.0f);
});

Tensor<

// Show general info

scalar.rank();  // 0
scalar.size(0);  // error
scalar.totalSize();  // 1

vector.rank();  // 1
vector.size(0);  // 4
vector.totalSize();  // 4

matrix.rank();  // 2
matrix.size(0);  // 2
matrix.totalSize();  // 6

// Reading data

scalar.get();  // true
vector.get(0);  // 1
matrix.get(0, 1);  // 5.0f

vector.stream();  // 1, 2, 3, 4
matrix.stream();  // 0.0f, 5.0f, 10.0f, 15.0f, 20.0f, 25.0f

// Working with slices

scalar.slice(0);  // error
vector.slice(0).get();  // rank-0 array 

```






class DoubleNdArray {
  
  // Read
  DoubleNdArray slice(Object... indices);  // returns a slice of this array across one or more dimensions
  DoubleVector vector();  // get this rank-1 array as a vector
  double scalar();  // get the scalar value of this rank-0 array
  long numElements();  // number of elements in this array
  DoubleStream stream();  // returns elements of this array as a stream
  void copyTo(DoubleBuffer buffer);  // copy elements of this array into `buffer`
  void copyTo(DoubleNdArray array);  // copy elements of this array into `array`
  
  // Write
  void scalar(double value);  // set the scalar value of this rank-0 array
  void copyFrom(DoubleStream stream);  // copy elements of `stream` into this array
  void copyFrom(DoubleBuffer buffer);  // copy elements of `buffer` into this array
  void copyFrom(DoubleNdArray array);  // copy elements of `array` into this array
}

class DoubleVector {
  
  // Read
  double get(int idx);  // return `idx`th value of this vector
  double get();  // return next value in this vector and increment current position
  void get(double[] dst);  // return values from the current position in `dst` array and increment current position
  long position();  // returns current position
  void position(int idx);  // resets current position to `idx`
  void reset();  // resets current position to `0`
  long numElements();  // number of elements in this vector
  DoubleStream stream();  // returns values of this vector as a stream
  void copyTo(DoubleBuffer buffer);  // copy elements of this vector to `buffer`
  void copyTo(DoubleVector vector);  // copy elements of this vector into `vector`

  // Write
  void set(int idx);  // sets `idx`th value of this vector
  void set(double value);  // sets next value in this vector and increment current position
  void set(double[] array);  // set values from the current position and increment current position
  void copyFrom(DoubleStream stream);  // copy elements of `stream` into this vector
  void copyFrom(DoubleBuffer buffer);  // copy elements of `buffer` into this vector
  void copyFrom(DoubleVector vector);  // copy elements of `vector` into this vector
}
```




```java
class DoubleNdArray {
  DoubleNdArray slice(Object... indices);
  DoubleVector vector(Object... indices);
  double scalar(Object... indices);
}

class DoubleVector




  t.booleanData().
  
  
 
  t.floatData().at(0, 2, 1).get();
  t.floatData().at(0, 2).get(1);
  t.floatData().at(0).at(2).get(1);
  t.floatData().at(0).at(2).at(1).get();
  
  t.floatData().at(0, 2).vector().get(1);
  
  for (TensorData data : t.data().at(0, 2)) {
    System.out.println(data.get()); // prints (0, 2, 0), (0, 2, 1)
  }
  for (TensorData data : t.data().at(0)) {
    System.out.println(data.get(1)); // prints (0, 0, 1), (0, 1, 1), (0, 2, 1)
  }
  
  t.data().at(0).get(2); // fails, non-scalar
  t.data().at(0, 2).get(); // fails, non-scalar
  t.data().at(0, 2).get(array);
  t.data().at(0, 2).stream();
  
  
  
  t.data().get(); // for points only 
  t.data().get(0, 2, 1);
  t.data().at(0, 2).get(1);
  t.data().at(0).get(2, 1);
  t.data().at(0).at(2).get(1);
  
  for (TensorData data : t.data().at(0, 2)) {
    System.out.println(point.get()); // prints (0, 2, 0), (0, 2, 1)
  }
  for (TensorData data : t.data().slice(0)) {
    System.out.println(data.point(1).get()); // prints (0, 0, 1), (0, 1, 1), (0, 2, 1)
  }
  
 
 
  t.data().at(0, 2, 1).point().get();
  t.data().at(0, 2).vector().get(1);
  t.data().at(0).at(2).vector().get(1);
  t.data().at(0).at(2).at(1).point().get();
  
  for (TensorDataPoint point : t.data().at(0, 2).vector()) {
    System.out.println(point.get()); // prints (0, 2, 0), (0, 2, 1)
  }
  for (TensorData data : t.data().at(0)) {
    System.out.println(data.vector().get(1)); // prints (0, 0, 1), (0, 1, 1), (0, 2, 1)
  } 
  
  
  
  
  
  
  t.data().scalar(0, 2, 1);
  t.data().vector(0, 2).get(1);
  t.data().at(0, all()).vector(2).get(1);
  t.data().at(0, all()).scalar(2, 1);
  
  
  t.data().vector().get(0);
  t.data().get(0);
  t.data().
  
  
  long position();
  double get();
  double get(long index);
  void get(double[] dst);
  
  
  DoubleStream stream();
  void copyTo(DoubleBuffer buffer);
  
  data.at(i, h, w, 0) = image.pixel(w + (h * w)).red();
  
  
            Tensor.createFloat(batch.shape(784), data -> {
            for (float[] image : batch.images()) {
              data.put(image);
            }
          });
          Tensor.createFloat(batch.shape(28, 28, 3) data -> {
            for (int i = 0; i < batchSize; ++i) {
              Image image = batch.nextImage();
              for (int h = 0; h < image.height; ++h) {
                for (int w = 0; w < image.width; ++w) {
                  Pixel pixel = image.pixel(w + (h*w));
                  FloatVector vector = data.vector(i, h, w);
                  FloatVector vector = data.slice(i, h, w).vector();
                  vector.put(pixel.red());
                  vector.put(pixel.green());
                  vector.put(pixel.blue());
                  
                  float red = vector.get();
                  float green = vector.get();
                  float blue = vector.get();
                }
              }
            }
          });
          Tensor.createFloat(batch.shape(28, 28, 3) data -> {
            for (int i = 0; i < batchSize; ++i) {
              Image image = batch.nextImage();
              for (int h = 0; h < image.height; ++h) {
                for (int w = 0; w < image.width; ++w) {
                  Pixel pixel = image.pixel(w + (h*w));
                  FloatNdArray slice = data.slice(i, h, w);
                  slice.put(pixel.red());
                  slice.put(pixel.green());
                  slice.put(pixel.blue());
                }
              }
            }
          });
          Tensor.createFloat(Shape.scalar, data -> {
            data.scalar(42);
          });
}
```


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
