# TensorFlow Keras (Java)

| Status        |Proposed     |
:-------------- |:---------------------------------------------------- |
| **Current Implementation** | [dhruvrajan/tensorflow-keras-java](https://github.com/dhruvrajan/tensorflow-keras-java) |
| **Author(s)** | Dhruv Rajan (dhruv@krishnaprem.com) |
| **Sponsor**   | Karl Lessard (karl.lessard@gmail.com)                 |
| **Updated**   | 2019-10-23                                          |

## Objective

To create a high-level API for building, training, and experimentation 
with TensorFlow models in Java, supporting the syntax and
standards of [tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras).

The design of the API and summary written here is based on
the open source code and documentation of 
[tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras).

## Motivation

The Keras deep learning API is now a widely-accepted standard; greatly simplfying
model implementations, and enabling efficient experimentation as well as rapid
production of trained models with cross-platform framework support. It has seen
adoption across the TensorFlow ecosystem (TF 2.0, tfjs).

Currently, the TensorFlow Java client requires direct manipulation of the
graph operand-by-operand to build and train models. While
`Graph` objects can be seralized and loaded in Java, the lack of a standard `Model`
framework makes TensorFlow Java projects custom, and slow to develop.

## User Benefit

Users of TensorFlow-Java focused on experimentation, production, and depolyment
of deep learning models will benefit greatly from the Keras API's conciseness and portability.

Combining the wide support of the JVM ecosystem and development environment with a 
practical interface to TensorFlow Java will allow users to easily integrate deep learning
into their existing development and production workflows.

## Design Proposal

In building this framework, the goal is to closely match the behaviors and workflows
Keras supports in its Python API, with a well designed Java API that can be easily
integrated into existing projects.

### Overview

Keras is built around a few core abstractions which comprise much of what is needed to build
deep learning models. These include layers, models, optimizers, activations, losses, metrics,
regularizers, and initializers.

To make our code clearer and more manageable, we distinguish explicitly between two steps:
Graph building (even in eager mode, graph "definition"), and Graph execution.



### Layers
Layers represent arbitrary graph computations on tensors. They are composable; meant to be "stacked"
on each other to build neural networks. Thus they must encode *differentiable* functions (i.e. built
using TF Operations).

Layers are defined by (1) a set of weights, and (2) a differentiable "transform" function which uses
these weights to transform input tensor(s). For example, a `Dense` layer contains a kernel
weight matrix *w_k* and a bias weight matrix *w_b*, and transforms an input *x* using the
function *x* -> *f*(*w_k* x + *w_b*) for a user-specified activaition *f*.

#### Defining Layers
To define a new type of layer, users can override 3 methods from the base `Layer<T>` class.
These are: 

1. `Layer.build(Ops tf, Shape inputShape)`

    The `build` method is intended for creating tensorflow variables which
    will be required to compute the output of a layer. For example, for a
    Dense layer implementation, `build` is used to create the kernel (*w_k*) and
    bias (*w_b*) weight matrices.

    To add weight variables to a layer, subclasses overriding the `build` method must call
    the protected method `Layer.addWeight`; an internal wrapper for `tf.variable`.
    See the Dense layer implementation for an example of this usage.

    There are two build methods; one that should be overridden in any `Layer` subclass,
    and a public method which wraps the overridden method, to be called by external
    users of any Layer class. 
    ```java
    // Override this in Layer subclasses. To access dtype
    // within this method, call this.getDtype()
    protected abstract void build(Ops tf, Shape inputShape)

    // Call this method when using a created Layer object externally.
    // Requires the user to provide a dtype, as some tensorflow 
    // operations require the explicit specification of a dtype
    // (e.g. tf.variable(Shape shape, Class<T> dtype) and tf.placeholder(Class<T> dtype))
    public final void build(Ops tf, Shape inputShape, Class<T> dtype)
    ```
2. `Layer.call(Ops tf, Operand<T>... inputs)`

    The `call` method transforms input operands, and returns a new tensorflow operand, using
    tensorflow operations. For example, in a Dense layer, this method runs the computation
    *f*(*w_k* x + *w_b*).

    As for `build`, the `call` method overridden within a Layer subclass is wrapped by an `apply`
    method meant for use external to the layer class:
    ```java
    // Override this method in a layer subclass
    protected abstract Operand<T> call(Ops tf, Operand<T>... inputs)

    // Call this method when using a created Layer object externally.
    // Performs additional checks on the input length / sizes.
    public final Operand<T> apply(Ops tf, Operand<T>... inputs)
    ```
3. `Layer.computeOutputShape(Shape inputShape)`

    Ensuring tensor dimensions are correct through a sequence of transformations
    is difficult. The `computeOutputShape` method is used to verify that the transformed
    output of `apply` has the correct shape.

#### Creating Layers

Standard and user-defined layers might require arguments that are optional, and have defaults.
To allow for this, the standard layer types each define `Options` and `Builder` classes which allow
the specification of default and optional arguments. Since using an `Options` class for
every layer can be cumbersome, static helper functions for creating standard layers
are defined in the `Layers` class.

Creating a `Dense` layer:
```java
// Using Layer Options Builder
Layer<Float> dense = new Dense<>(128, Dense.Options.builder()
    .setActivation(Activations.relu)
    .setKernelInitializer(Initializers.randomNormal)
    .setBiasInitializer(Initializers.zeros)
    .build())

// Using static helper
Layer<Float> dense = Layers.dense(128, Activations.relu, Initializers.randomNormal, Initializers.zeros)
```


## Detailed Design

### Managing the Graph and Ops Accessors
To use TensorFlow, we must have access to a `Graph` to build and run computations. In Python, this is constructed implicitly. In Java, the `Graph` and `Ops` objects must be created explicitly. Thus, we
leave this (below) for the user to write, and allow the `Ops` object to be passed
throughout keras `Layer` construction calls to provide access to core TensorFlow operations.

```java
try(Graph graph = new Graph) {
    Ops tf = Ops.create(graph);
    
    // Keras code here.
}
```

When using the Keras API, models are "compiled" and "fitted" within this block, and the `tf` object is passed through the internal method calls.

### Keras Identifiers
Standard Keras objects are very often referenced just by name, and don't all have
to be created explicitly by the user.
For example, in Python, the strings `"sgd"` or `"adam"` can be used to create different
kinds of `Optimizer`s, `"relu"` and `"softmax"` stand for specific `Activations`, etc.

In Java, the supported standard types are enumerated in various enum classes: `Activations`, 
`Intializers`, `Optimizers`, etc., so these types become `Optimizers.sgd`, `Activations.relu`,
etc. Using enums instead of strings seems to result in greater visibility into all possible
types, and reduces error-proneness.

Each of these classes has a `select` method which convers the enum value to an object of the
target type. For example, `Optimizers.select(Optimizers.sgd)` returns an `Optimizer`. However
the end users shouldn't need to interact with this.

### Keras Configuration
Currently, the `~/.keras` directory is used for storing downloaded files, and user configurations 
(for now, just datafiles for standard datasets). This works well when the Java keras client is used
in isolation, but potentially could cause conflicts when a user is (likely) using the Python client
for other projects.

### Standard Datasets
A few standard datasets are defined in `org.tensorflow.keras.datasets` (currently,
FashionMNIST and MNIST). These classes download datasets in their original format,
and define batched loaders for loading tensors. 


### Options and Builder Classes

To enable methods and constructors to use optional arguments with defaults,
`Options` classes are defined (most notably for `Layer` construction,
and `compile` and `fit` configuration for models). The structure for an `Options` class
of object `KerasObject` is standardized as follows; if changes can be made to this structure
so that it is cleaner and easier to work with, please comment!

```java
class KerasObject {

    static {
        Arg1 arg1;
        Arg2 arg2;
        
        // Create KerasObject from Options class
        KerasObject kerasObject =
            new KerasObject(KerasObject.Options.builder()           
                .setArg1(arg1)
                .setArg2(arg2)
                .build());
    }

    public KerasObject(Options options) {
        // construct KerasObject
    }    

    public static class Options() {
        Arg1 arg1;
        Arg2 arg2

        private void defaults() {
            return new Builder()
                .setArg1(DEFAULT_ARG1)
                .setArg2(DEFAULT_ARG2)
                .build()
        }

        public Builder builder() {
            return defaults();
        }

        public static class Builder() {
            Options options;

            public Builder() {
                options = new Options()
            }

            public Builder setArg1(Arg1 arg1) {
                options.arg1 = arg1;
                return this;
            }
            public void setArg2(Arg2 arg2) {
                options.arg2 = arg2;
                return this;
            }

            public Options build() {
                return options;
            }
        }
    }
}
```

### Comparing Java API to Python
#### Original Python MNIST
```python
import tensorflow as tf

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu', kernel_initializer="random_normal", bias_initializer="zeros"),
  tf.keras.layers.Dense(10, activation='softmax', kernel_initializer="random_normal", bias_initializer="zeros")
])

model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

(X_train, y_train), (X_val, y_val) = tf.keras.datasets.load_mnist()
model.fit(X_train, y_train, val_data=(X_val, y_val), epochs=10, batch_size=100)
```

#### MNIST using Java API

```java
public class MNISTKeras {
    private static Model<Float> model;

    static {
        model = Sequential.of(
            Float.class, // dtype which will be passed through layer.build
            Layers.input(28, 28),
            Layers.flatten(),
            Layers.dense(128, Activations.softmax, Initializers.randomNormal, Initializers.zeros),
            Layers.dense(10, Activations.softmax, Initializers.randomNormal, Initializers.zeros)
        );
    }

    public static Model<Float> train(Model<Float> model) throws Exception {
        try (Graph graph = new Graph()) {
            Ops tf = Ops.create(graph);
            model.compile(tf, Optimizers.sgd, Losses.sparseCategoricalCrossentropy, Metrics.accuracy);

            Pair<GraphLoader<Float>, GraphLoader<Float>> loaders = MNIST.graphLoaders2D();
            // GraphLoader objects contain AutoCloseable `Tensor` objects.
            try (GraphLoader<Float> train = loaders.first(); GraphLoader<Float> test = loaders.second()) {
                model.fit(tf, train, test, 10, 100);
            }
        }

        return model;
    }

    public static void main(String[] args) throws Exception {
        train(model);
    }
}

```

#### MNIST using Scala wrapper around Java API
```scala
object MNISTKeras {

  val model: Model[JFloat] = Sequential.of[JFloat](
    classOf[JFloat],
    input(28, 28),
    flatten(),
    dense(128, activation = relu, kernelInitializer = randomNormal, biasInitializer = zeros),
    dense(10, activation = softmax, kernelInitializer = randomNormal, biasInitializer = zeros)
  )

  def train(model: Model[JFloat]): Model[JFloat] = {
      Using.resource(new Graph()) { graph => {
        implicit val tf: Ops = Ops.create(graph)
        model.compile(optimizer = sgd, loss = sparseCategoricalCrossentropy, metrics = List(accuracy))
  
        val (trainLoader, testLoader): (GraphLoader[JFloat], GraphLoader[JFloat]) = MNIST.graphLoaders2D()
        // GraphLoader objects contain AutoCloseable `Tensors`.
        Using.resources(trainLoader, testLoader) { (train, test) => {
          model.fit(train, test, epochs = 10, batchSize = 100)
        }}
      }}
  
      model
  }

  def main(args: Array[String]): Unit = {
    train(model.self)
  }
}
```


## Questions and Discussion Topics

Seed this with open questions you require feedback on from the RFC process.
