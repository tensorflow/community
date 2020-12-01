# C++ Gradients 

| Status        | Proposed       |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [335](https://github.com/tensorflow/community/pull/335) |
| **Author(s)** | Saurabh Saxena (Google) |
| **Sponsor**   | Rohan Jain (Google)                 |
| **Updated**   | 2020-12-01                                           |


## Objective

We propose performing gradient computation entirely in C++. This aligns with TensorFlow Core team’s vision of providing first-class C++ APIs for building ML models. We mainly focus on reverse-mode autodiff here and leave forward-mode AD as future work. A C API for this implementation is also left as future work but we imagine it to be a straightforward wrapper over the provided C++ APIs.


## Motivation

**Performance**: Running op gradient functions in Python means we incur python overheads for a transformation pass that could entirely be done in C/C++. This most severely affects eager mode training of compute-light pipelines as we show in the performance impact section.

**Support C++ training**: Lack of autodiff support in C/C++ could be detrimental to high performance training pipelines at Google scale. This has often come up in discussions internally.

**Autodiff in non-python TF**: Relying on python gradient functions means we cannot support eager-mode autodiff from other language front-ends such as Java.

In addition, we try to address some shortcomings of the current GradientTape design:



1. The tape is currently in-charge of filling zeros for missing incoming gradients so that the gradient functions do not need any None-grad checks. This is sub-optimal since the tape now needs to keep around additional metadata(or actual tensors) from the forward pass around in order to be able to create these zeros. To prevent excessive memory usage we have a [hand-curated](https://github.com/tensorflow/tensorflow/blob/dfe6a8ea3725f57be1eaffc2d38c55c18cbd287a/tensorflow/c/eager/tape.h#L630) list of ops with indices of incoming gradients that are allowed to be None. This list is highly likely incomplete and is hard to keep up to date. We address this by moving this logic from the tape to individual gradient functions. Since most ops have a single output this is usually a no-op. 
2. We have a similar problem with inputs/outputs of the forward op. Since the tape does not know which of the op’s inputs/outputs the gradient function might require, it will need to [keep all tensors](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/python/eager/pywrap_tfe_src.cc;l=3196;drc=68d2777a170c0448ba9d853ef0f7327bdba1df2f) around which can be prohibitive. To limit the memory usage here we [auto-generate](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/python/eager/pywrap_gradient_exclusions.cc) a list of ops with indices of inputs/outputs that can be dropped by parsing ASTs of gradient functions. This again indicates that this should be handled on a per-op level. We address this by allowing individual gradient function instances to incref needed tensors from the forward pass.


## Design Overview

The gradients infrastructure will be built on top of the [abstract interfaces](https://cs.opensource.google/search?q=f:abstract_(context%7Coperation%7Ctensor_handle).h&sq=&ss=tensorflow%2Ftensorflow:tensorflow%2F) for op execution which provide a backend agnostic way of tracing and executing ops. We provide APIs for authoring `GradientFunction`s and registering them into a `GradientRegistry` for name based lookup. We provide a gradient `Tape` API that is close to python’s tf.GradientTape and shares most of the implementation with the existing tape.


## Detailed Design


### APIs


#### GradientFunction

An op’s gradient is defined by subclassing `GradientFunction`


```
class GradientFunction {
 public:
  virtual Status Compute(
      AbstractContext* ctx,
      absl::Span<AbstractTensorHandle* const> grad_outputs,
      absl::Span<AbstractTensorHandle*> grad_inputs) = 0;
  virtual ~GradientFunction() {}
};
```


`GradientFunction::Compute` receives gradients wrt op’s outputs in `grad_outputs` and needs to populate gradients wrt op’s inputs in `grad_inputs`. This is the same signature we use for authoring python gradients with the addition of an `AbstractContext`, which provides an API creating operations (eagerly or traced). In python this context is stored in a global variable and is implicitly captured. For the C++ API we chose to pass this context explicitly.

The reason `GradientFunction` is a class and not a callable is so that each op’s gradient function can keep the necessary state needed from forward pass for the gradient computation (see `ExpGradientFunction` below for an example).

Examples:


```
class AddGradientFunction : public GradientFunction {
 public:
  Status Compute(AbstractContext* ctx,
                 absl::Span<AbstractTensorHandle* const> grad_inputs,
                 absl::Span<AbstractTensorHandle*> grad_outputs) override {
    // Tape never calls a gradient function if there are no incoming grads.
    DCHECK(grad_inputs[0]);
    grad_outputs[0] = grad_inputs[0];
    grad_outputs[1] = grad_inputs[0];

    grad_outputs[0]->Ref();
    grad_outputs[1]->Ref();
    return Status::OK();
  }
  ~AddGradientFunction() override {}
};

class ExpGradientFunction : public GradientFunction {
 public:
  explicit ExpGradientFunction(AbstractTensorHandle* exp) : exp_(exp) {
    exp->Ref();
  }
  Status ExpGradientFunction::Compute(
      AbstractContext* ctx, absl::Span<AbstractTensorHandle* const> grad_inputs,
      absl::Span<AbstractTensorHandle*> grad_outputs) {
    vector<AbstractTensorHandle*> conj_outputs(1);
    TF_RETURN_IF_ERROR(
        Conj(ctx, {exp_.get()}, absl::MakeSpan(conj_outputs), "Conj_Exp_Grad"));
    AbstractTensorHandlePtr conj_output_releaser(conj_outputs[0]);

    TF_RETURN_IF_ERROR(
        Mul(ctx, {conj_outputs[0], grad_inputs[0]}, grad_outputs, "Mul_Exp_Grad"));
    return Status::OK();
  }

  ~ExpGradientFunction() override {}

 private:
  AbstractTensorHandlePtr exp_;
};
```


**C++ gen\_ops**

Authoring gradient functions requires calling elementary ops in C++. Using low level op building APIs everywhere would be very verbose so we plan to auto-generate C++ APIs for op-building using registered OpDefs, similar to how we do for python. This is part of a larger TensorFlow core team’s effort to provide efficient python bindings which are light wrappers for C++ API calls. For now we have been handwriting [C++ op building functions](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/c/experimental/ops/) using the unified execution APIs. Once ready we will replace these with auto-generated C++ op functions.


#### GradientRegistry

We provide a registry to store the mapping from op type to factory functions that return the `GradientFunction` for an op’s instance. The factory function takes as input the `ForwardOperation`, which contains metadata from the forward operation, and returns a `GradientFunction`. This allows gradient function authors to control which inputs/outputs of the forward op to keep around by increasing the ref-count on `AbstractTensorHandle`.

Additionally, we provide a utility function `RegisterNotDifferentiable` to mark an op as non-differentiable. This can be used to implement `tf.no_gradient`. We also provide a `NotDifferentiableGradientFunction` which returns nullptr output gradients. This can be used to implement `tf.stop_gradient`.


```
struct ForwardOperation {
 public:
  string op_name;
  std::vector<AbstractTensorHandle*> inputs;
  std::vector<AbstractTensorHandle*> outputs;
  std::vector<int64> skip_input_indices;
  AttrBuilder attrs;
};

using GradientFunctionFactory = std::function<GradientFunction*(const ForwardOperation& op)>;

// Map from op name to a GradientFunctionFactory.
class GradientRegistry {
 public:
  Status Register(const string& op,
                  GradientFunctionFactory gradient_function_factory);
  Status Lookup(const ForwardOperation& op,
                std::unique_ptr<GradientFunction>* gradient_function) const;

 private:
  absl::flat_hash_map<string, GradientFunctionFactory> registry_;
};

class NotDifferentiableGradientFunction : public GradientFunction {
  Status Compute(
      AbstractContext* ctx,
      absl::Span<AbstractTensorHandle* const> grad_outputs,
      absl::Span<AbstractTensorHandle*> grad_inputs) override;
};
Status RegisterNotDifferentiable(GradientRegistry* registry, const string& op);
```


Examples:


```
GradientFunction* AddRegisterer(const ForwardOperation& op) {
  return new AddGradientFunction;
}

GradientFunction* ExpRegisterer(const ForwardOperation& op) {
  return new ExpGradientFunction(op.outputs[0]);
}
```



#### Tape

The API for C++ `Tape` is very similar to python’s `tf.GradientTape`. The implementation for this interface is almost entirely shared with the C++ tape in `c/eager/tape.h`. 


```
class Tape {
 public:
  explicit Tape::Tape(bool is_persistent);

  // Adds this tensor to the list of watched tensors.
  //
  // This is a no-op if the tensor is already being watched either from an
  // earlier call to GradientTape::Watch or being an output of an op with
  // watched inputs.
  void Watch(const AbstractTensorHandle* tensor);

  // Records an operation with given inputs and outputs
  // on the tape and marks all its outputs as watched if at
  // least one input of the op is watched and has a trainable dtype.
  // op_name is optional and is used for debugging only.
  void RecordOperation(absl::Span<AbstractTensorHandle* const> inputs,
                       absl::Span<AbstractTensorHandle* const> outputs,
                       GradientFunction* gradient_function,
                       const string& op_name = "");

  // Returns whether any tensor in a list of tensors is being watched and has
  // a trainable dtype.
  bool ShouldRecord(absl::Span<const AbstractTensorHandle* const> tensors) const;

  // Unwatches this tensor on the tape. Mainly used for cleanup when deleting
  // eager tensors.
  void DeleteTrace(const AbstractTensorHandle*);

  // Computes the gradient and stores the result in result.
  // Raises an error if result is not the same length as sources.
  Status ComputeGradient(
      AbstractContext* ctx, absl::Span<AbstractTensorHandle* const> targets,
      absl::Span<AbstractTensorHandle* const> sources,
      absl::Span<AbstractTensorHandle* const> output_gradients,
      absl::Span<AbstractTensorHandle*> result);

  // Returns whether the tape is persistent, i.e., whether the tape will hold
  // onto its internal state after a call to ComputeGradient.
  bool IsPersistent() const;
};
```


Example:


```
// Computes
// y = tf.exp(inputs[0])
// outputs = grad(y, inputs)
Status ExpGradModel(AbstractContext* ctx,
                    absl::Span<AbstractTensorHandle* const> inputs,
                    absl::Span<AbstractTensorHandle*> outputs,
                    const GradientRegistry& registry) {
  Tape tape(/*persistent=*/false);
  tape.Watch(inputs[0]); 
  std::vector<AbstractTensorHandle*> exp_outputs(1);
  TF_RETURN_IF_ERROR(ops::Exp(ctx, inputs, absl::MakeSpan(exp_outputs), "Exp"));
  std::unique_ptr<GradientFunction> gradient_function;
  ForwardOperation forward_op;
  forward_op.op_name = “Exp”;
  forward_op.inputs.push_back(inputs[0]);
  forward_op.outputs.push_back(exp_outputs[0]);
  TF_RETURN_IF_ERROR(registry.Lookup(forward_op, &gradient_function));
  tape.RecordOperation(inputs, exp_outputs, gradient_function.release());
  TF_RETURN_IF_ERROR(tape.ComputeGradient(ctx,
                                           /*targets*/exp_outputs,
                                           /*sources=*/inputs,
                                           /*output_gradients=*/{},
                                           /*result=*/outputs));
  for (auto exp_output : exp_outputs) {
    exp_output->Unref();
  }
  return Status::OK();
}
```


**Some details on memory management**

`AbstractTensorHandle` provides `Ref` and `Unref` methods which can be used to manage its lifecycle. Gradient functions and the tape follow these guidelines for memory safety:



1. The tape releases tensors returned from gradient functions after they have been consumed.
2. Gradient functions which pass-through incoming gradients e.g. AddGradientFunction in the example above, should Ref each copy of the incoming grad since the tape is free to release the incoming grads passed to AddGradientFunction after it has been called.
3. It is the user’s responsibility to release tensors that are passed in as default gradients in the call to ComputeGradients.
4. More trivially, a gradient function should release any intermediate tensor that is created by it but not returned as an output gradient. This includes outputs of intermediate operations and operations added to the forward pass for reductions e.g. shape computation.

If manual management of ref-counts becomes too cumbersome we could consider adding a stack allocated wrapper for AbstractTensorHandle\* that manages its life-cycle. However, this change should probably happen at the level of the unified execution APIs.


### Use-cases


#### tf.custom\_gradient

A custom `GradientFunction` for a set of inputs/outputs can be registered using `Tape::RecordOperation` similar to a gradient function looked up from the gradient registry.

Example:


```
class CustomGradientFunction: public GradientFunction {
 public:
  Status ExpGradientFunction::Compute(
      AbstractContext* ctx, absl::Span<AbstractTensorHandle* const> grad_inputs,
      absl::Span<AbstractTensorHandle*> grad_outputs) {
    // Populate grad_outputs.
    return Status::OK();
  }

  ~CustomGradientFunction() override {}
};

Status ExpWithCustomGrad(AbstractContext* ctx,
                         absl::Span<AbstractTensorHandle* const> inputs,
                         absl::Span<AbstractTensorHandle*> outputs) {
  Tape tape(/*persistent=*/false);
  tape.Watch(inputs[0]); 
  std::vector<AbstractTensorHandle*> exp_outputs(1);
  TF_RETURN_IF_ERROR(ops::Exp(ctx, inputs, absl::MakeSpan(exp_outputs), "Exp"));
  std::unique_ptr<GradientFunction> gradient_function(new CustomGradientFunction);
  tape.RecordOperation(inputs, exp_outputs, gradient_function.release());
  TF_RETURN_IF_ERROR(tape.ComputeGradient(ctx,
                                          /*targets*/exp_outputs,
                                          /*sources=*/inputs,
                                          /*output_gradients=*/{},
                                          /*result=*/outputs));
  for (auto exp_output : exp_outputs) {
    exp_output->Unref();
  }
  return Status::OK();
}
```



#### tf.recompute\_grad

`tf.recompute_grad` is an application of `tf.custom_gradient` where we do not record the forward pass on the tape so that we are not holding on to forward pass tensors in memory. (In `tf.custom_gradient` we allow recording the forward pass on the tape in order for higher-order derivatives to work for cases where the custom gradient function uses intermediate tensors from the forward pass.) This is implemented by executing the forward pass outside the tape (managed by a higher layer) and registering a gradient function that re-runs the forward pass and computes gradients. The same behavior can be achieved using this tape.


#### Nested tapes and higher-order derivatives

Higher order derivatives are computed by either using a persistent tape or by tracing computations on nested tapes. The nesting is managed by a higher layer. This can be implemented using this tape.


#### Skipping gradients for certain op inputs (skip\_input\_indices)

A [small set](https://cs.opensource.google/search?q=f:py$%20skip_input_indices&sq=&ss=tensorflow%2Ftensorflow) of python gradient functions have been optimized to not return gradients for inputs which are not tracked under the tape. This is beneficial in eager mode where unneeded gradients cannot be pruned during execution. In the C++ tape, we support this by providing a `skip_input_indices` field on the `ForwardOperation` which stores the list of input indices which are either not watched or have an untrainable dtype. 


#### Automatic variable tracking

In python, if a variable is accessed inside a `tf.GradientTape`s scope it is automatically tracked, i.e. `Tape::Watch` is called for the `DT_RESOURCE` tensor backing the variable on behalf of the user. For now we will leave this out as a higher layer feature and require that variable handles are explicitly tracked by a higher layer. We can revisit this later if needed.


#### tf.function and functional control flow gradients [out of scope for now]

Eventually we plan to implement tf.function and functional control flow gradients in C++ but for now we will leave them in python and directly call the python gradient function.


#### IndexedSlices

Gradient function of a [gather](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/python/ops/array_grad.py;l=582;drc=d724cdbce69862cbb80617dd6573baa83bd3e819) returns `IndexedSlices` for efficiency. We need to support `IndexedSlices` as part of the input and output gradients of a gradient function. Currently there is no good C++ representation for these. One possible representation would be to wrap the component tensors in an `IndexedSlicesTensorHandle` that subclasses `AbstractTensorHandle`. This way IndexedSlices would be transparent to the tape. The C++ gen ops can choose to handle `IndexedSlices` appropriately or simply densify them by calling a C++ equivalent of `convert_to_tensor`.


```
class IndexedSlicesTensorHandle : public AbstractTensorHandle {
 protected:
  explicit IndexedSlicesTensorHandle(AbstractTensorHandle* values,
                                     AbstractTensorHandle* indices,
                                     const PartialTensorShape& dense_shape)
      : AbstractTensorHandle(kIndexedSlices),
        values_(values),
        indices_(indices),
        dense_shape_(dense_shape) {
    values->Ref();
    indices->Ref();
  }
  
  // Returns tensor dtype.
  DataType DataType() override {
    return values_->DataType();
  }
  // Returns tensor shape.
  Status Shape(PartialTensorShape* shape) override {
    *shape = dense_shape_;
  }

 public:
  // For LLVM style RTTI.
  static bool classof(const AbstractTensorHandle* ptr) {
    return ptr->getKind() == kIndexedSlices;
  }
  ~IndexedSlicesTensorHandle() {
    values_->Unref();
    indices_->Unref();
  }
 private:
  AbstractTensorHandle* values_;
  AbstractTensorHandle* indices_;
  PartialTensorShape dense_shape_;
}
```



## Performance Impact

Because gradient operations will no longer be run in Python, we expect the python framework overhead for usual NN models to be halved. This  should substantially improve both the graph building time and time to compute gradients in eager mode. 

[Initial tests](https://github.com/tensorflow/tensorflow/commit/ea83ee9a8297a742d4b95e34f9fe5e1de46befe9) on some MLPs of various depths for MNIST show ~2x speedup in eager mode.


## Implementation plan


#### Framework

The framework is a fairly lightweight implementation of the existing Tape interface in `c/eager/tape.h` which was already templated to support different C++ types for gradient functions and tensors. We have been making necessary improvements to the base framework to support this project, e.g., moving [default zeros creation logic](https://cs.opensource.google/tensorflow/tensorflow/+/ee95d88c4eb92311a8c57a8f78378235e1909d08) from the tape to respective gradient functions. 


#### Gradient functions

We plan to implement gradient functions under `tensorflow/c/gradients`. As a proof-of-concept we implemented an MLP for MNIST using an experimental python binding (see python/framework/experimental/tape.py). For that we implemented gradient functions for MatMul, Add, ReLu and Softmax. We are currently working on implementing gradient functions needed for ResNet50. 

We further plan to publish a guide for inviting contributions and setup a spreadsheet or some such for tracking completeness.


#### Python rollout

We plan to rollout C++ gradient functions incrementally. We will port the existing pybind C++ tape to use the new tape implementation. The `GradientFunction` for ops with registered C++ gradients will directly be called. For others,  we will simply register a `GradientFunction` that calls the python gradient function.

## Acknowledgements

@alextp motivated the design and provided an initial prototype for this project. @amturati implemented various gradient functions to get a MLP training on MNIST. @vnvo2409 has been working on making framework improvements and further implementing C++ gradient functions.
