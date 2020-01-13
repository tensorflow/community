# Single python code path for eager and graph

| Status        | Accepted      |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [184](https://github.com/tensorflow/community/pull/184) |
| **Author** | Saurabh Saxena (srbs@google.com) |
| **Sponsors**   | Alex Passos, Gaurav Jain                |
| **Updated**   | 2019-12-03                                           |


## Objective

This proposal discusses merging the graph building and eager op-dispatch code-paths in python and moving the FuncGraph capturing logic and gradient tape bookkeeping into C++.

## Motivation

### Graph building performance

Graph-building time performance has been a key bottleneck in enabling implementation of large models in TF2.

* Capturing external tensors: In analysis of graph-building time for [BERT](https://github.com/tensorflow/models/tree/master/official/nlp/bert) we found that ~20% time of building the body graph of a tf.while_loop is spent in `FuncGraph.capture`. We also extensively perform capturing when building gradients of functional ops since the backward function requires access to intermediate tensors of the forward function. This includes 2 parts, both of which we could potentially perform in C++.
  * [Creating](https://github.com/tensorflow/tensorflow/blob/6a70aa6d438259cabd23c09808db4cf51a2e5377/tensorflow/python/framework/func_graph.py#L1118) the placeholders. These can be many (154630 in BERT’s while loop). Building these in python means we incur the python op building overheads, Python->C SWIG costs and maintain the captures mapping in python.
  * [Copying](https://github.com/tensorflow/tensorflow/blob/6a70aa6d438259cabd23c09808db4cf51a2e5377/tensorflow/python/framework/func_graph.py#L1120) the handle data (for resources and variants). Handle data contains information about the shape and type of the _contained_ entity of a `DT_RESOURCE`/`DT_VARIANT` type. Copying handle data requires a python-c round-trip since the handle data is contained in either `EagerTensor._handle_data` (for EagerTensors) or `InferenceContext.output_handle_shapes_and_types` (for Graph tensors).
* Automatic control deps: We add control dependencies to a `tf.function`’s nodes as a post-processing step to make sure that any side-effects occur in program order. This can easily be done in C/C++.
*   Gradient Tape: The tape needs to keep track of the forward ops to build gradients (or actually compute the gradient in the case of forward-mode diff). This is currently triggered in gen_xyz_ops.py. We can move this to C++ as well.


### Cross-language support

There have been various [requests](https://github.com/tensorflow/tensorflow/issues/28195) for providing APIs for building `tf.function` and v2 control flow in non-python frontends. Moving capturing logic to the C/C++ layer is the first step towards enabling this. The full details for this will be fleshed out in follow-up proposals, however, we do analyse how this proposal addresses use-cases of `FuncGraph` later in this doc.


### Shape Inference

C++ shape inference in FuncGraphs fails if a shape tensor relies on the constant value of a captured placeholder because we do not have information about graph nesting available there. We currently work around this c6c1f2ff3bc979f420d8fffa2b6e02268f711bf6 by explicitly calling [maybe_set_static_shape](https://github.com/tensorflow/tensorflow/blob/15715cb2c8e877c18f8d969cc51a37ff26e8397b/tensorflow/python/ops/random_ops.py#L78) in Python because we have the graph hierarchy available there. One alternative @allenlavoie suggested was to replace the placeholders with their constant value tensors if possible, guarded by a size threshold but it was unclear what this threshold should be. Having information about the nested graphs and captures etc during shape inference could help avoid this problem.


### Consistent execution environments

(Contributed by @allenlavoie) We currently rely on Python exporting SavedModels which are compatible with Session-based execution, where the Session owns variable memory and it is retrieved by executing variable nodes with fixed names. TensorFlow Serving for example still uses Sessions. This compatibility mode is quite different than the 2.x Python eager execution memory model where the language bindings associate memory with variable objects, and is likely going to be a source of confusion and bugs. This effort lays necessary groundwork for implementing FuncGraph in C/C++ and hence brings us closer to executing SavedModels the same way during serving (from C++) that we execute them during development (TF 2.x Python).

References:

1. TODO(saxenasaurabh): Link to FuncGraph CUJs doc.

## Design Proposal

Basically we want to get rid of the graph-building part in gen_*_ops.py and get rid of gradient tape bookkeeping in both graph and eager modes. For example:


```diff
def batch_matrix_band_part(input, num_lower, num_upper, name=None):
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
- if tld.is_eager:
    try:  
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, tld.device_name, "BatchMatrixBandPart", name, 
        tld.op_callbacks, input, num_lower, num_upper)
      return _result
    except _core._FallbackException:
      try:  
        return batch_matrix_band_part_eager_fallback(
            input, num_lower, num_upper, name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
    except _core._NotOkStatusException as e: 
      _ops.raise_from_not_ok_status(e, name) 
- # Add nodes to the TensorFlow graph.
- _, _, _op, _outputs = _op_def_library._apply_op_helper(
-       "BatchMatrixBandPart", input=input, num_lower=num_lower,
-                              num_upper=num_upper, name=name)
- _result = _outputs[:]
- if _execute.must_record_gradient():
-   _attrs = ("T", _op._get_attr_type("T"))
-   _inputs_flat = _op.inputs
-   _execute.record_gradient(
-       "BatchMatrixBandPart", _inputs_flat, _attrs, _result)
- _result, = _result
- return _result

def batch_matrix_band_part_eager_fallback(input, num_lower, num_upper, name, ctx): 
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx)
  num_lower = _ops.convert_to_tensor(num_lower, _dtypes.int64)
  num_upper = _ops.convert_to_tensor(num_upper, _dtypes.int64)
  _inputs_flat = [input, num_lower, num_upper]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"BatchMatrixBandPart", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
- if _execute.must_record_gradient():
-   _execute.record_gradient(
-       "BatchMatrixBandPart", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result
```


1.  Graph building will implicitly happen in `TFE_Py_Execute` which is called from `xyz_eager_fallback`. 
1.  `TF_EagerContext` makes the call to `Tape.RecordGradient` so we no longer need to call it from Python.
1.  The Graph stack will be maintained in `TF_EagerContext` (see below) which includes the graph hierarchy and captures made from each graph.


## Detailed Design


### API
A high-level overview of the anticipated API.

**C API**


```
// TODO: control dependencies, auto control dependencies, callbacks

// A TF_EagerContext knows whether we're in eager mode or in graph mode, keeps
// track of gradient tapes, etc.
typedef struct TF_EagerContext TF_EagerContext;

TF_EagerContext* TF_NewEagerContext(TFE_Context* ctx);
void TF_DeleteEagerContext(TF_EagerContext* c);

// The context is executing eagerly if there are no graphs in the stack. We
// check when popping a graph from the stack that it is indeed the one we
// expected to avoid bugs.
int TF_EagerContextIsExecutingEagerly(TF_EagerContext* c);
void TF_EagerContextEnterGraph(TF_EagerContext* c, TF_Graph* g);
void TF_EagerContextExitGraph(TF_EagerContext* c, TF_Graph* g, TF_Status* s);
// Cleans up captures and other graph metadata in the eager context.
void TF_EagerContextDeleteGraph(TF_EagerContext* c, TF_Graph* g, TF_Status* s);

// A TF_TensorHandle is a union type of TFE_TensorHandle (eager tensor) and
// TF_Output (graph tensor).
typedef struct TF_TensorHandle TF_TensorHandle;

// Note: takes ownership of t.
TF_TensorHandle* TF_TensorHandleFromTensor(TFE_TensorHandle* t);
void TF_TensorHandleDecRef(TF_TensorHandle* t);
void TF_TensorHandleIncRef(TF_TensorHandle* t);
TFE_TensorHandle* TF_TensorHandleToTensor(TF_TensorHandle* t, TF_Status* s);
TF_Output TF_TensorHandleToGraphTensor(TF_TensorHandle* t, TF_Status* s);
int TF_TensorHandleHasValue(TF_TensorHandle* t);
TF_DataType TF_TensorHandleDataType(TF_TensorHandle* t);

// When in graph mode accessing a tensor from outside the graph will trigger
// capturing logic similar to what we have in FuncGraph. These methods let you
// inspect the capturing metadata before popping the graph from the graph stack.
int TF_EagerContextNumCaptures(TF_EagerContext* c, TF_Graph* g, TF_Status* s);
void TF_EagerContextCapturedValues(TF_EagerContext* c, TF_Graph* g,
                                   TF_TensorHandle** captures, TF_Status* s);
void TF_EagerContextCapturedPlaceholders(TF_EagerContext* c, TF_Graph* g,
                                         TF_Output* captures,
                                         TF_Status* s);

// Allows specifying a custom capturing function. To be use to implement
// custom capturing logic for tf.while_loop. `captured` must be in the current
// context graph.
typedef void(*CaptureCallback)(TF_EagerContext* c,
                               TF_Graph* source_graph,
                               TF_TensorHandle* source,
                               TF_TensorHandle** captured,
                               TF_Status* s);
void TF_EagerContextPushCaptureCallback(TF_EagerContext* c,
                                        CaptureCallback* callback,
                                        TF_Graph* graph, TF_Status* s);
void TF_EagerContextPopCaptureCallback(TF_EagerContext* c,
                                       TF_Graph* graph, TF_Status* s);

// Needed for updating the captured tensor in tf.function, tf.cond grad func, VariableTensor.
void TF_EagerContextUpdateCaptureForPlaceholder(TF_EagerContext* c, TF_Graph* g,
                                         TF_TensorHandle* placeholder,
                                         TF_TensorHandle* new_capture,
                                         TF_Status* s);

// TF_OutputList just lets us not specify the number of outputs of an operation
// beforehand. This forces a memory allocation in the runtime, which is bad, but
// it allows for generic code.
typedef struct TF_OutputList TF_OutputList;
TF_OutputList* TF_NewOutputList();
void TF_DeleteOutputList(TF_OutputList* o);
int TF_OutputListNumOutputs(TF_OutputList* o);
TF_TensorHandle* TF_OutputListOutput(TF_OutputList* o, int i);

// A TF_AbstractOp is the metadata we need to execute an operation in either
// eager or graph mode.
typedef struct TF_AbstractOp TF_AbstractOp;
TF_AbstractOp* TF_NewAbstractOp(TF_EagerContext* c, const char* const op_type,
                                const char* const op_name, TF_Status* s);
void TF_DeleteAbstractOp(TF_AbstractOp* op);

// TODO: we need a way of specifying attrs

// TF_ExecuteOperation will, if in eager mode, execute, if in graph mode, maybe
// capture some inputs and then add a node in the graph, and after
// execution/node creation it'll go and record things that happened in any tape
// which happens to be active.
void TF_ExecuteOperation(TF_AbstractOp* op, int num_inputs,
                         TF_TensorHandle* const * inputs, TF_Status* s,
                         TF_OutputList* o);

// TF_Tape is just a specialization of tensorflow::eager::Tape on
// TF_TensorHandle values and gradients.
typedef struct TF_Tape TF_Tape;
TF_Tape* TF_NewTape(TF_EagerContext* c, int persistent);
void TF_DeleteTape(TF_Tape* t);

void TF_ContextPushTape(TF_EagerContext* ctx, TF_Tape* tape);
void TF_ContextPopTape(TF_EagerContext* ctx, TF_Tape* tape, TF_Status* s);
void TF_TapeWatch(TF_EagerContext* ctx, TF_TensorHandle* t);
void TF_TapeGradient(TF_Tape* t, int num_sources, TF_TensorHandle** sources,
                     int num_targets, TF_TensorHandle** targets,
                     TF_OutputList* gradients, TF_Status* s);

// A GradientFunction is what we execute at runtime when computing a gradient;
// it takes some closure-captured values from the forward pass and the output
// gradients of the op and produces the input gradients of the op.
typedef void (*GradientFunction)(int num_output_gradients,
                                 TF_TensorHandle* const * output_gradients,
                                 TF_TensorHandle** input_gradients,
                                 TF_Status* s, void* closure);
typedef void (*GradientFunctionDeleter)(GradientFunction function,
                                        void* closure);

// A GradientFunctionRegisterer is the code that will run during the forward
// pass to find out which gradient function should be pushed into the tape. It
// has access to all inputs and outputs of an operation and gets to choose which
// ones to pack into the closure which will be available to the GradientFunction
// at runtime.
typedef void (*GradientFunctionRegisterer)(
    TF_EagerContext* c, int num_inputs, TF_TensorHandle* const* inputs,
    TF_OutputList* outputs, GradientFunction* gradient_function,
    GradientFunctionDeleter* gradient_function_deleter,
    void* registerer_closure, void** gradient_function_closure);

void TF_TapeCustomGradient(TF_EagerContext* ctx,
                           int num_inputs,
                           TF_TensorHandle** inputs,
                           int num_outputs,
                           TF_TensorHandle** outputs,
                           GradientFunctionRegisterer registerer,
                           void* registerer_closure);

// Registers a gradient function to run given an op name.
void TF_ContextRegisterGradientFunction(TF_EagerContext* ctx,
                                        const char* op_name,
                                        GradientFunctionRegisterer registerer,
                                        void* registerer_closure);
```


**Python API**


```
class EagerContextManager(object):
  def __init__(self, c_graph):
    self._c_graph = c_graph
  def __enter__(self):
    c_api.TF_EagerContextEnterGraph(ctx, self._c_graph)
  def __exit__(self):
    c_api.TF_EagerContextExitGraph(ctx, self._c_graph)

class _FuncGraphBase(object):
  def __init__():
    self._c_graph = c_api.TF_NewGraph()
  @contextmanager
  def as_default():
    # Note: This means that the graph hierarchy is no longer maintained in python.
    return EagerContextManager(self._c_graph)
```


We will implement a new subclass for `FuncGraph` that will replace `Graph`. We will try to keep as much of the logic as possible in C++ and expose that using pybind or somesuch. Here’s a discussion of some of the features that `FuncGraph` inherits from `Graph` which we will need to support. This list may not be exhaustive and we are hoping to add support for other things as needed.



1.  `apply_op_helper` + `create_op_internal` contain a lot of op _preparation_ logic which will need to be moved to C++. For example:
    1.  [Uniquifying op names](https://github.com/tensorflow/tensorflow/blob/41228d7f14496ff661e7c22361a987b0255cf812/tensorflow/python/framework/ops.py#L3297).
    1.  [Checking](https://github.com/tensorflow/tensorflow/blob/41228d7f14496ff661e7c22361a987b0255cf812/tensorflow/python/framework/op_def_library.py#L319-L327) deprecated op versions. [Graph version](https://github.com/tensorflow/tensorflow/blob/41228d7f14496ff661e7c22361a987b0255cf812/tensorflow/python/framework/ops.py#L2946) is already maintained in C++ so this should be fine.
    1.  [Type checking](https://github.com/tensorflow/tensorflow/blob/41228d7f14496ff661e7c22361a987b0255cf812/tensorflow/python/framework/op_def_library.py#L53).
    1.  There is a lot of logic for building attrs in python. For this we could possibly re-use the existing implementation in the pywrap C++ eager layer ([1](https://github.com/tensorflow/tensorflow/blob/41228d7f14496ff661e7c22361a987b0255cf812/tensorflow/python/eager/pywrap_tfe_src.cc#L755), [2](https://github.com/tensorflow/tensorflow/blob/41228d7f14496ff661e7c22361a987b0255cf812/tensorflow/python/eager/pywrap_tfe_src.cc#L850))
    1.  apply_op_helper calls [convert_to_tensor](https://github.com/tensorflow/tensorflow/blob/6d7926bb87c1a91ffd110aa3407c003b2ae54009/tensorflow/python/framework/op_def_library.py#L421) to convert python scalars to Tensors. This will happen in python for now and may move to a python specific C++ layer in the future.
1.  We need some form of context management to handle a variety of context managers we have in Graph e.g. [control dependencies](https://github.com/tensorflow/tensorflow/blob/6d7926bb87c1a91ffd110aa3407c003b2ae54009/tensorflow/python/framework/ops.py#L4345), control flow contexts (for XlaControlFlowContext), [colocate_with](https://github.com/tensorflow/tensorflow/blob/23275fb35cf17482d147f88ce7d8f4ce9c2376f3/tensorflow/python/framework/ops.py#L4115), [name_scope](https://github.com/tensorflow/tensorflow/blob/6d7926bb87c1a91ffd110aa3407c003b2ae54009/tensorflow/python/framework/ops.py#L3918), [_attr_scope_map](https://github.com/tensorflow/tensorflow/blob/6d7926bb87c1a91ffd110aa3407c003b2ae54009/tensorflow/python/framework/ops.py#L4587), [_kernel_label_map](https://github.com/tensorflow/tensorflow/blob/6d7926bb87c1a91ffd110aa3407c003b2ae54009/tensorflow/python/framework/ops.py#L4653) etc. We will look into whether this can be implemented using a generic op callback mechanism. The same mechanism can be used for implementing op callbacks as well.
1.  We will perform a case-by-case analysis of APIs of `Graph` to decide which of those should be supported in `_FuncGraphBase`.
    1.  Certain APIs related to [feeding and fetching](https://github.com/tensorflow/tensorflow/blob/6d7926bb87c1a91ffd110aa3407c003b2ae54009/tensorflow/python/framework/ops.py#L4788-L4805) probably don’t make sense for FuncGraph.
    1.  APIs for fetching Operations and Tensors: These APIs rely on a [dict of Operations](https://github.com/tensorflow/tensorflow/blob/6d7926bb87c1a91ffd110aa3407c003b2ae54009/tensorflow/python/framework/ops.py#L2721) maintained in Graph. Currently this dict is built _actively_ as operations are created in the graph. We could choose to populate this cache lazily as well.
    1.  In each Graph we maintain a [dict](https://github.com/tensorflow/tensorflow/blob/6d7926bb87c1a91ffd110aa3407c003b2ae54009/tensorflow/python/framework/ops.py#L2757) of EagerDefinedFunction/DefinedFunction used in the graph directly or in a sub-function. In nested functions we probably spend quadratic time in [copying](https://github.com/tensorflow/tensorflow/blob/23275fb35cf17482d147f88ce7d8f4ce9c2376f3/tensorflow/python/eager/function.py#L488-L497) the inner functions all the way to the eager context and use quadratic (in the number of functions) memory. Storing `_EagerDefinedFunction` references in the global graph has been a common source of memory leaks which @kkimdev has been valiantly fighting with.  I think we should try to register functions directly in the global eager context. We can just keep weakrefs to the _EagerDefinedFunction so that we don’t interfere with memory management. @kkimdev pointed out that we still need to maintain some reference to the list of functions used inside a ConcreteFunction so that we can add those to the [SavedModel](https://github.com/tensorflow/tensorflow/blob/23275fb35cf17482d147f88ce7d8f4ce9c2376f3/tensorflow/python/saved_model/save.py#L593).

Some implementation notes:

1.  Need to add RefCounting for eager tensor handles.
    1.  If a graph captures an EagerTensor, the code creating the EagerTensor should not delete it.
    1.  How do you write the gradient function of add, which just wants to forward the output gradient to the two inputs


### Analysing some FuncGraph CUJs


**tf.function**

When building the gradient (Stateful)PartitionedCall op, a captured tensor in the forward graph needs to be resolved to a forward call op’s output. This will still be possible to do in python.

**tf.cond/tf.switch_case**

Similar to tf.function, during gradient computation forward graph intermediates need to be mapped to forward op’s outputs. This currently updates the FuncGraph.captures map which can be done using `TF_EagerContextUpdateCaptureForPlaceholder`. Note however that tf.function does not actually update FuncGraph.captures and simply uses the new captures for building the gradient op. We may be able to avoid calling the API to update captures here if we do the same. Not sure if there any behavior relying on this though. Higher order derivatives using tf.gradients maybe?

**tf.while_loop**

tf.while_loop intercepts the default capturing mechanism of FuncGraph with custom behavior. In tf.while_loop, when a forward pass tensor needs to be captured we have to add an [accumulator](https://github.com/tensorflow/tensorflow/blob/6d7926bb87c1a91ffd110aa3407c003b2ae54009/tensorflow/python/ops/while_v2.py#L1012) and then capture the output of the While op corresponding to that accumulator.

To support this we will provide a `TF_EagerContext{Push|Pop}CaptureCallback` API which will register a callback function to perform the logic in [_WhileBodyGradFuncGraph._capture_helper](https://github.com/tensorflow/tensorflow/blob/6d7926bb87c1a91ffd110aa3407c003b2ae54009/tensorflow/python/ops/while_v2.py#L933).

We could leverage this to unify the gradient graph captures resolving behavior of `tf.function`/`tf.cond`/`tf.while_loop` all of which have their own recipes right now. 

### Automatic Control Dependencies

Automatic control dependencies (ACD) will move to C++ as well. However instead of being post-hoc it will now be performed _during_ graph building. The current design has certain limitations e.g. control dependencies across function boundaries are performed at the function level which is prohibitive for performance. There are ongoing discussions on ways to improve this. Other issues have come up in `tf.data` and `tf.distribute` for example because ACD only tracks direct dependencies. Ideally we should use this opportunity to address these shortcomings. However the details of this redesign are left to another doc to avoid diluting this doc.

### Open questions

1.  Keras seems to be using [non-public APIs](https://github.com/tensorflow/tensorflow/blob/6d7926bb87c1a91ffd110aa3407c003b2ae54009/tensorflow/python/keras/engine/base_layer.py#L2511) for directly building NodeDef and adding that to the graph. This is necessary for supporting Keras's Functional API (Model.add_loss, Model.add_metric, and auto-Lambda layers). We need to figure out if/how to support that. There are ongoing efforts to use just the public API of TF in tf.keras but the timelines for that are unclear.
    1. In the design review it was concluded that we should either be able to change Keras to use public python APIs or replace the internal python API calls with C API calls.


## Appendix

**Definition: Capturing**

Capturing is the process used to allow users to write functions which can reference tensors that are not directly passed as function inputs or are not passed as loop_vars in a call to tf.while_loop. In FuncGraph, when an external tensor is captured we create a [placeholder](https://github.com/tensorflow/tensorflow/blob/23275fb35cf17482d147f88ce7d8f4ce9c2376f3/tensorflow/python/framework/func_graph.py#L649) just like any other input and add that placeholder to the list of [FuncGraph.inputs](https://github.com/tensorflow/tensorflow/blob/23275fb35cf17482d147f88ce7d8f4ce9c2376f3/tensorflow/python/framework/func_graph.py#L672) and store the mapping from the external tensor to the placeholder in [FuncGraph._captures](https://github.com/tensorflow/tensorflow/blob/23275fb35cf17482d147f88ce7d8f4ce9c2376f3/tensorflow/python/framework/func_graph.py#L671).Capturing is triggered in `_create_op_internal` which is overridden in FuncGraph.

