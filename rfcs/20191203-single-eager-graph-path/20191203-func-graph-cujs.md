# CUJs for FuncGraph

| **Author** | Saurabh Saxena (srbs@google.com) |
:-------------- |:---------------------------------------------------- |
| **Updated**   | 2019-12-03                                           |


### tf.function


### **Forward**



1.  An empty FuncGraph is created.
1.  [Placeholders](https://github.com/tensorflow/tensorflow/blob/6a70aa6d438259cabd23c09808db4cf51a2e5377/tensorflow/python/framework/func_graph.py#L1205) are created in it corresponding to the input_signature. Note the signature can contain CompositeTensors which are flattened. The input structure is maintained in [structured_input_signature](https://github.com/tensorflow/tensorflow/blob/6a70aa6d438259cabd23c09808db4cf51a2e5377/tensorflow/python/framework/func_graph.py#L906).
    1.  We seem to be [always](https://github.com/tensorflow/tensorflow/blob/6a70aa6d438259cabd23c09808db4cf51a2e5377/tensorflow/python/framework/func_graph.py#L1237) capturing variables even though they are unused. Can that be avoided?
1.  The python_func is called with the above input placeholders as args. This can trigger creation of new placeholders by capturing. The captured tensors can be symbolic tensors from outer graphs or eager tensors.
1.  FuncGraph.structured_outputs is populated with the structured tensors(containing CompositeTensors, IndexedSlices etc.). FuncGraph.outputs is built by flattening the structure and CompositeTensors in structured_outputs and by removing any Nones.
    1.  We call [capture](https://github.com/tensorflow/tensorflow/blob/6a70aa6d438259cabd23c09808db4cf51a2e5377/tensorflow/python/framework/func_graph.py#L1015) on the tensors in the list of outputs to handle the case when the function is simply returning an external tensor. Solutions:
        1.  We could replace this with creating an Identity node in the forward graph which would implicitly capture the external tensor. However, these Identity nodes are not necessary and might cause performance problems later.
        1.  Can we avoid doing the capturing in func_graph_from_py_func? Idea: We keep Nones in the list of structured_outputs and not in the list of outputs. We could do the same for external outputs. These can get repackaged just like we [repackage](https://github.com/tensorflow/tensorflow/blob/6a70aa6d438259cabd23c09808db4cf51a2e5377/tensorflow/python/eager/function.py#L1911-L1913) Nones.

**Backward**



1.  An empty FuncGraph is created.
1.  input_signature is [constructed](https://github.com/tensorflow/tensorflow/blob/6a70aa6d438259cabd23c09808db4cf51a2e5377/tensorflow/python/eager/function.py#L644) from the incoming grads and [placeholders](https://github.com/tensorflow/tensorflow/blob/6a70aa6d438259cabd23c09808db4cf51a2e5377/tensorflow/python/framework/func_graph.py#L1205) are created in it corresponding to the input_signature.
1.  The gradient [function](https://github.com/tensorflow/tensorflow/blob/6a70aa6d438259cabd23c09808db4cf51a2e5377/tensorflow/python/eager/function.py#L649) is called in this FuncGraph. This triggers capturing of intermediate tensors in the forward FuncGraph or one of its outer graphs in case custom_gradients are involved. Note that we already created placeholders for incoming grads so those are not captured. When building the gradient PartitionedCall op, this external capture will be replaced with a Placeholder in the current graph if the capture is not already in the current graph. The external capture is now a capture in the current graph (graph containing the gradient PartitionedCall). There are a few cases in the resolution:
    1.  The external tensor is in one of the outer graphs of the current graph. In this case nothing needs to be done.
    1.  The external tensor is not in the current hierarchy.
        1.  If it is in the forward graph it gets [added](https://github.com/tensorflow/tensorflow/blob/6a70aa6d438259cabd23c09808db4cf51a2e5377/tensorflow/python/eager/function.py#L688) to the list of outputs and the forward op is [updated](https://github.com/tensorflow/tensorflow/blob/6a70aa6d438259cabd23c09808db4cf51a2e5377/tensorflow/python/eager/function.py#L715) with new outputs and this tensor is [resolved](https://github.com/tensorflow/tensorflow/blob/6a70aa6d438259cabd23c09808db4cf51a2e5377/tensorflow/python/eager/function.py#L723-L728) to an op output.
        1.  If it is in an outer graph of the forward graph, nothing needs to be done (yet).
        1.  If it is in an inner graph of the forward graph, an error is raised (this should never happen).
1.  FuncGraph.structured_outputs is populated with the structured tensors(containing CompositeTensors, IndexedSlices etc.). FuncGraph.outputs is built by flattening the structure and CompositeTensors in structured_outputs and by removing any Nones.


### tf.cond/tf.switch_case

**Forward**



1.  Build graphs for branch functions.
1.  Find the superset of input tensors needed by all branch functions and update signatures of all branch functions so that they [match](https://github.com/tensorflow/tensorflow/blob/f540109342f8b7cb9b96163dae455013249c3128/tensorflow/python/ops/cond_v2.py#L494) by creating dummy placeholders. This requires resetting FuncGraph.inputs and FuncGraph.captures.
    1.  Supporting this would require either ResetInputs, ResetCaptures APIs or adding new If/Case ops that don’t need this signature matching (b/143286622).
    1.  Another option is to not support resetting inputs and captures at all and let the consumers take care of this when generating the FunctionDef. However this would mean that the FunctionDef would not match the FuncGraph which may cause problems in [gradient computation](https://github.com/tensorflow/tensorflow/blob/f540109342f8b7cb9b96163dae455013249c3128/tensorflow/python/ops/cond_v2.py#L109) which use the forward cached FuncGraph and expects the forward op’s FunctionDef to be generated 1-1 from the forward FuncGraph.

**Backward**



1.  Build the grad func for each branch using tf.gradients.
1.  Similar to forward pass, add dummy inputs to make input signatures match.
1.  Any needed intermediates in the forward graph are wrapped in Optionals and are added to the list of forward graph [outputs](https://github.com/tensorflow/tensorflow/blob/f540109342f8b7cb9b96163dae455013249c3128/tensorflow/python/ops/cond_v2.py#L151-L152).
1.  Similar to tf.function, we resolve any external captures to the forward op’s outputs.


### tf.while_loop

**Forward**



1.  Build the [cond](https://github.com/tensorflow/tensorflow/blob/c29529aa7d55bc66b040917a36acdb5722231043/tensorflow/python/ops/while_v2.py#L141) FuncGraph using a signature built from the input loop vars. Cond function can capture external tensors which show up in cond_graph.external_captures.
1.  Build the [body](https://github.com/tensorflow/tensorflow/blob/c29529aa7d55bc66b040917a36acdb5722231043/tensorflow/python/ops/while_v2.py#L186) FuncGraph using the same signature as the cond. However in the body function [capture](https://github.com/tensorflow/tensorflow/blob/c29529aa7d55bc66b040917a36acdb5722231043/tensorflow/python/ops/while_v2.py#L162-L165) the external captures of cond first. At this point the full signature, i.e. original input signature with loop vars + captures, matches in cond and body.
    1.  The explicit capture is needed here to make the signatures of cond and body to match. This can be avoided if we allow signatures of cond and body to diverge.
1.  Now body_graph has some extra external captures. These are captured in the [cond_graph](https://github.com/tensorflow/tensorflow/blob/c29529aa7d55bc66b040917a36acdb5722231043/tensorflow/python/ops/while_v2.py#L206-L213). So in effect the external captures of body cond_graph and body_graph are effectively cond-graph-captures + body-graph-captures.

**Backward**



1.  Build the gradient graph for the forward graph just like for other functional ops.
1.  Since a while loop can run for multiple iterations, if the backwards pass needs to capture a forward tensor there are two cases:
    1.  If the tensor’s value potentially varies across iterations, in the forward graph the tensor is [accumulated](https://github.com/tensorflow/tensorflow/blob/c29529aa7d55bc66b040917a36acdb5722231043/tensorflow/python/ops/while_v2.py#L1012) in a TensorList (think: stack). Note: now the forward op has an extra input, the empty stack, and an extra output which contains the list of values of the tensor in multiple iterations. The forward graph stack is captured in the backward graph and a value is popped from it to use as the intermediate value for that tensor.
    1.  If the tensor’s value is invariant across loop iterations, we directly [capture](https://github.com/tensorflow/tensorflow/blob/c29529aa7d55bc66b040917a36acdb5722231043/tensorflow/python/ops/while_v2.py#L978) the forward tensor in the backward graph.


### Autograph

FuncGraph is used as a temporary graph to evaluate the type of a while loop’s conditional expression. See [while_stmt](https://github.com/tensorflow/tensorflow/blob/6a70aa6d438259cabd23c09808db4cf51a2e5377/tensorflow/python/autograph/operators/control_flow.py#L739). Created ops, if any, are discarded immediately - we only need to test whether the expression evaluates to a Tensor or not, and if a tf.while_loop is created, they will be created again by the while_loop itself.

This might not require a FuncGraph - any regular graph is suitable for this purpose.


### Serialization/SavedModel

Serialization



1.  The Trackable object graph is crawled to find all functions. An error is raised if trying to save an unsaveable FuncGraph.
    1.  FuncGraph has a `_saveable` property which is used to denote whether a FuncGraph can be saved to a SavedModel. This seems to have only [one usage](https://github.com/tensorflow/tensorflow/blob/99f0e90b384cfb255103a8965bec0d11a7995e49/tensorflow/python/keras/backend.py#L311) right now in Keras to mark functions that capture the symbolic learning phase to be unsaveable.
1.  For every ConcreteFunction
    1.  Its captured non-resource non-variant tensors are [converted](https://github.com/tensorflow/tensorflow/blob/23275fb35cf17482d147f88ce7d8f4ce9c2376f3/tensorflow/python/saved_model/save.py#L280-L298) into graph constants.
    1.  The graph is converted to a [FunctionDef](https://github.com/tensorflow/tensorflow/blob/23275fb35cf17482d147f88ce7d8f4ce9c2376f3/tensorflow/python/saved_model/save.py#L593) and is written to the MetaGraphDef graph’s function library.
    1.  An [entry](https://github.com/tensorflow/tensorflow/blob/99f0e90b384cfb255103a8965bec0d11a7995e49/tensorflow/core/protobuf/saved_object_graph.proto#L32) is added to the object graph proto which stores the node ids of the captured inputs in the object graph and the input/output structures.
1.  To enable loading the SavedModel with Sessions, placeholders are [created](https://github.com/tensorflow/tensorflow/blob/23275fb35cf17482d147f88ce7d8f4ce9c2376f3/tensorflow/python/saved_model/save.py#L341) in the graph for non-captured inputs. Then a (Stateful)PartitionedCall op is created in the graph, by feeding the placeholders + constants as inputs to the call op. A SignatureDef is then created for the call op and added to the MetaGraphDef.
    1.  This requires access to FuncGraph.inputs, captures and external_captures and assumes that placeholders for captures are at the rear of FuncGraph.inputs.

Deserialization



1.  Concrete functions are [created](https://github.com/tensorflow/tensorflow/blob/99f0e90b384cfb255103a8965bec0d11a7995e49/tensorflow/python/saved_model/load.py#L113-L115) for all graph library functions. 
    1.  This probably instantiates ConcreteFunctions for non-top-level functions as well. Is that necessary?
1.  The captures map is initialized by using the [bound_inputs](https://github.com/tensorflow/tensorflow/blob/99f0e90b384cfb255103a8965bec0d11a7995e49/tensorflow/core/protobuf/saved_object_graph.proto#L107) field of the SavedConcreteFunction proto.
    1.  This makes a call to [replace_capture](https://github.com/tensorflow/tensorflow/blob/99f0e90b384cfb255103a8965bec0d11a7995e49/tensorflow/python/saved_model/load.py#L184) and then a separate call to [capture](https://github.com/tensorflow/tensorflow/blob/99f0e90b384cfb255103a8965bec0d11a7995e49/tensorflow/python/saved_model/load.py#L200). This is done because we already have the internal placeholders created and we just need to update the captures map. The call to FuncGraph.capture records the capture on the tape.
1.  Input/output structures are [initialized](https://github.com/tensorflow/tensorflow/blob/99f0e90b384cfb255103a8965bec0d11a7995e49/tensorflow/python/saved_model/load.py#L155-L157).
    1.  Seems like structured_outputs only contains the structure but not really the tensors e.g. in the original FuncGraph.structured_outputs.
