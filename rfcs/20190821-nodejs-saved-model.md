# Native SavedModel execution in Node.js

| Status        | Proposed       |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | kangyizhang@google.com |
| **Sponsor**   | smilkov@google.com, nsthorat@google.com, piyu@google.com, kreeger@google.com |
| **Updated**   | 2019-08-21                                           |

## Objective

This project is aiming to enable native TF SavedModel execution in Node.js environment without conversion.

### Goals

*   Implement an API to load and execute TF SavedModel in Node.js for inference that is created in either TF 1.x or 2.0.
*   Wrap the loaded SavedModel as a new subtype implementing tf.InferenceModel, so users could call default signatureDef or other signatureDefs with tag
*   Enable the ability to inspect the SavedModel signatureDefs in Node.js with protocol buffers in JavaScript. This is a stage 2 implementation detail and will not be discussed deeply in this doc.

### Non-goals

*   Enable support for training a SavedModel
*   Enable support for exporting a SavedModel

## **Motivation**

[TensorFlow.js](https://www.tensorflow.org/js/) brings TensorFlow into the JavaScript world. Currently users could use [tfjs-converter](https://github.com/tensorflow/tfjs-converter) to convert TensorFlow SavedModel and TensorFlow Hub module to JS friendly format and to run inference through TensorFlow.js. The tfjs-converter tool requires developers to install Python TensorFlow and some parameters/configurations, which we have noticed users are struggling with.

The [tfjs-node](https://github.com/tensorflow/tfjs/tree/master/tfjs-node) project provides native TensorFlow execution in Node.js environment through TensorFlow C library under the hood. It provides the same API (140+ ops) as [TensorFlow.js](https://js.tensorflow.org/api/latest/), which is a subset of the TensorFlow ops (900+). The TensorFlow C library actually supports more ops and APIs beyond ops.

Here there is an opportunity to support native SavedModel execution in Node.js so that 1) tfjs-node can support models which contain ops that are not supported in tfjs yet, and 2) users do not need to go through extra conversion of the model.


## **Design Proposal**


### User-facing code

This project will provide a new API `tf.node.loadSavedModel` to load a SavedModel as a new class `TFSavedModel` in Node.js which can be used by the user to execute the SavedModel for inference.

The loadSavedModel API takes a `path` param, which is the absolute path to the SavedModel directory. It returns a `TFSavedModel` object, implementing [InferenceModel](https://github.com/tensorflow/tfjs/blob/81225adc2fcf6fcf633b4119e4b89a3bf55be824/tfjs-core/src/model_types.ts#L36) class.


```
const savedMode = tf.node.loadSavedModel(__dirname + 'saved_model_dir');
```


The returned TFSavedModel object has a `predict()` function to execute the SavedModel serving_default signature for inference. The param of this predict() function would be a single tensor if there is single input for the model, an array of tensors or named tensor map if the model has multiple inputs.

The TFSaveModel object also has an `execute()` function to execute the inference for the input tensors and return activation values for specified output node names.


```
const input = tensor1d([123], 'int32');
// Execute default signatureDef of the SavedModel
const output = savedMode.predict([input_tensors]);

// Execute multiple signatureDefs of the SavedModel
const output = savedMode.execute({'input_op_names':input_tensors}, ['output_op_names']);
```


The TFSavedModel object also has a `delete()` function to free the SavedModel related memory.


```
savedMode.delete()
// The following line will throw an exception saying the SavedModel has been deleted.
const output = savedMode.predict([input_tensors]);
```



### Internal Change


#### Load SavedModel

A [SavedModel](https://www.tensorflow.org/beta/guide/saved_model) is a directory containing serialized signatures and the states needed to run them.


```
assets  saved_model.pb  variables
```


The directory has a saved_model.pb (or saved_model.pbtxt) file containing a set of named signatures, each identifying a function. One of these signatures is keyed as “serving_default”  and used as the default function when user does not specify a signatureDef when executing the SavedModel. This saved_model.pb file contains encoded structure data that needs to be read as binary and parsed as [SavedModel](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/saved_model.proto) proto format.


##### Use TF C API TF_LoadSessionFromSavedModel to load SavedModel

The TensorFlow C library has a [TF_LoadSessionFromSavedModel](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h#L1211) API, which creates a new TF_Session and then initializes states. This API supports both TF 1.x and TF 2.0 models. So with this API, the same code in tfjs-node works for both TF 1.x and 2.0. The returned TF_Session can be run with [TF_SessionRun](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h#L1254) API to execute the graph associated with the session.


```
TF_Session *session = TF_LoadSessionFromSavedModel(
     session_options, run_options, export_dir, tags, tags_leng, graph,
     metagraph, tf_status.status);
```


Pros:

*   Using the available TF C API voids extra work to dive deep into the SavedModel structure and rebuild/execute the associated graph in JavaScript.
*   New changes of SavedModel will be updated in the TF C library. The tfjs-node APIs will not be broken by SavedModel change.

Cons:

*   C API does not provide details and metadata about SavedModel


##### Other option: parse saved_model.pb with protobuf in javascript

Google’s Protocol Buffers is also available in javascript. It provides a [Protocol compiler](https://github.com/protocolbuffers/protobuf/releases) to translate the xxx.proto file to js file, and a JavaScript Protocol Buffers runtime library [google-protobuf](https://www.npmjs.com/package/google-protobuf) to construct and parse the messages.

To use Protocol Buffers in javascript, first the saved_model.proto file need to be translated:


```
$ protoc --js_out=import_style=commonjs,binary:. saved_model.proto
```


The above command will translate the [saved_model.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/saved_model.proto) file to saved_model_pb.js file. However the saved_model.proto file depends on [meta_graph.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/meta_graph.proto). The meta_graph.proto file depends on [graph.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto), [op_def.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_def.proto), [tensor_shape.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor_shape.proto), [types.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.proto), [saved_object_graph.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/saved_object_graph.proto), [savef.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/saver.proto), and these proto files depend on more proto files in the tensorflow repo.

Thus almost all the proto files in tensorflow are required to be translated to js version. This can be achieved by clustering all the proto files in [tensorflow repo](https://github.com/tensorflow/tensorflow) together into one folder,  running the command above and moving the generated translated_pb.js files into tfjs-node repo.

Then in js code, the saved_model.pb file can be parsed as SavedModel object.


```
var messages = require('./tensorflow/core/protobuf/saved_model_pb');
var fs = require('fs');

var SavedModel = new messages.SavedModel();
const mobileModel = fs.readFileSync('./saved_model.pb');
const array = new Uint8Array(mobileModel);

const model = messages.SavedModel.deserializeBinary(array);

console.log(model.getSavedModelSchemaVersion());
console.log(model.getMetaGraphsList());
```


The op/node list can be get from the SavedModel, and we can rebuild the graph in js and run the ops with loaded checkpoint.

Pros:

*   SavedModel metadata and graph details are available in JS

Cons:

*   Extra effort to load signatures and checkpoint
*   Need to manage ops in graph
*   Need to update proto files occasionally

Using TF C API to load and execute SavedModel is preferred, because it requires smaller scope of work to reconstruct and execute the SavedModel and no need to actively update the code for future changes in TF SavedModel.

But bringing Protocol Buffers in JS might be useful to inspect SavedModel. It brings the ability to inspect SavedModel in Node.js. Further discussion is added in a later section.


#### Do inference through running the loaded Session

TF C API provides a [TF_SessionRun](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h#L1254) function to execute the graph associated with the input session, which can be loaded from the SavedModel through TF_LoadSessionFromSavedModel, as discussed above.


```
TF_CAPI_EXPORT extern void TF_SessionRun(
   TF_Session* session,
   // RunOptions, may be NULL
   const TF_Buffer* run_options,
   // Input tensors
   const TF_Output* inputs, TF_Tensor* const* input_values, int ninputs,
   // Output tensors
   const TF_Output* outputs, TF_Tensor** output_values, int noutputs,
   // Target operations
   const TF_Operation* const* target_opers, int ntargets,
   // RunMetadata, may be NULL
   TF_Buffer* run_metadata,
   // Output status
   TF_Status*);
```


If the session is successfully executed, the tensors corresponding to output ops are placed in output_values, which are TF_Tensor type. They will be converted to TFE_TensorHandle and registered in tfjs-node backend, which is the same as how tensor is managed currently in tfjs-node addon.


#### Input and output op names in SavedModel

When calling the TF_SessionRun method, input and output op names need to be provided. The default input and output op names for “serving_default” signatureDef can be parsed from the SavedModel proto object.  Following is an example of the structure of SavedModel proto:

SavedModel
*   SavedModelSchemaVersion
*   MetaGraphsList
    *   MetaInfoDef
        *   …
    *   GraphDef
        *   …
    *   …
    *   Signature Definition Map:
        *   Default SignatureDef
            *   Input op metadata
            *   Output op metadata
        *   SignatureDef 1
            *   Input op metadata
            *   Output op metadata
        *   SignatureDef 2
            *   Input op metadata
            *   Output op metadata

In TF Java TF go, TF_LoadSessionFromSavedModel and TF_SessionRun are also used to load and execute session from SavedModel. The SavedModel is parsed as protocol buffers in Java/go and input/output name is extracted from the default signature def.

Following are the options to get input/output op names in Node.js:


##### Option 1: ask user to provide the names

Input and output names can be provided as parameters when user want to execute SavedModel. The input/output op names can be obtained through the [saved_model_cli](https://www.tensorflow.org/beta/guide/saved_model#show_command).

This could be a solution for stage 1 of this project. Getting input/output names through tfjs-node API can be a stage 2 of this project and implement parallelly with other part of this project. Thus in long term users do not need to rely on TF Python or other tools to execute a SavedModel in tfjs-node.

Pros:

*   Getting input/output names will not block other part of this project
*   Users could start to execute SavedModel in Node.js sooner

Cons:

*   Users need to use saved_model_cli to inspect the SavedModel
*   Still need to figure out a solution to get default input/output name in long term so users do not need to know the details of the SavedModel


##### Option 2: add new TF C API to get input/output names from meta graph definition

Tfjs-node is running the TensorFlow C API, which provides TF C API header files and shared libraries. It does not require any Protocol Buffers to call the library. So adding new TF API to get serving_default input/output name string from a SavedModel would enable projects like tfjs-node to execute a session without bringing Protocol Buffers onboard.

Pros:

*   No need to add Protocol Buffers in tfjs-node

Cons:

*   Require review from TF team and extra release cycle for TF C shared libraries


##### Option 3: add Protocol Buffers in JavaScript

Like discussed earlier, Protocol Buffers is available in JavaScript and SavedModel proto object can be parsed. So the default signatureDef can be parsed in JavaScript and be used when executing SavedModel session. More details of the SavedModel are available for users to inspect through pure JavaScript API and all the signatureDefs can be executed.

Pros:

*   Users do not need to be familiar with the details of SavedModel or deal with other TF tools
*   Users could execute any signature definition they want

Cons:

*   Need to compile and maintain SavedModel proto in JS


##### Option 4: add Protocol Buffers in tfjs-node c++ addon

Another option similar to option 3 is to add Protocol Buffers in the node native C++ addon. But it requires the tfjs-node package to deliver protocol buffer runtime, which is different for each operating system.

Pros:

*   Users do not need to be familiar with the details of SavedModel or deal with other TF tools
*   Users could execute any signature definition they want

Cons:

*   Heavy maintenance duty for tfjs-node
*   Extra tfjs_bindings API to maintain
*   Need to copy all proto files from TensorFlow to tfjs-node


##### Decision

Option 1, ask user to provide the input/output names, could be the temp solution for stage 1 of this project.

Option 2, add function in TF C API to get input/output op names from SavedModel, is a good long term solution. It separates tfjs-node from TF C library details and keeps tfjs_bindings API simple, but it requires extra release cycle for the next TensorFlow C library.

Option 3, bringing protocol buffer in JavaScript could enable users to fully inspect the SavedModel and run any signatureDef. This option will be implemented as the stage 2 solution.


#### TFJSBackend new functions in C++

Several new functions and members are added into TFJSBackend to support SavedModel execution in node.


##### Tf_savedmodel_map and InsertSavedModel()

A map is added in the TFJSBackend to manage the loaded session from SavedModel. Similar to tfe_handle_map, the key of this map is a number of savedmodel_id. The value of this map is a pair of the loaded TF_Session and TF_Graph from SavedModel.


```
std::map<int32_t, std::pair<TF_Session*, TF_Graph*>> tf_savedmodel_map_;
```



##### LoadSavedModel

LoadSavedModel function is added to load a SavedModel from a path. It will get TF_Session and TF_Graph from the SavedModel and insert them into the tf_savedmodel_map.


```
 // load a SavedModel from a path:
 // - export_dir (string)
 napi_value LoadSavedModel(napi_env env, napi_value export_dir);
```



##### RunSavedModel

The backend will need savedmodel id, input tensor ids, and input/output names to execute the TF_Session. TF_Graph will be used to check if the input/output names exist in the graph to avoid segmentation fault.


```
 // Execute a session with the provided input/output name:
 // - savedmodel_id (number)
 // - input_tensor_ids (array of input tensor IDs)
 // - input_op_names (string)
 // - output_op_names (string)
 napi_value RunSavedModel(napi_env env, napi_value savedmodel_id,
                       napi_value input_tensor_ids, napi_value input_op_names,
                       napi_value output_op_names);
```



##### DeleteSavedModel

When user does not need the SaveModel, DeleteSaveModel needs to be called to delete the corresponding TF_Session and TF_Graph to release the memory.


```
 // Delete the corresponding TF_Session and TF_Graph
 // - savedmodel_id (number)
 void DeleteSavedModel(napi_env env, napi_value savedmodel_id);
```



##### TFJSBinding API

The TFJSBinding will have corresponding functions to load, run and delete the SavedModel.


#### TFSavedModel class and tf.node.loadSavedModel API in JS

To manage and execute loaded sesion from SavedModel, a new TFSavedModel class is added nodejs_kernel_backend.


```
class TFSavedModel implement InferenceModel {
 private readonly id: number;
 private deleted: boolean;

 constructor(id: number, backend: NodeJSKernelBackend) {}

 predict(inputs: Tensor|Tensor[]|NamedTensorMap, config: ModelPredictConfig):
     Tensor|Tensor[]|NamedTensorMap;

 execute(inputs: Tensor|Tensor[]|NamedTensorMap, outputs: string|string[]):
     Tensor|Tensor[];


 delete() {}
}
```


The instance of TFSavedModel could only be created in nodejs_kernel_backend object when calling TFJSBinding’s LoadSavedModel function. And the id value is the number returned from the TFJSBachend.

Following is how user will use SavedModel in tfjs-node:


```
const model = tf.node.loadSavedModel(__dirname + 'saved_model');

const input = tensor1d([123], 'int32');

const output = model.predict([input_tensor]);

const output = savedMode.execute({'input_op_names':input_tensors}, ['output_op_names']);

model.delete();
```



### Test Plan

Several different types of SavedModel will be added as test artifacts. And tests will run against these real SavedModel. These tests will also cover memory leaking checks to make sure corresponding memories are released when a SavedModel is deleted in Node.js runtime.


### Benchmarking

A job to benchmark executing SavedModel (probably mobilenet) in tfjs-node vs tf python will be added to the current benchmark infrastructure.
