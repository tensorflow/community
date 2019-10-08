# Native SavedModel execution in Node.js

| Status        | Accepted       |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | kangyizhang@google.com |
| **Sponsor**   | smilkov@google.com, nsthorat@google.com, piyu@google.com |
| **Updated**   | 2019-09-27                                           |

## Objective

This project is aiming to enable native TF SavedModel execution for inference in Node.js environment without conversion.

### Goals

*   Implement an API to load and execute TF SavedModel Signature for inference only in Node.js.
*   This API should works for SavedModel exported in both TF 1.x and 2.0
*   Wrap the loaded SavedModel as a new subtype implementing [tf.InferenceModel](https://github.com/tensorflow/tfjs/blob/81225adc2fcf6fcf633b4119e4b89a3bf55be824/tfjs-core/src/model_types.ts#L36)
*   Enable the ability to inspect the SavedModel metaGraph and signature in Node.js with protocol buffers in JavaScript.

### Non-goals

*   Enable execution tf.function in Node.js
*   Enable support for training a SavedModel
*   Enable support for exporting a SavedModel

## **Motivation**

TensorFlow.js brings TensorFlow into the JavaScript world. It provides APIs to develop and train models, and also tools to convert models trained in other languages.

Currently users could use [tfjs-converter](https://github.com/tensorflow/tfjs-converter) to convert TensorFlow SavedModel and TensorFlow Hub module to js friendly format and run inference through TensorFlow.js through the following steps:


1.  Install tf-nightly-2.0-preview and tensorflowjs pip packages
2.  Run the converter script to convert the model to js friendly format

```
tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    --signature_name=serving_default \
    --saved_model_tags=serve \
    /mobilenet/saved_model \
    /mobilenet/web_model
```

3.  Load and run the converted model in javascript through [tf.loadGraphModel()](https://js.tensorflow.org/api/latest/#loadGraphModel) or [tf.loadLayersModel()](https://js.tensorflow.org/api/latest/#loadLayersModel) API based on the model type

```
const model = await tf.loadGraphModel(MODEL_URL);
const cat = document.getElementById('cat');
model.predict(tf.browser.fromPixels(cat));
```

The above steps require developers to install Python TensorFlow package and some parameters/configurations, which we have noticed users are struggling with.

The tfjs-node repository provides native TensorFlow execution in Node.js environment through TensorFlow C library under the hood. It provides the same API (190+ ops) as [TensorFlow.js](https://js.tensorflow.org/api/latest/), which is a subset of the TensorFlow ops (900+).

Here there is an opportunity to support native SavedModel execution in Node.js with TensorFlow C library so that 1) tfjs-node can support models which contain ops that are not supported in TensorFlow.js yet, and 2) users do not need to go through the model conversion process.

This project uses the non-eager APIs in TensorFlow C library to enable loading and executing TF SavedModel for inference in Node.js environment without conversion.


## **Design Proposal**


### User-facing code

This project will provide a new API `tf.node.loadSavedModel` to load a Signature in SavedModel as a new class `TFSavedModel` in Node.js, which can be used to execute the SavedModel Signature for inference.

The loadSavedModel API takes a `path`, which is the absolute path to the SavedModel directory, a `tag_set` to identify which MetaGraph to load, and `signature` name as params. It returns a `TFSavedModel` object, implementing [tf.InferenceModel](https://github.com/tensorflow/tfjs/blob/81225adc2fcf6fcf633b4119e4b89a3bf55be824/tfjs-core/src/model_types.ts#L36) class.


```
const savedModel = tf.node.loadSavedModel(__dirname + 'saved_model_dir', 'tag_set', 'signature_def_name');
```


The returned TFSavedModel object has a `predict()` function to execute the SavedModel signature for inference. The param of this predict() function would be a single tensor if there is single input for the model or an array of tensors if the model has multiple inputs.

The TFSaveModel object also has an `execute()` function to execute the inference for the input tensors and return activation values for specified output node names.


```
const input = tensor1d([123], 'int32');
// Execute the loaded signatureDef of the SavedModel
const output = savedModel.predict([input_tensors]);
```


The TFSavedModel object also has a `delete()` function to free the SavedModel related memory.


```
savedModel.delete()
// The following line will throw an exception saying the SavedModel has been deleted.
const output = savedModel.predict([input_tensors]);
```



### Internal Change


#### Load SavedModel

A [SavedModel](https://www.tensorflow.org/beta/guide/saved_model) is a directory containing serialized signatures and the states needed to run them.


```
assets  saved_model.pb  variables
```


The directory has a saved_model.pb (or saved_model.pbtxt) file containing a set of named signatures, each identifying a function.

SavedModels may contain multiple sets of signatures (multiple MetaGraphs, identified with the tag-sets). When serving a model for inference, usually only one signature is used.


#### Designate the MetaGraph and Signature to execute

Though the C API supports loading multiple MetaGraph, and one loaded MetaGraph may have several SignatureDefs, this project only supports loading one MetaGraph and executing one SignatureDef through the JavaScript API, so that it’s clear to users that the loaded SavedModel is only using the specified Signature for inference. This also aligns with the current TensorFlow.js [models API](https://js.tensorflow.org/api/latest/#class:GraphModel), and the current workflow with tfjs-converter.

Users are able to load multiple signature from the same SavedModel by calling JavaScript API multiple times. The detailed discussion is provided later in SavedModel management section.

#### Deserialize saved_model.pb with protobuf in javascript to get MetaGraph and Signature info

For JavaScript developers, who do not have a lot of machine learning experience, their use case might be that they find an open source model and they want to use it in their Node.js project. The MetaGraph and Signatures are unclear to them and they don’t know how to get the model metadata in saved_model.pb file.

While TensorFlow provide the [SavedModel CLI tool](https://www.tensorflow.org/beta/guide/saved_model#details_of_the_savedmodel_command_line_interface) to inspect and execute a SavedModel, this project will make it convenient for JS developers to do all the work in JavaScript.

A new API will be added in TensorFlow.js node environment to allow users to inspect SavedModel, similar to [saved_model_cli show](https://www.tensorflow.org/beta/guide/saved_model#show_command), so that users can know what value to provide as MetaGraph and Signature.


```
const modelInfo = tf.node.inspectSavedModel(__dirname + 'saved_model_dir');

console.log(modelInfo);
/* The modelInfo should include the following information:
{
  tags: ['serve'],
  signatureDef: {
    serving_default: {
      'inputs': {
        x: {
          'name': 'serving_default_x:0',
          'dtype': ...,
          'tensorShape': ...
        }
      },
      'outputs': {
        output_0: {
          'name': 'StatefulPartitionedCall:0',
          'dtype': ...,
          'tensorShape': ...
        }
      },
      'methodName': 'tensorflow/serving/predict'
    }
  }
}
*/
```


Google’s Protocol Buffers is also available in javascript. It provides a [Protocol compiler](https://github.com/protocolbuffers/protobuf/releases) to translate the xxx.proto file to js file, and a JavaScript Protocol Buffers runtime library [google-protobuf](https://www.npmjs.com/package/google-protobuf) to construct and parse the messages.

To use Protocol Buffers in javascript, first the saved_model.proto file need to be translated:


```
$ protoc --js_out=import_style=commonjs,binary:. saved_model.proto
```


The above command will translate the [saved_model.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/saved_model.proto) file to saved_model_pb.js file. Then in js code, the saved_model.pb file can be parsed as SavedModel object through the translated js file.


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


With protobuf in JavaScript, the MetaGraphDef tag-sets and SignatureDef keys in SavedModel are available to be retrieved in JavaScript.


#### Use TF C API TF_LoadSessionFromSavedModel to load SavedModel

The TensorFlow C library has a [TF_LoadSessionFromSavedModel](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h#L1211) API, which creates a new TF_Session and then initializes states. This API supports both TF 1.x and TF 2.0 models. So with this API, the same code in tfjs-node works for both TF 1.x and 2.0. The `export_dir` and `tag` parameters are the `path` and `tag_set` value provided by users in javascript API.


```
TF_Session *session = TF_LoadSessionFromSavedModel(
     session_options, run_options, export_dir, tags, tags_leng, graph,
     metagraph, tf_status.status);
```


The returned TF_Session can be run with [TF_SessionRun](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h#L1254) API to execute the graph associated with the session.


### Do inference through running the loaded Session

TF C API provides a [TF_SessionRun](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h#L1254) function to execute the graph associated with the input session, which is loaded from the SavedModel through TF_LoadSessionFromSavedModel, as discussed above.


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

When running the session, input and output op names are the input/output names of the Signature provided when loading the SavedModel.


### New functions in node C++ addon (TFJSBackend)

Several new functions and members are added into [TFJSBackend](https://github.com/tensorflow/tfjs/blob/master/tfjs-node/binding/tfjs_backend.h#L29) to support SavedModel execution in node.


#### Tf_savedmodel_map and InsertSavedModel()

A map is added in the TFJSBackend to manage the loaded session from SavedModel. Similar to tfe_handle_map, the key of this map is a number of savedmodel_id. The value of this map is a pair of the loaded TF_Session and TF_Graph from SavedModel.


```
std::map<int32_t, std::pair<TF_Session*, TF_Graph*>> tf_savedmodel_map_;
```


#### LoadSavedModel

LoadSavedModel function is added to load a SavedModel from a path. It will get TF_Session from the SavedModel and insert the session into tf_savedmodel_map.


```
 // load a SavedModel from a path:
 // - export_dir (string)
 napi_value LoadSavedModel(napi_env env, napi_value export_dir, napi_value tag_set);
```


#### RunSavedModel

The backend will need savedmodel id, input tensor ids, and input/output names to execute the TF_Session.


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


#### DeleteSavedModel

When user does not need the SaveModel, DeleteSaveModel needs to be called to delete the corresponding TF_Session to release the memory.


```
 // Delete the corresponding TF_Session and TF_Graph
 // - savedmodel_id (number)
 void DeleteSavedModel(napi_env env, napi_value savedmodel_id);
```



#### TFJSBinding API

The [TFJSBinding](https://github.com/tensorflow/tfjs/blob/master/tfjs-node/src/tfjs_binding.ts#L32) interface will have corresponding functions to load, run and delete the SavedModel in JavaScript.



### Manage SavedModel in JavaScript

To manage and execute loaded sesion from SavedModel, a new TFSavedModel javascript class is added [nodejs_kernel_backend](https://github.com/tensorflow/tfjs/blob/master/tfjs-node/src/nodejs_kernel_backend.ts#L38).


```
class TFSavedModel implement InferenceModel {
 private readonly id: number;
 private deleted: boolean;
 private readonly inputOpName: string[];
 private readonly outputOpName: string[];

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
const model = tf.node.loadSavedModel(__dirname + 'saved_model', 'serve', 'serving_default');

const input = tensor1d([123], 'int32');

const output = model.predict([input_tensor]);

const output = savedModel.execute({'input_op_names':input_tensors}, ['output_op_names']);

model.delete();
```



#### Load multiple signatures from the same SavedModel

If users want to use multiple signatures from the same SavedModel, they can call tf.node.loadSavedModel() API several times to get multiple instances. The node backend will keep track of SavedModel paths that have been loaded. When doing a new loading, if the path to SavedModel have been loaded, the node backend will use the existing Session in addon module and will not load new thing through TF C API again.


# Test Plan

Several different types of SavedModel will be added as test artifacts. And tests will run against these real SavedModel. These tests will also cover memory leaking checks to make sure corresponding memories are released when a SavedModel is deleted in Node.js runtime.


# Benchmarking

A job to benchmark executing SavedModel (probably mobilenet) in tfjs-node vs tf python will be added to the current [benchmark infrastructure](https://github.com/tensorflow/tfjs/tree/master/tfjs/integration_tests#running-tfjs-node-benchmarks).
