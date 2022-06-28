# Title of RFC

| Status        | Proposed                                                                                      |
:-------------- |:----------------------------------------------------------------------------------------------|
| **RFC #**     | [NNN](https://github.com/tensorflow/community/pull/NNN) (update when you have community PR #) |
| **Author(s)** | Abin Thomas (abin.thomas@blueyonder.com), Iain Stitt (iain.stitt@blueyonder.com)              |
| **Sponsor**   | Robert Crowe (robertcrowe@google.com)                                                         |
| **Updated**   | 2020-06-20                                                                                    |

## Objective

Modify [BulkInferrer](https://github.com/tensorflow/tfx/tree/master/tfx/components/bulk_inferrer) TFX component.

Changes :-
* Store only a subset of features in `output_examples` artifact.
* Support inference on multiple models.

## Motivation

A BulkInferrer TFX component is used to perform batch inference on unlabeled tf.Examples. 
The generated output examples contains the original features and the prediction results. 
Keeping all original features in the output is troubling when dealing with feature heavy models. 
For most of the use cases we only require example identifiers and the predictions in the output. 

In machine learning, it is a common practice to train multiple models using the same feature set to perform different tasks (sometimes same tasks). 
It will be convenient to have a multimodel inference feature in bulk-inferrer. The component should take a list of models and produce predictions for all models.

## User Benefit

Filtering down the number of features in the output helps to reduce storage space for artifcats. 
It allows us to use larger batch sizes in downstream processing and reduces the chance of OOM issue.

Multimodel inference when done separately requires joining of the outputs on some identifiers, which is computationally and otherwise expensive. 
With this update the user can do post-processing directly without joining different outputs.

## Design Proposal

This is the meat of the document, where you explain your proposal. If you have
multiple alternatives, be sure to use sub-sections for better separation of the
idea, and list pros/cons to each approach. If there are alternatives that you
have eliminated, you should also list those here, and explain why you believe
your chosen approach is superior.

### Filter Output Features

The component decides whether to keep all the features or not based on an additional field in `OutputExampleSpec` proto.
The updated proto will look like this:-
```protobuf
message OutputExampleSpec {
  // Defines how the inferrence results map to columns in output example.
  repeated OutputColumnsSpec output_columns_spec = 3;
  repeated string example_features = 5;

  reserved 1, 2, 4;
}
```
`example_features` expects a list of feature names to be persisted in the output. Component will not filter if an empty list is provided.
The check and filtering will be performed in the [prediction_to_example_utils.py](https://github.com/tensorflow/tfx/blob/master/tfx/components/bulk_inferrer/prediction_to_example_utils.py#L86).

Check:- 
```python
def convert(prediction_log: prediction_log_pb2.PredictionLog,
            output_example_spec: _OutputExampleSpecType) -> tf.train.Example:
  
  ⁞
  
  if len(output_example_spec.example_features) > 0:
    example = _filter_columns(example, output_example_spec)
  
  return _add_columns(example, output_features)
```
`_filter_columns` function:- 
```python
def _filter_columns(example: tf.train.Example,
                    output_example_spec: _OutputExampleSpecType) -> tf.train.Example:
  """Remove features not in output_example_spec.example_features"""
  all_features = list(example.features.feature)
  for feature in all_features:
    if feature not in output_example_spec.example_features:
      del example.features.feature[feature]
  return example
```

### Mulitmodel Inference

For muli-model inference, the component will expect a union channel of models as input. 
[RunInference](https://github.com/tensorflow/tfx/blob/master/tfx/components/bulk_inferrer/executor.py#L253) will be performed using [RunInferencePerModel](https://github.com/tensorflow/tfx-bsl/blob/master/tfx_bsl/public/beam/run_inference.py#L101) method from tfx-bsl. 
This method will return a tuple of prediction logs instead of one single log. 
In subsequent steps these multiple logs will be merged to produce one single tf.Example. 
If raw inference_results are expected, then the component will save the predictions logs in inference_result subdirectories.

#### Changes to input protos

`model_spec` and `output_example_spec` parameters expect `ModelSpec` and `OutputExampleSpec` protos respectively.
For supporting multiple models and also keeping in mind backward compatibility, self referencing proto definitions can be used.   

`model_spec` : -
```protobuf
message ModelSpec {
  // Specifies the signature name to run the inference with. If multiple
  // signature names are specified (ordering doesn't matter), inference is done
  // as a multi head model. If nothing is specified, default serving signature
  // is used as a single head model.
  repeated string model_signature_name = 2;

  // Tags to select metagraph from the saved model. If unspecified, the default
  // tag selects metagraph to run inference on CPUs. See some valid values in
  // tensorflow.saved_model.tag_constants.
  repeated string tag = 5;

  // handle multiple ModelSpec
  repeated ModelSpec model_specs = 7;

  reserved 1, 3, 4, 6;
}
```

`output_example_spec` : -
```protobuf
message OutputExampleSpec {
  // Defines how the inferrence results map to columns in output example.
  repeated OutputColumnsSpec output_columns_spec = 3;

  // List of features to maintain in the output_examples
  repeated string example_features = 5;

  // handle multiple OutputExampleSpec
  repeated OutputExampleSpec output_example_specs = 6;

  reserved 1, 2, 4;
}
```
Parsing both protos requires additional validation checks to figure out single model spec or multiple model spec.

#### Changes to input channels

`model` and `model_blessing` parameters can be either of the type [BaseChannel](https://github.com/tensorflow/tfx/blob/master/tfx/types/channel.py#L51) or [UnionChannel](https://github.com/tensorflow/tfx/blob/master/tfx/types/channel.py#L363).
If BaseChannel is passed as input, the component will convert it to a single item UnionChanel before invoking the executor.
```python
    if model and (not isinstance(model, types.channel.UnionChannel)):
        model = types.channel.union([model])
    if model_blessing and (not isinstance(model_blessing, types.channel.UnionChannel)):
        model_blessing = types.channel.union([model_blessing])
```
If any of the model is not blessed the executor will return without doing inference.


#### Changes to write `inference_result` beam pipeline

If raw inference_results are expected, then the component will save the predictions logs in inference_result subdirectories.
```python
      if inference_result:
        data = (
                data_list
                | 'FlattenInferenceResult' >> beam.Flatten(pipeline=pipeline))
        for i in range(len(inference_endpoints)):
            _ = (
                data
                | 'SelectPredictionLog[{}]'.format(i) >> beam.Map(lambda x: x[i])
                | 'WritePredictionLogs[{}]'.format(i) >> beam.io.WriteToTFRecord(
                    os.path.join(inference_result.uri, str(i), _PREDICTION_LOGS_FILE_NAME),
                    file_name_suffix='.gz', coder=beam.coders.ProtoCoder(prediction_log_pb2.PredictionLog)))
```

#### Changes to prediction to examples convert function

In case of multiple prediction logs, the features are extracted from the first one.
```python
def convert(prediction_logs: Tuple[prediction_log_pb2.PredictionLog, ...],
            output_example_spec: _OutputExampleSpecType) -> tf.train.Example:
  """Converts given `prediction_log` to a `tf.train.Example`.

  Args:
    prediction_logs: The input prediction log.
    output_example_spec: The spec for how to map prediction results to columns
      in example.

  Returns:
    A `tf.train.Example` converted from the given prediction_log.
  Raises:
    ValueError: If the inference type or signature name in spec does not match
    that in prediction_log.
  """
  is_single_output_example_spec = bool(output_example_spec.output_columns_spec)
  is_multiple_output_example_spec = bool(output_example_spec.output_example_specs)

  if (not is_single_output_example_spec) and (not is_multiple_output_example_spec):
    raise ValueError('Invalid output_example spec')
  elif is_single_output_example_spec and (not is_multiple_output_example_spec):
    specs = [output_example_spec]
  elif (not is_single_output_example_spec) and is_multiple_output_example_spec:
    specs = output_example_spec.output_example_specs
    if len(prediction_logs) != len(specs):
      raise ValueError('inference result, spec length mismatch '
                       'output_example_spec: %s' % output_example_spec)
  else:
    raise ValueError('Invalid output_example spec')

  example = _parse_examples(prediction_logs[0])
  output_features = [_parse_output_feature(prediction_log, example_spec.output_columns_spec)
                     for prediction_log, example_spec in zip(prediction_logs, specs)]

  if len(output_example_spec.example_features) > 0:
    example = _filter_columns(example, output_example_spec)

  return _add_columns(example, output_features)
```

### Alternatives Considered

### Performance Implications
Neutral

### Dependencies
No new dependencies introduced.

### Engineering Impact

### Platforms and Environments
No special considerations across different platforms and environments.

### Best Practices
No change in best practices.

### Tutorials and Examples
API docs will be updated.

### Compatibility
Proto and input changes are backward compatible.

### User Impact

## Questions and Discussion Topics
* Is it okay to use self-referencing proto definitions for backward compatibility?