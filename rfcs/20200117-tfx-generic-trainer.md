# TFX Generic Trainer

| Status        | Accepted                                                  |
| :------------ | :-------------------------------------------------------- |
| **Author(s)** | Jiayi Zhao (jyzhao@google.com)                            |
| **Sponsor**   | Konstantinos Katsiapis (katsiapis@google.com), Zhitao Li (zhitaoli@google.com), Karmel Allison (karmel@google.com) |
| **Updated**   | 2020-01-17                                                |

## Objective

### Goal

*   Support any TensorFlow Training loop in TFX Trainer in addition to
    tf.estimator, primarily focused on native Keras model.

### Non Goal

*   Natively support multi-worker distributed training by the system.
*   Non-TF training that generates savedmodel.

## Background and Motivation

In current TFX Trainer component, only tf.estimator is supported for training
and generating models. User provides a module file which contains a
`trainer_fn`, trainer will call the function to get the estimator model and
related spec for training, and generate a saved model by
`tf.estimator.train_and_evaluate`.

[tf.keras](https://www.tensorflow.org/guide/keras) is TensorFlow's high-level
API for building and training models. It’s currently supported in TFX by using
`tf.keras.estimator.model_to_estimator` in module file. User can create keras
model in their `trainer_fn` but need to convert it to estimator for return (for
example,
[cifar10](https://github.com/tensorflow/tfx/blob/r0.15/tfx/examples/cifar10/cifar10_utils.py)).

This doc will focus on native Keras support (without model_to_estimator) in TFX.
We propose changing the user facing API to be more generic so that users can do
(single node) native Keras model training within TFX.

## User Benefit

*   Allows non estimator based training, especially Keras as TensorFlow is
    establishing Keras as the
    [Standardized high-level API](https://medium.com/tensorflow/standardizing-on-keras-guidance-on-high-level-apis-in-tensorflow-2-0-bad2b04c819a).
*   Allows
    [custom training](https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough)
    for customization of training loop.

## Detailed Design

Below shows the pseudo code for current TFX Trainer’s executor:

```python
class Executor(base_executor.BaseExecutor):

 def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Uses a user-supplied tf.estimator to train a tf model locally."""
    trainer_fn = self._GetFn(exec_properties) # load from module file
    trainer_fn_args = self._GetFnArgs(
        input_dict, output_dict, exec_properties)

    training_spec = trainer_fn(trainer_fn_args)
    tf.estimator.train_and_evaluate(training_spec['estimator'], ...)
    # For TFMA (downstream evaluator and model validator component).
    tfma.export.export_eval_savedmodel(training_spec['estimator'], ...)
```

And the user supplied module file contains a function called `trainer_fn` which
returns an estimator:

```python
def _build_keras_model() -> tf.keras.Model:
  model = keras.XXX
  model.compile(...)
  return model

def trainer_fn(
    trainer_fn_args: trainer.executor.TrainerFnArgs) -> Dict[Text, Any]:
  """Build the estimator using the high level API.

  Args:
    trainer_fn_args: Holds args used to train the model as name/value pairs.

  Returns:
    A dict of the following:
      - estimator: The estimator that will be used for training and eval.
      - train_spec: Spec for training.
      - eval_spec: Spec for eval.
      - eval_input_receiver_fn: Input function for eval.
  """
  ...

  estimator = tf.keras.estimator.model_to_estimator(
     keras_model=_build_keras_model(), ...)

  return {
      'estimator': estimator,
      'train_spec': ...,
      'eval_spec': ...,
      'eval_input_receiver_fn': ...
  }

```

We propose that in generic trainer's module file, user not only need to provide
the model, but also control how the model is trained (`train_and_evaluate` for
estimator and `model.fit` for keras will be in user module file instead of in
executor), thus executor can be generic to model, and users can customize the
[training loop](https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough#training_loop).
The executor pseudo code would look like below:

```python
class Executor(base_executor.BaseExecutor):

 def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Train a user-supplied tf model."""
    run_fn = self._GetRunFn(exec_properties) # load from module file

    # run_fn_args contains
    #   1. input train and eval data path.
    #   2. desired output model path for the trained savedmodel.
    #   3. training args, e.g., train/eval steps.
    #   4. optional base model.
    #   5. optional tuning result (kerastuner.HyperParameters config).
    #   6. optional custom config for passing params from component.
    run_fn_args = self._GetRunFnArgs(
        input_dict, output_dict, exec_properties)

    run_fn(run_fn_args)
    # Validates the existence of run_fn's output savedmodel.
    ...
```

In module file, user needs to provide `run_fn` instead of previous `trainer_fn`.
The `trainer_fn` was responsible for creating the model, in addition to that,
`run_fn` also needs to handle training part and output the trained model to a
desired location given by run args:

```python
def run_fn(args: trainer.executor.TrainerFnArgs) -> None:
  """Build the TF model and train it."""
  model = _build_keras_model()
  model.fit(...)
  # Save model to args.serving_model_dir.
  model.save(...)
```

In generic trainer, executor is mainly for handling the
[artifact](https://github.com/tensorflow/tfx/blob/r0.21/docs/guide/index.md#artifacts)
(a unit of data that is passed between components), all model related logic is
user supplied.

A separate GenericExecutor will be created, and the existing trainer executor
will be sunsetted. We plan to keep estimator based executor for one more version
and then deprecate it.

### How to convert current estimator based module file

To convert the current estimator based module file (e.g.,
[iris](https://github.com/tensorflow/tfx/blob/r0.15/tfx/examples/iris/iris_utils.py))
for generic trainer, simply add a run_fn that calls the trainer_fn and train the
returned model (code that used to be in the trainer.executor.Do).

```python
def run_fn(fn_args: executor.TrainerFnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  schema = io_utils.parse_pbtxt_file(fn_args.schema_file, schema_pb2.Schema())

  # Reuse the trainer_fn.
  training_spec = trainer_fn(fn_args, schema)

  # Train the model
  absl.logging.info('Training model.')
  tf.estimator.train_and_evaluate(training_spec['estimator'],
                                  training_spec['train_spec'],
                                  training_spec['eval_spec'])
  absl.logging.info('Training complete.  Model written to %s',
                    fn_args.serving_model_dir)

  # Export an eval savedmodel for TFMA, note that for keras, eval savedmodel is
  # not needed as TFMA2 can use serving model for evaluation.
  absl.logging.info('Exporting eval_savedmodel for TFMA.')
  tfma.export.export_eval_savedmodel(
      estimator=training_spec['estimator'],
      export_dir_base=fn_args.eval_model_dir,
      eval_input_receiver_fn=training_spec['eval_input_receiver_fn'])

  absl.logging.info('Exported eval_savedmodel to %s.', fn_args.eval_model_dir)
```

### tf.distribute.Strategy

Distribution strategy will be user module's responsibilty with the new generic
trainer interface. To use it, user needs to modify the `run_fn()` in the module
file, below shows the pseudo code example for single worker and multi-worker
distribute strategy.

For single worker distribute strategy, you need to create an appropriate
[tf.distribute.Strategy](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy),
and move the creation and compiling of Keras model inside `strategy.scope`:

```python
def run_fn(args: trainer.executor.TrainerFnArgs) -> None:
  """Build the TF model and train it."""
  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    model = _build_keras_model()
  model.fit(...)
  model.save(...)
```

For multi-worker distribution strategy, the TFX Trainer does not have ability to
spawn multi-worker cluster by
[current executor](https://github.com/tensorflow/tfx/blob/r0.21/tfx/components/trainer/executor.py),
hence not covered in the scope of this RFC. If the execution environment of an
implementation of TFX Trainer has the ability to bring up the cluster of worker
machines, and execute user funtion in the workers with correct
[TF_CONFIG setup](https://www.tensorflow.org/guide/distributed_training#setting_up_tf_config_environment_variable),
such as GCP AI Platform Training service via
[extensions/google_cloud_ai_platform/trainer/executor.py](https://github.com/tensorflow/tfx/blob/r0.21/tfx/extensions/google_cloud_ai_platform/trainer/executor.py),
the `run_fn()` would look like below:

```python
def _is_chief() -> bool:
  """Decide whether the current worker's role is chief."""
  # Check TF_CONFIG (set by TFX when bring up the worker) in execution env.
  ...

def run_fn(args: trainer.executor.TrainerFnArgs) -> None:
  """Build the TF model and train it."""
  ps_strategy = tf.distribute.experimental.ParameterServerStrategy()
  with ps_strategy.scope():
    model = _build_keras_model()
  model.fit(...)
  if _is_chief():
    model.save(...)
```

For details about `tf.distribute.Strategy`, please refer to
[here](https://www.tensorflow.org/guide/distributed_training).

## Future work

*   Examples for custom training loop.
*   Native support for multi-worker distribution.
