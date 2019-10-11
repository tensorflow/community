# Unify RNN Inteface

| Status        | Accepted                                             |
| ------------- |:---------------------------------------------------- |
| **Author(s)** | @qlzh727 (scottzhu@google.com)                       |
| **Sponsor**   | @ebrevdo (ebrevdo@google.com), @drpng (drpng@google.com)|
| **Updated**   | 2018-08-24                                           |

## Objective

Unify RNN (LSTM/GRU/other recurrent models) interfaces between TF RNN and Keras RNN at the release 
of TensorFlow 2.0.

## Motivation

Recurrent neural networks [RNN](https://en.wikipedia.org/wiki/Recurrent_neural_network) is the deep 
neural network that works great with sequential input data like text and voice. There are two sets 
of APIs in TensorFlow that allows users to build a RNN. The APIs are a bit different from each 
other, but trying to solve the same problem. This situation creates some user confusions about:

1. What's the recommended API to use.
1. What's the relationship between those 2 APIs and how they work together (which they don't 
currently).

Ideally those two APIs should be unified into "one true API", which should be:
1. Intuitive to use (hide unnecessary complexity from user).
1. Covers most of the common use cases.
1. Easy to migrate from either side.

Keras has become the recommended high level API for Tensorflow since its easy to understand and use.
We would prefer to unify the final API that is similar to existing Keras API, and port 
functionalities from TF RNN to Keras.


## Design Proposal

1. Update RNN cells in Keras and TF RNN, so that all of them will work with Keras RNN layer and TF 
RNN layer.
1. Remove the duplicated RNN cells between Keras and TF, update the documentation for the migration 
path.
    1. tf.nn.rnn_cell.BasicRNNCell -> tf.keras.SimpleRNNCell
    1. tf.nn.rnn_cell.BasicLSTMCell,  tf.keras.LSTMCell -> tf.nn.rnn_cell.LSTMCell
    1. tf.nn.rnn_cell.GRUCell -> tf.keras.GRUCell
    1. tf.nn.rnn_cell.MultiRNNCell -> tf.keras.StackedRNNCells
1. Move existing tf.nn.rnn* code from tensorflow/python/ops/ to tensorflow/python/rnn. The RNN layer
and cell are not tensorflow ops, and is a bit weird to stay under ops. Externally the tf.export will
ensure the object/function stays the same.
1. Migrate selected RNN cell from contrib to core (tf/python/rnn). See detailed section below.
1. Still keep both TF and Keras RNN layer API in TF 2.0, since both APIs are heavily used. Update 
documentation to state that Keras is the preferred API to use, and encourage user to use Keras. Add 
warning message in TF RNN layer, point user to documentation for suggested high level RNN layer API.
1. (Stretch goal) unify CuDNN implementation (LSTM and GRU) with standard implementation by Defun 
approach proposed by drpng@.

## Detailed Design

There are two parts of overall RNN APIs: <b>RNN layer (interface)</b> and <b>RNN cell</b>.

<b>RNN layer</b> is composed by RNN cell, and it connects the output and states from previous 
timestep and feed them to the cell. It drives the run loop and responsible for returning the final 
output and states. User has to use RNN layer interface to get the output tensor. 

<b>RNN cell</b> is the individual component within RNN layer. It calculates the output for the 
current timestep based on the input from previous timestep. It contains the weights within itself 
and defines the actual behavior of the numerical calculation. User cannot use RNN cell alone to get 
output, and it has to work with RNN layer interface. 


### RNN Layer
Following table lists out the current status for RNN layer between 2 APIs:


| TF RNN                                        | Keras RNN                  |
|:--------------------------------------------- |:-------------------------- |
|tf.nn.raw_rnn                                  | tf.keras.layers.RNN        |
|tf.nn.static_rnn                               | tf.keras.layers.SimpleRNN  |
|tf.nn.static_state_saving_rnn                  | tf.keras.layers.LSTM       |
|tf.nn.static_bidirectional_rnn                 | tf.keras.layers.GRU        |
|tf.nn.dynamic_rnn                              | tf.keras.layers.ConvLSTM2D |
|tf.nn.bidirectional_dynamic_rnn                |                            |
|tf.contrib.rnn.stack_bidirectional_rnn         |                            |
|tf.contrib.rnn.stack_bidirectional_dynamic_rnn |                            |

Clearly the two API are very different, and there isn't any direct translation between them.

TF RNN API is based on a combination of:
1. static/dynamic: 
    1. Static RNN has the length of timesteps fixed and uses a symbolic loop to iterate 
       over the timestep. 
    1. Dynamic RNN can have various timestep length within the same batch and use tf.while_loop to 
    parallelize the loop, which is faster compared to static, but use more memory. 
    
    Both static and dynamic allows the run loop to stop early for the shorter input data within the 
    batch to gain performance.
1. single_direction/bidirectional: 
    1. bidirectional RNN is unifying the output of two single_direction RNN together, and one of 
    them has the input/output in reversed order.

1. tf.nn.raw_rnn allows user to define the run loop given the input data and output states from each
step. It provides some flexibility to user, but we don't really see a real concrete use case for 
this. We will probably remove this in the public API in 2.0.

1. tf.nn.static_state_saving_rnn has been quite commonly used. This is used when input sequence 
length is long and cannot fit into the memory. The state saver allows end states from previous batch 
to be fed to the next batch as init state, which is equivalent to dividing long sequence, and concat
them together. Keras has similar implementation with "stateful" RNN layer. The state of the Keras 
RNN layer need to be reset every time when the whole trunk of batches of input are processed, which 
requires some callback/listener.  

1. Keras RNN has 5 public layers, within which, 4 of them are derived from the base one by inputting 
different cell type, which is tf.keras.layers.RNN. It achieves the static/dynamic functionality by 
the parameter unroll, and bidirectional via Bidirectional wrapper.

So semantically:

| TF RNN                                        | Keras RNN                  |
|:----------------------------------------------|:-------------------------- |
|tf.nn.static_rnn(cell)                                 |tf.keras.layers.RNN(cell, unroll=True)    |
|tf.nn.dynamic_rnn(cell)                                |tf.keras.layers.RNN(cell)                 |
|tf.nn.static_bidirectional_rnn(cell_fw, cell_bw)       |keras.layers.Bidirectional(tf.keras.layers.RNN(cell, unroll=True)) |
|tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw)      |keras.layers.Bidirectional(tf.keras.layers.RNN(cell))              |
|tf.nn.static_state_saving_rnn(cell)                    |tf.keras.layers.RNN(cell, stateful=True)  |
|tf.contrib.rnn.stack_bidirectional_rnn(cells_fw, cells_bw)|for i in range(num_layers):<br>model.add(bidirectional(RNN(cell, unroll=True)))|
|tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw)|for i in range(num_layers):<br>model.add(bidirectional(RNN(cell)))|

Apart from the semantic similarity, the two APIs are different from each other in following aspects:
 
1. TF RNN is implemented as function while Keras RNN is object based. This allows Keras to preserve 
states by layer itself, which kind of fulfill the use case like tf.nn.static_state_saving_rnn.

1. TF RNN expect inputs as list of tensor with size T (timestep), and each of the tensors has shape 
[batch_size, input_dim], while Keras requires input to be a 3D tensor with shape 
[batch_size, timestep, input_dim].

1. TF RNN accepts input data with more than 3 dimensions, eg [timestep, batch_size, ...], where […] 
is the real dimension for data entity. Keras RNN expect input rank to be 3, but the underlying code 
that loops through all the timesteps of the input data is supporting highly dimension inputs.

1. The output of TF RNN is a pair with (outputs, state), where outputs are the list of the output 
for each of the timestep, state is the final state for the last timestep. The outputs also changes 
in the dynamic RNN based on time_major params. Keras RNN return type is different based on the init 
parameter. By default it only returns the output for the last timestep, and user can choose to 
return additional output for all timesteps, as well as the final states for the last timestep.

#### Changes to make
1. Add support nested input/output/state for Keras.RNN layer, which is one of the feature gap between
   Keras RNN and TF RNN.
1. Add support for time-major input tensor (time, batch, feature).
1. Provide clear example about how to convert from existing TF RNN to Keras RNN interface.
1. Make TF RNN as deprecated, but still available in tf.v1_compatible, eventually delete it 
   in 2.X.
       
### RNN Cell
Following table lists out the current status for RNN cells between 2 APIs:

| Keras RNN | TF RNN (tf.nn.rnn_cell) | Comment |
|:----------|:----------------------- |:------  |
| |RNNCell|Abstract base class.|
| |LayerRNNCell|Not publicly exposed. Layer support for RNNCell.|
|SimpleRNNCell|BasicRNNCell|Identical implementation of vanilla RNN, the underlying weights are structured differently.|
|LSTMCell|BasicLSTMCell|No peephole, clipping, projection. Keras allows kernel_activation to be customized (default=hard_sigmoid)|
| |LSTMCell|Support peephole, clipping and projection.|
|GRUCell|GRUCell|Identical implementation, the underlying weights are structured differently.|
|StackedRNNCells|MultiRNNCell|Identical implementation|
| |DeviceWrapper| |
| |DropoutWrapper|Keras support dropout at the cell level and configured by the init parameters. All of the Keras cell support input and state dropout. DropoutWrapper support output dropout as well. Both Keras and DropoutWrapper support variational dropout|
| |ResidualWrapper| |
|BidirectionalWrapper| | |

Apart from the cells in tf.nn.rnn_cell, there are more cells/wrappers in tf.contrib.rnn. Part of the 
cell/wrapper are duplicate with existing tf.nn.rnn_cell, and the rest part provides extra 
functionality. See sections below.

The interface between TF RNN and Keras RNN APIs are actually similar.

Both of them requires:

```python
def call(self, inputs, prev_state):
  # Calculate the output and new_state based on the cell type.
  return output, new_state
```

```python
@property	
def state_size(self):
  # The size of the internal state for each step. 
  return state_size
```

In addition, TF RNN cell requires:

```python
@property
def output_size(self):
  # The size of output.


def zero_state(self, batch_size, dtype):
   # return the initial state for timestep 0, which is used when caller didn't 
   # specify any initial state when constructing the layer.
```

Keras RNN cell requires cell to implement the following method:

```python
def get_config(self):
  # return the dict of the cell attribute, which can be used to rebuild the cell.
```

#### Changes to make
1. Update the Keras RNN cell to support the extra two methods required for TF RNN cell. This will 
enable all the cells to be used by both layer interfaces.
1. Deprecate the duplicated RNN cells in TF RNN and suggest user to use the Keras equivalent, 
namely:
    1. nn.BasicRNNCell -> keras.SimpleRNNCell
    1. (nn.BasicLSTMCell, nn.LSTMCell) -> keras.LSTMCell
    1. nn.GRUCell -> keras.GRUCell
    1. nn.MultiRNNCell -> keras.StackedRNNCells
    1. (Optional) Remove the individual dropout param from Keras cells and preferred the DropWrapper 
        from TF RNN.

1. All the RNN cells need to have a unified namespace. 

The side effects of deleting cell is that previously saved checkpoint will not work, and maybe TF 
2.0 is a good time to do this.

### RNN Cell in tf.contrib.rnn

There are extra implementations for various RNN cells and wrappers in tf.contrib.rnn. Since 
tf.contrib is going away in TF 2.0. It is a good time to do the housekeeping and decide whether they
should be moved to core, or move to newly introduce SIG-Addon repository.

|Name|Usage|Comment|
|:---|:----|:------|
|EmbeddingWrapper|low usage within google|Leave in v1.compat due to low usage and performance|
|InputProjectionWrapper|low usage within google|Leave in v1.compat due to low usage and performance|
|OutputProjectionWrapper|low usage within google|Leave in v1.compat due to low usage and performance|
|FusedRNNCellAdaptor|low usage within google|Leave in v1.compat due to low usage|
|TimeReversedFusedRNN|low usage within google|Leave in v1.compat due to low usage|
|GRUBlockCell|Deprecated, parent class for GRUBlockCellV2.| |
|GRUBlockCellV2|low usage within google|Has a customized ops and grad implemented in c. Much better in performance except using XLA. Leave in v1.compat.|
|LSTMBlockCell|Medium usage within google|Similar as GRUBlockCell|
|LSTMBlockFusedCell|Medium usage within google|Even better than LSTMBlockCell, which further fuse the timestep and put the whole LSTM into one op.|
|LayerNormBasicLSTMCell|High usage within google|This class should actually be replaced with LayerNormLSTMCell which supports more features.|
|LayerNormLSTMCell|low usage within google|Should move to core and replace LayerNormBasicLSTMCell.|
|CoupledInputForgetGateLSTMCell|low usage within Google|Leave in v1.compat due to low usage|
|TimeFreqLSTMCell|low usage within Google|Leave in v1.compat due to low usage|
|GridLSTMCell|low usage within Google|Leave in v1.compat due to low usage|
|BidirectionalGridLSTMCell|low usage within Google|Leave in v1.compat due to low usage|
|NASCell|low usage within Google|Move to new Addon repository|
|UGRNNCell|low usage within Google|Move to new Addon repository|
|IntersectionRNNCell|low usage within Google|Move to new Addon repository|
|PhasedLSTMCell|low usage within Google|Leave in v1.compat due to low usage|
|ConvLSTMCell<br>Conv1DLSTMCell<br>Conv2DLSTMCell<br>Conv3DLSTMCell|low usage within Google|Keras has a Conv2DLSTM implementation and this is more generic. Probably port this core|
|GLSTMCell|low usage within Google|Move to new Addon repository|
|SRUCell|low usage within Google|Move to new Addon repository|
|IndRNNCell|low usage within Google|Move to new Addon repository|
|IndyGRUCell|low usage within Google|Move to new Addon repository|
|IndyLSTMCell|low usage within Google|Move to new Addon repository|
|AttentionCellWrapper|Medium usage|Move to core|
|HighwayWrapper|low usage within Google|Move to new Addon repository|


### CuDNN implementation vs Normal implementation

NVidia published the CuDNN support for RNN in the [blog post](https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/), 
which provides a fast underlying implementation for LSTM and GRU. The downside however, is that it 
does not support all the features that extended from various varieties of LSTM and GRU.

Currently both TF RNN and Keras RNN are exposing CuDNN cells as different type from the basic LSTM 
and GRU cell. One of the goals here is to hide this details from user and choose smartly for user. 
This idea is original proposed by drpng@.

The CuDNN implementation fused several inputs and internal steps together to achieve the performance
gain:

1. Merge 4 internal matmul into 1 within LSTM cell
1. the timestep loop
1. multiple layers stacked together
1. bidirectional RNN

It also has few differences from the original LSTM/GRU implementation:

1. The output projection in CuDNN has an extra bias, which cause the weights of the CuDNN 
incompatible with the standard LSTM/GRU. There are internal effort to convert the weights between 
a CuDNN implementation and normal TF implementation. See CudnnLSTMSaveable.
1. CuDNN does not support variational recurrent dropout, which is a quite important feature.
1. CuDNN implementation only support TANH activation which is also the default implementation in the 
LSTM paper. The Keras one support more activation choices if user don't want the default behavior.

With that, it means when users specify their LSTM/GRU layer, the underlying implementation could be 
swapped to use CuDNN version if it meets certain criteria:

```python
class LSTM(RNN):
  def __init__(self, .....):
     if activation == 'tan' and dropout == 0 and use_bias == True:
       self.could_use_cudnn = True
     else:
       self.could_use_cudnn = False
     ....


  def build(self):
    # TODO: since the real implementation is unknown at this point, maybe do 
    # the init of weights for standard implementation here. At call time, if the
    # weights are loaded from standard LSTM but the implementation is CuDnn, some
    # transformation like CudnnLSTMSaveable is needed back and forth.

  def call(self, input, mask=None, initial_state=None, ....):
    if self.could_use_cudnn:
      @tfe.function.Defun(..., api_interface="LSTM", perferred_device="GPU")
      def cudnn_impl(lstm_layer, input, mask, initial_state):
        # reuse the existing cudnn impl
        return [output] + states
      result = cudnn_impl(self, input, mask, initial_state)
    @tfe.function.Defun(..., api_interface="LSTM", perferred_device=None)
    def generic_impl(lstm_layer, input, mask, initial_state):
      # reuse the existing standard LSTM impl
      return [output] + states
    
    result = generic_impl(self, input, mask, initial_state)
    # Note that the `self` python instance is also passed in here. The instance is only used to 
    # access the functions in the parent class or current instance. The attribute of the instance 
    # should be accessible within the function, but is not writable. Ideally the defun body should 
    # be stateless, and the only thing it need to access is the weights tied to the layer/cell.
    
    return result
  # Note that both implementation is called which means both will be created in the
  # graph as subgraph. The grappler plugin will pick one of them based on the 
  # hardware availability and remove the other one from graph. 
```

An extra grappler plugin is needed to pick the real implementation based on the hardware.

#### Extend to multi layer and bidirectional
The snippet above illustrates the single layer LSTM with one direction. The CuDNN implementation 
actually allows bidirectional layer and multiple LSTM layers being fused together to achieve further 
performance gain. 

An example of implementation can be found in CuDNNLSTM in contrib. With this implementation, it 
provides more dense API that user only need to specify the number of layers, number of cells for 
each layer, and whether its unidirectional or bidirectional. 

## Testing
The unit test should be added to cover the correctness of the implementation. Since most of the 
layers/cells already have unit test, the change should ensure those existing tests are still 
passing.

Performance test is also needed for the stage 3 change to make sure "Defun" does not introduce extra 
performance overhead.

## Questions and Discussion Topics

Please add comment to the PR and author will populate them here.
