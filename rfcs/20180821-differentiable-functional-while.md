# Functional while_loop
| Status        | Accepted                                             |
:---------------|:-----------------------------------------------------|
| **Author** | Saurabh Saxena (Google) |
| **Sponsor**   | Skye Wanderman-Milne (Google)                 |
| **Updated**   | 2018-03-01                                           |
| **Implementation** | [while_v2.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/while_v2.py) |


## Objective

This proposal talks about an implementation of [while_loop](https://www.tensorflow.org/api_docs/python/tf/while_loop) which adds a single While op to the GraphDef as opposed to the current implementation that uses [lower level primitives](https://arxiv.org/abs/1805.01772). The goal is to simplify debugging and other analysis and to make it easier for compiler backends like XLA to [recognize](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/tf2xla/functionalize_while.cc) the while loop in the GraphDef. At runtime, a C++ optimization pass will lower this op to the primitive dataflow ops for feature parity with the current implementation similar to how we do for the [If op](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/lower_if_op.cc).


## Motivation

TensorFlow provide two flavours of control flow constructs which differ widely in the way they manifest themselves in the GraphDef:



1.  Functional ops which create a single node in the Graph: [If](https://github.com/tensorflow/tensorflow/blob/fc4504edb1ab419ae59b0ebb9ff8d943beb61117/tensorflow/core/ops/functional_ops.cc#L104), [While](https://github.com/tensorflow/tensorflow/blob/fc4504edb1ab419ae59b0ebb9ff8d943beb61117/tensorflow/core/ops/functional_ops.cc#L147).
1.  Non-functional ops which make use of primitive control flow constructs namely Enter, Exit, Switch, Merge and NextIteration: [tf.cond](https://www.tensorflow.org/api_docs/python/tf/cond), [tf.while_loop](https://www.tensorflow.org/api_docs/python/tf/while_loop).

Both approaches have their merits and demerits. The functional representation emits a single node in the GraphDef thus making it easy to recognize such ops in processing pipelines that operate on the GraphDef, which is not the case when control flow is represented using lower level primitives. The functional representation is however not easily differentiable and requires using the [SymbolicGradient](https://github.com/tensorflow/tensorflow/blob/a0e76ce73c5f095fc61e06c19ff8e653cfd2965c/tensorflow/core/ops/functional_ops.cc#L24) op which recomputes the forward pass(slow) and needs symbolic gradients defined for all ops in the function body which can be complicated to implement. Also since we force a strict execution of functions, i.e., a function can start executing only after its inputs are all ready, the functional ops may not be that performant. The current representation solved these problems at the cost of a slightly complicated GraphDef. In this proposal, we try to achieve the best of both worlds.

We recently added a differentiable version of the [functional If/cond op](https://github.com/tensorflow/community/blob/master/rfcs/20180507-cond-v2.md). As with functional cond, a key challenge here is to figure out gradient computation. For cond, we could expose the [intermediate tensors](https://github.com/tensorflow/tensorflow/blob/51100a8de57ef53e36a8a9f5a9829cbd33fbed04/tensorflow/python/ops/cond_v2_impl.py#L114) as op outputs so that they could be used for computing gradients. We cannot directly do the same for while loops since we would need the intermediate values _for all iterations_ and not just the values after the last iteration. Hence, some sort of accumulator is required. We use TensorLists for accumulating the loop body intermediates. Since while loops may run for a large number of iterations, e.g. long RNNs,  we need to be mindful of the memory usage by accumulators.


## Design Proposal


### Computing gradients

Excerpt from white paper on [Control Flow in TensorFlow](http://download.tensorflow.org/paper/white_paper_tf_control_flow_implementation_2017_11_1.pdf):

> Intuitively, the gradient of `while_loop(pred, body)` is just a while loop of the form:
>
>
> ```
> def pred(i, _): return i < N
> while_loop(pred, g_body, [0] + g_vars)
> ```
>
> Where `N` is the number of iterations that the forward while loop runs, `g_body` is the gradient of the forward loop body, and `g_vars` is the initial values for the loop variables. As we will see later, `g_vars` includes the initial gradients for the loop variables of the forward while loop.

We use the same logic here as well. To compute *g_body* we use the [gradients_impl._GradientsHelper](https://github.com/tensorflow/tensorflow/blob/600caf99897e82cd0db8665acca5e7630ec1a292/tensorflow/python/ops/gradients_impl.py#L599) function which supports computing the gradient of a given [src_graph](https://github.com/tensorflow/tensorflow/blob/600caf99897e82cd0db8665acca5e7630ec1a292/tensorflow/python/ops/gradients_impl.py#L607) in another graph, which in this case is a [FuncGraph](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/func_graph.py#L117). This gradient graph captures references to the intermediate values of the forward graph (the src_graph). Each iteration of *g_body* needs to use the intermediate values from the corresponding iteration of *body*; thus, we accumulate the needed forward values and replace these references with values from the accumulators. See the *Accumulating intermediates* section below.

To get a count of the number of forward iterations we add an integer counter to every while_loop which is initialized to 0 and is incremented in the loop body. Note that we just need the total number of iterations for the gradient pass so we do not need to accumulate the intermediate values of the counter. This counter is always the first output of the While op.

Note that the [While gradient function](https://github.com/tensorflow/tensorflow/blob/ec1effdb69d33c947f30a5155c5cc4104c07a87e/tensorflow/python/ops/while_v2.py#L248) assumes that the first loop output is the
number of loop iterations.  The While op generated by the gradient function satisfies the above constraints and hence can be differentiated again to generate higher-order derivatives. However, arbitrary While ops generated outside of this design may be differentiated incorrectly.


### Accumulating intermediates


#### Stack vs TensorArray vs TensorList

The current implementation uses [Stacks](https://github.com/tensorflow/tensorflow/blob/51100a8de57ef53e36a8a9f5a9829cbd33fbed04/tensorflow/python/ops/control_flow_ops.py#L1002) for accumulating intermediate values from the forward pass that may be needed for gradient computation. The new implementation uses [TensorLists](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/list_ops.cc)(TL) instead which, unlike Stack and TensorArray, do not have a mutable internal state making them easy to differentiate.


#### Algorithm

After constructing the gradient function of the loop body, we rewrite the loop body function and forward While op to output the needed intermediates. Specifically, for each intermediate tensor of the loop body function that's be needed for the gradient computation, we create an empty TensorList (TL) and add it to the list of forward input loop_vars. In the forward loop body, we then push the intermediate values to the TL using the [TensorListPushBack](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/list_ops.cc#L40) op, and output the final list of accumulated intermediates. The gradient computation then takes the TL as input and pops intermediate values off the TL.

In the case of nested while loops, we will accumulate the intermediate values of inner while loops in nested TensorLists.

To-be-implemented improvements to this algorithm:
*   Do not accumulate Const nodes. We can lift these outside the while loop.
*   Do not accumulate loop vars that are passed-through unchanged.

Alternatives considered:
*  Output every possible intermediate and rely on pruning to clean it up. The original implementation did this, but we changed it to match tf.function.
*   Expose only those intermediate values that are required by the backward pass by building the gradient graph in the forward pass.
    *   This will increase graph building time.


### Memory management

tf.while_loop swaps the tensors from GPU to CPU when the [swap_memory](https://github.com/tensorflow/tensorflow/blob/600caf99897e82cd0db8665acca5e7630ec1a292/tensorflow/python/ops/control_flow_ops.py#L3046) flag is set. Section 5.3 of the control flow [paper](https://arxiv.org/abs/1805.01772) mentions that with memory swapping they were able to handle an RNN with 2x the unrolled length(1000 vs 500) with little overhead. The heuristics for memory swapping are implemented in the [StackPush](https://github.com/tensorflow/tensorflow/blob/600caf99897e82cd0db8665acca5e7630ec1a292/tensorflow/core/kernels/stack_ops.cc#L289) and [StackPop](https://github.com/tensorflow/tensorflow/blob/600caf99897e82cd0db8665acca5e7630ec1a292/tensorflow/core/kernels/stack_ops.cc#L411) ops. We will need to support similar functionality for TensorListPushBack and TensorListPopBack ops.


### Lowering pass

In order to get feature parity with the current implementation we will lower the While op to the current while loop graph representation as a grappler pass similar to the one for [if_op](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/lower_if_op.cc). This gets us around some of the issues with the current functional op:



1.  We can perform parallel iterations which are not possible due to the strict mode execution of functions which requires that all inputs to the function must be ready before the function can start executing. We will need to add a `parallel_iterations` attr to the While op.
1.  We can perform global grappler optimizations without needing to cross function boundaries. E.g. we can remove accumulators for intermediate values which are not consumed downstream.


### Example

```python

x = tf.constant(2.)

ret = while_loop(lambda v: v < 8., lambda v: v * v, [x])

grad = tf.gradients(ret, [x])

```

**Current implementation**



![alt_text](20180821-differentiable-functional-while/while_v1.png "image_tooltip")


**New implementation**



![alt_text](20180821-differentiable-functional-while/while_v2.png "image_tooltip")


The forward functional while op is highlighted in <span style="color:#ff0000;">red</span>. Note that it takes 2 `Const` nodes as inputs. One of the `Const` nodes is `x` with value 2. The other `Const` node is the initial value of the loop counter which is set to 0. There are also 2 `EmptyTensorList` nodes which are used for accumulating intermediate values.

*while_cond*

The loop condition function is fairly trivial. It expects the extra args for the loop counter and accumulators but doesn't actually use them.



![alt_text](20180821-differentiable-functional-while/while_cond.png "image_tooltip")


*while_body*

The loop body contains the extra nodes for updating the counter and accumulating intermediates.



![alt_text](20180821-differentiable-functional-while/while_body.png "image_tooltip")


`arg0` is the loop counter which gets initialized to 0. This is always the first argument.

`arg1` is the value of x at the start of each iteration.

`add_0` is the counter update node and `y` is the increment `Const` node with value 1.

`mul_0` performs `x * x`


Accumulators:

`tensorlist0` <- `arg1`, the value of `x` at the start of the loop.

`tensorlist1` <- Output of `mul_0`.

## Discussion notes

Please see notes in [tensorflow/community#13](https://github.com/tensorflow/community/pull/13#issuecomment-422591773).

## Update log

2019-03-1: Updated some sections to reflect what was built and added link to implementation.
