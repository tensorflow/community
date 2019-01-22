# More Flexible Concurrency

| Status        | (Proposed)                                           |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Gabriel Perdue (gnperdue@gmail.com),                 |
|               | Christopher Jones (cdj@fnal.gov),                    |
|               | Matti Kortelainen (matti@fnal.gov)                   |
| **Sponsor**   | Ravi Chirravuri (crk@google.com)                     |
| **Updated**   | 2018-10-29                                           |

## Objective

This document presents a proposal to implement a more flexible thread
scheduling system. This would enable users to plug in a different thread
management system (e.g., Intel TBB) as opposed to using only libraries
TensorFlow is dependent on.

## Motivation

Many users that work in large scale distributed environments are bound to
different multithreading solutions, typically due to the need to operate within
a specific software framework. When this sort of user utilizes TensorFlow from
within their regular resource management framework, a lack of flexibility in
the TensorFlow concurrency system can make it difficult to utilize resources
optimally.

For example, in High Energy Physics (HEP), we do a lot of very large scale
batch processing. Our goal is generally "high throughput computing" - we want
to process an enormous number of data chunks as fast as possible, but we don't
care about the order in which they are processed at all. We would like more
hooks to manage threads, something like a plug-in mechanism for concurrency
engines.

Each chunk needs to be processed by a large number of tasks (from a few tens to
a few thousand) where each task is doing a transformation on a given chunk. Our
processes use multiple threads where we can process multiple chunks in parallel
as well as multiple independent tasks all transforming parts of the same data
chunk. A major constraint is our processes must restrict the number of
processing threads they use to a number assigned to them at process start. This
is needed since we use computing resources shared by multiple users with each
user assigned a certain number of computing cores they are allowed to use.
(NOTE: a small number of primarily 'waiting' threads is permissible, as long as
the processing time used by those threads is negligible.)

We use the Intel Thread Building Blocks (TBB) library to manage a thread pool
of processing tasks. We tell TBB the total number of cores it is allowed to use
and that sets our processing limits for the process. In order to use TensorFlow
based inferences in this system, we modified the Session class to not use any
threads. This allowed the inference to be evaluated completely on the TBB
managed thread which is running the task needed the inference is using. This
works, but we would like to be able to do better. There are cases where the
total number of TBB tasks available to the process at a given time is less than
the number of threads available to TBB. Under those cases, we would like to be
able to use the underutilized threads to speed up the TensorFlow inference
calculation. This could be accomplished if TensorFlow had a mechanism to allow
alternative concurrency engines to be 'plugged into' the system. We would then
use those APIs to implement a TBB based concurrency engine. Such a facility
could also be used to allow OpenMP, another common scientific concurrency
engine, to be developed as well.

To be clear - we are not asking for TBB support, but for a way to add it
ourselves as a pull request. This would also enable other users to write code
to support, for example, OpenMP.

## Design Proposal

The most important features we require are:

1. Setup state for the execution engine on each call to Session::Run. For TBB
 this session would include the creation of a `tbb::task_arena` and a `tbb::task_group`.
 The `tbb::task_arena` is used to isolate the tasks associated with this instance
 of a TensorFlow inference from all other running TBB tasks. In that way if a TBB task
 pauses and allows other TBB tasks to be run on its threads, a TensorFlow task will never
 be used (similarly if a TensorFlow task pauses and allows other TBB tasks to run, only 
 TensorFlow tasks from that same inference will be run on that thread). Such behavior avoids
 possible deadlocks as well as performance degredation if a 'stolen' task takes much longer
 than the remaining tasks for a given inference. The `tbb::task_group` provides a mechanism
 to wait for all TBB tasks which were started for a given inference.
2. Use that state as part of the args given to 'executor->RunAsync'. For example, the 
   state could be contained within the `default_runner`. This state is then used to 
   schedule each closure to be run. For the TBB case, each `std::function<void()>` closure
   would be executed within the proper tbb::task_arena and as part of the proper
   `tbb::task_group`.
3. Use that state as part of the Session::WaitForNotification to allow the thread 
 doing the Session::Run call to be used for processing closures associated with
 the inference as well as to wait for all outstanding tasks associated with
 the inference to finish. This also would be the time for the state to be cleaned
 up. For the TBB case, the wait would be done using the `tbb::task_group` and
 make use of the `tbb::task_arena`. Waiting on a `tbb::task_group` allows TBB
 to run other TBB tasks associated with that thread's `tbb::task_arena` on that
 thread. Using the `tbb::task_arena` from the state guarantees that only TBB
 tasks associated with the given inference will be run on the thread which called
 Session::Run.

We believe it is possible to leverage the newly introduced `RunHandlerPool`
instead of a new session Class. At a high level this would introduce a new
implementation of `RunHandlerPool` that works with TBB instead of Eigen, and
returns a `TBBRunHandler` on `RunHandlerPool::Get()`. We may want to make the
existing `RunHandler` an interface with a pure virtual method for
`ScheduleInterOpClosure()`. This would produce our required control over
running inter-op closures. This interface currently doesn't support intra-op
parallelism, so we assume our second feature is satisfied in this case.

Each `RunHandler` object is active only during a session run, so we should get
control before the destruction.

## Detailed Design

(This section is optional. Elaborate on details if they’re important to
understanding the design, but would make it hard to read the proposal section
above.)

## Questions and Discussion Topics

(Seed this with open questions you require feedback on from the RFC process.)
