# Title of RFC

| Status        | (Proposed / Accepted / Implemented / Obsolete)       |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | My Name (me@example.org), AN Other (you@example.org) |
| **Sponsor**   | A N Expert (whomever@tensorflow.org)                 |
| **Obsoletes** | TF-RFC it replaces, else remove this header          |

## Objective

What are we doing and why? What problem will this solve? What are the goals and
non-goals? This is your executive summary; keep it short, elaborate below.

## Motivation

Why this is a valuable problem to solve? What background information is needed
to show how this design addresses the problem?

How will users (or other contributors) benefit from this work? What would be the
headline in the release notes or blog post? What data supports
this? What related work exists?

## Proposed Solution
Describe your solution to the problem. Provide examples and describe how they work. Show how your solution is better than current workarounds (if there are any).

## Detailed Design

Describe the design of the solution in detail. The detail in this section should be sufficient for someone who is not one of the authors to be able to reasonably implement the feature. 

If the design affects API (new, changed, removed, upgraded from experimental), please describe the changes in detail. Here is more [information](https://github.com/tensorflow/community/blob/master/governance/api-reviews.md) about what API owners are looking for. 

For new changed API, show the full API and its documentation comments detailing what it does along with end-to-end examples (ideally, a tutorial) showing it's use. 
* The minimum requirements for this are to consider how this would be used in a Keras-based workflow, as well as a non-Keras (low-level) workflow. If either isn’t applicable, explain why.
* It should show the usage of the new feature in an end to end example (from data reading to serving, if applicable). Many new features have unexpected effects in parts far away from the place of change that can be found by running through an end-to-end example. TFX [Examples](https://github.com/tensorflow/tfx/tree/master/tfx/examples) have historically been good in identifying such unexpected side-effects and are as such one recommended path for testing things end-to-end.
* This should be written as if it is documentation of the new feature, i.e., consumable by a user, not a TensorFlow developer. 
* The code does not need to work (since the feature is not implemented yet) but the expectation is that the code does work before the feature can be merged.

## User Impact
What are the user-facing changes? How will this feature be rolled out?

## Engineering Impact
* Do you expect changes to binary size / startup time / build time / test times?
* Who will maintain this code? Is this code in its own buildable unit? Can this code be tested in its own? Is visibility suitably restricted to only a small API surface for others to use?

## Performance Implications
* Do you expect any changes to performance (speed / memory)? How will you confirm?
* There should be microbenchmarks. Are there?
* There should be end-to-end tests and benchmarks. If there are not (since this is still a design), how will you track that these will be created?

## Compatibility
* Does the design conform to the backwards & forwards compatibility [requirements](https://www.tensorflow.org/programmers_guide/version_compat)?
* How will this proposal interact with other parts of the TensorFlow Ecosystem?
    - How will it work with TFLite?
    - How will it work with distribution strategies?
    - How will it interact with tf.function?
    - Will this work on GPU/TPU?
    - How will it serialize to a SavedModel?

## Dependencies
Does this proposal add any new dependencies to TensorFlow? Are there other areas of TensorFlow or things that use TensorFlow (TFX/pipelines, TensorBoard, etc.) that this affects? How have you identified these dependencies and are you sure they are complete? If there are dependencies, how are you managing those changes?

## Platforms and Environments
* Platforms: does this work on all platforms supported by TensorFlow? If not, why is that ok? Will it work on embedded/mobile? Does it impact automatic code generation or mobile stripping tooling? Will it work with transformation tools?
* Execution environments (Cloud services, accelerator hardware): what impact do you expect and how will you confirm?

## Best Practices
Does this proposal change best practices for some aspect of using/developing TensorFlow? How will these changes be communicated/enforced?

## Tutorials, Examples and Documentation
What is the plan for documentation? 

## Alternatives
If there are alternatives that you have eliminated, you should also list those here, and explain why you believe
your chosen approach is superior. List pros/cons to each approach. 

## Questions and Discussion Topics
Seed this with open questions you require feedback on from the RFC process.

## Design Review Notes
Please post the notes from the Design Review here.
