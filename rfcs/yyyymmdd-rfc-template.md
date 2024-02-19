# Build Managment System Of Freelancers By Registering 

| Status        | (Proposed / Accepted / Implemented / Obsolete)       |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [NNN](https://github.com/tensorflow/community/pull/NNN) (update when you have community PR #)|
| **Author(s)** | Joe (hyeon.woo@ideamood.co.kr), AN Other (you@example.org) |
| **Sponsor**   | A N Expert (whomever@tensorflow.org)                 |
| **Updated**   | 2023-07-22                                           |
| **Obsoletes** | TF-RFC it replaces, else remove this header          |

## Objective
To manage freelancers efficiently by giving benefits them once freelancers register on our website(ideamood.co.kr)

## Motivation
Freelancers we need could consider us(ideamood) random beloning organization so I want to solve that problem by giving sense of belonginig to them

## User Benefit
Our organization(ideamood) can prepare for the situation when we need freelancers and the freelancers can get opportunity to work at the same time

## Design Proposal
If we build and develope the system of registration section for freelancers internally, we can react for the situation when we need them as soon as possible by using their ability.
And we can manage and control the freelancers more efficieontly by creating connection which give benfits each other If the freelancers register on our website(ideamood.co.kr).
Also we approvce and accept the freelancers by setting standard.

### Alternatives Considered
* We can take many advantages as a prepared team and organization firmly

### Performance Implications
* We can use freelancer's ability by filtering and selecting us(ideamood), whichh means we can be trust oraganization more and more for prospect and potencail clients, not only our clients we already got signed contract.

### Dependencies
* Dependencies: does this proposal add any new dependencies to TensorFlow?
* Dependent projects: are there other areas of TensorFlow or things that use TensorFlow (TFX/pipelines, TensorBoard, etc.) that this affects? How have you identified these dependencies and are you sure they are complete? If there are dependencies, how are you managing those changes?

### Engineering Impact
* Do you expect changes to binary size / startup time / build time / test times?
* Who will maintain this code? Is this code in its own buildable unit? Can this code be tested in its own? Is visibility suitably restricted to only a small API surface for others to use?

### Platforms and Environments
* Platforms: does this work on all platforms supported by TensorFlow? If not, why is that ok? Will it work on embedded/mobile? Does it impact automatic code generation or mobile stripping tooling? Will it work with transformation tools?
* Execution environments (Cloud services, accelerator hardware): what impact do you expect and how will you confirm?

### Best Practices
* Does this proposal change best practices for some aspect of using/developing TensorFlow? How will these changes be communicated/enforced?

### Tutorials and Examples
* If design changes existing API or creates new ones, the design owner should create end-to-end examples (ideally, a tutorial) which reflects how new feature will be used. Some things to consider related to the tutorial:
    - The minimum requirements for this are to consider how this would be used in a Keras-based workflow, as well as a non-Keras (low-level) workflow. If either isn’t applicable, explain why.
    - It should show the usage of the new feature in an end to end example (from data reading to serving, if applicable). Many new features have unexpected effects in parts far away from the place of change that can be found by running through an end-to-end example. TFX [Examples](https://github.com/tensorflow/tfx/tree/master/tfx/examples) have historically been good in identifying such unexpected side-effects and are as such one recommended path for testing things end-to-end.
    - This should be written as if it is documentation of the new feature, i.e., consumable by a user, not a TensorFlow developer. 
    - The code does not need to work (since the feature is not implemented yet) but the expectation is that the code does work before the feature can be merged. 

### Compatibility
* Does the design conform to the backwards & forwards compatibility [requirements](https://www.tensorflow.org/programmers_guide/version_compat)?
* How will this proposal interact with other parts of the TensorFlow Ecosystem?
    - How will it work with TFLite?
    - How will it work with distribution strategies?
    - How will it interact with tf.function?
    - Will this work on GPU/TPU?
    - How will it serialize to a SavedModel?

### User Impact
* What are the user-facing changes? How will this feature be rolled out?

## Detailed Design

This section is optional. Elaborate on details if they’re important to
understanding the design, but would make it hard to read the proposal section
above.

## Questions and Discussion Topics

Seed this with open questions you require feedback on from the RFC process.
