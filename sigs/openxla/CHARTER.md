## OpenXLA Project Charter

## Objective

The OpenXLA Project ([GitHub](https://github.com/openxla)) will be a community-driven and modular open source compiler ecosystem. It will enable efficient lowering, optimization and deployment of ML models from most major frameworks to any hardware backend notably CPUs, GPUs, and ML ASICs. This work will be done collaboratively with major ML frameworks and hardware vendors.

SIG OpenXLA will focus on creating the OpenXLA project, including the extraction of XLA from TensorFlow into a standalone project. SIG discussions will facilitate coordination around roadmap, design evolution, and new workflows to be created in OpenXLA. 

### Goals

* Accelerate industry collaboration around XLA and build a vibrant OSS community.
* Share and receive feedback on the technical direction for OpenXLA and ensure it meets the needs of major users and contributors.
* Set up a new XLA repository or organization with independent build/test, with infra to more easily accept PRs, and that is hardware and framework independent. 
* Ensure the extraction of XLA from TensorFlow is minimally disruptive to existing users and contributors.  
* Create a product identity with its own brand, website, docs, and communication channels.
* Discuss establishment of governance outside TensorFlow. 

### Membership

Everyone involved in developing or integrating with XLA is welcome to participate in discussions. To participate, members can request an invitation to join the GitHub organization [TBA] and SIG Discord [TBD]. 

Creating a successful OpenXLA project will also benefit from a collaborative effort from key representatives from ML frameworks, hardware platforms, users and integrators. The following organizations have agreed to participate in SIG discussions and provide resources allowing the SIG to reach its goals (in alphabetical order):

* AMD
* Apple
* ARM
* AWS
* Google (TensorFlow, JAX, Pytorch/XLA)
* Intel
* Meta (Pytorch)
* NVIDIA

Individual SIG members will be added via PR in the following directory [TBA]. Members are expected to regularly attend meetings, participate in technical discussions, and make regular technical contributions to the project. 

### Communication

SIG OpenXLA will hold at minimum monthly virtual meetings for roadmap sharing, design discussion, and SIG coordination. Agendas will be open to contribution and shared in advance. Meeting minutes will be shared to the SIG [mailing list]. 
Asynchronous communications will happen in GitHub Discussions in the OpenXLA GitHub organization (until it's possible to migrate to an independent Discourse forum), including design proposals and roadmap update announcements. 

### Collaboration & Governance 

**Future Governance** 

An explicit workstream within the SIG in 2023 will be to establish collaboration principles, code review processes, and community infrastructure for OpenXLA when the XLA project moves out of the TensorFlow organization. Discussions to prepare for this work will begin in 2022.

The SIG aims to establish an open governance model drawing from standards such as LLVM, with particular emphasis on open design/roadmap discussions, public process for gaining technical steering rights, and neutral docs & repo governance (eg location, CLA, etc). repo location.
Near-term Governance

Here we define near-term avenues for collaboration & governance given the current location of XLA in TensorFlow. SIG OpenXLA will be launched under the TensorFlow governance umbrella, and leverage existing TensorFlow community infrastructure to more efficiently bootstrap collaboration.

**Code**

Code contribution to XLA in its current location will be released under the Apache 2.0 license, governed by TensorFlow’s collaboration rules and contributed under the Google CLA, where the initial maintainers will be the existing maintainers of XLA.

**Design Reviews & Roadmap**

Once launched, SIG OpenXLA will immediately begin technical design conversations with publicly available archives. All significant design proposals and reviews will use a public proposal process established by the SIG (or the TensorFlow RFC process for changes impacting tensorflow/compiler/XLA.  

**Technical Governance**

A priority of the SIG will be to establish a path for community members to take on technical leadership roles in the project. During the bootstrapping phase of the project in 2022, Google engineers will assume responsibility for the technical leadership of the project.

### Contacts
* For technical questions, contact Mehdi Amini - aminim at google  
* For administrative questions, contact Thea Lamkin - thealamkin at google
### Resources
* GitHub ((current)[https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/xla])
* Discord TBA
* Community proposals TBA
* [Community meetings](https://github.com/openxla/community/wiki/OpenXLA-Community-Meetings)

### Code of Conduct
While under TensorFlow governance, all community spaces for SIG OpenXLA are subject to the [TensorFlow Code of Conduct](https://github.com/tensorflow/tensorflow/blob/master/CODE_OF_CONDUCT.md). 
