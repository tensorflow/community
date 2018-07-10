# Community Supported TensorFlow Builds and Releases


## Overview

TensorFlow is used in many more environments and configurations than is practical for the core team to regularly test and support: so we need a way to include federated third party testing and builds.

This document describes a process for creating third party builds of TensorFlow, federating tests and builds, and making the build artifacts available to users. Examples of such builds include those optimized for particular hardware configurations, operating system environments, or other specific applications.

There are three major phases of the process:



1.  Engagement — connect with the TensorFlow core team and work on a plan for integration, tests, documentation and support
1.  Testing — set up continuous integration and connect to GitHub webhooks
1.  Building — once tests exist and pass, and builds are available, they will be linked as community supported builds from the official TensorFlow site


## Phase 1: Engagement

You should first join the [SIG Build interest group](https://groups.google.com/a/tensorflow.org/forum/#!forum/build): this community is the main way coordination happens around building, testing and releasing TensorFlow.

To start the process, reach out with a description of your intent to build a particular flavor or release of TensorFlow to the SIG Build community: include a tracking bug filed in GitHub.

A TensorFlow team member will reply and start the planning process with you. Together, we will create a plan to get the work to "community supported" status. We discuss how to integrate your code, what the TensorFlow team needs from you, and set expectations for both sides.

In particular we will need to ensure there is:



*   A testing plan, to make sure the build is periodically tested by you, with our help. The TensorFlow team won't run these tests. We also will not add tests that will block merging code to the central TensorFlow repository.
*   Documentation and examples. You should plan to provide sufficient documentation to let people install, setup, and use the artifacts you have created.
*   A support plan. Before we link the build artifacts from the web site, you will need to provide a contact for support and maintenance of the packages.

The TensorFlow team will periodically review community supported efforts, and highlight them in collaboration with you through various promotional channels on a case-by-case basis: for example, through blog posts or conference presentations.


## Phase 2: Testing

In this phase, we agree what configurations should be tested based on what the community needs and what you are willing to contribute. Usually, this should be a discussion conducted within SIG Build.

The TensorFlow team will work with you to set up continuous testing of your build:



*   There is no mandated CI system: you can choose what CI system you would like to use (e.g. Jenkins, Travis, custom)
*   We recommend running as many unit tests as possible
*   Continuous testing of the master branch is required
*   Testing release branches at least once after each branch cut is highly recommended

The TensorFlow team will create "webhooks" in our GitHub repository to enable automated triggering of tests in your CI.

Once the tests are up and running, we will link to the CI build status under community supported builds on GitHub, as is the case for the IBM CI links [here](https://github.com/tensorflow/tensorflow/blob/master/README.md)!


## Phase 3: Building

At this stage, we must be sure that the continuous integration is configured, all tests pass, and that the CI setup proves stable.

You will set up a destination download and documentation site, and the TensorFlow web site will add a link to it, highlighting that this is a community supported build, with credit to you and your organization.

To be listed as a build, you must also provide:



*   One or more GitHub users to assign issues to
*   Support details for users to report bugs to you
*   Documentation as discussed in Phase 1

If the build remains broken for an extended period of time, the TensorFlow team may remove it from the community builds list until the requirements for phase 3 are once again met.


## Comments and questions

Please feel free to ask further about this process on the [build@tensorflow.org](mailto:build@tensorflow.org) mailing list.

