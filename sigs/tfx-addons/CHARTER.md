# SIG TFX-Addons

## Overview
SIG TFX-Addons aims to facilitate community contributions of community-supported software and tools which can be used with TFX to build a production ML solution. This document outlines the context, goals, and engagement plan for the proposed Special Interest Group.

## Context
Machine learning in production environments is a mission-critical part of a growing number of products and services across many industries. To become an [AI First company](https://ai.google/), Google required a state-of-the-art production ML infrastructure framework, and created TensorFlow Extended (TFX). Google open-sourced TFX in 2019 to enable other developers worldwide to benefit from and help us improve the TFX framework, and established open layers within the TFX architecture specifically focused at customization for a wide range of developer needs. These include custom pipeline components, containers, templates, and orchestrator support.

In order to accelerate the sharing of community customizations and additions to TFX, the TFX team would like to encourage, enable, and organize community contributions to help continue to meet the needs of production ML, expand the vision, and help drive new directions for TFX and the ML community.

## Goals & Objectives
We welcome community contributions on any area of TFX, but this SIG will initially focus on the following goals:
- Driving the development of high-quality custom pipeline components, including Python function-based components, container-based components, and fully custom components.
- Shaping a standardized set of descriptive metadata for community-contributed components to enable easy understanding, comparison, and sharing of components during discovery.
- Similarly driving the development of templates, libraries, visualizations, and other useful additions to TFX.

These projects will begin as proposals to the SIG, and upon approval will be led and maintained by the community members involved in the project and assigned a project folder, with high-level consultation from the TFX team.

### In-Scope, Out of Scope
Although TFX is an open-source project and we welcome contributions to TFX itself, **this SIG does not include contributions or additions to core TFX**.  It is focused only on building community-contributed and maintained additions on top of core TFX.  [Core TFX has its own repo](https://github.com/tensorflow/tfx), and PRs and issues will continue to be managed there. **In addition, all contributions must not violate the [Google AI Principles](https://ai.google/principles/) or [Responsible AI Practices](https://ai.google/responsibilities/responsible-ai-practices/).**

## Membership
We encourage any developers working in production ML environments, infrastructure, or applications to join and participate in the activities of the SIG. Whether you are working on advancing the platform, prototyping or building specific applications, or authoring new components, templates, libraries, and/or orchestrator support, we welcome your feedback on and contributions to TFX and its tooling, and are eager to hear about any downstream results, implementations, and extensions.

We have multiple channels for participation, and publicly archive discussions in our user group mailing list:
- tfx@tensorflow.org – our general mailing list that all are welcome to join (archive group: https://groups.google.com/a/tensorflow.org/g/tfx)
- tfx-addons-core@tensorflow.org – Google group for core team (to be created)

We will create a new mailing list for TFX SIG members.

Other Resources
- SIG Repository: http://github.com/tensorflow/tfx-addons (to be created)
- Documentation: https://www.tensorflow.org/tfx

## Organization and Governance
A central Github repo will be created under the [TensorFlow organization](https://github.com/tensorflow) for individual SIG projects and contributions.  The central repo will also contain overall SIG documents and resources, which will be managed by the TensorFlow team.  Individual contribution projects will begin as proposals to the SIG, and once approved a folder will be created for the project, and project leaders assigned permissions to manage the folder.  **Projects will be led, maintained, and be the responsibility of community project leaders. Google and the TensorFlow team will not provide user support or maintenance for contributed addons. The TFX team will support community maintainers in SIG operations and contribution infrastructure.**

Categories of projects will be grouped under folders at the top level of the central SIG repo, including folders for components and examples. Individual projects will be assigned a new folder under the proper category folder, where all project materials will live. For all community-contributed projects the source of truth will be those project folders. Project leaders will be identified using OWNERS files at the top level of their project folder. New project leaders will be recruited for abandoned projects, or if new leaders are not found then projects will be deprecated and archived. Statistics will be generated and reported per-project.
This conversation was marked as resolved by theadactyl

SIG TFX-Addons is a community-led open source project. As such, the project depends on public contributions, bug fixes, and documentation. This project adheres to the [TensorFlow Code of Conduct](https://github.com/tensorflow/tensorflow/blob/master/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Project Approvals
1. Project proposals will be submitted to the SIG and published for open review and comment by SIG members for 2 weeks.
2. Core team members will vote either in person or offline on whether to approve or reject project proposals.
   - There will be a maximum of 5 voting core team members from any organization
   - Voting core team members should be actively contributing to central SIG infrastructure (documentation, facilitation of SIG activities, CI, testing, vulnerabilities, upgrades, etc)
3. All projects must meet the following criteria:
   - Team members must be named in the proposal
   - All team members must have completed a [Contributor License Agreement](https://cla.developers.google.com/)
   - The project must not violate the [TensorFlow Code of Conduct](https://github.com/tensorflow/tensorflow/blob/master/CODE_OF_CONDUCT.md), [Google AI Principles](https://ai.google/principles/) or [Responsible AI Practices](https://ai.google/responsibilities/responsible-ai-practices/).
4. Projects must code to supported open interfaces only, and not reach into core TFX to make changes or rely on private classes, methods, properties, or interfaces.
5. **Google retains the right to reject any proposal.**
6. Projects are approved with a simple majority vote of the core team, and Google approval.

## Contacts
- Project Lead(s):
  - Robert Crowe (Google)
- Community Lead(s)
  - Hannes Hapke (Digits)
- Core Team Members:
  - Gerard Casas Saez (Twitter)
  - Suzen Fylke (Twitter)
  - Wihan Booyse (Kriterion)
  - Badrul Chowdhury (Apple)
  - Kshitijaa Jaglan (IIITD)
- Founding Core Team Members:
  - Paul Selden (OpenX)
  - Gerard Casas Saez (Twitter)
  - Newton Le (Twitter)
  - Marc Romeyn (Spotify)
  - Ryan Clough (Spotify)
  - Samuel Ngahane (Spotify)
  - Michal Brys (OpenX)
  - Baris Can Durak (maiot)
  - Hamza Tahir (maiot)
  - Larry Price (OpenX)
- SIG members:
  - <add yourself here to participate in SIG meetings & discussion>
- Administrative questions:
  - Thea Lamkin (Google): thealamkin at google dot com
  - Joana Carrasqueira (Google): joanafilipa at google dot com
  - tf-community at tensorflow dot org

Meeting cadence:
- Bi-weekly Wednesdays 9:00am PST / 5:00pm GMT / 9:30pm IST

The current list of project ideas for TFX Addons is available in the [TFX Addons repo](https://github.com/tensorflow/tfx-addons/issues?q=is%3Aissue+is%3Aopen+label%3A%22Project%3A+Idea%22).

The notes from our last meetings can be found at [here](https://docs.google.com/document/d/1T0uZPoZhwNStuKkeCNsfE-kfc-PINISKIitYxkTK3Gw/edit?usp=sharing&resourcekey=0-N9vT9Tn171wYplyYn4IPjQ).

