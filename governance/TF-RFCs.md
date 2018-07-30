# TensorFlow Request for Comments (TF-RFC)

The purpose of a TensorFlow RFC is to engage the TensorFlow community in
development, by getting feedback from stakeholders and experts, and
communicating design changes broadly.

## Who is involved?

Any **community member** may help by providing feedback on whether the RFC will
meet their needs.

An **RFC author** is one or more community member who writes an RFC and is
committed to championing it through the  process.

An **RFC sponsor** is any maintainer who sponsors the RFC and will shepherd it
through the RFC review process.

A **review committee** is a group of maintainers who have the responsibility of
recommending the adoption of the RFC.

## What is a TensorFlow RFC?

An RFC is a document that describes a requirement and the proposed changes that
will solve it. Specifically, the RFC will:

* be formatted according to the RFC template
* be submitted as a pull request to the community/rfcs directory
* be subject to discussion and a review meeting prior to acceptance

## RFC process

Before submitting an RFC, it is a good idea to discuss your aims with project
contributors and maintainers and get early feedback. Use the developer mailing
list for the project concerned (developers@tensorflow.org, or the list for the
relevant SIG). After writing the RFC draft, get feedback from these
experts before submitting it.

1. Submit your RFC as a pull request to community/rfcs. 

   Name your RFC file using the [template](https://github.com/tensorflow/community/blob/master/rfcs/yyyymmdd-rfc-template.md) `YYYYMMDD-descriptive-name.md`, where
   YYYYMMDD is the date of submission, and ‘descriptive-name’ relates to the
   title of your RFC. For instance, if your RFC is titled “Parallel Widgets API”,
   you might use the filename `20180531-parallel-widgets.md`. If you have images
   or other auxiliary files, create a directory of the form `YYYYMMDD-descriptive-name`
   in which to store those files.

2. Recruit a sponsor from the maintainers of the project which your RFC concerns.
   Identify them in the PR, ideally before posting the PR. If no sponsor is found
   within a month of posting the PR, it will be closed.

3. Email the developer mailing list with a brief description, and a link to the
   PR and a request for review.

4. The sponsor will request a review committee meeting, no sooner than two weeks
   after the RFC PR is posted. If discussion is lively, wait until it has
   settled before going to review. The goal of the review meeting is to resolve
   minor issues; consensus should be reached on major issues beforehand.

5. The meeting may approve the RFC, reject it, or require changes before it
   can be considered again. Approved RFCs will be merged into community/rfcs, and
   rejected RFCs will have their PRs closed.

6. Implementations of a successful RFC should reference it in their
   documentation, and work with the sponsor to successfully land the code.

While implementation code is not necessary to start the RFC process, its
existence in full or part may help the design discussion.

## Community members

As the purpose of RFCs is to ensure the community is well represented and served
by new changes to TensorFlow, it is the responsibility of community members to
participate in reviewing RFCs where they have an interest in the outcome.

Community members should:

* provide feedback as soon as possible to allow adequate time for consideration
* read RFCs thoroughly before providing feedback
* be civil and constructive

## Review committees

The constitution of a review committee may change according to the particular
governance style and leadership of each project. For core TensorFlow, the
committee will exist of contributors to the TensorFlow project, who have
expertise in the domain area concerned.

Review committees must:

* ensure that substantive items of public feedback have been accounted for
* add their meeting notes as comments to the PR
* provide reasons for their decisions

If a review committee requires changes before acceptance, it is the
responsibility of the sponsor to ensure these are made and seek subsequent
approval from the committee members.

## RFC sponsors

A sponsor is a project maintainer responsible for ensuring the best possible
outcome of the RFC process. In particular this includes:

* advocating for the proposed design
* guiding the RFC to adhere to existing design and style conventions
* guiding the review committee to come to a productive consensus
* if the RFC moves to implementation:
  * ensuring proposed implementation adheres to the design
  * liaison with appropriate parties to successfully land implementation

## Keeping the bar high

While we encourage and celebrate every contributor, the bar for RFC acceptance
should be kept intentionally high. A design may be rejected or need significant
revision at any one of these stages:

* initial design conversations on the relevant mailing list
* failure to recruit a sponsor
* critical objections during the feedback phase
* failure to achieve consensus during the design review
* concerns raised during implementation (e.g., inability to achieve backwards
  compatibility, concerns about maintenance appearing once a partial implementation
  is available)

If this process is functioning well, RFCs are expected to fail in the earlier,
rather than later, stages.

An approved RFC is no guarantee of a commitment to implement, and acceptance of
a proposed RFC implementation is still subject to the usual code review
process.

## RFC Template

Use the template [from
GitHub](https://github.com/tensorflow/community/blob/master/rfcs/yyyymmdd-rfc-template.md),
being sure to follow the naming conventions described above.
