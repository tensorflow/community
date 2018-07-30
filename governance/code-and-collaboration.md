# TensorFlow Governance: Code and Collaboration

## Projects

A project is the primary unit of collaboration. It can either have its own repo,
preferably, or be a part of another repo (e.g. a directory in
_tensorflow/models_, or a _tensorflow/contrib_ module)


## Contributors

Anyone can submit a PR contribution to any project, as long as they have signed
the CLA and follow the guidelines in CONTRIBUTING.md. Their code must be
reviewed by a maintainer, and their code must pass all applicable tests.

Code reviews check for code quality and style, including documentation, and
enforce API compatibility guarantees and other policies. Contributions may be
rejected for strategic reasons unrelated to the code in question. For instance,
because a feature may be too costly to maintain, or because it would duplicate
APIs.

## Maintainers

A project has one or more maintainers. If the project is in its own repo, the
maintainers are those with write access to the repo[^1]. If a repo is shared
between many projects, we use GitHub's CODEOWNERS to identify owners and route
PRs to them for review[^2].

Once there are more than a couple of maintainers for a project, we will create a
GitHub team for the project maintainers. This allows for easier maintenance, and
opens up some [GitHub
tooling](https://help.github.com/articles/about-team-discussions/) for
communication. Larger projects can facilitate coordination and contribution
through establishing a
[TensorFlow SIG](SIGS.md).

Maintainers have write access to the repo containing their project. That means
they can review PRs—an approving review will allow PRs to be merged. They can
also change labels (which means they can trigger tests), add assignees and
reviewers, and they can be assigned to issues and PRs.

### Repositories requiring synchronization

For some projects initiated by Google (including the _tensorflow/tensorflow_
repo), the infrastructure which synchronizes and merges internal and external
changes requires that all merges are performed by a Google employee. In such
cases, Google sets up an on-call rotation which merges PRs once they pass tests
(and a specific label is applied to the PR in order to notify the rotation to
merge it). This does not preclude non-Google contributors from becoming
maintainers. In this case, the maintainers of the project decide on what should
be merged, then the actual merging is performed as a service. In some cases,
Google-internal tests may fail and may have to be fixed: the Google employee
will work with the submitter to achieve this.


### Achieving maintainer status

Maintainers may elevate a contributor to maintainer status, on evidence of
previous contributions and established trust.

## Collaboration

Maintainers are free to agree on their preferred form of collaboration and
decision making, with the requirement that their material discussions and
decisions about the project must be made publicly accessible—this can happen
after the fact, for example in the form of publishing meeting minutes, reviews,
or decisions. Communication about topics such as admitting other maintainers, or
as of yet undisclosed security issues, can be kept confidential.

If significant engagement from multiple parties is encountered, the group may
request the formation of a SIG to formalize collaboration and cooperation. The
threshold for SIG formation includes:

*   A clearly stated purpose
*   Two or more non-maintainers willing to contribute code, and evidence of
    existing demand for the group
*   Project maintainers willing to be in the group and shepherd contributions

For further details on SIGs, read [TensorFlow SIGs](SIGS.md).

As with most structures, a project doesn't need a SIG to get started, but should
find a home in one if it has proven itself as an ongoing concern, as SIGs are
the primary organizational vehicle for the contributor community.
