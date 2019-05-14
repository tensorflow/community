# Sustainability of TensorFlow Addons

| Status        | Accepted       |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Paige Bailey (webpaige@google.com), Sean Morgan (seanmorgan@outlook.com), Yan Facai (facai.yan@gmail.com) |
| **Sponsor**   | Karmel Allison (karmel@google.com)                 |
| **Updated**   | 2019-05-13                                         |

## Objective
 
[TensorFlow Addons](https://github.com/tensorflow/addons) is a repository of 
contributions that conform to well-established API patterns, but implement new 
functionality not available in core TensorFlow. For TensorFlow Addons to 
support [code that is moved from `tf.contrib` to `addons`](https://github.com/tensorflow/community/blob/ef626896f30130dfc3b5e75126c94624b689a943/rfcs/20181214-move-to-addons.md#code-to-be-moved-from-tfcontrib-to-addons),
as well as new functionality, requirements for maintaining and retiring those 
submodules must be defined and enforced.

This document details requirements and responsibilities for owning a 
subpackage/submodule that is included in TensorFlow Addons, as well as the 
periodic review process for all components of Addons.

## Motivation

In this RFC, we are soliciting discussion regarding Addons maintainership 
and periodic review. This RFC discussion will help the SIG Addons team 
determine appropriate roles and responsibilities for proxy maintainers.

## Design Proposal 

### Repository Structure

TensorFlow Addons is structured in a into a hierarchy of subpackages 
and submodules as described in the [python docs](https://docs.python.org/3/tutorial/modules.html#packages).

A newly proposed subpackage will be required to have at least one active proxy 
maintainer or organization; and the inclusion must be 
[discussed with the community at large](https://github.com/tensorflow/addons/issues/58) 
to see if its matches the addons paradigm. A newly proposed submodule or code 
addition is required to get approval from the governing subpackage maintainers. 
If a contributor is not willing to maintain a piece of code, 
subpackage owners will decide if they are willing to maintain 
the code going forward and if it should be accepted into future addons releases.

TensorFlow Addons would like to build as inclusive of a community as 
possible. More than one maintainer is encouraged for subpackages and submodules. 
The Addons core team will send out monthly emails to the addons [Google Group](https://groups.google.com/a/tensorflow.org/forum/#!forum/addons) detailing which subpackages/submodules are light on maintainers and encouraging 
community support.

### Subpackages 

To date, the `tensorflow/addons` [subpackages](https://github.com/tensorflow/addons#maintainers) include:

| Subpackage    | Maintainers  | Contact Info                        |
|:----------------------- |:----------- |:----------------------------|
| tfa.activations | SIG-Addons | @facaiy @seanpmorgan | 
| tfa.image | SIG-Addons | @windqaq @facaiy |
| tfa.layers | SIG-Addons | @seanpmorgan @facaiy |
| tfa.losses | SIG-Addons | @facaiy @windqaq   |
| tfa.optimizers | SIG-Addons | @facaiy @windqaq |
| tfa.rnn | Google | @qlzh727 |
| tfa.seq2seq | Google | @qlzh727 |
| tfa.text |  SIG-Addons |  @seanpmorgan @facaiy |

## Maintainership 

### How to Become a Proxied Maintainer

The easiest way to become a proxy maintainer is to respond to our monthly 
"request for help" emails. Subpackage maintainership will only be 
granted after substantial contribution has been made in order to limit the 
number of users with write permission. Contributions can come in the form of 
issue closings, bug fixes, documentation, new code, or optimizing existing code. 
Submodule maintainership can be granted with a lower barrier for entry as this will not 
include write permissions to the repo.

Additionally, Proxy maintainers are encouraged to co-maintain existing TensorFlow 
Addons subpackages/submodules. If a potential maintainer chooses to do so, 
there are a few things to consider first:

* If the subpackage has an existing maintainer, the potential maintainer 
must communicate with them first. The existing maintainer would need to 
approve the co-maintainer role, and might give the potential maintainer 
a specific task to complete before granting co-maintainership.

* If the subpackage has major bugs, the potential maintainer 
can provide fixes and expedite their inclusion as a listed
proxy maintainer.

### Privileges and Responsibitilies of Proxied Maintainers

Subpackage maintainers will be granted write access to the repo and will be 
designated in the [CODEOWNERS](https://help.github.com/en/articles/about-code-owners) 
file.

### Resolving Bugs

Proxy maintainers are expected to handle bugs against the subpackages 
they own. This includes resolving those issues once their fix is 
merged by a proxy maintainer.

If a proxy maintainer fails to provide sufficient support for their
subpackage they will be designated as inactive. This status will be taken into 
account during the periodic review of their subpackage/submodule.

### Retiring from Proxy Maintainership

If a proxy maintainer decides that they no longer wish to maintain one 
or more of their subpackages, they **must** commit to the following procedure:

* Send an email to addons@tensorflow.org and list all code that they are 
no longer able to maintain. It is usually a good idea to shortly describe the 
state of code, e.g. whether the packages have open bugs, whether they are difficult 
to maintain, etc.

* The proxy maintainer must remove themselves from the `CODEOWNERS.md` 
file, with `<!--maintainer-needed-->` listed in the place of 
maintainers. This will help other developers to find subpackages with no 
maintainers.

## Repository Growth and Review 
### Periodic Review

Given the nature of this repository, subpackages and submodules may become less 
and less useful to the community as time goes on. In order to keep the 
repository sustainable, we'll be performing bi-annual reviews of our code to 
ensure everything still belongs within the repo. Contributing factors to this 
review will be:

1. Number of active maintainers
2. Amount of OSS use
3. Amount of issues or bugs attributed to the code
4. A better solution becomes available

Functionality within TensorFlow Addons can be categorized into three groups:

* **Suggested**: well-maintained API; use is encouraged.
* **Discouraged**: a better alternative is available; the API is kept for historic reasons; or the API requires maintenance and is the waiting period to be deprecated.
* **Deprecated**: use at your own risk; subject to be deleted.

The status change between these three groups is: Suggested <-> Discouraged -> Deprecated.

The period between an API being marked as deprecated and being deleted will be 90 days.

1. In the event that TensorFlow Addons releases monthly, there will be 2-3 releases before an API is deleted. The release notes could give user enough warning.

2. 90 days gives maintainers ample time to fix their code.

3. Google often gives a 90 day window for security fixes, which is a trade off between risk and time to respond. For more comparison of different time window, please see [this link](https://googleprojectzero.blogspot.com/search?q=90+days).

### Adding a New Subpackage

The SIG Addons core team will review the submitted files 
(for details on coding style, requirements, and testing, please refer 
to [`CONTRIBUTING.md`](https://github.com/tensorflow/addons/blob/master/CONTRIBUTING.md)) 
and help bring the contribution to an acceptable quality. Once the 
package is ready, it will be merged and the relevant bugs will be 
closed (if there are any).

Please note that the Addons team reserves the right to reject new 
subpackages, especially if they would otherwise qualify for removal per 
its quality standards. Example reasons for rejection include:

* Having known major bugs or security issues.
* Duplicating functionality in the TensorFlow core API.
* Having deprecated upstream dependencies.
* Failling to obtain an appropriate code review.
* Having no general usefulness for TensorFlow users.

### Moving a Subpackage from tf.contrib

If a potential maintainer wishes to migrate a subpackage that was deprecated 
as part of `tf.contrib`, they must explicitly note that on their 
submission. A potential maintainer would be required to:

* Demonstrate that the code is useful to the community at large.
* Modify the code to utilize TF 2.x functionality

**Useful Links:**
* [SIG Addons RFC](https://github.com/tensorflow/community/blob/ef626896f30130dfc3b5e75126c94624b689a943/rfcs/20181214-move-to-addons.md).
* [SIG Addons Charter](https://github.com/tensorflow/community/blob/master/sigs/addons/CHARTER.md).

## Questions and Discussion Topics

* What is a fair way to define "sufficient support" for a subpackage/submodule.
* Should we limit the number of write permissions per subpackage?
* What channels of communication should be available for SIG Addons Proxy Maintainers?
* How will the periodic review affect production users of TF-Addons?
* How often should the periodic reviews be conducted?
