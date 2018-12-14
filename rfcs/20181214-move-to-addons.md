# Move code from tf.contrib to tensorflow/addons

| Status        | Proposed       |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Sean Morgan (seanmorgan@outlook.com), Armando Fandango (armando@neurasights.com) |
| **Sponsor**   | Karmel Allison (karmel@google.com)                 |
| **Updated**   | 2018-12-12                                           |

## Objective

As part of our initial lift to [tensorflow/addons](https://github.com/tensorflow/addons),
the Addons SIG plans to move a substantial amount of code from tf.contrib
into the new repository.

This document details what functionality the SIG plans to move and invites
discussion around the decisions.


## Motivation

We wish to solicit discussion regarding what tf.contrib code will be moved.
Specifically, there are many contributions that are feasible to move,
but we were unable to determine if there is any value in doing so.

## Design Proposal

### Criteria for moving
1) The code follows an established API pattern
1) There is sufficient interested in the community to...


### Code slated to be moved to addons

| Module (tf.contrib)     | Class/Function   | Rationale                               |
|:----------------------- |:----------- |:------------------------------------ |
| opt.external_optimizer  | ExternalOptimizerInferface  | Base class for external optimizers used in OSS projects |
| opt.external_optimizer | ScipyOptimizerInterface      | Significant usage is OSS projects |
| crf.crf      | ALL MODULES   | Heavily used by the NLP community |


### Code that will be removed pending commentary

| Module (tf.contrib)     | Class/Function   | Rationale                               |
|:----------------------- |:----------- |:------------------------------------ |
| opt.addsign  | AddSignOptimizer  | No OSS uses found |
| opt.agn_optimizer | AGNOptimizer      | No OSS uses found |


**Note: The details of our larger code review can be found in [this spreadsheet](https://docs.google.com/spreadsheets/d/1hYJchHp1y1t2U6htq5UXxMxWlGxxtOyyNHDF8_qhtQQ/edit#gid=185512613)**

## Questions and Discussion Topics

* Are there any modules being excluded from the move that you feel have substantial value to the community?
*