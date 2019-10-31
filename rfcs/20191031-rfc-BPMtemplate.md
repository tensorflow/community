# Title of RFC

| Status        | (Proposed)       |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [NNN](https://github.com/tensorflow/community/pull/NNN) (update when you have community PR #)|
| **Author(s)** | My Name (bojan.ploj@egmail.com),                     |
| **Sponsor**   | A N Expert (whomever@tensorflow.org)                 |
| **Updated**   | 2019-10-31                                           |
| **Obsoletes** |           |

## Objective

I have invented Border Pairs Method (BPM) - a new ML algorithm for NN. It has more than ten advantages over Backpropagation.
My goal is to add this ML algorithm into TF.

Link:
https://www.researchgate.net/publication/249011199_Border_Pairs_Method-constructive_MLP_learning_classification_algorithm
https://www.researchgate.net/publication/322617800_New_Deep_Learning_Algorithms_beyond_Backpropagation_IBM_Developers_UnConference_2018_Zurich

## Motivation

BPM advantages over Backpropagation:

1) BPM is non iterative, it is done in one step (epoch) without the guesing and gradient
2) BPM is constructive, it finds near optimal NN construction,
3) BPM uses only useful patterns,
4) BPM know how good are learning patterns,
5) BPM is accurate,
6) BPM is 100% reliable - no local minimum and early stoping problems,
7) BPM evaluates the patterns quality,
8) BPM is suitable for noise cancelation,
9) BPM is suitable for online learnig (adding of patterns),
10) BPM is overfitting resitant
11) and more (see given links)...

This will very improve learning procedure which will be now simpler, faster, reliable, accurate,...

All NN users are affected by this improvement.
BPM is supporting all kind of data. 

## User Benefit

ML procedure will be simpler, faster, reliable, accurate,...
Headline in the release notes or blog post: Reliable nongradient NN learning in one step/epoch.

## Design Proposal

BPM will be a new tf.fit() function which finds the NN model by its self. The coding proces will
so be simpler and the learning of NN will be faster.

Factors to consider include:

* better performance - see Motivation chapter above
* no dependences
* suitable for all platforms and environments 
* this will simplyfy proces of coding


## Questions and Discussion Topics

I am new with Python and TF, I need a masive help. I am lookong for coding BPM Seed this with open questions you require feedback on from the RFC process.
