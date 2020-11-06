# Creating SIG Recommenders

| Status        | Proposed                                             |
:-------------- |:---------------------------------------------------- |
| **RFC #**     ||                                                  
| **Author(s)** | Shuangfeng Li (shuangfeng@google.com), Yuefeng Zhou (yuefengz@google.com), Zhenyu Tan (tanzheny@google.com), Derek Cheng (zcheng@google.com) |
| **Sponsors(s)** |Thea Lamkin (thealamkin@google.com), Joana Carrasqueira (joanafilipa@google.com) |
| **Updated**   | 2020-10-23 |                                        

## What is this group for?

For discussion and collaborations using TensorFlow for large scale recommendation systems (Recommenders), which are one of most common and impactful use cases in the industry. We hope to encourage sharing of best practices in the industry, get consensus and product feedback to help evolve TensorFlow better, and facilitate the contributions of RFCs and PRs in this domain.

It might touch various aspects of the TensorFlow ecosystem, including:

- Training with scale: How to train from super large sparse features? How to deal with dynamic embedding?
- Serving with efficiency: Given recommendation models are usually pretty large, how to serve super large models easily, and how to serve efficiently?
- Modeling with SoTA techniques: online learning, multi-target learning, deal with quality inconsistent among online and offline, model understandability, GNN etc.
- End-to-end pipeline: how to train continuously, e.g. integrate with platforms like TFX.
- Vendor specific extensions and platform integrations: for example, runtime specific frameworks (e.g. NVIDIA Merlin, …), and integrations with Cloud services (e.g. GCP, AWS, Azure…)

Notice that TensorFlow has open-sourced [TensorFlow Recommenders](https://blog.tensorflow.org/2020/09/introducing-tensorflow-recommenders.html), an open-source TensorFlow package that makes building, evaluating, and serving sophisticated recommender models easy. Github: 
[github.com/tensorflow/recommenders](http://github.com/tensorflow/recommenders)

Further, we plan to create a tensorflow repo dedicated for community contributions and maintained by SIG as well, under: 
**github.com/tensorflow/recommenders-addons** (to be created).
SIG Recommenders can contributes more addons as complementary to TensorFlow Recommenders, or any helpful libraries related to recommendation systems using TensorFlow. The contribution areas can be broad and don't limit to the topic listed above. We hope this can make community contributions much easier.

## Who will be part of it?

Membership will be entirely public. Everybody with an interest in using TensorFlow to build recommendation systems are welcome to join the SIG. To participate, request an invitation to join the mailing list. Archives of the mailing list are publicly accessible.

Initially SIG leads will be below and welcome 
new members to request to be SIG leads.

* Community leads
   * Bo Liu, Pinterest
   * Haidong Rong, Tencent
   * Yong Li, Alibaba
* Co-leads and sponsors from Google
   * Yuefeng Zhou, TensorFlow
   * Zhenyu Tan, TensorFlow
   * Wei Wei, TensorFlow
   * Derek Cheng, Google Brain

For administrative questions, contact Joana Carrasqueira‎ (joanafilipa@google.com), or Thea Lamkin (thealamkin@google.com), or tf-community@tensorflow.org.

From an initial survey over a small group of people from about 10+ leading internet companies, most of them have expressed interests to join the discussions, and close to half of them are interested to contribute to the code.

## What initial problems will the group tackle?

Create and prioritize the roadmap for development. Divide up high priority work between contributors. Share support and help between contributors.

We start with training with scale. We already have RFCs that got widely interests in the community, for example two RFCs below, and we can contribute them into SIG repo as the starting tasks:
- [Sparse Domain Isolation RFC](https://github.com/tensorflow/community/pull/237
), [code](https://github.com/tensorflow/tensorflow/pull/41371), from Tencent
- [Embedding Variable RFC](https://docs.google.com/document/d/1odez6-69YH-eFcp8rKndDHTNGxZgdFFRJufsW94_gl4/edit#heading=h.tik7lgjxnl78), from Alibaba


## What modes of communication do you intend to use?
- SIG Recommenders mailing list (recommenders@tensorflow.org, or [google groups](https://groups.google.com/a/tensorflow.org/forum/#!forum/recommenders), to be created)
- New TensorFlow github maintained by SIG Recommenders
(github.com/tensorflow/recommenders-addons, to be created)
- SIG monthly meeting agenda and [notes](https://docs.google.com/document/d/1-jLPffS_MhOd50WScfjFpVNC1DGaIwWxMQPSl5YIJYo/edit#)
- SIG Gitter chat channel (https://gitter.im/tensorflow/sig-recommenders, to be created)

## Code of Conduct

As with all forums and spaces related to TensorFlow, SIG Recommenders is subject to
the [TensorFlow Code of Conduct](https://github.com/tensorflow/tensorflow/blob/master/CODE_OF_CONDUCT.md).


## Launch plan

- Create email list and welcome SIG members.
- Create SIG owned repo.
- SIG added to community pages on tensorflow.org.
- Starts off mailing list discussion about initial work items.
- Run kickoff meetings, working on initial projects.
- Run regular meetings
- Write a blog post about SIG with the initial achievements, welcome more members.
