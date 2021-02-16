

# SIG Recommenders

## Objective

This SIG will drive discussion and collaborations around using TensorFlow for large scale recommendation systems (Recommenders), which are one of most common and impactful use cases in the industry. We hope to encourage sharing of best practices in the industry, get consensus and product feedback to help evolve TensorFlow better, and facilitate the contributions of RFCs and PRs in this domain.

## Scope

See approved TensorFlow RFC #[313](https://github.com/tensorflow/community/pull/313). 

Workstreams might touch various aspects of the TensorFlow ecosystem, including:

- Training with scale: How to train from super large sparse features? How to deal with dynamic embedding?
- Serving with efficiency: Given recommendation models are usually pretty large, how to serve super large models easily, and how to serve efficiently?
- Modeling with SoTA techniques: online learning, multi-target learning, deal with quality inconsistent among online and offline, model understandability, GNN etc.
- End-to-end pipeline: how to train continuously, e.g. integrate with platforms like TFX.
- Vendor specific extensions and platform integrations: for example, runtime specific frameworks (e.g. NVIDIA Merlin, …), and integrations with Cloud services (e.g. GCP, AWS, Azure…)

Notice that TensorFlow has open-sourced [TensorFlow Recommenders](https://blog.tensorflow.org/2020/09/introducing-tensorflow-recommenders.html), an open-source TensorFlow package that makes building, evaluating, and serving sophisticated recommender models easy. Github: 
[github.com/tensorflow/recommenders](http://github.com/tensorflow/recommenders)

Further, we maintain a Tensorflow repo dedicated for community contributions and maintained by SIG as well, under: 
**github.com/tensorflow/recommenders-addons** (to be created).
SIG Recommenders can contributes more addons as complementary to TensorFlow Recommenders, or any helpful libraries related to recommendation systems using TensorFlow. The contribution areas can be broad and don't limit to the topic listed above. We hope this can make community contributions much easier.

## Membership

Everybody with an interest in using TensorFlow to build recommendation systems are welcome to join the SIG. To participate, request an invitation to join the mailing list. Archives of the mailing list are publicly accessible.

## Resources

- SIG Recommenders [mailing list](https://groups.google.com/u/1/a/tensorflow.org/g/recommenders)
- New TensorFlow github maintained by SIG Recommenders
(github.com/tensorflow/recommenders-addons, to be created)
- SIG monthly meeting agenda and [notes](https://docs.google.com/document/d/1-jLPffS_MhOd50WScfjFpVNC1DGaIwWxMQPSl5YIJYo/edit#)
- SIG Gitter chat channel (https://gitter.im/tensorflow/sig-recommenders, to be created)

## Contacts

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

For administrative questions, contact tf-community-team@tensorflow.org.

## Code of Conduct

As with all forums and spaces related to TensorFlow, SIG Recommenders is subject to
the [TensorFlow Code of Conduct](https://github.com/tensorflow/tensorflow/blob/master/CODE_OF_CONDUCT.md).
