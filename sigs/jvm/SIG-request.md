# Request for SIG

## What is this group for?

As many sources claim, Java is still on of the (if not the) most popular language found in small to large enterprises around the world. For TensorFlow, it makes it a strategic choice to reach a large audience of programmers who are interested to embed machine learning algorithms in their applications.

TensorFlow already has a Java client out-of-the-box, which was originaly conceived to support inference on Android devices. As many contributions were made thereafter to make it a good choice even for building, training and serving models, there is still a lot of work to be done before it reaches a level of maturity comparable to Python.

The purpose of this group is to establish an official communication channel for discussions and suggestions related to TensorFlow support in Java or any other JVM-based languages, such as Kotlin or Scala.

In addition, the group will develop and maintain a new set of high-level utility libraries on top of TensorFlow for providing better tools to developers interested in machine learning on JVMs, with their own distinct repositories and release life cycles.

## Who will be part of it?

* Group leader: Karl Lessard
* Second for the leader: Christian Tzolov

Anyone interested to help are invited to join by requesting access to the mailing list, whether they are developers who wants get involve in the code or just individuals who wants to take part of the discussions.

Members of the [unofficial Java group](https://groups.google.com/forum/#!forum/tensorflow-java-dev-unofficial) might be interested to join this new official one, as the former will be closed.

## What initial problems will the group tackle?

* Providing a new set of high-level API and features in Java or other languages running on JVMs
    * New repositories under the TensorFlow organization will host the code for the new libraries, with their own release dates
    * Features could be merged faster by allowing some members of the community to do code reviews 
* Establish an official communication channel for discussion related to TensorFlow on JVM
    * Right now, discussions are spread on different channels, such as the unofficial google group or on top of different GitHub issues
    
The outcome of the initial discussions of this group will be to identify what features should be addressed first. Here is some ideas:

* Eager execution mode in the Java core client
* `tensorflow-java`: Utility library on top of the core client to simplify usage of TensorFlow in Java, such as multi-dimensional array accessors.
* `tensorflow-java-models`: Modelisation in Java of pretrained TensorFlow models.

## What modes of communication do you intend to use?

* A dedicated mailing list backed by Google Groups
* A Slack channel
* VC calls on Google Hangout could be organized on demand
* StackOverflow

## Launch plan

1. Exposing the present plan and charter to the community for review
2. SIG set up with initial group members
3. SIG added to community pages on tensorflow.org
4. Leader starts off mailing list discussions about initial work items
5. Creation of repository(ies) for the inception of the first high-level libraries to be develop by this group 

# Charter

Here's the link to the group [charter](CHARTER.md).

