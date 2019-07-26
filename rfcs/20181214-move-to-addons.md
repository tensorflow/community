# Move from tf.contrib to tensorflow/addons

| Status        | Accepted       |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Sean Morgan (seanmorgan@outlook.com), Armando Fandango (armando@neurasights.com) |
| **Sponsor**   | Karmel Allison (karmel@google.com)                 |
| **Updated**   | 2018-12-16                                           |

## Objective

With the upcoming removal of tf.contrib in TF 2.0, we are in the process
of deciding what existing functionality will be moved and maintained in
the [tensorflow/addons](https://github.com/tensorflow/addons)
repository.

This document details what functionality the SIG plans to move and
invites discussion around the decisions.


## Motivation

In this RFC, we are soliciting discussion regarding what tf.contrib code
should be moved to tensorflow/addons. This RFC discussion will help us
determine the value of moving code and their respective
maintainability aspects.

## Design Proposal

### Criteria for moving
1) The functionality is not otherwise available in TensorFlow
1) There is sufficient interest in the community to maintain the code being moved
1) The code conforms to an established API pattern (Some pieces can be refactored if needed)

It is worth noting that just because some functionality isn't part of
the initial move, does not mean it won't be eventually part of addons
if there is value. We will begin reviewing pull requests to the
repository after the directory structure is shaped during the initial move.

### Code to be moved from tf.contrib to addons

| Module (tf.contrib)     | Class/Function   | Rationale                               |
|:----------------------- |:----------- |:------------------------------------ |
| opt.external_optimizer  | ExternalOptimizerInferface  | Base class for external optimizers used in OSS projects |
| opt.external_optimizer | ScipyOptimizerInterface  | Significant usage in OSS projects |
| opt.lazy_adam_optimizer | LazyAdamOptimizer | Significant usage in OSS projects / discussions |
| opt.moving_average_optimizer | MovingAverageOptimizer | Significant usage in OSS projects |
| layers.layers | dense_to_sparse | Useful functionality and discussion around it |
| layers.layers | layer_norm | Heavily used is OSS projects / From impactful paper |
| layers.layers | maxout | From impactful paper |
| layers.layers | poincare_normalize | Functionality not available / Useful for hyperbolic embeddings |
| layers.normalization | instance_norm | Heavily used is OSS projects / Used for style xfer |
| layers.normalization | group_norm | Will be moved as a generalized case of layer_norm and instance_norm |
| losses.metric_loss_ops | pairwise_distance | Useful functionality not otherwise available  |
| losses.metric_loss_ops | contrastive_loss | Useful functionality not otherwise available |
| losses.metric_loss_ops | masked_maximum | Useful functionality not otherwise available |
| losses.metric_loss_ops | masked_minimum | Useful functionality not otherwise available |
| losses.metric_loss_ops | triplet_semihard_loss | Useful functionality not otherwise available / From impactful paper |
| losses.metric_loss_ops | npairs_loss | Useful functionality not otherwise available |
| losses.metric_loss_ops | npairs_loss_multilabel | Useful functionality not otherwise available |
| losses.metric_loss_ops | lifted_struct_loss | Useful functionality not otherwise available |
| sparsemax.sparsemax | ALL | Useful functionality not otherwise available / Volunteers to maintain |
| image.dense_image_warp | dense_image_warp | Useful functionality not otherwise available |
| image.distort_image_ops | random_hsv_in_yiq | Useful functionality not otherwise available |
| image.distort_image_ops | adjust_hsv_in_yiq | Useful functionality not otherwise available |
| image.image_ops | rotate | Useful functionality not otherwise available / Several uses in OSS found |
| image.image_ops | translate  | Useful functionality not otherwise available |
| image.image_ops | angles_to_projective_transforms | Useful functionality not otherwise available / Several uses in OSS found |
| image.image_ops | translations_to_projective_transforms | Useful functionality not otherwise available  |
| image.image_ops | transform  | Useful functionality not otherwise available / Several uses in OSS found |
| image.image_ops | compose_transforms | Useful functionality not otherwise available / Several uses in OSS found |
| image.image_ops | flat_transforms_to_matrices | Helper util used a few times in module |
| image.image_ops | matrices_to_flat_transforms | Helper util used a few times in module |
| image.image_ops | connected_components | Useful functionality not otherwise available |
| text.skip_gram_ops | ALL | Useful functionality not otherwise available |
| crf.crf | ALL | Heavily used by the NLP community |
| opt.weight_decay_optimizers | DecoupledWeightDecayExtension | ~SOTA convergence speeds / Needs refactored as Wrapper subclass |
| opt.weight_decay_optimizers | AdamWOptimizer  | ~SOTA convergence speeds / Needs refactored as wrapper + keras Adam |
| opt.weight_decay_optimizers | MomentumWOptimizer | ~SOTA convergence speeds / Needs refactored as wrapper + keras SGD|

### Code that will not be moved from tf.contrib pending objections

| Module (tf.contrib)     | Class/Function   | Rationale                               |
|:----------------------- |:----------- |:------------------------------------ |
| opt.addsign  | AddSignOptimizer  | No OSS uses found / Needs refactored as OptimizerV2 subclass |
| opt.agn_optimizer | AGNOptimizer      | No OSS uses found / Needs refactored as OptimizerV2 subclass |
| opt.drop_stale_gradient_optimizer | DropStaleGradientOptimizer | No OSS uses found / Needs refactored as Wrapper subclass |
| opt.elastic_average_optimizer | ElasticAverageOptimizer | No OSS uses found / Needs refactored as Wrapper subclass |
| opt.ggt | GGTOptimizer | No OSS uses found |
| opt.lars_optimizer | LARSOptimizer | No OSS uses found / Needs refactored as OptimizerV2 subclass |
| opt.shampoo | ShampooOptimizer | No OSS uses found / Needs refactored as OptimizerV2 subclass |
| opt.matrix_functions | matrix_inverse_pth_root  | Used in opt.shampoo |
| opt.model_average_optimizer | ModelAverageOptimizer | No OSS uses found / Needs refactored as Wrapper subclass |
| opt.multitask_optimizer_wrapper | MultitaskOptimizerWrapper | No OSS uses found / Needs refactored as Wrapper subclass |
| opt.multitask_optimizer_wrapper | clip_gradients_by_global_norm | No OSS uses found / Specific to MultitaskOptimizers / At least partly covered in Keras optimizer |
| opt.powersign | PowerSignOptimizer | No OSS uses found / Needs refactored as OptimizerV2 subclass |
| opt.sign_decay | get_linear_decay_fn | No OSS usage / Used in AddSign & PowerSign |
| opt.sign_decay | get_cosine_decay_fn | No OSS usage / Not an optimizer |
| opt.sign_decay | get_restart_decay_fn | No OSS usage / Not an optimizer |
| opt.reg_adagrad_optimizer | RegAdagradOptimizer | No OSS uses found / Needs refactored as keraas Adagrad subclass |
| opt.variable_clipping_optimizer | VariableClippingOptimizer  | No OSS uses found / Needs refactored as Wrapper subclass / partial covered by keras norm clip |
| opt.weight_decay_optimizers | ShampooWOptimizer | No OSS uses found |
| opt.weight_decay_optimizers | extend_with_decoupled_weight_decay | No OSS uses found /  Functional paradigm - factory function |
| layers.embedding_ops | scattered_embedding_lookup_sparse | No OSS uses found |
| layers.embedding_ops | embedding_lookup_unique | No OSS uses found |
| layers.encoders | bow_encoder | Creates variables, but does not subclass Layer |
| layers.encoders | embed_sequence | Creates variables, but does not subclass Layer |
| layers.layers | convolution2d_in_plane | No OSS uses found |
| layers.layers | GDN | No OSS uses found |
| layers.layers | scale_gradient | No OSS uses found |
| layers.layers | sequence_to_images | No OSS uses found |
| layers.layers | spatial_softmax  | One OSS project found / Needs refactored as base Layer subclass / Uses get_variable_collections |
| layers.optimizers | optimize_loss | Concience wrapper to build a training op / Would need refactor to stick to TF2.0 APIs |
| layers.optimizers | adaptive_clipping_fn | No OSS uses found |
| layers.rev_block_lib | RevBlock | No OSS uses found |
| layers.rev_block_lib | recompute_grad | No OSS uses found |
| layers.summaries | summarize_tensor  | One OSS project found / Very simple wrapper |
| layers.utils | constant_value | Simple wrapper... need a good reason to support |
| nn.alpha_dropout | alpha_dropout | No OSS uses found / Needs refactored as base Layer subclass |
| nn.fwd_gradients | fwd_gradients | No OSS uses found |
| nn.sampling_ops | rank_sampled_softmax_loss | One OSS use found / Needs to utilize sampled_softmax_loss_v2 |
| nn.sampling_ops | sampled_sparse_softmax_loss  | No OSS uses found / Needs to utilize sampled_softmax_loss_v2 |
| nn.scaled_softplus | scaled_softplus | No OSS uses found |
| losses.metric_loss_ops | update_1d_tensor | No OSS uses found / Large amount of code related to cluster_loss |
| losses.metric_loss_ops | get_cluster_assignment | No OSS uses found / Large amount of code related to cluster_loss |
| losses.metric_loss_ops | compute_facility_energy  | No OSS uses found / Large amount of code related to cluster_loss |
| losses.metric_loss_ops | compute_clustering_score | No OSS uses found / Large amount of code related to cluster_loss |
| losses.metric_loss_ops | compute_augmented_facility_locations | No OSS uses found / Large amount of code related to cluster_loss |
| losses.metric_loss_ops | update_medoid_per_cluster | No OSS uses found / Large amount of code related to cluster_loss |
| losses.metric_loss_ops | update_all_medoids | No OSS uses found / Large amount of code related to cluster_loss |
| losses.metric_loss_ops | compute_augmented_facility_locations_pam | No OSS uses found / Large amount of code related to cluster_loss |
| losses.metric_loss_ops | compute_gt_cluster_score | No OSS uses found / Large amount of code related to cluster_loss |
| losses.metric_loss_ops | cluster_loss | No OSS uses found / Large amount of code related to cluster_loss |
| image.image_ops | bipartite_match | No OSS uses found / Should live in linalg or somewhere else? |
| image.interpolate_spline | interpolate_spline | One OSS uses found / Should live in tf.signal? |
| image.single_image_random_dot_stereograms | single_image_random_dot_stereograms | No OSS uses found |
| image.parse_image_wrap | sparse_image_warp | No OSS uses found |
| resampler.resampler_ops | ALL | Pending community interest |
| solvers | ALL | Pending community interest to maintain |
| integrate | ALL | Pending community interest to maintain |

### Code that will not be copied from tf.contrib to addons and hence would not be available in either of tf.contrib or addons

| Module (tf.contrib)     | Class/Function   | Rationale                               |
|:----------------------- |:----------- |:------------------------------------ |
| opt.adamax  | AdaMaxOptimizer  | Available in tf.keras.optimizers |
| opt.matrix_functions | matrix_square_root | Available as linalg_ops.matrix_square_root  |
| opt.nadam_optimizer | NadamOptimizer | Available in tf.keras.optimizers  |
| layers.embedding_ops | safe_embedding_lookup_sparse | Exists as tf.nn.safe_embedding_lookup_sparse |
| layers.embedding_ops | embedding_lookup_sparse_with_distributed_aggregation | Replaced by emedding_lookup_sparse_v2 |
| layers.feature_column  | ALL | Better version available in tf.feature_column |
| layers.initizalizers | xavier_initializer  | tf.keras has a glorot_normal and glorot_uniform |
| layers.initizalizers | variance_scaling_initializer | Exists in tf.keras.initializers |
| layers.layers | avg_pool2d  | Exists in tf.keras.layers |
| layers.layers | avg_pool3d | Exists in tf.keras.layers |
| layers.layers | batch_norm | Exists in tf.keras.layers |
| layers.layers | bias_add | Exists in tf.keras.layers |
| layers.layers | conv1d  | Exists in tf.keras.layers |
| layers.layers | conv2d  | Exists in tf.keras.layers |
| layers.layers | conv3d  | Exists in tf.keras.layers |
| layers.layers | conv2d_in_plane | Functional Alias |
| layers.layers | conv2d_transpose | Exists in tf.keras.layers |
| layers.layers | conv3d_transpose | Exists in tf.keras.layers |
| layers.layers | convolution | Exists in tf.keras.layers |
| layers.layers | convolution1d | Exists in tf.keras.layers |
| layers.layers | convolution2d  | Exists in tf.keras.layers |
| layers.layers | convolution2d_transpose | Exists in tf.keras.layers |
| layers.layers | convolution3d | Exists in tf.keras.layers |
| layers.layers | convolution3d_transpose | Exists in tf.keras.layers |
| layers.layers | dropout | Exists in tf.keras.layers |
| layers.layers | elu | Exists in tf.keras.layers |
| layers.layers | flatten | Exists in tf.keras.layers |
| layers.layers | fully_connected | Exists in tf.keras.layers |
| layers.layers | gdn | Functional interface of GDN |
| layers.layers | images_to_sequence | No OSS uses found /  Functional paradigm |
| layers.layers | linear | Exists in tf.keras.layers |
| layers.layers | pool | Exists in tf.keras.layers |
| layers.layers | max_pool2d| Exists in tf.keras.layers |
| layers.layers | max_pool3d | Exists in tf.keras.layers |
| layers.layers | one_hot_encoding | Exists in tf.keras  / Uses collections |
| layers.layers | relu | Exists in tf.keras.layers |
| layers.layers | relu6 | Exists in tf.keras.layers |
| layers.layers | repeat | Exists as sequential model |
| layers.layers | separable_conv2d | Exists in tf.keras.layers |
| layers.layers | separable_convolution2d | Exists in tf.keras.layers |
| layers.layers | softmax  | Exists in tf.keras.layers |
| layers.layers | stack | Exists as sequential model / Uses variable scoping |
| layers.layers | unit_norm | Exists in linalg |
| layers.layers | legacy_fully_connected | Legacy layer |
| layers.layers | legacy_linear | Legacy layer |
| layers.layers | legacy_relu  | Legacy layer |
| layers.regularizers | l1_regularizer  | Available in tf.keras.regularizers |
| layers.regularizers | l2_regularizer | Available in tf.keras.regularizers |
| layers.regularizers | l1_l2_regularizer | Available in tf.keras.regularizers |
| layers.regularizers | sum_regularizer | Trivial convience wrapper |
| layers.regularizers | apply_regularization | Uses collections |
| layers.rev_block_lib | rev_block | Functional paradigm for RevBlock |
| layers.summaries | summarize_tensors  | Trivial list comprehension |
| layers.summaries | summarize_collection | Uses collections |
| layers.summaries | summarize_activations | Uses collecftions |
| layers.target_column | ALL | Deprecated since Estimators |
| layers.utils | collect_named_output | Unsupported tensor alias API  |
| layers.utils | append_tensor_alias | Unsupported tensor alias API |
| layers.utils | gather_tensors_aliases | Unsupported tensor alias API |
| layers.utils | get_tensor_aliases | Unsupported tensor alias API |
| layers.utils | convert_collection_to_dict | Uses collections |
| layers.utils | static_cond | Simple wrapper / No OSS use |
| layers.utils | smart_cond | Simple wrapper / Little OSS use |
| layers.utils | get_variable_collections | Uses collections |
| layers.utils | channel_dimension | Simple wrapper / No OSS use |
| layers.utils | last_dimension  | Simple wrapper / No OSS use |
| layers.utils | two_element_tuple  | No OSS use |
| layers.utils | n_positive_integers | No OSS use |
| nn.cross_entropy | ALL | Deprecated Losses |
| losses.loss_ops | ALL | Available in core tf.losses |



**Notes:**
* More details of our code review can be found in [this spreadsheet](https://docs.google.com/spreadsheets/d/1hYJchHp1y1t2U6htq5UXxMxWlGxxtOyyNHDF8_qhtQQ/edit#gid=185512613)
* We used [this analysis tool](https://tf-contrib-analyzer.herokuapp.com/) to detect OSS usage.

## Questions and Discussion Topics

* Are there any modules being excluded from the move that you feel have substantial value to the community?
* Are there any new modules that you feel should be added to addons from somewhere else apart from tf.contrib
* We're actively collecting volunteers to help move, refactor and/or maintain in Addons (Please reachout to our [mailing list](https://groups.google.com/a/tensorflow.org/forum/#!forum/addons)
or [gitter channel](https://gitter.im/tensorflow/sig-addons) if you have interest in helping our community.

## After Request Notes
* Now that the review period has ended, please post all suggested
 additions/removals directly to the tensorflow/addons [issues page](https://github.com/tensorflow/addons/issues)