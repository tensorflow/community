# Migration From TF-Addons To TensorFlow Core

### In-Progress & Previous Migrations:
https://github.com/tensorflow/addons/projects/2/

### Process 
1. Create an issue in TensorFlow Addons for a candidate that you think should be 
migrated. 
2. The SIG will evaluate the request and add it to the `Potential Candidates` section 
of our GitHub project.
3. If it's agreed that a migration makes sense, an RFC needs to be written to discuss 
the move with a larger community audience. 
    * If the transition will impact tf-core and Keras then submit the RFC to 
    [TensorFlow Community](https://github.com/tensorflow/community)
    * Additions which only subclass Keras APIs should submit their migration proposals to 
    [Keras Governance](https://github.com/keras-team/governance)
    
4. A sponsor from the TF/Keras team must agree to shepard the transition.
   * If no sponsor is obtained after 45 days the RFC will be rejected and will remain 
   as part of Addons.
5. If a sponsor is obtained, and the RFC is approved, a pull request must move the 
addon along with proper tests.
6. After merging, the addition will be replaced with an alias to the core function 
if possible. If an alias is not possible (e.g. large parameter changes), then a deprecation 
warning will be added and will be removed from TFA after 2 releases. 


### Criteria for Migration
* The addition is widely used throughout the community, or has high academic significance.
    * Metrics must be reported in the RFC (OSS usage, H5 index, etc.)
* The addition is unlikely to have API changes as time progresses
* The addition is well written / tested
