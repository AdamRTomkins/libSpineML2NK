This version of the photo receptor is a 'halfway house' between the model presented Cosine 2014, and the latest model which includes the LMC.

Because of this, the model will never 100% match either the old, or new model implemented by carlos.

This model contains:
    An updated mean generation mechanism from the 2016 model with an LMC
    Updated parameters from the 2016 model

However as this model does not contain the lmc, as such the parameters are not 100% corect.

At this point to 100% reproduce a photoreceptor in NK, we must either
1) Implement the full LMC model
    Time Consuming
    Not certain how this can fit into the neurokernel LPU models

2) Go back to the poster model
    Find the correct parameters
    Change the Mean Generation mechanism
        But this is not the model we would want to continue working with.

In the meantime, we have a half house result which produces sensible Narx output, but will not 100% reproduce any single model.
