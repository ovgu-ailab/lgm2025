---
layout: default
title: Assignment 5
id: ass5
---


# Assignment 5: Inspection Through Inception
**Discussion: November 28th**  
**Deadline: November 27th, 18:00**


This week, we do not get to know any new models.
Rather, we will look at how to _evaluate_ the ones we already have.
Also, we will use the opportunity to introduce a new concept, namely _conditional generation_, where we want to steer the generated outputs into some direction e.g. by supplying a specific class to generate, rather than completely random outputs.

There will be little work to do in terms of basic implementations;
rather, the focus will be on experiments.
We will first discuss the points of evaluation and conditional generation in more detail;
your task will follow towards the end.


## Evaluating Generative Models

We provide two evaluation metrics in `lgm/evaluation.py`,
namely the [FID](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance) and 
[Inception Score](https://en.wikipedia.org/wiki/Inception_score) (IS).
Out of these, the latter is considered somewhat outdated, and FID should be preferred.
Still, it can be instructive to evaluate both and compare the results.

Both measures require a pretrained classifier network.
Their names come from the fact that traditionally, an Inception net is used for this purpose.
However, the methods themselves don't depend on a specific architecture, and forcing other datasets into the correct format (Inception was trained on ImageNet) can be cumbersome.

For this reason, we provide pretrained classifiers for various datasets on E-Learning (not in the repository due to file sizes -- check under "Additional Files").
It's important to use one specific architecture and set of weights to have comparable results!
Note that, for this reason, your results will not be comparable to anything you might find in the literature.
But within our little course, we can compare numbers. :)

The starter notebook discussed in the next section contains usage examples for both metrics, as does the "solution" notebook for the GANs from last week.


## Conditional Generation

Being able to provide some "direction" to what is generated is the basis of modern text-to-image/video/music models.
We will start small:
We want to provide a single _class label_ and have the model generate a sample of that class, instead of just any random sample from the dataset.
There are several things to work out here:

1. Obviously we need a dataset with labels.
This precludes a few of the ones we have in the repository, such as Flickr or STL-10.
2. We need to update our training and generation code to take in those labels.
This is relatively simple, but if we want to preserve the option to also train and run _unconditional_ models, the code will end up somewhat more complex.
3. We need to actually _process_ the conditioning information in our models somehow.
There is no single correct way of doing this, but in any case it will lead to more complex neural network layers.

### Conditional VAE
We have updated `lgm/vae` to support _optional_ conditional generation.
This means you can still train unconditional VAEs if you want.
Note that we could easily update the GAN code in a similar manner.
However, we have decided not to include this for now, so as to keep it a bit simpler.
You could of course try to do this yourself! :)  

Furthermore, `lgm.layers` has been updated with _adaptive normalization_ (discussed in the exercise) for integrating the conditioning information into the networks.

There is also a new starter notebook `06_vae_conditional.ipynb` that showcases how to train such a model, as well as using the evaluation methods.

**NOTE:** The old VAE notebook (`04_vae`) unfortunately will no longer work with the updated code.
However, you can also train unconditional models as before with the new code.


## Comparing Generative Models

Your task this week is to revisit the models we have gotten to know thus far, define a sensible "experiment" that evaluates and compares different models, and report on the results.
This means:

1. Define one or more interesting questions you want to focus on. Some examples:
    - Investigating the effect of `beta` (the weight on the KL-loss) in VAEs.
    - Comparing different likelihoods (corresponding to reconstruction losses) in VAEs.
    - Comparing GANs vs VAEs.
    - Evaluating whether GAN "tricks" like scaling data to [-1, 1] vs [0, 1] actually make a measurable difference.
    - Comparing conditional VAEs and unconditional VAEs, all else being the same.
    - ...
2. Define a hypothesis, i.e. a belief about the answer to the question(s).
3. Train the necessary models.
4. Evaluate each model using FID and/or IS, as well as visually inspecting samples.

Then you should produce a "report" (does not need to be formal) on your experiment(s).
You could use Markdown cells in a notebook for this; you don't need to submit all the model training code, but can do so if you wish.
You should discuss the following points:
- What was your expected result before starting?
- Did your experiments confirm your expectations or not? Did you make other observations besides your original hypothesis?
- Do the "objective" measures like FID match up with your subjective impression of sample quality?
