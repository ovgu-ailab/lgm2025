---
layout: default
title: Assignment 3
id: ass3
---


# Assignment 3: ELBOs Out
**Discussion: November 14th**  
**Optional Deadline: November 13th, 18:00**

This week kicks off our "marathon" of implementing various kinds of generative models.
We start with Variational Autoencoders (VAEs), a classic model that has lost significance somewhat, but is still important in various contexts, and also forms the basis of other models we will see later.


## General Remarks

These assignments will usually follow a pattern:
1. Implement the model in question, along with a training procedure.
2. Test the model on a small dataset (like MNIST) to make sure it works.
3. Either
    1. Try to scale the model to a larger/more complex dataset, or
    2. Run more detailed experiments with various parameter settings.

Of course, you can also do both. :)

For each assignment, we will give you significant help via the course repository:
- A dedicated file with code related to the model in the `lgm` module.
This will take care of boilerplate as well as some of the more complex issues, or those easy to get wrong.
However, there will generally be details missing that you have to fill in.
This is usually about understanding the theory and being able to put it into practice.
These are crucial skills if you want to work on more recent state-of-the-art models, so you have to practice them!
This is why we heavily recommend _against_ using generative AI or complete tutorials for help, as you will not build self-sufficiency this way.
Please also review the general notes in Assignment 0.
Note that the code in our repository is exempt from our usual plagiarism rules -- it's there for you to use!
- A notebook with some starter code taking care of more boilerplate such as loading data or building neural network architectures.
You can use this as a starting point for your experiments.

In general, you could ignore this help and do everything yourself.
You will learn much more this way, but it will also be _much_ more work.
As such, make sure you have gotten access to [the course repository](https://code.ovgu.de/ai-lab/teaching/lgm/lgm2025) as well as sufficient compute resources (we are offering GPU access as detailed on Mattermost).


## Implementing VAEs

As the name implies, VAEs are very similar to standard Autoencoders.
As such, the code for Assignment 0, as well as `lgm/autoencoder.py` should be helpful here.
Make sure you understand that code and try it out.
Then, we can consider how to implement VAEs just in terms of the difference to standard AEs.
There is also already a file `lgm/vae.py` with just a few pieces missing.
These are marked by throwing a `NotImplementedError`.
We further have a notebook `04_vae_starter.ipynb` that sets up a basic VAE model, again with some parts missing.
The main changes to AEs you need to consider are as follows.

### Encoder returning probability distributions
The "encoder" in a VAE actually represents the _variational posterior q(z|x)_.
This means we need a conditional distribution over _z_ given an input _x_.
Thus, we need to decide what form our distributions should take.
The easiest choice by far is a Gaussian.
Such a distribution is fully characterized by just two parameters, namely the mean and the variance.
This means our encoder has to return _two_ values for each code dimension, which we can interpret as the two parameters, respectively.
- You will need to split the output into two parts. See `torch.split`.
- Make sure the variances are positive values, for example by applying `torch.exp`.

NOTE: Recall the discussion in the exercise.
We are technically free to choose whether our model returns variances, standard deviations, precisions, log-variances...
There is no "correct" choice here.
We just have to make sure to be consistent here and convert to whatever values we need.
For example, the sampler (see below) requires the standard deviations.
The KL-divergence requires variances and log-variances.

### Sampler for the latent space
We need a method to _sample_ values for _z_ from the distribution _q(z|x)_.
Recall the "reparameterization trick" from the lecture.
This module should be given means and standard deviations as input (as returned by the encoder), draw a random sample from a Standard Gaussian distribution of the same size as the input, multiply by the per-dimension standard deviations, and add the means.
Thus we have sampled from _q(z|x)_ with the correct parameters, while keeping the gradients with respect to those parameters (recall these are the encoder outputs) intact.

### Kullback-Leibler Divergence
One reason to choose a Gaussian variational posterior, and a Gaussian prior as well, is that the KL-divergence between the two can be computed easily in closed form.
The formula has already been implemented in `vae.py`!
You just have to figure out how to get the correct values from the encoder to put into the function. :)

### Putting it together
Implementing the above three points should give you a functioning VAE model.
Try training one and see if it works.
The code is already set up to regularly plot reconstructions as well as random generations.
Make sure that both aspects look reasonable!
It's possible to have good reconstructions, but bad generations (the other way around is unlikely).
See the AE code for Assignment 0 on how to choose a reconstruction loss corresponding to a data likelihood.
You can go with `"gaussian_fixed_sigma"` as a reasonable starting point.
You will also have to implement the actual neural network architectures; again the basic autoencoder code can serve as inspiration here.

Finally, note that you also have to use some kind of reconstruction loss corresponding to the decoder likelihood.
If you are confused about this topic, [we wrote a blog post](https://ovgu-ailab.github.io/blog/methods/2025/10/05/vae-reconstruction.html) that might help.


## Going Further

After you have a basic model going, we recommend that you try it out further.
Some ideas for experiments:
- One extension is the [beta-VAE](https://openreview.net/pdf?id=Sy2fzU9gl).
This amounts to simply multiplying the KL-divergence by a fixed weight `beta`, usually larger than 1.
This is trivial to implement and has already been done in the code.
Try different values for `beta` and observe how the model changes (be sure to create a new model and train from scratch each time).
    - Can you find the "best" value for `beta`? Is it close to 1? Larger? Smaller? How do you even evaluate this?
    - You can manually try some values, or construct a loop over a whole range and just let it run for some hours on the server.
    This gives you a more complete picture.
- How does the VAE behavior change as you increase or decrease the size of the latent space?
- How does choosing different likelihoods/reconstruction losses affect the model?
Note, this will heavily interact with the choice of `beta` above!
Different likelihoods will have different optimal `beta` values.
- How well do basic VAEs scale to more complex datasets?
For example, you could try something like CIFAR.
Be sure to increase the size of the model, as this is much more complex than MNIST!
    - At the very least, we recommend that you try a model for FashionMNIST or SVHN.
    Are you happy with the generated outputs?
    Do they look sharp and detailed to you?
- Finally, there is some code already in the notebook investigating interpolating in the latent space, as well as how much
the VAE is actually using the latent space dimensions.
