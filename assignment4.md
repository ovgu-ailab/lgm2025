---
layout: default
title: Assignment 4
id: ass4
---


# Assignment 4: Generative Adversarial Notworks
**Discussion: November 21st**  
**Deadline: November 20th, 18:00**


When deep generative models first started emerging, the biggest competitors were VAEs and GANs.
GANs are still among the top contenders for high-quality outputs, however their training is rather difficult.
In this assignment, we will implement basic GANs and see if we can improve them a little.


## Implementing GANs

Compared to VAEs, GANs are somewhat light on theory, making the implementation more straightforward:
We don't have to think as much about how to put mathematical equations into practice.
However, difficulties arise from other sources:
- Correctly and efficiently implementing the two-model training loop, ideally without having to re-write too much code and abandoning our `TrainerBase` completely. 
In fact, GANs will be the _only_ model where we cannot just implement a `core_step`; we will have to overwrite the entire `train_step` instead.
- Actually getting the model to generate anything sensible.
This is by no means guaranteed by a correct implementation of the basic GAN training loop; we will need to employ some advanced methods.

### Training Loop
We once again have a starter notebook (`05_gan_starter.ipynb`) and parts of the model/training code (`lgm/gan.py`) given.
This has been optimized somewhat by creating a single `GAN` object that contains both the generator (G) and discriminator (D);
this allows us to get away with a single `model`/`optimizer` input to the `Trainer` class.
Otherwise, we would have to create a completely new class that doesn't inherit from `TrainerBase` to take two separate model/optimizer inputs.

This also puts some constraints on our training loop.
There is a "naive" version of GAN training where you would train the models alternatingly.
This would require calling each model twice (once for D training, once for G training).
We instead opt for a more optimized version that only calls each model once and updates them in parallel.
This is faster, and only differs slightly from the alternating version.
It's also the only version we can really use with a single optimizer object.
On the flipside, this version is perhaps slightly more difficult to understand than the "naive" one.

### Architecture Considerations
You can view D similarly to the encoder in a VAE, and G similar to the decoder.
This makes it relatively easy to build neural network architectures.
A few things to keep in mind:
- D should end with a _single output_ for binary classification.
Since we usually use logit-based cross-entropy losses, there should be no output activation.
- [Due to reasons](https://ovgu-ailab.github.io/blog/methods/2022/07/07/batchnorm-gans.html), D should not use batch normalization.
We can use `nn.GroupNorm` instead.
- G can have an output activation that forces generations into the correct, range, such as [0, 1] for images.
Thus, a sigmoid activation on G could make sense.
You could also use no output activation and thus force G to handle the correct output range by itself.
- We once again need a _prior distribution_ for the latent space.
But other than for VAEs, we never need to backpropagate through the latents.
This means you are technically free to choose any distribution at all -- even discrete ones!
But a sensible default could be a standard normal distribution once again.


## Making it work

Even after you got a technically functioning training loop, chances are the results are very bad.
Thus, we really need to think about employing some advanced methods to improve performance.
At the same time, we want to keep the training relatively simple/close to the original GAN formulation.

First, let's start with a collection of simple tricks:

### Optimizer improvements
Many GAN training loops make some changes to the standard optimizer setup:
- Reduce the learning rate.
Adam mostly as a default set to `0.001`.
Try reducing this by a factor of 5-10, e.g. `0.0002`.
- It seems useful to reduce or completely remove momentum.
Intuitvely, this might be because of the "moving target" we have in GAN training -- lower momentum allow each network to more quickly adapt to changes in the other one.

### Rescaling data
It is often advocated to scale data such that it lies in the range [-1, 1] rathern than [0, 1].
This is trivially implemented using the `Normalize` transform from `lgm.data`.
An example is given in the starter notebook.
Note that we will need to revert this transformation whenever plotting data.
Also, G needs to output data in the correct range, but this can be enforced using `nn.Tanh` as an output activation.

### One-sided label smoothing
To prevent the D loss from saturating too much, we can employ _label smoothing_, replacing the 0/1 labels by soft versions.
[Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498) proposes to only smooth the _positive_ labels away from 1 (e.g. 0.9), but leave the negative labels as 0.

### Adding noise
We can add small amounts of noise to the data, as well as generated outputs, before putting them into D.
This makes the job slightly harder for the latter.

### EMA
A popular technique for modern generative models is to use an _exponential moving average (EMA)_ over model parameters.
During training, we keep an average of the parameters at each step.
After training, the parameters of the final learning step are replaced by the EMA parameters.
This supposed to smoothen the variance introduced by singular training steps, and can reduce issues with models "going in circles".

This is already implemented in `lgm.common.EMA` class.
You just have to create the object and pass it to the Trainer class.
Afterwards, one function call applys the EMA parameters.
It also creates a backup of the original parameters, so in case the EMA parameters don't perform well, it's easy to revert.

### LSGAN
There are other possibilities to use for the loss function besides the
classical cross-entropy.
One is the Least-Squares-GAN (LSGAN).
While in principle there is quite a bit of theory as well as different possible setups, the basic idea is just to use the squared error for the D loss.
That's it!

### Batch features
We can help D enforce variety in the generated batches by explicitly computing features _over the batch_, rather than the typical per-input features.
A simple example is the `SDLayer` given in the provided code.
This can easily be added into D.

### Feature Matching
This is the most important of all these techniques.
The intuition is quite simple.
If we had a good G that was producing samples whose overall distribution looks like the data distribution, then
the features computed by D in the hidden layers should _on average_ be similar between the real data and samples from G.
This means:
- Compute features for real data and average.
- Compute features for generated data and average.
- Compute the squared difference between these averages.
Note: NOT the average difference -- rather the difference of averages!
- Since this difference should be small, we can use it as a loss function for G!

In principle, we can just add this so-called _feature matching loss_ to the regular G loss, or even replace the old one completely.
This is a very powerful technique that most modern GANs will employ in some way.

There is just one problem:
How do we get the features?
Turns out we must adapt our D model to return _hidden layer outputs_ (i.e. "features") in addition to its regular classification output.
This can be achieved by a change to our `layers` module.
We discuss this in the exercise; otherwise there is a usage example in the starter notebook.
Which features to use is completely arbitray.
We decide to return all _block outputs_.
Then the difference outlined above is computed for each block output, and everything is summed together.
This ensures that both early, mid and late features from D give a training signal to G.

Note with this setup it's quite likely that D achieves pretty much perfect classification, and yet G is still learning fine.
We don't _have to_ balance the "game" between the two networks, we just have to make sure there is a good training signal for G, and this is much
simpler with feature matching.
Try it out!


## Summary

All in all, we need to 
- Implement the basic training loop and make sure it runs.
- Implement further "tricks" to get good outputs.

Ideally, you would add these tricks one-by-one and train a new GAN each time.
But this will take quite long.
So you may bundle a few of them up and add each "package" at once.
That way, you might get 2-4 or so iterations on your model.
If you are short on time, the most impactful one should be feature matching, so definitely try that one!

At the end, you should reflect on which methods worked particularly well, and which seemed to make little or no difference.
Once again, we have the issue of how to evaluate the models (this will be alleviated next week).
For now, you can judge the visual sample quality.
Also look out for signs of low mode coverage:
Generate a large batch of images, say 100.
Can you see different images that look almost or exactly the same?

Finally, in the starter notebook there is some code to achieve a kind of "approximate inference" using GANs, even though they lack the encoder network of VAEs.
This can be considered more of a bonus.
