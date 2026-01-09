---
layout: default
title: Assignment 10
id: ass10
---


# Assignment 10: Ultimate Guide to FID < 1
**Discussion: January 16th**  
**Deadline: January 15th, 20:00**

This week, we look at classic ["denoising diffusion probabilistic models"](https://arxiv.org/pdf/2006.11239), or DDPMs for short.
As it turns out, they are extremely similar to score-based models, to the point that they can be implemented in the same framework, as we will also see in this class.
For this reason, we will not focus so much on the basic implementation, although as usual, there are details left for you to fill in, which should hopefully simple given the similary to the score-based models considered previously.

Really, almost all you need is given in the [original paper](https://arxiv.org/pdf/2006.11239), algorithms 1 and 2 on page 4.
One open question is how to choose `sigma_t` for the sampler, but this is discussed in section 3.2 (page 3).
The noise schedule `beta_t` is given at the start of section 4 (page 5).
One complication comes from the fact that DDPMs are usually conditioned on the "time step" `t`.
Recall that in score-based models, this was solved by simply dividing the model output by the standard deviation.
However, for DDPMs, this seems insufficient in practice.
Therefore, the time step is provided to the model as a conditioning variable.
This is handled in exactly the same way as the class conditioning back when we looked at conditional VAEs!
The only difference is that we have a continuous variable rather than discrete classes, and this is encoded via a _positional encoding_ similar to Transformers, followed by a small fully-connected network.

As usual, these details are also provided in the associated notebook.
Thus, a basic implementation of DDPMs hopefully shouldn't be an issue for you.


## Guided Diffusion

Given the relatively simple implementation, we will use the additional capacity to revisit _conditional_ generation.
This means we want a model of `p(x|c)`, where `c` can be a class identity, for example.
It turns out this works in exactly the same way as for the conditional VAEs we implemented a while back.
But recall what we stated above:
We already need to provide _time conditioning_ to the model.
This makes it even easier to switch to a class-conditonal model:
Since even an "unconditional" DDPM already receives time conditioning, we can simply add in the class conditioning on top!

A class-conditional DDPM should provide a very strong baseline for labeled datasets like FashionMNIST or CIFAR.
In fact, it should easily eclipse anything you have implemented in this class so far in terms of sample quality and evaluation measures like FID, all while being relatively simple and stable to train.
Of course, we inherit the same disadvantage as score-based models:
Sampling is **very** slow.

Before we look at sampling speed, we want to introduce a technique that is absolutely instrumental in high-quality conditional generation nowadays: Guided diffusion.
If you want to read up on the details you can check [the paper introducing classifier guidance](https://arxiv.org/pdf/2105.05233), and the more popular [classifier-free guidance](https://arxiv.org/pdf/2207.12598).
But the resulting sampling procedure is very simple:

- Define a _guidance weight_ `w`.
- Replace any conditional model call `model(x, c)` by `(w + 1) * model(x, c) - w * model(x, null)`.

Here, `null` is some kind of "empy conditioning", which denotes an _unconditional model_.
This is implemented by randomly dropping the class conditoning `c` during training, and replacing it by `null`.
This way, you are training a conditional and unconditonal model at the same time, using only one network.
We also get some special cases:
- `w = -1` results in an unconditional model, although this will usually perform worse than one that has been trained fully in an unconditional manner.
- `w = 0` is a standard conditional model.
- `w > 0` (usually `w >= 1`, but you can try values < 1) activates guidance.

You can expect guidance to produce outputs that are often higher quality on average, and adhere more closely to the class conditioning, while incurring a reduction in diversity.
Setting `w` too high will usually result in outputs degrading:
Colors become oversaturated, and output diversity collapses to a few modes per class.
Although guidance is much more essential for larger, more complex datasets, it's a very good idea to acquaint yourself with this concept now.


## Efficient Sampling

One "small" issue with guided diffusion is that we require two model calls per sampling step.
This makes the already slow sequential sampling procedure up to twice as slow.
At this point, we should investigate different sampling methods for diffusion models, all of which can be applied to the _same_ trained model.

First off, the paper actually gives two choices for the reverse process variance `sigma_t**2`.
They claim to have gotten similar results from both, but this is generally only true in the limit of many time steps.
Often, the second choice given in the paper gives _much_ better results.
The provided code already implements both versions.

Another popular sampler is proved by ["implicit diffusion" (DDIM)](https://arxiv.org/pdf/2010.02502).
This provides a _deterministic_ sampler, meaning that there is no additional noise added during the sampling step.
This sampler often works much better for very few sampling steps.
The theory for this method is very complex, but the sampling procedure can still be implemented straight from the paper, and this has already been done.

You can compare the different samplers at different noise schedules (i.e. number of sampling steps).
In fact, you can try different schedules with the _same_ trained model; no need for retraining.
The notebook shows how this is done.


## Your Task: Investigating Diffusion Sampling

You should form your own impression diffusion sampling through hands-on experience.
In the notebook, you can find code that tests
- a): Different sampling methods at different schedules
- b): Different guidance weights

You should try reasonable parameter ranges and see how this affects the results.
Both subjective impression of the generated samples as well as quantitative evaluation measures are good.
Try to answer the following questions for yourself (and feel free to put your answers in your submission):
- How do the different sampling methods compare? Is one clearly superior/inferior?
- How many sampling steps do we really need? Does it have to be hundreds or even thousands? Does this depend on the sampling method?
- What guidance weight is good? Can we trust the quantitative measures like FID here?
- Do the answers change when switching to more complex datasets?

This is rather open-ended in that you likely won't have enough time to run all the evaluations that might be interesting to you.
But do try to get started early, so that you can allot plenty of time for the evalution to run.
Running many different setups is easy once you have a basic loop going; it really just requires enough compute.


## Please Read: Important Note on Breaking Architecture Changes

There was a bug in our model code that lead to slightly incorrect/inconsistent downsampling.
This has been fixed; however, this makes old model checkpoints invalid, as they correspond to a slightly different architecture.
Crucially, this also affects the classifiers we have been using as feature extractors for evaluation measures like FID!
We trained new models with the fixed architecture, so you should download the new checkpoints.
They can be found [in a cloud folder](https://cloud.ovgu.de/s/2YSsxHKHxC4MDWm) which is also linked via E-Learning as before.
It will be easy to see whether you got the correct weights, as the old ones will not load with an up-to-date version of the repository.

Unfortunately, using different models from before also means that the evaluation results are not comparable to previous weeks.
You would have to re-train new generative models (since they will also be affected by the downsampling change), and evaluate them again.
However, as long as you use the new models for this assignment, the evaluation is at least internally consistent.
Sorry for the inconvenience!

If you _really_ need to use the old models for some reason, you can use the repository before commit `9af4aa9696fc9dc706bd82e6d0f6716716c4364d` on January 8th (marked with the `BREAKING` message), or revert the change in line 286 in `layers/conv.py` made in that commit to restore the old, slightly incorrect architecture.
