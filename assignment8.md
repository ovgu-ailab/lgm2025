---
layout: default
title: Assignment 8
id: ass8
---


# Assignment 8: Flow, Thanks
**Discussion: December 19th**  
**Deadline: December 18th, 20:00**


This week, we will get to know Flow-based generative models.
These are a bit special in that they do not really provide state-of-the-art models, yet other model types based on the _idea of flows_ are currently among the top performing models.
As such, you should likely scale back your expectations compared to autoregressive models.
On the other hand, trainings tends to be faster and is more straightforward, as we only have to train a single model, rather than the two-step process of VQVAE -> Autoregressor.

Unfortunately, the relatively simple "classic" Flow models really don't perform well.
On the other hand, more complex extensions tend to very involved in terms of mathematics and implementation.
As such, we will have to settle for a compromise and just explore that a bit.
The code can be found in the repository as usual, in `lgm.flow` and associated notebooks.


## NICE

The OG deep Flow model is [NICE](https://arxiv.org/abs/1410.8516).
We include this in the repository for reference, but this model does not perform on par with pretty much any other framework we have seen so far in the class.
Still, it is comparetively simple and can be implemented without too many issues, so it serves as a good benchmark for your understanding of the general flow paradigm.

As such, your first task is to implement the missing pieces of the model.
This consists of two parts:
- The full model, which is a sequence of _additive coupling layers_ plus a scaling layer at the end.
- The individual coupling layers, which split the input into two parts and transform one based on a function of the other, leaving the latter unchanged.

This allows us to transform inputs from a complex distribution (i.e. the data) into a simple prior, like a standard Gaussian.
Training is then simple:
We minimize the _negative log-likelihood_, which is easy to compute thanks to the flow framework.
Some relevant sections in the paper include:
- The end of section 1, or section 3.2 on the coupling layers.
- Section 3.3 on the scaling layer.
- Sections 2 and/or 3.3 on the negative log-likelihood.
- Section 3.4 on choosing a prior.
- Figure 3 (section 5) on architectures.

We include a notebook with some simple 2D toy data you can use to test your implementation.
A correct setup should be able to fit this easily.
You can also move on to "real" datasets, but don't expect good results.
The paper shows some samples:
Even MNIST doesn't work too well, and anything beyond that is pretty much hopeless.


## Glow

To let you experience a Flow model that works somewhat better, we also implement the [Glow architecture](https://arxiv.org/abs/1807.03039).
This is essentially an extension of [RealNVP](https://arxiv.org/abs/1605.08803), which in turn is an improvement over NICE.
The main differences to NICE are:
- Convolutions instead of MLPs.
- Affine coupling layers (shift and scale) instead of additive (shift only).
- Normalization layers in between coupling layers.
- A _multi-scale_ architecture that factors out variables at each scale for efficiency.
- _Learned mixing of channels_ via invertible 1x1 convolutions instead of predetermined splitting of the inputs.

In terms of implementation, this is likely the most difficult model we will consider in this class.
As such, this has been fully implemented already, allowing you to focus on experimentation.
You should try to simply get a strong generative model on a dataset of your choice.
Some issues to tackle include:
- Flow models tend to be extremely size- and parameter-hungry for some reason.
  - For example, even the NICE architecture on MNIST has around 20 million parameters.
  - Glow architectures are even worse, with the paper (see tables in Appendix C) using _at least_ 32 coupling layers **per level** of the multi-scale architecture, with at least 3 levels depending on image size, and 3 convolutions per coupling layer.
This implies _at least_ 300 convolutional layers in total, and they also choose them to be fairly wide (512 channels throughout).
These models are absurdly large, so prepare to go pretty big.
  - The Glow authors also seem to train their models for close to 2000 epochs. :)
- Unfortunately, large and deep models tend to lead to instability, which seems especially problematic with the various scaling components in Flow models.
You may need to rein in the training process through stabilizing changes such as smaller learning rates.
  - Note that _weight decay_ can sometimes lead to its own underfitting/stability issues, as it seems to negatively affect the flexibility of the scaling components.
  - You could try changing the `scale_fn` to something besides the classic `exp`.
  For example, [the official OpenAI code](https://github.com/openai/glow/blob/master/model.py#L395) has actually swapped this out for `sigmoid`, directly contradicting the paper...
- RealNVP actually models images in logit space rather than pixel values directly. Examples of this are included in the notebook.
- Similarly, normalizing the data might help the model learn the correct scaling more easily.

You can be bold here; if many people try many different things, we might end up with a pretty decent model at the end. :)
We have been able to achieve FashionMNIST FIDs below 30, although inconsistently, which would be comparable to conditional VAEs from earlier in the class.
Or you can read the Glow paper and try to stick als close as possible to all their choices.
Up to you!
