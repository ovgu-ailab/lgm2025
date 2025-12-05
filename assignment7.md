---
layout: default
title: Assignment 7
id: ass7
---


# Assignment 7: Go Big or Go Home
**Discussion: December 12th**  
**Deadline: December 11th, 20:00**

This week, we will combine our VQVAE with an _autoregressive_ model that can generate latent codes, which can then easily be decoded back to data space.
This is a massive gain over naive autoregressive models trained directly in data space, as these will be _very_ slow for high-dimensional data.
At the same time, this offers more stable training than GANs and less restricted models than VAEs.


## Autoregressive Models

Chances are (especially if you took last semester's IDL class), you have already seen or worked with autoregressive models without knowing it.
In particular, the common _language models_, whether based on Transformers, RNNs, or even CNNs, are usually autoregressive models!

These models generate one "token" after another, which is the main reason for why they are so slow to run.
It doesn't matter what a token is -- a piece of language, or the index of a codebook vector for the VQVAE.
Thus, autoregressive models are extremely general and can be easily applied to all sorts of problems.
For most of you, the implementation should merely be a bit of repetition!
The interesting part starts when we _combine_ this with a VQVAE in a holistic framework.

The relevant code can be found in the `lgm.larp` module and the associated notebook.
As usual, there are a few things left for you do to, but these should be relatively straightforward.

### Training a Latent Autoregressive Prior
The recipe to follow is quite simple:
- Use a pretrained VQVAE to encode a dataset into codebook indices.
    - You can either encode the entire dataset and use that for training, or do the encoding on-the-fly for each batch. The latter will be much slower, however.
- Train an autoregressive model on those encoded indices, as you would do e.g. on word indices for language modeling.
- Generation proceeds by first generating code indices autoregressively, then decoding the resulting "image" using the VQVAE.

The result is a reliable, stable training process and (usually) fairly high-quality samples.
Thus, the most interesting aspects are how to _optimize_ training and generations.


## LARPing Harder

### Preparing the VQVAE
Some of the choices we make for our VQVAE will significantly affect how easy of a time the autoregressive model will have.
In particular:
- We should aim for as small as a grid as possible. For example, an 8x8 grid is much easier to model than a 16x16 grid, and training will also be faster and require less memory, which in turn unlocks larger/more powerful models.
- A smaller codebook implies fewer classes for the autoregressive model to predict, again making the task easier.

Unfortunately, stronger compression also means worse reconstructions, so in practice we will have to settle for some compromise.
However, we can counteract this to some extent by a more powerful VQVAE model (more layers, filters etc.).
Since autoregressive training works purely in the code space, a larger/slower VQVAE will not affect performance here.

You can of course use your own models from last assignment, or even train new ones.
But we also supply checkpoints for Flickr, STL-10, CIFAR10 and FashionMNIST.
You can find these on E-Learning under "Additional files".
All these models map to 8x8 grids, but with different networks and codebook sizes.
Please see the notebook for... notes.
You can expect the best results from Flickr, as the more regular structure of the dataset (everything is a face) makes modeling much easier, or from FashionMNIST due to the relative simplicity of the dataset as a whole.

### Scaling and Controlling the LARP
When training a small or medium-sized Transformer model, you may see the following:
- Training loss keeps decreasing, but stays relatively high overall.
- Validation loss decreases for a while, then either bottoms out or starts increasing again (for larger models).

We have a bit of a problem here: We are overfitting (gap between training and validation; validation loss increases), but also underfitting (high training loss).
Perhaps paradoxically, the main solution can be to _use a bigger model_.
Pushing down the training loss will often push the validation loss along with it.
Furthermore, lower training loss, even at the cost of increased validation loss, tends to improve generations _even when evaluated against the validation set_.
Finally, scaling laws tell us that, given a certain amount of compute (i.e. time), training a _larger model for fewer steps_ tends to be more effective than training a smaller model for more steps.

Finally, classification performance (cross-entropy or accuracy) can be misleading:
A model may propose a vector very close to the correct one, but this is just was "wrong" as predicting a vector that is very far away.
A better idea can be gained by measures like top-k accuracy, which count a prediction as correct if the true answer is in the `k` highest outputs.

The overall message here is: Create big models. You most likely _cannot go too big_, as our resource constraints will stop us before this becomes a concern.
Of course, you should still start with small models for prototyping and making everything work.
But after that, few model types scale as well with size as autoregressive Transformer models, so this is the best chance to get a good bang for your buck.  
Remember that you can reduce training batch size to make space for larger models!
You may still be able to run large models (100,000,000+ parameters) on a 2080Ti that way, assuming you are using 8x8 image encodings.

Beyond that, there are a few simple regularization techniques we can make use of to alleviate overfitting at least slightly.
We mainly provide some light data augmentation (horizontal flips), as well as a "attention dropout" where we randomly mask positions in the attention matrix.
This should result in more robust attention patterns.

### Improving Generations
Even with a finished model, we can turn a few knobs to sometimes improve generations. These are:
- Reducing _temperature_: Dividing logits by a number `t < 1` will push them further apart, resulting in stronger peaks for high probabilities, further biasing predictions towards those vectors.
- Top-k sampling: Only sampling among the `k` highest-probability vectors makes it impossible to accidentally sample a low-probability one.

Both of these techniques can be combined, and will each result in improved quality, at the cost of reduced diversity.
For extreme values, outputs will often degrade.
As such, there is usually a sweet spot for the best values, which you can find (for example) using FID evaluation.
Note that in some cases, applying any amount of these techniques can hurt.
We found inconsistent results, from helpful (CIFAR) over barely noticeable (STL) to generally hurtful (Flickr, Fashion).

Note that we also implemented a new metric, [precision & recall](https://research.nvidia.com/sites/default/files/pubs/2019-12_Improved-Precision-and/kynkaanniemi2019metric_paper.pdf).
You can use this as another point of guidance.
Generally, reducing `t` or `k` will improve precision (up to a point), but reduce recall.


## Putting it All Together

All in all, a well-trained autoregressive model with a decent VQ decoder should give you the best results so far.
Remember to evaluate your models using measures like FID.
Note that autoregressive models, as well as the VQ component, could be further improved using conditional generation, as well.
But to keep the code clean, we will only be implementing this in certain cases going forward.
