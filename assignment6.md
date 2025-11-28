---
layout: default
title: Assignment 6
id: ass6
---


# Assignment 6: Auto-Quantized Vectorial Variation Encoders
**Discussion: December 5th**  
**Deadline: December 4th, 18:00**


This will be the first of a two-part project where we will combine two distinct model types for a unified generative model.
In particular, we will _not_ be generating anything this week, rather laying the groundwork and familiarizing ourselves with vector-quantized autoencoders, which are very strong for _compressing_ data.


## Building VQVAEs

As usual, we will first need to implement the basic model and training loop.
There is the `lgm.vqgan` module (the name will become clear in the next section) and a corresponding notebook.
Besides the standard encoder and decoder models, we need a _quantizer_ that maps incoming vectors to the closest entries in a _codebook_.
While this can be done in a couple lines of code, these lines are not straightforward and make use of several tricks.
For this reason, the `VectorQuantizer` is already fully finished.
However, it is highly recommend that you closely study the code!
In particular, **read the notes on updating/resetting the codebook!**
Else you might set up your model incorrectly.

There are also **some questions in the code** that you should answer for yourself to test your understanding.
This is not obligatory, but it is recommeded to at least try.
You can put your answers in your submission.
The questions are helpfully asked via `ValueError` so you don't forget about them. :)
You can remove the errors afterwards; you don't have to implement anything there.

With the quantizer given, the VQVAE class is simple to implement.
It functions like a standard autoencoder, except for the quantization step after the encoder.

### Training
Training is straightforward, optimizing the reconstruction and _commitment_ losses.
The latter is intended to have the encoder outputs "stick" to chosen codebook vectors, and also alleviates issues caused by straight-through estimation (STE).
Speaking of which, the main issue with training VQVAEs is that the gradients don't flow through the quantization step.
The most common fix is to simply "skip" this step in the backward pass.
This causes some inaccuracies, but usually works well when combined with other steps to stabilize the process.
The implementation is simple, but not very intuitive, and has already been done as part of the quantizer.

### Architecture
A VQVAE that encodes inputs to a single vector, which is then quantized, would be extremely restrictive.
As such, the network should be fully convolutional.
For example, a 64x64 image may be encoded to an 8x8 "image" of latent vectors, which can then be decoded again.
The final convolution can have a relatively small number of filters, as the codebook will have trouble filling out a high-dimensional space, so the excess dimensions would be wasted.
As luck would have it, [we have a blog post about that](https://ovgu-ailab.github.io/blog/methods/2024/05/28/vqvae-compression.html) if you are interested. :)


## VQGAN

All autoencoders suffer from some degree of bluriness, likely due to the simplistic element-wise reconstruction loss functions.
But this issue gets much worse when we decrease the capacity of the model, such as through vector quantization.
Since we later want to use the decoder as part of a generative model, we should strive for maximum reconstruction quality.

As it turns out, a highly effective approach is to use an additional GAN loss which judges the realism of the reconstructions.
Since blurry outputs are very clearly fake, this forces the decoder to make them sharper and more detailed.
The reconstruction loss still gives guidance on the global shape/coherence of the outputs.
Because of this, we can use a so-called _PatchGAN_, where the discriminator is fully-convolutional and outptus an "image" of predictions at a smaller scale.
These local real/fake predictions should then focus more on smaller patterns like edges or textures.

It is recommended that you start developing the VQVAE without the GAN loss, as this additional component makes training _much_ slower, as well as more memory-intensive.
Still, after you have a working VQVAE, you should integrate the GAN, as well.
Once tuned properly, this will significantly improve output quality.


## Overall Task

You should try to build a model that achieves the best possible output quality under significant compression constraints.
The smaller the code images and the fewer codebook vectors, the easier next week's autoregressive model will be to train.
With a GAN loss, it should be possible to encode 64x64 input images down to an 8x8 latent grid with serviceable image quality.
16x16 will give better outputs, but will make next week's model much slower, and likely produce worse generations.

As for the codebook size, we mainly tested 1024 codebook vectors, but reducing this may also be possible without too much quality loss.
This should yet again make next week's task easier.
For example, a quick test indicated that 512 codebook vectors also works well for Flickr -- perhaps we could go even lower!

Be sure to _save_ your resulting model so that you can use it next week.
Set aside plenty of time for the final training process, as this will take a while when including a GAN.
