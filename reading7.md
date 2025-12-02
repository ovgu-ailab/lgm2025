---
layout: default
title: Reading 7
id: reading7
---


# Reading Assignment 7: Autoregressive Models & LLMs

## Autoregressive Models

[Chapter 22 from Probabilistic Machine Learning: Advanced 
Methods](https://probml.github.io/pml-book/book2.html) gives a good overview of
autoregressive models. Since these models are conceptually very straightforward,
there is not too much deep theory here.

(The [Bishop Book](https://www.bishopbook.com/) only briefly covers the topic in Section 12.2.4.)

### Optional Reading: Classic Success Stories

Consider these optional reading in case you want to get some more concrete examples.
Some of these may be treated in more detail later in the class.

- [PixelRNN](https://arxiv.org/pdf/1601.06759.pdf) models images directly on the 
pixel level. As the CNN variant was used a lot more, you can skip the details of
the RNNs. There is also a [follow-up](https://arxiv.org/pdf/1606.05328.pdf), and
[another one](https://arxiv.org/pdf/1701.05517.pdf).
- [Wavenet](https://www.deepmind.com/blog/wavenet-a-generative-model-for-raw-audio)
was a revolutionary model that generated audio in an autoregressive manner. It's
basically PixelCNN in 1D.
- [VQ-VAE](https://arxiv.org/pdf/1711.00937.pdf) encodes data in a lower-dimensional
space and then uses an autoregressive model on that space. There is also 
[VQ-VAE2](https://arxiv.org/pdf/1906.00446.pdf) which uses a multi-level approach to generate very high-quality images.


## Large Language Models

There are _many_ language models being developed by large companies and research
groups. Most of these function similarly. We will look at a select few only.
As before, this is a _lot_ to read, so the optional papers below are mainly included as a reference.

Section 12.3.5 of the [Bishop Book](https://www.bishopbook.com/) provides a high-level overview.

For more details, refer to this HUGE overview paper with [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223).

Finally, a very important piece of research, providing justification for this
research agenda, is [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361.pdf).


### Optional: Model Examples

GPT series of papers:
- [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [GPT 2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [GPT 3](https://arxiv.org/pdf/2005.14165.pdf)
- GPT 4
  - [Paper](https://arxiv.org/pdf/2303.12712.pdf)
  - [Technical Report](https://arxiv.org/pdf/2303.08774.pdf)

As these are all developed by OpenAI, you could also check [PaLM](https://arxiv.org/pdf/2204.02311.pdf)
by Google.


### Reinforcement Learning with Human Feedback

An important technique in fine-tuning LLMs, especially for human interaction.

- [A paper](https://arxiv.org/pdf/1909.08593.pdf)
- [Another one](https://arxiv.org/pdf/2203.02155.pdf)
- [A blog post](https://huggingface.co/blog/rlhf)


