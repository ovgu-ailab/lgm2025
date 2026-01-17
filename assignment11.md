---
layout: default
title: Assignment 11
id: ass11
---


# Assignment 11: The Most Ordinary Model
**Discussion: January 23rd**  
**Deadline: January 22nd, 20:00**


As we approach the end of the class, we will now consider one of the most recent new model types in the deep generative modeling space.
Flow matching promises flexible ODE-based samplers that get away with relatively few steps, while maintaining good sample quality.
We will only consider a basic setup for this assignment, paying special attention to the similarities and differences to previous score-based and diffusion models.


## Flow Matching Implementation

You know the drill by now.
We have an implementation with missing parts in `lgm.flow_matching`, and an associated notebook.
There is nothing to say about the neural network aspect this time;
we can use the exact same architectures as for diffusion models.
Note that we will not consider conditional models this week, as we want to focus on the flow matching procedure itself.
The Trainer class is also simple and familiar:
Turn the data batch into a partially "noisified" version, put that into the model, and compare the output to a target.
In this case, the model will learn the ODE velocity (i.e. the direction and "speed" of movement) directly, instead of a score.

The model is more complex, relatively speaking.
However, we only expect you to implement the basic "Optimal Transport" flows.
The missing parts are marked in the code, as usual.
You just have to implement a few formulas.
While [the original paper](https://arxiv.org/pdf/2210.02747) is very dense and complicated, you should still be able to extract everything you need from there;
the relevant parts are given in the missing code sections.

As an advanced and highly optional task, you could also try implementing other paths besides optimal transport, like ones corresponding to classic diffusion models, and compare the results.


## Evaluation

You can do "the usual things" with Flow Matching models:
Generate samples and evaluate them subjectively and/or with objective measures, or compare quality with different numbers of sampling steps.
Note that the `generate` function allows you to change the latter very easily.

We encourage you to experiment with the sampling procedures again.
Due to the differential equation setup, different discretizations (number of sampling steps) are very easy to implement.
The basic solver is the [Euler method](https://en.wikipedia.org/wiki/Euler_method).
We also provide a second-order sampler via [Heun's method](https://en.wikipedia.org/wiki/Heun%27s_method).
In principle, any black-box ODE solver can be used, such as provided by [scipy](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.integrate.solve_ivp.html).
However, this is very cumbersome in practice, as you have to switch between GPU tensors and CPU numpy arrays each step; this is also very slow.

In general, don't expect this to compete with the diffusion models from last time:
- We are only using unconditional models to keep the focus on the basic Flow Matching procedure.
- If anything, the ODE-based sampler has to be compared to other deterministic samplers (such as DDIM), not the standard stochastic diffusion sampler.


## Latent Variables with ODEs

There is also some code showcasing how you can deterministically encode/decode data using the ODE, and do things like interpolating between images in the latent space.
This works best with more "uniform" data such as faces.


## Mini Extra: Mixed Precision

Chances are you haven't been too happy with the time it takes to evaluate diffusion models -- generating enough samples takes ages due to the sequential sampling process.
This week, we'll get to know one "trick" to (potentially) improve those times a good bit.

The idea of "mixed precision" is to reduce floating point numbers from the usual 32 bits to just 16 bits where appropriate.
This generally includes expensive operations such as convolutions or matrix multiplications.
Torch has this implemented with just a couple lines of code, and this has been integrated into our repository.
The notebook gives further instructions.

Note that these capabilities heavily rely on specific parts of the GPU architecture that have been massively expanded upon in the last few generations.
We have found reductions in training & sampling time of up to 33%, but unfortunately, there is a chance that the impact on a 2080Ti will not be as significant.
