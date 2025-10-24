---
layout: default
title: Assignment 2
id: ass2
---


# Assignment 2: Let's Go to Monte Carlo
**Discussion: November 7th**  
**Deadline: November 6th, 18:00**

In this assignment, we will be investigating Monte Carlo Methods with a few simple examples. 
There is a lot of text for explanation, but the actual tasks are rather compact.

**Some starter code can be found on Gitlab! Please see `assignment02_starter.ipynb`**.


## Markov Chain Monte Carlo

We first want to practice running a Markov chain for a simple example.
This is easiest when you have a small number of discrete states (numbered 0 to _n_), and a given _transition matrix_ (we call this _A_) giving the probabilities to move from one state to the next.
The starter notebook already defines such a matrix, but you can make your own if you like.
The one requirement is that each colum sums up to 1, as a column gives the probabilities to move _from_ a given state to others.

Now, run a Markov chain:
- Start off with an arbitrary state _x(0)_.
- Repeatedly take the column from _A_ that corresponds to the current state _x(i)_
 (i.e. if the current state is 3, you take the 3rd column). Sample a new state
 _x(i+1)_ from this probability distribution. Do this many times (like, thousands of times)
and collect all samples in a list or something similar.
  - Sampling can be done via `np.random.choice`.
  You don't need `torch` for this first part at all.
- At the end, plot a histogram of your samples over time. 
This gives you an empirical distribution over the samples. 
Done!
By running your chain you are sampling from the distribution represented by _A_.

To go a little bit deeper, we consider some theoretical analysis in 
[section 17.3 of the deep learning book](https://www.deeplearningbook.org/).
Try the following (you don't have to read the section):
- Create an arbitrary initial distribution _v(0)_, i.e. an _n_-element vector with elements summing to 1.
This represents the _chance_ of being in each state at time step 0, for example if you were running many Markov chains in parallel.
- Multiply the transition matrix with _v(0)_, the result is _v(1)_.
This gives, for each state, the probability of being in that state at time step 1, i.e. after the first transition from step 0.
Can you work out why this is the case?
- Repeatedly multiply the matrix _A_ with the vector to get the distribution for each time step.
The result should converge very quickly (just a handful of steps) to a vector _v'_. 
This vector represents the probability distribution that this Markov chain (represented by matrix _A_) will converge to!
- Try different starting values of _v(0)_. How does this influence _v'_ ?


Does the vector _v'_ match the empirical distribution computed above, when you actually ran a Markov chain? 
If not, you might need to run the chain for more steps -- or you made a mistake somewhere. :)


## Gibbs Sampling & Mixing

In most of our use cases, the situation is not as in the example above: 
We don't have a transition distribution given and just start running a Markov chain on it to get some marginal distribution.
Instead, we have a _desired target distribution_ (e.g. our model distribution) and need to figure out how to get there, i.e. how to sample from it.

Let's try to sample from a mixture of Gaussians via Gibbs sampling.
- Set up a distribution. 
Gibbs sampling makes use of conditional distributions.
In order to be able to do any conditioning, the full distribution needs to be multivariate. 
For simplicity (and easy plotting), stick to a 2D distribution.
Also, we want at least two components;
these can be simple independent Gaussians.
- Start with an arbitrary initial sample (e.g. a vector of 0s). 
Now, repeatedly do the following: 
Sample a new value for _x_ given the value for _y_. 
Then, sample a new value for _y_ given the (new!) value for _x_. 
Since we now have sampled new values for both dimensions, we have essentially taken a new sample of our 2D distribution.
- You can use `torch.distributions` to build the distributions.
You will likely want to use `MixtureSameFamily` of `Normal` distributions.
- The hardest part is figuring out how to take a conditional sample. 
It turns out that in this case, where _p(x,y)_ is a mixture of Gaussians, the conditional distribution _p(x|y)_ (as well
as the other way around) is a mixture of Gaussians as well! 
You can find a derivation [here](https://stats.stackexchange.com/questions/348941/general-conditional-distributions-for-multivariate-gaussian-mixtures).
Being able to translate such formulae into code is important!
However, the derivation linked above is for a fairly general one, and can be simplified considerably in our case.
By using independent Gaussians as components, the only thing that actually changes in the conditional distribution are 
the mixture coefficients!
- You can use `torch.distributions` to sample from the conditional, one-dimensional, mixtures of Gaussians as mentioned 
above. 
You might say: 
Why not just sample from a two-dimensional mixture of Gaussians in the first place? 
The reason we don't do this is that this is a contrived example to show the concept of Gibbs sampling. ;)

Note that many of the steps outlined here have already been implemented in the starter notebook.
You don't need to do much more!

You should collect a reasonable number of samples (1000 or more) and plot both the target distribution (mixture of
Gaussians) and your samples. 
Do the samples reflect the distribution well? 
In particular are both modes of the Gaussian mixture covered equally? 
You can do this visually and/or using statistics. 
Also, experiment with different locations/scales for the Gaussians.
That is, move the components further apart or closer together and repeat the sampling process each time.
The quality of the samples should vary dramatically based on the distance between components!
