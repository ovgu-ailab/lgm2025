---
layout: default
title: Assignment 1
id: ass1
---


# Assignment 1: Definitely Not COVID
**Discussion: October 24th**  
**Deadline: October 23rd, 18:00**

**If you haven't done so, please see the [general assignment notes](https://ovgu-ailab.github.io/lgm2025/assignment0.html)
posted with Assignment 0.**

This assignment builds on concepts discussed in the first exercise as well as the reading (Chapter 2 in the Bishop book),
so if you struggle, try doing the reading first!


## Testing Illnesses

Consider a dangerous and/or common illness that people are being tested for to recognize it early (e.g. cancer) 
and/or prevent its spread (e.g. COVID). 
The test is either positive or negative. We make the following assumptions:

- About 1% of the population has the illness. 
That is, any given person has a 1% "a priori" probability of being sick.
- If a sick person is tested, the test returns a positive result 99.9% of the time.
- If a healthy person is tested, the test still returns a false positive result 1% of the time.

You take part in a study where a random, representative sample of the population is tested for the illness. 
Your test result is positive. 
What is the probability that you have the illness?
1. **(Submission)** Solve this via simulation. 
   1. Take a "population sample" of a specific size (experiment with different sizes!) where every "generated person" 
   has a 1% chance of turning out sick.  
   2. Test your "people" -- if they are sick, the test should have a 99.9% chance of returning a positive result; 
   if they are healthy, it should be 1%.
   3. Out of all people that have been _tested sick_, get the proportion of people that _are actually sick_.
2. **(Submission)** Solve this via mathematics. 
This requires a basic grasp of marginal and conditional probabilities as well as Bayes' theorem. 
These are fundamental concepts without which you will be lost in this class! 
[The corresponding wiki article](https://en.wikipedia.org/wiki/Bayes%27_theorem) should be sufficient.

For the simulation, note that you can generate uniform random numbers in the range [0, 1] via functions such as
`np.random.rand`, and such random numbers will be smaller than another fixed number `p` with probability `p`.
For example, the chance of the uniform random number being smaller than `p=0.9` is 90%.
   
Next (mathematical solution is sufficient, no need for more simulation):
1. **(Submission)** Conversely, assume the test result is negative. 
What is the probability that you have the illness anyway?
2. **(Submission)** To bullet-proof the results of their study, the researchers decide to administer _two_ tests to each
participant.
The second test has the following properties:
   - If a sick person is tested, the test returns a positive result 96% of the time.
   - If a healthy person is tested, the test still returns a positive result 2% of the time.  
     
   As we can see, the second test is much more prone to errors than the first.
   However, assume that the results of the second test are _conditionally independent_ of the first. 
   That is, whether the second test makes an error does not depend on whether there is an error on the first test and 
   vice versa, _given whether a person is sick or not_.  
   Now, _both_ of your tests come back positive. 
   Given this information, what is the probability that you are indeed sick?
   

## A More Complex Coin Toss

The purpose of this part is for you to walk through a basic probabilistic modeling task yourself.
This can be considered somewhat more advanced, so if you struggle with this, focus on the first part above.

In the exercise, we looked at how one can model a coin toss using a Bernoulli distribution, and using this model and
given some data, decide whether a coin is fair or not.

Here we extend this to a slightly more complex scenario:
Say I have a coin, and I flip it.
If it lands tails up, I flip again.
If it lands heads up, I stop.
Once it lands, heads up, I record the number of times I saw tails before I got heads.
This is one trial.
Then I repeat this many more times to get more trials.
You can find a numpy array with results uploaded on E-Learning.
Given these results, do you think this is a fair coin?

You can proceed as follows:
- The appropriate probability distribution here is a 
[geometric distribution](https://en.wikipedia.org/wiki/Geometric_distribution).
Given a parameter `p`, this gives the probability of seeing a "success" (heads) after *exactly* `k` failures (tails).
   - **Be careful**, there is also another "version" of this distribution which gives the probability of success after exactly
  `k` *trials* (i.e. `k-1` failures and one success). 
  This is slightly different!
  The data linked above is the number of failures, _without_ the success!
  If you mix this up, you will get slightly different results from the intended one (not a big deal).
- Derive the log probability for one trial and write the sum over trials -- this is the quantity we want to maximize.
- Use basic calculus to get the derivative of this quantity with respect to `p`.
- Now you have two options:
  1. Set the derivative to 0 and solve for `p`.
  2. Implement gradient descent to find the optimal `p` iteratively.

Either way, this should get you the `p` that best explains the data according to the maximum likelihood principle.
Is this close to 0.5?

You might want to review the derivation for the Bernoulli case in 
[our blog post on this topic](https://ovgu-ailab.github.io/blog/methods/2025/09/08/probabilistic-models.html).
