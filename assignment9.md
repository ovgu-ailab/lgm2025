---
layout: default
title: Assignment 9
id: ass9
---


# Assignment 9: High Score!
**Discussion: January 9th**  
**Deadline: January 8th, 20:00**


At long last, we have arrived at the model class that forms most of today's state-of-the-art generative models for structured data like images.
Score-based models offer high flexibility for architectures, simple implementation, and (relatively) fast and stable training.
Their main downside is _very_ slow generation.
The next couple of assignments will look at different variants of these models and slowly build up our toolkit.
This first week serves to set up a good baseline.
Still, with decent networks and enough training time, these models should outperform anything you have implemented so far.

As usual, there is a notebook plus an unfinished implementation where you have to fill in the remaining steps (`lgm.score`).
Training is very simple:
- At each step, sample the noise level `sigma` that we want to train.
    - In theory, we would sum over _all_ `sigma` in our noise schedule, but this is prohibitively slow.
- Create a noisy sample from the current batch of data.
- Put the noisy batch into the model.
- Compute the squared difference between the model prediction and the _target score_ -- this is your loss.


## Noise Schedule

A major question to consider is how many noise levels we need, and how they should be arranged relative to each other.
Sampling will proceed by going through these levels from highest to lowest, so the more levels we have, the longer it takes to sample.
There are some well-functioning guidelines on how to choose this schedule.
Details can be found in the notebook.
**Getting this right is crucial for well-performing models!**


## The Rest

The standard architecture for score-based models is based on [U-Nets](https://arxiv.org/pdf/1505.04597).
The repository has been updated with functionalities for the _encoder-decoder skip connections_ these networks employ.
All parts of the implementation should be relatively short and simple, especially compared to last week's Glow model.
With this, you should be able to build strong generative models on pretty much any of the datasets we offer.
The notebook also contains some pointers to usages besides "raw" generation, such as inpainting or creating "variations" on existing data points.

### What to watch out for
- Take your time tuning the noise schedule and generation process, especially the step size `epsilon`.
This is really important to get right, as even small deviations can result in complete failure to generate proper outputs.
    - Note that training is independent of `epsilon`, so you don't need to retrain the model to try out different values.
- In terms of how many noise levels and generation steps, rather use too many than too few.
- Put aside plenty of time for various generation tasks and evaluation; the sequential generation process is very slow.
- Similarly, score-based models seem to require rather long training. Papers usually report a number of training steps in the hundreds of thousands, even for smaller datasets like CIFAR.
    - Keep in mind, for a dataset with 60,000 samples and batch size 256, there are only slightly above 200 steps per epoch.
    So you would require close to 500 epochs for just 100,000 steps. You could of course reduce the batch size to cover more steps per epoch.
- The latter point holds even if the loss seems to plateau relatively early. Just keep going!

All in all, there are a few more implementation details to fill in this time than usual, to make sure you really engage with the concepts.
Understanding these will be important to be able to follow the concepts of the next couple of weeks.
After getting it to run, just try to achieve a good model for a dataset of your choice and check some of the advanced concepts!
