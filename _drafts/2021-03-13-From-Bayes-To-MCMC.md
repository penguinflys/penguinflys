---
layout: single
title: From Bayes To MCMC - Sampling Techniques
toc: true
toc_sticky: true
categories:
  - Fundamental
tags:
  - Reinforcement Learning
  - SLAM
comments: true
---
<!-- > Markov decision process (MDP) is a <span style="color:red">discrete-time</span> <span style="color:green">stochastic</span> control process, which is very useful in the robotic application especially when uncertainty should be modeled in the process. This post include Monte Carlo method, Markov Chain Monte Carlo(MCMC) and Markov decision process. Then short mathmatical application shall be discussed in later in SLAM section and RL section. What need to mentioned is that Markov methods are not only useful in Reinforcement Learning but also in SLAM. -->

> This post describes probabilistic fundamentals in sampling techniques. It is very useful in taking samples with non-analytical form distribution. It is assumed that you already know Bayesian theory, or at least heard about it.
Bayesian theory is firstly introduced and then followed by the functional description of MCMC without mathematical equations.

## Bayes Inference

In SLAM and state machine:

$$
P(X|Z) = \frac{P(Z|X)P(X)}{P(Z)} = \frac{P(Z|X)P(X)}{\sum_{i}P(Z|x_i)P(x_i)} \\ \Rightarrow \underbrace{P(X|Z)}_{\substack{Posterior}} \propto \underbrace{P(Z|X)}_{\substack{Likelihood}} \underbrace{P(X)}_{\substack{Prior}}
$$

Where $Z$ is the __observation__ or __input feature__, the set of all possible observation which forms the observation space or feature space. If you take a sample from $Z$, this sample shall denoted as $z$. 

$X$ can be seen as __state__ in state machine or the __model__ in machine learning to prediction with hidden internal parameters, which forms the prediction space.

What Bayes Inference describes is a inverse of __Cause and Effect Theorem__. $X$ is the cause which can be abstract and hard to be explained(especially in the context of machine learning), such as being a woman, or a robot pose, and corresponding $Z$ may be the lang hair or environment of the robot.

__Likelihood__ describes the probability(sometimes also called model) that a woman might have lang hair and robot which might be in the specific space(with little uncertainty) observing a environment setting. It can also be interpreted as _because_ ..., ... such as because she is a woman, there is a high probability she has lang hair; because the robot is in a special pose, there is a high probability that it observes the universe in a special specific fixed viewpoint. As the probabilistic theory has a property of being stochastic, you can try not think universe deterministic, this may gives you a easy to understand why probabilistic theory is so important.

__Posterior__ switched the the condition and variable compared to Likelihood, and it describes how probably that a lang hair person is a woman, and how probably a robot is in the special pose, given its observation. Posterior is important because it gives a way to describe analyse model or state with observation, and the observation is relatively easy to be obtained. However the way it works is still not generally accepted.

> Bayesian credible intervals can be quite different from frequentist confidence intervals for two reasons:[wiki](https://en.wikipedia.org/wiki/Credible_interval#:~:text=credible%20intervals%20incorporate%20problem%2Dspecific,parameters%20in%20radically%20different%20ways.)
* credible intervals incorporate problem-specific contextual information from the prior distribution whereas confidence intervals are based only on the data;
* credible intervals and confidence intervals treat nuisance parameters in radically different ways.

__Prior__ is the trouble maker, because a priori distribution is your __subjective judgment__ about the probability distribution of a parameter before you obtain experimental observations. subjective judgment is the reason why Bayesian statistics are not well recognized, and science prefer objective. However it is actually very useful. 

### An Example
You might still feel confused about Bayes Inference, Lets have a simple example to explain Prior and Likelihood:

Suppose you are producing something, and there are 2 procedures to produce posterior, the likelihood and prior distribution are both gaussian distribution(this is made on purpose to form a conjugate prior so that posterior is also a gaussian):

$$
\underbrace{P(X|Z)}_{\substack{Posterior}} \propto \underbrace{P(Z|X)}_{\substack{Likelihood}} \underbrace{P(X)}_{\substack{Prior}} = N(z|\mu, \sigma^2) \cdot N(\mu|\mu_0, \sigma_0^2)
$$

What prior samples is the $\mu$ and $\mu$ is the parameter for sampling $x$, in other words, with different sampled $\mu$, the likelihood can be very different! as the result, Posterior vaires with the prior distribution. This is the key why Prior is the trouble maker, This is why the mathmatikers are trying so hard to find a way to determine prior.

In the sampling of Prior distribution(process of production), the "real" feature/state $x^*$($\mu$) is a sample, which __should__ be subject to Prior distribution $N(\mu \| \mu_0, \sigma_0^2)$.

In the sampling of likelihood distribution(process of measurement), $\hat{z}$ is the measurement and is subject to $N(z\|\mu, \sigma^2)$. In the following figure, $\mu = 102$ and observation has $\sigma = 3$.

![image-center]({{ "/assets/images/MCMC/Joint.PNG" | relative_url }}){: .align-center}

*Fig. Visualization of joint probability distribution(notation difference: $x$ in the figure correspond to measurement), source: Claus Brenner*
{: .text-center}


Now you can easily find that what posterior is for, posterior is more interested in the $X$ with observation $Z$, in the example, it means how does our knowledge about $\mu$ changes after a measurement, corresponding to "horizontal" conditional density. In the following figure, measurement $x = 97$

![image-center]({{ "/assets/images/MCMC/posterior.PNG" | relative_url }}){: .align-center}

*Fig. Visualization of posterior distribution(notation difference: $x$ in the figure correspond to measurement), source: Claus Brenner*
{: .text-center}

In above case, P(\mu|x) = $N(\mu|\mu_{post}, \sigma_{post}^2)$ with $\mu_{post} = 99.0769$ and $\sigma_{post} = 1.66410$
## General Case

Analytical from is not always the available, gaussian in the example is a special case. Generally we are interested in certain quantities such as __expectation of a function__ with variable in the probability distribution or __Maximum a posteriori (MAP) estimation__, However, because of unknown prior density of $p(x)$, it is usually approximated by

$$
E[\hat{f}] \approx \frac{1}{m} \sum_{j=1}^{m}f(x_{j}) \text{   with  }  x_j \sim p(x)
$$

$$
Var[\hat{f}] \approx \frac{1}{m} E[(f - E[f])^2]
$$

## Task Definition and Methods

Thus, the task is sampling from a (complex) distribution, this is exactly why MCMC is included in this post.
We need a random sample generator for a random variable with density p(x)

### Inversion Trick

Suppose you already know [CDF and PDF](http://reliawiki.org/index.php/Basic_Statistical_Background), inversion trick is realized by the inverse function to transfer the known distribution to unknown CDF.

Suppose $U \sim Uniform(0,1)$ and $V:= F^{-1}(U)$, then it follows:

$$
F(v) = P(V \leq x) = P(F^{-1}(U) \leq x) = P(U \leq F(x)) = F(x)
$$

So, to sample $x \sim f(x)$ we can draw $U \sim Uniform(0,1)$ and then compute $x = F^{-1}(U)$

![image-center]({{ "/assets/images/MCMC/inverse.PNG" | relative_url }}){: .align-center}

*Fig. Inverse trick example, source: Claus Brenner*
{: .text-center}

__Limitation__: 
* $x = F^{-1}(U)$ has to be able to be computed, you can try a multiple mode gaussian distribution to find out whether it is okay.
* not suitable for distributions which are not given in a usable form or for which an inverse CDF cannot be computed or cannot be computed efficiently.


### Accept-Reject Method

### Importance Sampling

### Gradient ascent(only MAP)

### MCMC

### Metropolis-Hastings

### Gibbs

### reversible jump MCMC

<!-- <span style="color:blue"> What is Markov Decision Process(MDP)? </span>


<span style="color:blue"> What is the difference between MDP and MC? </span>


<span style="color:blue"> What is Markov chain Monte Carlo?  </span> -->