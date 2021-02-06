---
layout: post
comments: false
title: "Policy Gradient Algorithms"
date: 2018-04-08 00:15:06
tags: reinforcement-learning review
---

> Abstract: In this post, we are going to look deep into policy gradient, why it works, and many new policy gradient algorithms proposed in recent years: vanilla policy gradient, actor-critic, off-policy actor-critic, A3C, A2C, DPG, DDPG, MADDPG, TRPO, PPO, ACER, and ACTKR.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}


## What is Policy Gradient

Policy gradient is an approach to solve reinforcement learning problems. If you haven’t looked into the field of reinforcement learning, please first read the section ["A (Long) Peek into Reinforcement Learning >> Key Concepts"]({{ site.baseurl }}{% post_url 2018-02-19-a-long-peek-into-reinforcement-learning %}#key-concepts) for the problem definition and key concepts.


### Notations

Here is a list of notations to help you read through equations in the post easily.

{: class="info"}
| Symbol | Meaning |
| ----------------------------- | ------------- |
| $$s \in \mathcal{S}$$ | States. |
| $$a \in \mathcal{A}$$ | Actions. |
| $$r \in \mathcal{R}$$ | Rewards. |
| $$S_t, A_t, R_t$$ | State, action and reward at time step t of one trajectory. I may occasionally use $$s_t, a_t, r_t$$ as well. |
| $$\gamma$$ | Discount factor; penalty to uncertainty of future rewards; $$0<\gamma \leq 1$$. |
| $$G_t$$ | Return; or discounted future reward; $$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$. |
| $$P(s’, r \vert s, a)$$ | Transition probability of getting to the next state s’ from the current state s with action a and reward r. |
| $$\pi(a \vert s)$$ | Stochastic policy (agent behavior strategy); $$\pi_\theta(.)$$ is a policy parameterized by θ. |
| $$\mu(a)$$ | Deterministic policy; we can also label this as $$\pi(s)$$, but using a different letter gives better distinction so that we can easily tell when the policy is stochastic or deterministic without further explanation :) Both π and μ are what a reinforcement learning algorithm targets to learn. |
| $$V(s)$$ | State-value function measures the expected return of state s; $$V_w(.)$$ is a value function parameterized by w.|
| $$V^\pi(s)$$ | The value of state s when we follow a policy π; $$V^\pi (s) = \mathbb{E}_{a\sim \pi} [G_t \vert S_t = s]$$. |
| $$Q(s, a)$$ | Action-value function is similar to $$V(s)$$, but it assesses the expected return of a pair of state and action (s, a); $$Q_w(.)$$ is a action value function parameterized by w. |
| $$Q^\pi(s, a)$$ | Similar to $$V_\pi(.)$$, the value of (state, action) pair when we follow a policy π; $$Q^\pi(s, a) = \mathbb{E}_{a\sim \pi} [G_t \vert S_t = s, A_t = a]$$. |
| $$A(s, a)$$ | Advantage function, $$A(s, a) = Q(s, a) - V(s)$$; it can be considered as another version of Q-value with lower variance by taking the state-value off as the baseline. |



### Policy Gradient

The goal of reinforcement learning is to find an optimal behavior strategy for the agent to obtain optimal rewards. The **policy gradient** methods target at modeling and optimizing the policy directly. The policy is usually modeled with a parameterized function respect to θ, $$\pi_\theta(a \vert s)$$. The value of the reward (objective) function depends on this policy and then various algorithms can be applied to optimize θ for the best reward.

The reward function is defined as:


$$
J(\theta) 
= \sum_{s \in \mathcal{S}} d^\pi(s) V^\pi(s) 
= \sum_{s \in \mathcal{S}} d^\pi(s) \sum_{a \in \mathcal{A}} \pi_\theta(a \vert s) Q^\pi(s, a)
$$

where $$d^\pi(s)$$ is the stationary distribution of Markov chain for $$\pi_\theta$$ (on-policy state distribution under π). 
For simplicity, the θ parameter would be omitted for the policy $$\pi_\theta$$ when the policy is present in the subscript of other functions; for example, $$d^{\pi}$$ and $$Q^\pi$$ should be $$d^{\pi_\theta}$$ and $$Q^{\pi_\theta}$$ if written in full.
Imagine that you can travel along the Markov chain’s states forever, and eventually, as the time progresses, the probability of you ending up with one state becomes unchanged --- this is the stationary probability for $$\pi_\theta$$. $$d^\pi(s) = \lim_{t \to \infty} P(s_t = s \vert s_0, \pi_\theta)$$ is the probability that $$s_t=s$$ when starting from $$s_0$$ and following policy $$\pi_\theta$$ for t steps. Actually, the existence of the stationary distribution of Markov chain is one main reason for why PageRank algorithm works. If you want to read more, check [this](https://jeremykun.com/2015/04/06/markov-chain-monte-carlo-without-all-the-bullshit/).

It is natural to expect policy-based methods are more useful in the continuous space. Because there is an infinite number of actions and (or) states to estimate the values for and hence value-based approaches are way too expensive computationally in the continuous space. For example, in [generalized policy iteration]({{ site.baseurl }}{% post_url 2018-02-19-a-long-peek-into-reinforcement-learning %}#policy-iteration), the policy improvement step $$\arg\max_{a \in \mathcal{A}} Q^\pi(s, a)$$ requires a full scan of the action space, suffering from the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality).

Using *gradient ascent*, we can move θ toward the direction suggested by the gradient $$\nabla_\theta J(\theta)$$ to find the best θ for $$\pi_\theta$$ that produces the highest return. 


### Policy Gradient Theorem

Computing the gradient $$\nabla_\theta J(\theta)$$ is tricky because it depends on both the action selection (directly determined by $$\pi_\theta$$) and the stationary distribution of states following the target selection behavior (indirectly determined by $$\pi_\theta$$). Given that the environment is generally unknown, it is difficult to estimate the effect on the state distribution by a policy update.

Luckily, the **policy gradient theorem** comes to save the world! Woohoo! It provides a nice reformation of the derivative of the objective function to not involve the derivative of the state distribution $$d^\pi(.)$$ and simplify the gradient computation $$\nabla_\theta J(\theta)$$ a lot.


$$
\begin{aligned}
\nabla_\theta J(\theta) 
&= \nabla_\theta \sum_{s \in \mathcal{S}} d^\pi(s) \sum_{a \in \mathcal{A}} Q^\pi(s, a) \pi_\theta(a \vert s) \\
&\propto \sum_{s \in \mathcal{S}} d^\pi(s) \sum_{a \in \mathcal{A}} Q^\pi(s, a) \nabla_\theta \pi_\theta(a \vert s) 
\end{aligned}
$$


### Proof of Policy Gradient Theorem

This session is pretty dense, as it is the time for us to go through the proof ([Sutton & Barto, 2017](http://incompleteideas.net/book/bookdraft2017nov5.pdf); Sec. 13.1) and figure out why the policy gradient theorem is correct.

We first start with the derivative of the state value function:

$$
\begin{aligned}
& \nabla_\theta V^\pi(s) \\
=& \nabla_\theta \Big(\sum_{a \in \mathcal{A}} \pi_\theta(a \vert s)Q^\pi(s, a) \Big) & \\
=& \sum_{a \in \mathcal{A}} \Big( \nabla_\theta \pi_\theta(a \vert s)Q^\pi(s, a) + \pi_\theta(a \vert s) \color{red}{\nabla_\theta Q^\pi(s, a)} \Big) & \scriptstyle{\text{; Derivative product rule.}} \\
=& \sum_{a \in \mathcal{A}} \Big( \nabla_\theta \pi_\theta(a \vert s)Q^\pi(s, a) + \pi_\theta(a \vert s) \color{red}{\nabla_\theta \sum_{s', r} P(s',r \vert s,a)(r + V^\pi(s'))} \Big) & \scriptstyle{\text{; Extend } Q^\pi \text{ with future state value.}} \\
=& \sum_{a \in \mathcal{A}} \Big( \nabla_\theta \pi_\theta(a \vert s)Q^\pi(s, a) + \pi_\theta(a \vert s) \color{red}{\sum_{s', r} P(s',r \vert s,a) \nabla_\theta V^\pi(s')} \Big) & \scriptstyle{P(s',r \vert s,a) \text{ or } r \text{ is not a func of }\theta}\\
=& \sum_{a \in \mathcal{A}} \Big( \nabla_\theta \pi_\theta(a \vert s)Q^\pi(s, a) + \pi_\theta(a \vert s) \color{red}{\sum_{s'} P(s' \vert s,a) \nabla_\theta V^\pi(s')} \Big) & \scriptstyle{\text{; Because }  P(s' \vert s, a) = \sum_r P(s', r \vert s, a)}
\end{aligned}
$$


Now we have:

$$
\color{red}{\nabla_\theta V^\pi(s)} 
= \sum_{a \in \mathcal{A}} \Big( \nabla_\theta \pi_\theta(a \vert s)Q^\pi(s, a) + \pi_\theta(a \vert s) \sum_{s'} P(s' \vert s,a) \color{red}{\nabla_\theta V^\pi(s')} \Big)
$$


This equation has a nice recursive form (see the red parts!) and the future state value function $$V^\pi(s')$$ can be repeated unrolled by following the same equation.

Let’s consider the following visitation sequence and label the probability of transitioning from state s to state x with policy $$\pi_\theta$$ after k step as $$\rho^\pi(s \to x, k)$$.

$$
s \xrightarrow[]{a \sim \pi_\theta(.\vert s)} s' \xrightarrow[]{a \sim \pi_\theta(.\vert s')} s'' \xrightarrow[]{a \sim \pi_\theta(.\vert s'')} \dots
$$

* When k = 0: $$\rho^\pi(s \to s, k=0) = 1$$.
* When k = 1, we scan through all possible actions and sum up the transition probabilities to the target state: $$\rho^\pi(s \to s’, k=1) = \sum_a \pi_\theta(a \vert s) P(s’ \vert s, a)$$.
* Imagine that the goal is to go from state s to x after k+1 steps while following policy $$\pi_\theta$$. We can first travel from s to a middle point s’ (any state can be a middle point, $$s’ \in \mathcal{S}$$) after k steps and then go to the final state x during the last step. In this way, we are able to update the visitation probability recursively: $$\rho^\pi(s \to x, k+1) = \sum_{s'} \rho^\pi(s \to s', k) \rho^\pi(s' \to x, 1)$$.

Then we go back to unroll the recursive representation of $$\nabla_\theta V^\pi(s)$$! Let $$\phi(s) = \sum_{a \in \mathcal{A}} \nabla_\theta \pi_\theta(a \vert s)Q^\pi(s, a)$$ to simplify the maths. If we keep on extending $$\nabla_\theta V^\pi(.)$$ infinitely, it is easy to find out that we can transition from the starting state s to any state after any number of steps in this unrolling process and by summing up all the visitation probabilities, we get $$\nabla_\theta V^\pi(s)$$! 

$$
\begin{aligned}
& \color{red}{\nabla_\theta V^\pi(s)} \\
=& \phi(s) + \sum_a \pi_\theta(a \vert s) \sum_{s'} P(s' \vert s,a) \color{red}{\nabla_\theta V^\pi(s')} \\
=& \phi(s) + \sum_{s'} \sum_a \pi_\theta(a \vert s) P(s' \vert s,a) \color{red}{\nabla_\theta V^\pi(s')} \\
=& \phi(s) + \sum_{s'} \rho^\pi(s \to s', 1) \color{red}{\nabla_\theta V^\pi(s')} \\
=& \phi(s) + \sum_{s'} \rho^\pi(s \to s', 1) \color{red}{\nabla_\theta V^\pi(s')} \\
=& \phi(s) + \sum_{s'} \rho^\pi(s \to s', 1) \color{red}{[ \phi(s') + \sum_{s''} \rho^\pi(s' \to s'', 1) \nabla_\theta V^\pi(s'')]} \\
=& \phi(s) + \sum_{s'} \rho^\pi(s \to s', 1) \phi(s') + \sum_{s''} \rho^\pi(s \to s'', 2)\color{red}{\nabla_\theta V^\pi(s'')} \scriptstyle{\text{ ; Consider }s'\text{ as the middle point for }s \to s''}\\
=& \phi(s) + \sum_{s'} \rho^\pi(s \to s', 1) \phi(s') + \sum_{s''} \rho^\pi(s \to s'', 2)\phi(s'') + \sum_{s'''} \rho^\pi(s \to s''', 3)\color{red}{\nabla_\theta V^\pi(s''')} \\
=& \dots \scriptstyle{\text{; Repeatedly unrolling the part of }\nabla_\theta V^\pi(.)} \\
=& \sum_{x\in\mathcal{S}}\sum_{k=0}^\infty \rho^\pi(s \to x, k) \phi(x)
\end{aligned}
$$


The nice rewriting above allows us to exclude the derivative of Q-value function, $$\nabla_\theta Q^\pi(s, a)$$. By plugging it into the objective function $$J(\theta)$$, we are getting the following:


$$
\begin{aligned}
\nabla_\theta J(\theta)
&= \nabla_\theta V^\pi(s_0) & \scriptstyle{\text{; Starting from a random state } s_0} \\
&= \sum_{s}\color{blue}{\sum_{k=0}^\infty \rho^\pi(s_0 \to s, k)} \phi(s) &\scriptstyle{\text{; Let }\color{blue}{\eta(s) = \sum_{k=0}^\infty \rho^\pi(s_0 \to s, k)}} \\
&= \sum_{s}\eta(s) \phi(s) & \\
&= \Big( {\sum_s \eta(s)} \Big)\sum_{s}\frac{\eta(s)}{\sum_s \eta(s)} \phi(s) & \scriptstyle{\text{; Normalize } \eta(s), s\in\mathcal{S} \text{ to be a probability distribution.}}\\
&\propto \sum_s \frac{\eta(s)}{\sum_s \eta(s)} \phi(s) & \scriptstyle{\sum_s \eta(s)\text{  is a constant}} \\
&= \sum_s d^\pi(s) \sum_a \nabla_\theta \pi_\theta(a \vert s)Q^\pi(s, a) & \scriptstyle{d^\pi(s) = \frac{\eta(s)}{\sum_s \eta(s)}\text{ is stationary distribution.}}
\end{aligned}
$$

In the episodic case, the constant of proportionality ($$\sum_s \eta(s)$$) is the average length of an episode; in the continuing case, it is 1 ([Sutton & Barto, 2017](http://incompleteideas.net/book/bookdraft2017nov5.pdf); Sec. 13.1). The gradient can be further written as:

$$
\begin{aligned}
\nabla_\theta J(\theta) 
&\propto \sum_{s \in \mathcal{S}} d^\pi(s) \sum_{a \in \mathcal{A}} Q^\pi(s, a) \nabla_\theta \pi_\theta(a \vert s)  &\\
&= \sum_{s \in \mathcal{S}} d^\pi(s) \sum_{a \in \mathcal{A}} \pi_\theta(a \vert s) Q^\pi(s, a) \frac{\nabla_\theta \pi_\theta(a \vert s)}{\pi_\theta(a \vert s)} &\\
&= \mathbb{E}_\pi [Q^\pi(s, a) \nabla_\theta \ln \pi_\theta(a \vert s)] & \scriptstyle{\text{; Because } (\ln x)' = 1/x}
\end{aligned}
$$

Where $$\mathbb{E}_\pi$$ refers to $$\mathbb{E}_{s \sim d_\pi, a \sim \pi_\theta}$$ when both state and action distributions follow the policy $$\pi_\theta$$ (on policy).

The policy gradient theorem lays the theoretical foundation for various policy gradient algorithms. This vanilla policy gradient update has no bias but high variance. Many following algorithms were proposed to reduce the variance while keeping the bias unchanged.

$$
\nabla_\theta J(\theta)  = \mathbb{E}_\pi [Q^\pi(s, a) \nabla_\theta \ln \pi_\theta(a \vert s)]
$$

Here is a nice summary of a general form of policy gradient methods borrowed from the [GAE](https://arxiv.org/pdf/1506.02438.pdf) (general advantage estimation) paper (Schulman et al., 2016):

![General form]({{ '/assets/images/general_form_policy_gradient.png' | relative_url }})
{: class="center"}
*Fig. 1. A general form of policy gradient methods. (Image source: [Schulman et al., 2016](https://arxiv.org/pdf/1506.02438.pdf))*



## Policy Gradient Algorithms

Tons of policy gradient algorithms have been proposed during recent years and there is no way for me to exhaust them. I'm introducing some of them that I happened to know and read about.


### REINFORCE

**REINFORCE** (Monte-Carlo policy gradient) relies on an estimated return by [Monte-Carlo]({{ site.baseurl }}{% post_url 2018-02-19-a-long-peek-into-reinforcement-learning %}#monte-carlo-methods) methods using episode samples to update the policy parameter $$\theta$$. REINFORCE works because the expectation of the sample gradient is equal to the actual gradient:


$$
\begin{aligned}
\nabla_\theta J(\theta)
&= \mathbb{E}_\pi [Q^\pi(s, a) \nabla_\theta \ln \pi_\theta(a \vert s)] & \\
&= \mathbb{E}_\pi [G_t \nabla_\theta \ln \pi_\theta(A_t \vert S_t)] & \scriptstyle{\text{; Because } Q^\pi(S_t, A_t) = \mathbb{E}_\pi[G_t \vert S_t, A_t]}
\end{aligned}
$$

Therefore we are able to measure $$G_t$$ from real sample trajectories and use that to update our policy gradient. It relies on a full trajectory and that’s why it is a Monte-Carlo method. 

The process is pretty straightforward:
1. Initialize the policy parameter θ at random.
2. Generate one trajectory on policy $$\pi_\theta$$: $$S_1, A_1, R_2, S_2, A_2, \dots, S_T$$.
3. For t=1, 2, ... , T:
    1. Estimate the the return $$G_t$$;
    2. Update policy parameters: $$\theta \leftarrow \theta + \alpha \gamma^t G_t \nabla_\theta \ln \pi_\theta(A_t \vert S_t)$$

A widely used variation of REINFORCE is to subtract a baseline value from the return $$G_t$$ to *reduce the variance of gradient estimation while keeping the bias unchanged* (Remember we always want to do this when possible). For example, a common baseline is to subtract state-value from action-value, and if applied, we would use advantage $$A(s, a) = Q(s, a) - V(s)$$ in the gradient ascent update.



### Actor-Critic

Two main components in policy gradient are the policy model and the value function. It makes a lot of sense to learn the value function in addition to the policy, since the value function is typically unknown, and that is exactly what the **Actor-Critic** method does. 

Actor-critic is consist of two models:
* **Critic** updates the value function parameters w and depending on the algorithm it could be action-value $$Q_w(a \vert s)$$ or state-value $$V_w(s)$$.
* **Actor** updates the policy parameters θ for $$\pi_\theta(a \vert s)$$, in the direction suggested by the critic.

Let’s see how it works in a simple action-value actor-critic algorithm.
1. Initialize s, θ, w at random; sample $$a \sim \pi_\theta(a \vert s)$$.
2. For $$t = 1 \dots T$$:
    1. Sample reward $$r_t \sim R(s, a)$$ and next state $$s’ \sim P(s’ \vert s, a)$$;
    2. Then sample the next action $$a’ \sim \pi_\theta(a’ \vert s’)$$;
    3. Update the policy parameters: $$\theta \leftarrow \theta + \alpha_\theta Q_w(s, a) \nabla_\theta \ln \pi_\theta(a \vert s)$$;
    4. Compute the correction (TD error) for action-value at time t: <br/>$$\delta_t = r_t + \gamma Q_w(s’, a’) - Q_w(s, a)$$ <br/>and use it to update the parameters of action-value function:<br/> $$w \leftarrow w + \alpha_w \delta_t \nabla_w Q_w(s, a)$$
    5. Update $$a \leftarrow a’$$ and $$s \leftarrow s’$$.

Two learning rates, $$\alpha_\theta$$ and $$\alpha_w$$, are predefined for policy and value function parameter updates respectively.


### Off-Policy Policy Gradient

Both REINFORCE and the vanilla version of actor-critic method are on-policy: training samples are collected according to the target policy --- the very same policy that we try to optimize for. It lack of several advantages that we could harvest if switch to *off-policy*:
1. The off-policy approach does not require full trajectories and can reuse any past episodes ([“experience replay”]({{ site.baseurl }}{% post_url 2018-02-19-a-long-peek-into-reinforcement-learning %}#deep-q-network)) for much better sample efficiency.
2. The sample collection follows a behavior policy different from the target policy, bringing better [exploration]({{ site.baseurl }}{% post_url 2018-02-19-a-long-peek-into-reinforcement-learning %}#exploration-exploitation-dilemma).

Now let’s see how off-policy policy gradient is computed. The behavior policy for collecting samples is a known policy (predefined just like a hyperparameter), labelled as $$\beta(a \vert s)$$. The objective function sums up the reward over the state distribution defined by this behavior policy:


$$
J(\theta)
= \sum_{s \in \mathcal{S}} d^\beta(s) \sum_{a \in \mathcal{A}} Q^\pi(s, a) \pi_\theta(a \vert s)
= \mathbb{E}_{s \sim d^\beta} \big[ \sum_{a \in \mathcal{A}} Q^\pi(s, a) \pi_\theta(a \vert s) \big]
$$

where $$d^\beta(s)$$ is the stationary distribution of the behavior policy β; recall that $$d^\beta(s) = \lim_{t \to \infty} P(S_t = s \vert S_0, \beta)$$; and $$Q^\pi$$ is the action-value function estimated with regard to the target policy π (not the behavior policy!).

Given that the training observations are sampled by $$a \sim \beta(a \vert s)$$, we can rewrite the gradient as:


$$
\begin{aligned}
\nabla_\theta J(\theta)
&= \nabla_\theta \mathbb{E}_{s \sim d^\beta} \Big[ \sum_{a \in \mathcal{A}} Q^\pi(s, a) \pi_\theta(a \vert s)  \Big] & \\ 
&= \mathbb{E}_{s \sim d^\beta} \Big[ \sum_{a \in \mathcal{A}} \big( Q^\pi(s, a) \nabla_\theta \pi_\theta(a \vert s) + \color{red}{\pi_\theta(a \vert s) \nabla_\theta Q^\pi(s, a)} \big) \Big] & \scriptstyle{\text{; Derivative product rule.}}\\
&\stackrel{(i)}{\approx} \mathbb{E}_{s \sim d^\beta} \Big[ \sum_{a \in \mathcal{A}} Q^\pi(s, a) \nabla_\theta \pi_\theta(a \vert s) \Big] & \scriptstyle{\text{; Ignore the red part: } \color{red}{\pi_\theta(a \vert s) \nabla_\theta Q^\pi(s, a)}}. \\
&= \mathbb{E}_{s \sim d^\beta} \Big[ \sum_{a \in \mathcal{A}} \beta(a \vert s) \frac{\pi_\theta(a \vert s)}{\beta(a \vert s)} Q^\pi(s, a) \frac{\nabla_\theta \pi_\theta(a \vert s)}{\pi_\theta(a \vert s)} \Big] & \\
&= \mathbb{E}_\beta \Big[\frac{\color{blue}{\pi_\theta(a \vert s)}}{\color{blue}{\beta(a \vert s)}} Q^\pi(s, a) \nabla_\theta \ln \pi_\theta(a \vert s) \Big] & \scriptstyle{\text{; The blue part is the importance weight.}}
\end{aligned}
$$

where $$\frac{\pi_\theta(a \vert s)}{\beta(a \vert s)}$$ is the [importance weight](http://timvieira.github.io/blog/post/2014/12/21/importance-sampling/). Because $$Q^\pi$$ is a function of the target policy and thus a function of policy parameter θ,  we should take the derivative of $$\nabla_\theta Q^\pi(s, a)$$ as well according to the product rule. However, it is super hard to compute $$\nabla_\theta Q^\pi(s, a)$$ in reality. Fortunately if we use an approximated gradient with the gradient of Q ignored, we still guarantee the policy improvement and eventually achieve the true local minimum. This is justified in the proof [here](https://arxiv.org/pdf/1205.4839.pdf) (Degris, White & Sutton, 2012).

In summary, when applying policy gradient in the off-policy setting, we can simple adjust it with a weighted sum and the weight is the ratio of the target policy to the behavior policy, $$\frac{\pi_\theta(a \vert s)}{\beta(a \vert s)}$$.


### A3C

[[paper](https://arxiv.org/abs/1602.01783)\|code]

**Asynchronous Advantage Actor-Critic** ([Mnih et al., 2016](https://arxiv.org/abs/1602.01783)), short for **A3C**, is a classic policy gradient method with a special focus on parallel training. 

In A3C, the critics learn the value function while multiple actors are trained in parallel and get synced with global parameters from time to time. Hence, A3C is designed to work well for parallel training.

Let’s use the state-value function as an example. The loss function for state value is to minimize the mean squared error, $$J_v(w) = (G_t - V_w(s))^2$$ and gradient descent can be applied to find the optimal w. This state-value function is used as the baseline in the policy gradient update.

Here is the algorithm outline:
1. We have global parameters, θ and w; similar thread-specific parameters, θ’ and w’.
2. Initialize the time step $$t = 1$$
3. While $$T <= T_\text{MAX}$$:
    1. Reset gradient: dθ = 0 and dw = 0.
    2. Synchronize thread-specific parameters with global ones: θ’ = θ and w’ = w.
    3. $$t_\text{start}$$ = t and sample a starting state $$s_t$$.
    4. While ($$s_t$$ != TERMINAL) and $$t - t_\text{start} <= t_\text{max}$$:
        1. Pick the action $$A_t \sim \pi_{\theta’}(A_t \vert S_t)$$ and receive a new reward $$R_t$$ and a new state $$s_{t+1}$$.
        2. Update t = t + 1 and T = T + 1
    5. Initialize the variable that holds the return estimation $$R = \begin{cases} 
	0 & \text{if } s_t \text{ is TERMINAL} \\
	V_{w’}(s_t) & \text{otherwise}
	\end{cases}
	$$
    6. For $$i = t-1, \dots, t_\text{start}$$:
        1. $$R \leftarrow \gamma R + R_i$$; here R is a MC measure of $$G_i$$.
        2. Accumulate gradients w.r.t. θ’: $$d\theta \leftarrow d\theta + \nabla_{\theta’} \log \pi_{\theta’}(a_i \vert s_i)(R - V_{w'}(s_i))$$;<br/>Accumulate gradients w.r.t. w’: $$dw \leftarrow dw + 2 (R - V_{w’}(s_i)) \nabla_{w’} (R - V_{w’}(s_i))$$.
    7. Update synchronously θ using dθ, and w using dw.

A3C enables the parallelism in multiple agent training. The gradient accumulation step (6.2) can be considered as a parallelized reformation of minibatch-based stochastic gradient update: the values of w or θ get corrected by a little bit in the direction of each training thread independently.


### A2C

[paper?\|[code](https://github.com/openai/baselines/blob/master/baselines/a2c/a2c.py)]

**A2C** is a synchronous, deterministic version of A3C; that’s why it is named as “A2C” with the first “A” (“asynchronous”) removed. In A3C each agent talks to the global parameters independently, so it is possible sometimes the thread-specific agents would be playing with policies of different versions and therefore the aggregated update would not be optimal. To resolve the inconsistency, a coordinator in A2C waits for all the parallel actors to finish their work before updating the global parameters and then in the next iteration parallel actors starts from the same policy. The synchronized gradient update keeps the training more cohesive and potentially to make convergence faster. 

A2C has been [shown](https://blog.openai.com/baselines-acktr-a2c/) to be able to utilize GPUs more efficiently and work better with large batch sizes while achieving same or better performance than A3C.

![A2C]({{ '/assets/images/A3C_vs_A2C.png' | relative_url }})
{: class="center"}
*Fig. 2. The architecture of A3C versus A2C.*


### DPG

[[paper](https://hal.inria.fr/file/index/docid/938992/filename/dpg-icml2014.pdf)\|code]

In methods described above, the policy function $$\pi(. \vert s)$$ is always modeled as a probability distribution over actions $$\mathcal{A}$$ given the current state and thus it is *stochastic*. **Deterministic policy gradient (DPG)** instead models the policy as a deterministic decision: $$a = \mu(s)$$. It may look bizarre --- how can you calculate the gradient of the policy function when it outputs a single action? Let’s look into it step by step.

Refresh on a few notations to facilitate the discussion:
* $$\rho_0(s)$$: The initial distribution over states
* $$\rho^\mu(s \to s', k)$$: Starting from state s, the visitation probability density at state s’ after moving k steps by policy μ.
* $$\rho^\mu(s')$$: Discounted state distribution, defined as $$\rho^\mu(s') = \int_\mathcal{S} \sum_{k=1}^\infty \gamma^{k-1} \rho_0(s) \rho^\mu(s \to s', k) ds$$.


The objective function to optimize for is listed as follows:

$$
J(\theta) = \int_\mathcal{S} \rho^\mu(s) Q(s, \mu_\theta(s)) ds
$$

**Deterministic policy gradient theorem**: Now it is the time to compute the gradient! According to the chain rule, we first take the gradient of Q w.r.t. the action a and then take the gradient of the deterministic policy function μ w.r.t. θ:


$$
\begin{aligned}
\nabla_\theta J(\theta) 
&= \int_\mathcal{S} \rho^\mu(s) \nabla_a Q^\mu(s, a) \nabla_\theta \mu_\theta(s) \rvert_{a=\mu_\theta(s)} ds \\
&= \mathbb{E}_{s \sim \rho^\mu} [\nabla_a Q^\mu(s, a) \nabla_\theta \mu_\theta(s) \rvert_{a=\mu_\theta(s)}]
\end{aligned}
$$

We can consider the deterministic policy as a *special case* of the stochastic one, when the probability distribution contains only one extreme non-zero value over one action. Actually, in the DPG [paper](https://hal.inria.fr/file/index/docid/938992/filename/dpg-icml2014.pdf), the authors have shown that if the stochastic policy $$\pi_{\mu_\theta, \sigma}$$ is re-parameterized by a deterministic policy $$\mu_\theta$$ and a variation variable $$\sigma$$, the stochastic policy is eventually equivalent to the deterministic case when $$\sigma=0$$. Compared to the deterministic policy, we expect the stochastic policy to require more samples as it integrates the data over the whole state and action space.

The deterministic policy gradient theorem can be plugged into common policy gradient frameworks. 

Let’s consider an example of on-policy actor-critic algorithm to showcase the procedure. In each iteration of on-policy actor-critic, two actions are taken deterministically $$a = \mu_\theta(s)$$ and the [SARSA]({{ site.baseurl }}{% post_url 2018-02-19-a-long-peek-into-reinforcement-learning%}#sarsa-on-policy-td-control) update on policy parameters relies on the new gradient that we just computed above:


$$
\begin{aligned}
\delta_t &= R_t + \gamma Q_w(s_{t+1}, a_{t+1}) - Q_w(s_t, a_t) & \scriptstyle{\text{; TD error in SARSA}}\\
w_{t+1} &= w_t + \alpha_w \delta_t \nabla_w Q_w(s_t, a_t) & \\
\theta_{t+1} &= \theta_t + \alpha_\theta \color{red}{\nabla_a Q_w(s_t, a_t) \nabla_\theta \mu_\theta(s) \rvert_{a=\mu_\theta(s)}} & \scriptstyle{\text{; Deterministic policy gradient theorem}}
\end{aligned}
$$

However, unless there is sufficient noise in the environment, it is very hard to guarantee enough [exploration]({{ site.baseurl }}{% post_url 2018-02-19-a-long-peek-into-reinforcement-learning%}#exploration-exploitation-dilemma) due to the determinacy of the policy. We can either add noise into the policy (ironically this makes it nondeterministic!) or learn it off-policy-ly by following a different stochastic behavior policy to collect samples. 

Say, in the off-policy approach, the training trajectories are generated by a stochastic policy $$\beta(a \vert s)$$ and thus the state distribution follows the corresponding discounted state density $$\rho^\beta$$:


$$
\begin{aligned}
J_\beta(\theta) &= \int_\mathcal{S} \rho^\beta Q^\mu(s, \mu_\theta(s)) ds \\
\nabla_\theta J_\beta(\theta) &= \mathbb{E}_{s \sim \rho^\beta} [\nabla_a Q^\mu(s, a) \nabla_\theta \mu_\theta(s)  \rvert_{a=\mu_\theta(s)} ]
\end{aligned}
$$

Note that because the policy is deterministic, we only need $$Q^\mu(s, \mu_\theta(s))$$ rather than $$\sum_a \pi(a \vert s) Q^\pi(s, a)$$ as the estimated reward of a given state s.
In the off-policy approach with a stochastic policy, importance sampling is often used to correct the mismatch between behavior and target policies, as what we have described [above](#off-policy-policy-gradient). However, because the deterministic policy gradient removes the integral over actions, we can avoid importance sampling.


### DDPG

[[paper](https://arxiv.org/pdf/1509.02971.pdf)\|[code](https://github.com/openai/baselines/tree/master/baselines/ddpg)]

**DDPG** ([Lillicrap, et al., 2015](https://arxiv.org/pdf/1509.02971.pdf)), short for **Deep Deterministic Policy Gradient**, is a model-free off-policy actor-critic algorithm, combining DPG with [DQN]({{ site.baseurl }}{% post_url 2018-02-19-a-long-peek-into-reinforcement-learning%}#deep-q-network). Recall that DQN (Deep Q-Network) stabilizes the learning of Q-function by experience replay and the frozen target network. The original DQN works in discrete space, and DDPG extends it to continuous space with the actor-critic framework while learning a deterministic policy.

In order to do better exploration, an exploration policy μ' is constructed by adding noise $$\mathcal{N}$$:

$$
\mu'(s) = \mu_\theta(s) + \mathcal{N}
$$

In addition, DDPG does soft updates ("conservative policy iteration") on the parameters of both actor and critic, with $$\tau \ll 1$$: $$\theta’ \leftarrow \tau \theta + (1 - \tau) \theta’$$. In this way, the target network values are constrained to change slowly, different from the design in DQN that the target network stays frozen for some period of time.

One detail in the paper that is particularly useful in robotics is on how to normalize the different physical units of low dimensional features. For example, a model is designed to learn a policy with the robot’s positions and velocities as input; these physical statistics are different by nature and even statistics of the same type may vary a lot across multiple robots. [Batch normalization](http://proceedings.mlr.press/v37/ioffe15.pdf) is applied to fix it by normalizing every dimension across samples in one minibatch.

![DDPG]({{ '/assets/images/DDPG_algo.png' | relative_url }})
{: class="center"}
*Fig 3. DDPG Algorithm. (Image source: [Lillicrap, et al., 2015](https://arxiv.org/pdf/1509.02971.pdf))*



### MADDPG

[[paper](https://arxiv.org/pdf/1706.02275.pdf)\|[code](https://github.com/openai/maddpg)]

**Multi-agent DDPG** (**MADDPG**) ([Lowe et al., 2017](https://arxiv.org/pdf/1706.02275.pdf))extends DDPG to an environment where multiple agents are coordinating to complete tasks with only local information. In the viewpoint of one agent, the environment is non-stationary as policies of other agents are quickly upgraded and remain unknown. MADDPG is an actor-critic model redesigned particularly for handling such a changing environment and interactions between agents.

The problem can be formalized in the multi-agent version of MDP, also known as *Markov games*. Say, there are N agents in total with a set of states $$\mathcal{S}$$. Each agent owns a set of possible action, $$\mathcal{A}_1, \dots, \mathcal{A}_N$$, and a set of observation, $$\mathcal{O}_1, \dots, \mathcal{O}_N$$. The state transition function involves all states, action and observation spaces  $$\mathcal{T}: \mathcal{S} \times \mathcal{A}_1 \times \dots \mathcal{A}_N \mapsto \mathcal{S}$$. Each agent’s stochastic policy only involves its own state and action: $$\pi_{\theta_i}: \mathcal{O}_i \times \mathcal{A}_i \mapsto [0, 1]$$, a probability distribution over actions given its own observation, or a deterministic policy: $$\mu_{\theta_i}: \mathcal{O}_i \mapsto \mathcal{A}_i$$.



Let $$\vec{o} = {o_1, \dots, o_N}$$, $$\vec{\mu} = {\mu_1, \dots, \mu_N}$$ and the policies are parameterized by $$\vec{\theta} = {\theta_1, \dots, \theta_N}$$.

The critic in MADDPG learns a centralized action-value function $$Q^\vec{\mu}_i(\vec{o}, a_1, \dots, a_N)$$ for the i-th agent, where $$a_1 \in \mathcal{A}_1, \dots, a_N \in \mathcal{A}_N$$ are actions of all agents. Each $$Q^\vec{\mu}_i$$ is learned separately for $$i=1, \dots, N$$ and therefore multiple agents can have arbitrary reward structures, including conflicting rewards in a competitive setting. Meanwhile, multiple actors, one for each agent, are exploring and upgrading the policy parameters $$\theta_i$$ on their own. 


**Actor update**:

$$
\nabla_{\theta_i} J(\theta_i) = \mathbb{E}_{\vec{o}, a \sim \mathcal{D}} [\nabla_{a_i} Q^{\vec{\mu}}_i (\vec{o}, a_1, \dots, a_N) \nabla_{\theta_i} \mu_{\theta_i}(o_i) \rvert_{a_i=\mu_{\theta_i}(o_i)} ]
$$

Where $$\mathcal{D}$$ is the memory buffer for experience replay, containing multiple episode samples $$(\vec{o}, a_1, \dots, a_N, r_1, \dots, r_N, \vec{o}')$$ --- given current observation $$\vec{o}$$, agents take action $$a_1, \dots, a_N$$ and get rewards $$r_1, \dots, r_N$$, leading to the new observation $$\vec{o}'$$.


**Critic update**:

$$
\begin{aligned}
\mathcal{L}(\theta_i) &= \mathbb{E}_{\vec{o}, a_1, \dots, a_N, r_1, \dots, r_N, \vec{o}'}[ (Q^{\vec{\mu}}_i(\vec{o}, a_1, \dots, a_N) - y)^2 ] & \\
\text{where } y &= r_i + \gamma Q^{\vec{\mu}'}_i (\vec{o}', a'_1, \dots, a'_N) \rvert_{a'_j = \mu'_{\theta_j}} & \scriptstyle{\text{; TD target!}}
\end{aligned}
$$

where $$\vec{\mu}’$$ are the target policies with delayed softly-updated parameters.

If the policies $$\vec{\mu}$$ are unknown during the critic update, we can ask each agent to learn and evolve its own approximation of others’ policies. Using the approximated policies, MADDPG still can learn efficiently although the inferred policies might not be accurate.

To mitigate the high variance triggered by the interaction between competing or collaborating agents in the environment, MADDPG proposed one more element - *policy ensembles*: 
1. Train K policies for one agent;
2. Pick a random policy for episode rollouts;
3. Take an ensemble of these K policies to do gradient update.

In summary, MADDPG added three additional ingredients on top of DDPG to make it adapt to the multi-agent environment:
* Centralized critic + decentralized actors;
* Actors are able to use estimated policies of other agents for learning;
* Policy ensembling is good for reducing variance.


![DDPG]({{ '/assets/images/MADDPG.png' | relative_url }})
{: class="center" style="width: 70%;"}
*Fig. 4. The architecture design of MADDPG. (Image source: [Lowe et al., 2017](https://arxiv.org/pdf/1706.02275.pdf))**


### TRPO

[[paper](https://arxiv.org/pdf/1502.05477.pdf)\|[code](https://github.com/openai/baselines/tree/master/baselines/trpo_mpi)]

To improve training stability, we should avoid parameter updates that change the policy too much at one step. **Trust region policy optimization (TRPO)** ([Schulman, et al., 2015](https://arxiv.org/pdf/1502.05477.pdf)) carries out this idea by enforcing a [KL divergence]({{ site.baseurl }}{% post_url 2017-08-20-from-GAN-to-WGAN %}#kullbackleibler-and-jensenshannon-divergence) constraint on the size of policy update at each iteration. 

If off policy, the objective function measures the total advantage over the state visitation distribution and actions, while the rollout is following a different behavior policy $$\beta(a \vert s)$$:


$$
\begin{aligned}
J(\theta)
&= \sum_{s \in \mathcal{S}} \rho^{\pi_{\theta_\text{old}}} \sum_{a \in \mathcal{A}} \big( \pi_\theta(a \vert s) \hat{A}_{\theta_\text{old}}(s, a) \big) & \\
&= \sum_{s \in \mathcal{S}} \rho^{\pi_{\theta_\text{old}}} \sum_{a \in \mathcal{A}} \big( \beta(a \vert s) \frac{\pi_\theta(a \vert s)}{\beta(a \vert s)} \hat{A}_{\theta_\text{old}}(s, a) \big) & \scriptstyle{\text{; Importance sampling}} \\
&= \mathbb{E}_{s \sim \rho^{\pi_{\theta_\text{old}}}, a \sim \beta} \big[ \frac{\pi_\theta(a \vert s)}{\beta(a \vert s)} \hat{A}_{\theta_\text{old}}(s, a) \big] &
\end{aligned}
$$

where $$\theta_\text{old}$$ is the policy parameters before the update and thus known to us; $$\rho^{\pi_{\theta_\text{old}}}$$ is defined in the same way as [above](#dpg); $$\beta(a \vert s)$$ is the behavior policy for collecting trajectories. Noted that we use an estimated advantage $$\hat{A}(.)$$ rather than the true advantage function $$A(.)$$ because the true rewards are usually unknown.

If on policy,  the behavior policy is $$\pi_{\theta_\text{old}}(a \vert s)$$:

$$
J(\theta) = \mathbb{E}_{s \sim \rho^{\pi_{\theta_\text{old}}}, a \sim \pi_{\theta_\text{old}}} \big[ \frac{\pi_\theta(a \vert s)}{\pi_{\theta_\text{old}}(a \vert s)} \hat{A}_{\theta_\text{old}}(s, a) \big]
$$

TRPO aims to maximize the objective function $$J(\theta)$$ subject to, *trust region constraint* which enforces the distance between old and new policies measured by [KL-divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) to be small enough, within a parameter δ:


$$
\mathbb{E}_{s \sim \rho^{\pi_{\theta_\text{old}}}} [D_\text{KL}(\pi_{\theta_\text{old}}(.\vert s) \| \pi_\theta(.\vert s)] \leq \delta
$$

In this way, the old and new policies would not diverge too much when this hard constraint is met. While still, TRPO can guarantee a monotonic improvement over policy iteration (Neat, right?). Please read the proof in the [paper](https://arxiv.org/pdf/1502.05477.pdf) if interested :)


### PPO

[[paper](https://arxiv.org/pdf/1707.06347.pdf)\|[code](https://github.com/openai/baselines/tree/master/baselines/ppo1)]

Given that TRPO is relatively complicated and we still want to implement a similar constraint, **proximal policy optimization (PPO)** simplifies it by using a clipped surrogate objective while retaining similar performance. 

First, let's denote the probability ratio between old and new policies as:

$$
r(\theta) = \frac{\pi_\theta(a \vert s)}{\pi_{\theta_\text{old}}(a \vert s)}
$$

Then, the objective function of TRPO (on policy) becomes:

$$
J^\text{TRPO} (\theta) = \mathbb{E} [ r(\theta) \hat{A}_{\theta_\text{old}}(s, a) ]
$$

Without a limitation on the distance between $$\theta_\text{old}$$ and $$\theta$$, to maximize $$J^\text{TRPO} (\theta)$$ would lead to instability with extremely large parameter updates and big policy ratios. PPO imposes the constraint by forcing r(θ) to stay within a small interval around 1, precisely [1-ε, 1+ε], where ε is a hyperparameter.


$$
J^\text{CLIP} (\theta) = \mathbb{E} [ \min( r(\theta) \hat{A}_{\theta_\text{old}}(s, a), \text{clip}(r(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_{\theta_\text{old}}(s, a))]
$$

The function $$\text{clip}(r(\theta), 1 - \epsilon, 1 + \epsilon)$$ clips the ratio within [1-ε, 1+ε]. The objective function of PPO takes the minimum one between the original value and the clipped version and therefore we lose the motivation for increasing the policy update to extremes for better rewards.

When applying PPO on the network architecture with shared parameters for both policy (actor) and value (critic) functions, in addition to the clipped reward, the objective function is augmented with an error term on the value estimation (formula in red) and an entropy term (formula in blue) to encourage sufficient exploration. 


$$
J^\text{CLIP'} (\theta) = \mathbb{E} [ J^\text{CLIP} (\theta) - \color{red}{c_1 (V_\theta(s) - V_\text{target})^2} + \color{blue}{c_2 H(s, \pi_\theta(.))} ]
$$

where Both $$c_1$$ and $$c_2$$ are two hyperparameter constants.

PPO has been tested on a set of benchmark tasks and proved to produce awesome results with much greater simplicity.


### ACER

[[paper](https://arxiv.org/pdf/1611.01224.pdf)\|[code](https://github.com/openai/baselines/tree/master/baselines/acer)]

**ACER**, short for **actor-critic with experience replay** ([Wang, et al., 2017](https://arxiv.org/pdf/1611.01224.pdf)), is an off-policy actor-critic model with experience replay, greatly increasing the sample efficiency and decreasing the data correlation. A3C builds up the foundation for ACER, but it is on policy; ACER is A3C’s off-policy counterpart. The major obstacle to making A3C off policy is how to control the stability of the off-policy estimator. ACER proposes three designs to overcome it:
* Use Retrace Q-value estimation;
* Truncate the importance weights with bias correction;
* Apply efficient TRPO.


**Retrace Q-value Estimation**

[*Retrace*](http://papers.nips.cc/paper/6538-safe-and-efficient-off-policy-reinforcement-learning.pdf) is an off-policy return-based Q-value estimation algorithm with a nice guarantee for convergence for any target and behavior policy pair (π, β), plus good data efficiency.

Recall how TD learning works for prediction:
1. Compute TD error: $$\delta_t = R_t + \gamma \mathbb{E}_{a \sim \pi} Q(S_{t+1}, a) - Q(S_t, A_t)$$; the term $$r_t + \gamma \mathbb{E}_{a \sim \pi} Q(s_{t+1}, a) $$ is known as “TD target”. The expectation $$\mathbb{E}_{a \sim \pi}$$ is used because for the future step the best estimation we can make is what the return would be if we follow the current policy π. 
2. Update the value by correcting the error to move toward the goal: $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \delta_t$$. In other words, the incremental update on Q is proportional to the TD error: $$\Delta Q(S_t, A_t) = \alpha \delta_t$$.

When the rollout is off policy, we need to apply importance sampling on the Q update:

$$
\Delta Q^\text{imp}(S_t, A_t) 
= \gamma^t \prod_{1 \leq \tau \leq t} \frac{\pi(A_\tau \vert S_\tau)}{\beta(A_\tau \vert S_\tau)} \delta_t
$$

The product of importance weights looks pretty scary when we start imagining how it can cause super high variance and even explode. Retrace Q-value estimation method modifies $$\Delta Q$$ to have importance weights truncated by no more than a constant c:

$$
\Delta Q^\text{ret}(S_t, A_t) 
= \gamma^t \prod_{1 \leq \tau \leq t} \min(c, \frac{\pi(A_\tau \vert S_\tau)}{\beta(A_\tau \vert S_\tau)})  \delta_t
$$

ACER uses $$Q^\text{ret}$$ as the target to train the critic by minimizing the L2 error term: $$(Q^\text{ret}(s, a) - Q(s, a))^2$$.


**Importance weights truncation**

To reduce the high variance of the policy gradient $$\hat{g}$$, ACER truncates the importance weights by a constant c, plus a correction term. The label $$\hat{g}_t^\text{acer}$$ is the ACER policy gradient at time t.


$$
\begin{aligned}
\hat{g}_t^\text{acer}
= & \omega_t \big( Q^\text{ret}(S_t, A_t) - V_{\theta_v}(S_t) \big) \nabla_\theta \ln \pi_\theta(A_t \vert S_t) 
  & \scriptstyle{\text{; Let }\omega_t=\frac{\pi(A_t \vert S_t)}{\beta(A_t \vert S_t)}} \\
= & \color{blue}{\min(c, \omega_t) \big( Q^\text{ret}(S_t, A_t) - V_w(S_t) \big) \nabla_\theta \ln \pi_\theta(A_t \vert S_t)} \\
  & + \color{red}{\mathbb{E}_{a \sim \pi} \big[ \max(0, \frac{\omega_t(a) - c}{\omega_t(a)}) \big( Q_w(S_t, a) - V_w(S_t) \big) \nabla_\theta \ln \pi_\theta(a \vert S_t) \big]}
  & \scriptstyle{\text{; Let }\omega_t (a) =\frac{\pi(a \vert S_t)}{\beta(a \vert S_t)}}
\end{aligned}
$$

where $$Q_w(.)$$ and $$V_w(.)$$ are value functions predicted by the critic with parameter w. The first term (blue) contains the clipped important weight. The clipping helps reduce the variance, in addition to subtracting state value function $$V_w(.)$$ as a baseline. The second term (red) makes a correction to achieve unbiased estimation.


**Efficient TRPO**

Furthermore, ACER adopts the idea of TRPO but with a small adjustment to make it more computationally efficient: rather than measuring the KL divergence between policies before and after one update, ACER maintains a running average of past policies and forces the updated policy to not deviate far from this average.

The ACER [paper](https://arxiv.org/pdf/1611.01224.pdf) is pretty dense with many equations. Hopefully, with the prior knowledge on TD learning, Q-learning, importance sampling and TRPO, you will find the [paper](https://arxiv.org/pdf/1611.01224.pdf) slightly easier to follow :)


### ACTKR

[[paper](https://arxiv.org/pdf/1708.05144.pdf)\|[code](https://github.com/openai/baselines/tree/master/baselines/acktr)]

**ACKTR (actor-critic using Kronecker-factored trust region)** ([Yuhuai Wu, et al., 2017](https://arxiv.org/pdf/1708.05144.pdf)) proposed to use Kronecker-factored approximation curvature ([K-FAC](https://arxiv.org/pdf/1503.05671.pdf)) to do the gradient update for both the critic and actor. K-FAC made an improvement on the computation of *natural gradient*, which is quite different from our *standard gradient*. [Here](http://kvfrans.com/a-intuitive-explanation-of-natural-gradient-descent/) is a nice, intuitive explanation of natural gradient. One sentence summary is probably:

> “we first consider all combinations of parameters that result in a new network a constant KL divergence away from the old network. This constant value can be viewed as the step size or learning rate. Out of all these possible combinations, we choose the one that minimizes our loss function.”

I listed ACTKR here mainly for the completeness of this post, but I would not dive into details, as it involves a lot of theoretical knowledge on natural gradient and optimization methods. If interested, check these papers/posts, before reading the ACKTR paper:
* Amari. [Natural Gradient Works Efficiently in Learning](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.452.7280&rep=rep1&type=pdf). 1998
* Kakade. [A Natural Policy Gradient](https://papers.nips.cc/paper/2073-a-natural-policy-gradient.pdf). 2002
* [A intuitive explanation of natural gradient descent](http://kvfrans.com/a-intuitive-explanation-of-natural-gradient-descent/)
* [Wiki: Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product)
* Martens & Grosse. [Optimizing neural networks with kronecker-factored approximate curvature.](http://proceedings.mlr.press/v37/martens15.pdf) 2015.


Here is a high level summary from the K-FAC [paper](https://arxiv.org/pdf/1503.05671.pdf):

> “This approximation is built in two stages. In the first, the rows and columns of the Fisher are divided into groups, each of which corresponds to all the weights in a given layer, and this gives rise to a block-partitioning of the matrix. These blocks are then approximated as Kronecker products between much smaller matrices, which we show is equivalent to making certain approximating assumptions regarding the statistics of the network’s gradients. 

> In the second stage, this matrix is further approximated as having an inverse which is either block-diagonal or block-tridiagonal. We justify this approximation through a careful examination of the relationships between inverse covariances, tree-structured graphical models, and linear regression. Notably, this justification doesn’t apply to the Fisher itself, and our experiments confirm that while the inverse Fisher does indeed possess this structure (approximately), the Fisher itself does not.”


## Quick Summary

After reading through all the algorithms above, I list a few building blocks or principles that seem to be common among them:

* Try to reduce the variance and keep the bias unchanged.
* Off-policy gives us better exploration and helps us use data samples more efficiently.
* Experience replay (training data sampled from a replay memory buffer);
* Target network that is either frozen periodically or updated slower than the actively learnt policy network;
* Batch normalization;
* The critic and actor can share lower layer parameters of the network and two output heads for policy and value functions.
* It is possible to learn with deterministic policy rather than stochastic one.
* Put constraint on the divergence between policy updates.
* New optimization methods (such as K-FAC).



## References

[1] jeremykun.com [Markov Chain Monte Carlo Without all the Bullshit](https://jeremykun.com/2015/04/06/markov-chain-monte-carlo-without-all-the-bullshit/)

[2] Richard S. Sutton and Andrew G. Barto. [Reinforcement Learning: An Introduction; 2nd Edition](http://incompleteideas.net/book/bookdraft2017nov5.pdf). 2017.

[3] John Schulman, et al. ["High-dimensional continuous control using generalized advantage estimation."](https://arxiv.org/pdf/1506.02438.pdf) ICLR 2016.

[4] Thomas Degris, Martha White, and Richard S. Sutton. ["Off-policy actor-critic."](https://arxiv.org/pdf/1205.4839.pdf) ICML 2012.

[5] timvieira.github.io [Importance sampling](http://timvieira.github.io/blog/post/2014/12/21/importance-sampling/)

[6] Mnih, Volodymyr, et al. ["Asynchronous methods for deep reinforcement learning."](https://arxiv.org/abs/1602.01783) ICML. 2016. 

[7] David Silver, et al. ["Deterministic policy gradient algorithms."](https://hal.inria.fr/file/index/docid/938992/filename/dpg-icml2014.pdf) ICML. 2014.

[8] Timothy P. Lillicrap, et al. ["Continuous control with deep reinforcement learning."](https://arxiv.org/pdf/1509.02971.pdf) arXiv preprint arXiv:1509.02971 (2015).

[9] Ryan Lowe, et al. ["Multi-agent actor-critic for mixed cooperative-competitive environments."](https://arxiv.org/pdf/1706.02275.pdf) NIPS. 2017.

[10] John Schulman, et al. ["Trust region policy optimization."](https://arxiv.org/pdf/1502.05477.pdf) ICML. 2015.

[11] Ziyu Wang, et al. ["Sample efficient actor-critic with experience replay."](https://arxiv.org/pdf/1611.01224.pdf) ICLR 2017.

[12] Rémi Munos, Tom Stepleton, Anna Harutyunyan, and Marc Bellemare. ["Safe and efficient off-policy reinforcement learning"](http://papers.nips.cc/paper/6538-safe-and-efficient-off-policy-reinforcement-learning.pdf) NIPS. 2016.

[13] Yuhuai Wu, et al. ["Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation."](https://arxiv.org/pdf/1708.05144.pdf) NIPS. 2017.

[14] kvfrans.com [A intuitive explanation of natural gradient descent](http://kvfrans.com/a-intuitive-explanation-of-natural-gradient-descent/)

[15] Sham Kakade. ["A Natural Policy Gradient."](https://papers.nips.cc/paper/2073-a-natural-policy-gradient.pdf). NIPS. 2002.


