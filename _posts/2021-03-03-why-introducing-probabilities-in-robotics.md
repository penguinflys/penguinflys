---
layout: single
toc: true
toc_sticky: true
categories:
  - Fundamental
tags:
  - SLAM
published: true
---

> Uncertainty happens during observation in the real world. Robotics observe the world using sensors, and the uncertainty of sensors is modelled by probabilistic models, such as gaussian distribution. 

## Uncertainty in Robotics

* <span style="color:blue">What does Robotics do?</span>  
Robotics is a science about __controlling__ mechanical devices (automatically/intelligently) to __perceive__ and __manipulate__ the physical __environment__.

| Word        | Description                                                                                                                      |
|-------------|----------------------------------------------------------------------------------------------------------------------------------|
| control     | also referred as execute or actuate, which is the function that makes robots interact with the environment.                      |
| perceive    | also referred as percept or observe, which is the function to transfer environment information to the computing unit of a robot. |
| manipulate  | which means change the environment into desired state.                                                                           |
| environment | also referred as world, which can be variable or fixed, including the robot itself and environment independent with the robot    |

Intelligence of robotic shall be mentioned in another posts. we only talk about uncertainty in this post.
Taking a assembly line as an example, the location of the robot is stationary and environment is structured, the operation of robot arms is qualified as long as the operational accuracy is within manageable range.
However, the new robot systems operates in a increasingly unstructured environments that are inherently unpredictable, which makes sensors very important in the process of perception.

* <span style="color:blue">What is Uncertainty and why it exist?</span>  

    Uncertainty happens in mobile robotics in the following ways:
    * _Environments_: compared with assembly line, environments of roads are highly __dynamic__ and __unpredictable__.
    * _Sensors_: sensors has limited __resolution__ and measurement contains __noise__.
    * _Robots_: control __noise__ and __wear-and-tear__.
    * _Models_: models are __abstraction__ of the real world, and abstracted model and not model the world fully, which means the robotics system are crude.
    * _Computation_: due to the requirement of being a real-time system, algorithms usually are approximate to achieve timely response.
<!-- Depending on the degree of handling complexity of the environment, robotics can be in different intelligent levels while manipulating the world. For example, a robot repetitively do the same operations is hard to be seen as "intelligent", such as a toy train in your childhood. On the contrast a service robot in your home organizing and clean everything in your life is "intelligent", and an autonomous car on the road adjusting the speed and follow the principle of traffic rules to let pedestrians go first is "intelligent". -->

## Probabilistic Model
* <span style="color:blue">How to model the uncertainty?</span>  

The key idea of probabilistic robotics is to represent uncertainty explicitly, using the calculus of probability theory. Instead of find the best guess of what happens in the real world, the probabilistic model represent the information by probability distribution over the space of possible hypothesis. With this model, all sources of uncertainty can be fused mathematically, detailed information shall be mentioned later in later posts, such as kalman filtering.

<span style="color:red"> The most important mindset of applying probabilistic model is that the scalars describing the state of the environment is extended to probabilistic space, instead of being a number.</span>
## References

<div id="refs"></div>

Source: [Probabilistic Robotics](https://docs.ufpr.br/~danielsantos/ProbabilisticRobotics.pdf)