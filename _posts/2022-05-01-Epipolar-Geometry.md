---
layout: single
title: Essential Matrix and Fundamental Matrix from Epipolar Geometry
toc: true
toc_sticky: true
categories:
  - SLAM
tags:
  - Essential Matrix
  - Fundmental Matrix
  - Epipolar Geometry
published: true
comments: true
---

> Epipolar geometry delivers constraints of same observation (point in the world) from 2 images with overlaps, which brings the possibility to estimate geometrical transformation between cameras.

![image](../assets/images/posts/Epipolar_geometry.svg)

As seen from above illustration, $O_L$ and $O_R$ represents 2 camera centers. While Object $X$ in the world prejected to image space with $X_L$ and $X_R$. However due to the loss of depth in camera projection model. How deep is $X$ in left view is not clear, but what is known is that it must in the line $O_LX$, same applys to right view. With the help of right view, you can find that although depth in left view is unknown, you can still find a line $e_RX_R$ projecting $O_LX$, X locates on the line.

With Point pairs $X_L$ and $X_R$ you can create relation between landmark $X$ and cameras coordiante transformation from $O_L$ to $O_R$.

$$X_R^TK^{-T}\hat{t}RK^{-1}X_R = 0$$

## Essential Matrix

$$E = \hat{t}R$$

## Fundamental Matrix

$$F = K^{-T}\hat{t}RK^{-1}$$

