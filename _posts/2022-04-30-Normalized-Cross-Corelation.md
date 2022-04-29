---
layout: single
title: Normalized Cross Corelation
toc: true
toc_sticky: true
categories:
  - Fundamental
tags:
  - Direct Method
  - Template Matching
published: false
comments: true
---

> Normalized Normalized Cross Corelation(NCC)

Take an image patch (usually sliding window, e.g. 9x9) $X$ from Image $I_1$ and $Y$ from $I_2$, X and Y are the same size. We suppose that $X$ is the template and want to find the correspond batch in $I_2$.
$$NCC(X,Y) = \frac{Cov(X,Y)}{\sqrt{Var(X)Var(Y)}}\tag{1}$$

where $Cov(X, Y)$ is the covariance of $X$ and $Y$ and $Var(X)$ and $Var(Y)$ are variance.

$NCC(X,Y)$ ranges in $[-1, 1]$, $NCC(X,Y) = 1$ when $X == Y$, meaning X and Y are the same.
$NCC(X,Y) = -1$ when $X == -Y$, meaning X and Y are point to the contrary direction.

$NCC(X,Y) = 0$ when there is no corelation between $X$ and $Y$, meaning X and Y are point to the contrary direction.

### Assumption on using NCC

Images are allowed to have changes in

* translation
* brightness
* contrast
* ~~rotation~~

Pure NCC is mostly only limited in translation estimation, rotation estimation is also possible see [link](https://www.researchgate.net/publication/224641323_Image_Matching_by_Normalized_Cross-Correlation).

### Offset Estimation

$$I_1(p,q) == a + bI_2(i,j)$$
where $p$ and $q$ are the template location of $I_1$, $i$ and $j$ candidate location in $I_2$. $a$ predict brightness change and $b$ predict contrast change. NCC is invariant to brightness changes and contrast changes, by taking $a$ and $b$ in $(1)$, $a$ and $b$ would appear in the result.

$$\binom{p}{q} = \binom{i}{j} - \binom{u}{v}$$

where $u$ and $v$ are to be estimated.

NCC shall be maximized with sliding window in $I_2$.

### Pymramid Refinment

Matching from the high level of pyrimid and limit search area to the next level of pyrimid.
