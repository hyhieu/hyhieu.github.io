---
layout: post
date: 2023-12-31
---

On CuTe layouts
===============

**[work in progress. many citations missing, and many writings here could be wrong.]**

[[Hieu's personal blog index](./index)]

## Introduction

[CuTe](https://github.com/NVIDIA/cutlass/tree/main/include/cute) is a
library in NVIDIA's [CUTLASS](https://github.com/nvidia/cutlass).
CuTe makes GPU programming flexible and generalizable across different GPU
architectures, such as
[Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/),
[Ampere](https://www.nvidia.com/en-us/data-center/ampere-architecture/), and
[Hopper](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/).
Thanks to these benefits, CuTe plays the central role in the development of many
efficient deep learning libraries, including but not limited to
[FlashAttention](https://github.com/Dao-AILab/flash-attention),
[FasterTransformer](https://github.com/NVIDIA/FasterTransformer), and
[xFormers](https://github.com/facebookresearch/xformers).

CuTe's benefits stem from the ingenious design of its central concept:
*layout.* A layout represents a map from a $D$-dimensional coordinate to an
integer. Despite this very basic intuition, layouts accompanied by the
operations on them have significantly improved the experience of writing
performant GPU programs.

## Motivations

Despite the benefits of CuTe, we find CuTe's
[original documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute)
somewhat inadequate, especially at building a foundational understanding of
CuTe's concepts. For instance, the complement operation is a fundamental one
with CuTe layout, but the doc does not explain *how* to compute a complement.
While it is okay to treat these low-level operations as black boxes and just
move on writing good GPU programs with CuTe, we do feel the need to have a
rigorous foundation of the concepts when writing our programs.

**Goal.**
Building such rigurous foundation is the goal of this blog post. In particular,
we attempt to formalize the definition of CuTe layout and a few of its
accompanying operations, most importantly [complementation](#complement),
[composition](#composition), and [logical division](#logical-division).

**Our source of inspiration.**
This blog post is not the first attempt at building foundational understanding of CuTe.
We are inspired by
[Jay Shah's note](https://research.colfax-intl.com/a-note-on-the-algebra-of-cute-layouts)
on the same topic. Jay's note elicits the concepts around layouts very well, perhaps
better than what we are doing, and we recommend the note to anyone who wants to deepen
their undersatnding of CuTe.

In this post, we provide a different treatment of CuTe's layout.
[Jay's note](https://research.colfax-intl.com/a-note-on-the-algebra-of-cute-layouts)
states concise standards to test which layouts are *"admissable for
complementation"* or *"composition".* In order to derive such standards, they
introduced some new concepts, such as sorting the modes of layouts. A
consequence of their new standards anc concepts is that some layouts are not
admissable for complement or composition. Ruling out such layouts makes sense
practically, since we rarely encounter them while programming, but doing so
means that the theory constructed around layouts does not faithfully reflect the
behaviors of layouts in the CuTe library. We do not say such is a bad
consequence. Instead, augmented by the intuitions harvested from
[Jay's note](https://research.colfax-intl.com/a-note-on-the-algebra-of-cute-layouts),
here we provide an *algorithmic* treatment of layouts, which we hope more
closely reflects their behaviors in the CuTe library.

**Our roadmap.**
We treat a layout as a way to represent function $f: \mathbb{N} \to
\mathbb{N}$ that maps an integer to a multilinear function's output defined by
the layout. We explain that obtaining such a function from a layout is
straightforward, and develop an algorithm to find a layout corresponding a
certain function (it's a deterministic algorithm, but it does not elicit a
concise standard like
[Jay's treatment](https://research.colfax-intl.com/a-note-on-the-algebra-of-cute-layouts)).
Then, using the correspondence between layouts and these functions via an
algorithm, and then use this correspondence to construct layout operations such
as [complementation](#complement), [composition](#composition), and
[logical division](#logical-division).

TODO: How about coalesce, left/right inverse? Are these fundamental enough to cover?

## Layout

<div class="statement" markdown="1" id="layout-def">

**Definition 1. (Layout)** Let $D$ be a positive integer. A layout $L = N :
S$ is a pair of tuples, each with $D$ positive integers:

$$
\begin{aligned}
N &= (n_0, n_1, ..., n_{D-1}) \\
S &= (s_0, s_1, ..., s_{D-1})
\end{aligned}
$$

</div>

<details markdown="1">
<summary>So what is the goal of layouts?</summary>

In this note, we are not very concerned with the goal of layouts. We are much
rather focus on their mathematical constructions and the operations on them.

That said, the TL;DR of layout's goal is that they are used to represent
*offsets* of elements in CuTe tensors -- that is, how far is each element from
the tensor's first element in its memory.

For instance, a contiguous row-major matrix of size `(m, n)` as we typically see
in `numpy` has the layout $(m, n) : (n, 1)$, meaning that its $(i, j)$ entry is
$ni + j$ memory cells away from its $(0, 0)$ element.

Later in this note, with a better understanding of
[canonical functions](#canonical-functions), we will see that the layout $(m, n) : (n, 1)$
is *reasonably* the same as the layout $(n, m) : (1, n)$, but the latter behaves much
better with many operations.

</details> <!-- So what is the goal of layouts? -->

There are some terminologies associated with the definition of layout:


- Each element in the size tuple is sometimes called an *extent* of $L$.

- $\text{size}(L)$ is simply the product of all extents of $L$:
$\text{size}(L) = n_0 n_1 \cdots n_{D-1}$.

- The tuple $S$ is called the layout's *stride.*

- The maximum offset that the layout can represent,
i.e., $1 + \sum_{i=0}^{D-1}s_i (n_i - 1)$, is the layout's *cosize* and is denoted by
$\text{cosize}(L)$.

- The pair $(n_i, s_i)$, sometimes written $(n_i) : (s_i)$, is called the
$i^\text{th}$ *mode* of $L$.

In the next section, we will study the *canonical functions* of layouts, which
represent how to map coordinates to offsets in layouts.

## Canonical functions

<div class="statement" markdown="1">

**Definition 2. (Canonical multivariate function)**

A layout $L$ represents a multivariable function $g_L : [0, n_0) \times [0,
n_1) \times \cdots \times [0, n_{D - 1}) \subseteq \mathbb{N}^{D} \to
\mathbb{N}$, defined by:

$$
g_L(x_0, x_1, ..., x_{D-1}) := s_0 x_0 + s_1 x_1 + \cdots + s_{D-1} x_{D-1}
$$

We call $g_L$ the *canonical multivariate function* of $L$.

</div>

For brevity, when clear from context, we drop the word "canonical", and might
also use the overloaded notation:

$$
L(x_0, x_1, ..., x_{D-1}) := s_0 x_0 + s_1 x_1 + \cdots + s_{D-1} x_{D-1}
$$

Other than the canonical multivariate function, we are also interested in the
*canonical singlevariate function* of a layout. This singlevariate function is
constructed from the layout's multivariate function via the natural isomorphism
between $[0, n_0 n_1 \cdots n_{D-1})$ and $[0, n_0) \times [0, n_1) \times
\cdots \times [0, n_{D - 1})$. Intuitively, this isomorphism is the enumerating
of the points on the integral lattice $[0, n_0) \times [0, n_1) \times \cdots
\times [0, n_{D - 1})$. Here comes the tricky piece -- *this isomorphism is itself a layout:*

$$
\begin{aligned}
\text{MultiToSingle}
    &= (n_0, n_1, n_2, ..., n_{D-1}) \\
    &:~~(1, n_0, n_0 n_1, n_0 n_1 n_2, ..., n_0 n_1 \cdots n_{D-2})
\end{aligned}
$$

As $\text{MultiToSingle}$ is an isomorphism, its invert $\text{SingleToMulti}$
is *mathematically* well-defined. Here is a self-contained formula for
$\text{MultiToSingle}$ and $\text{SingleToMulti}$:

$$
\begin{aligned}
\text{MultiToSingle}(x_0, x_1, ..., x_{D-1})
  &= x_0
    + n_0 \cdot x_1
    + n_0 n_1 \cdot x_2
    + n_0 n_1 n_2 \cdot x_3
    + \cdots
    + n_0 n_1 \cdots n_{D-2} \cdot x_{D-1} \\
\text{SingleToMulti}(x)
  &= \left(
    x~\text{mod}~n_0,
    \left\lfloor \frac{x}{n_0} \right\rfloor~\text{mod}~n_1,
    \left\lfloor \frac{x}{n_0 n_1} \right\rfloor~\text{mod}~n_2,
    ...,
    \left\lfloor \frac{x}{n_0 n_1 \cdots n_{D-2}} \right\rfloor~\text{mod}~n_{D-1}
  \right)
\end{aligned}
$$

Using the $\text{SingleToMulti}$ function above, we can define a layout's
canonical singlevariate function:

<div class="statement" markdown="1">

**Definition 3. (Canonical singlevariate function)**

Let $L = (n_0, n_1, ..., n_{D-1}) : (s_0, s_1, ..., s_{D-1})$ be a layout.  Let
$M = n_0 n_1 \cdots n_{D-1}$ be $L$'s size.  The canonical singlevariate
function of $L$ is $f_L: [0, M) \to \mathbb{N}$ defined by:

$$
\begin{aligned}
f_L(x)
  &:= \text{SingleToMulti}(x)^\top \cdot (s_0, s_1, ..., s_{D-1}) \\
  &= \sum_{i=0}^{D-1} s_i \cdot \left(\left\lfloor \frac{x}{n_0 n_1 \cdots n_{i-1}} \right\rfloor~\text{mod}~n_i\right)
\end{aligned}
$$

</div>

Similar to our treatment of canonical multivariate functions, we also drop the
terms "canonical" when clear from context. We might also write $L(x)$ instead of
$f_L(x)$.

<details>
<summary><b>Note:</b> the terms "canonical single/multi-variate functions" are a bit
different from the <a href="https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/02_layout_algebra.md">CuTe's original docmentation</a>.</summary>

They simply use the term "function" to refer to "canonical singlevariate
function". We find that when clear from context, it is okay to just say "the
layout's function", or "the layout's associated function".

</details>

<details markdown="1">
<summary><b>Digression:</b> column-major vs. row-major.</summary>

The way we define the singlevariate function of a layout corresponds to how we
traverse the layout's $D$-dimensional coordinate space in the increasing order
of its modes. This traversal is sometimes called the *column-major* traversal.

Column-major traversal is used in MATLAB and Fortran. In contrast, most modern
deep learning frameworks like `numpy`, `torch`, and `jax` use row-major
traversal.

It is possible to redefine the entire theory on layouts using
row-major traversal, but we choose to follow CuTe's original choice of being
column-major.

</details>

Given any layout $L$, we can easily construct its singlevariate function. The
reverse question is much less trivial: for which functions $f: \mathbb{N} \to
\mathbb{N}$ there is a layout whose singlevariate function is $f$?

In the next section, we will discuss a general question regarding the canonical
functions.  For a function $f: [0, M) \to \mathbb{N}$, we say that $L$ *admits*
$f$ if $L(x) = f(x)$ for all $x \in [0, M)$. Under that definition, which
function $f: \mathbb{N} \to \mathbb{N}$ is admitted by a layout?

<!-- HIEU HAS PROOFREAD UNTIL THIS POINT -->

It can be seen from the definition that mode-coalescing does not change the
layout's singlevariate function.

[Later in this note](#what-function-can-be-admitted-by-a-layout), we will discuss an
algorithm to determine which functions can be expressed by a layout.

## Layout extensions

Now that we have discussed the canonical functions, we can discuss a layout's
*extension*. This concept is introduced in
[Jay's note](https://research.colfax-intl.com/a-note-on-the-algebra-of-cute-layouts).
While the concept is not presented in
[CuTe's original definition](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute),
we find it very natural once we have understood the basics of layouts. More
importantly, the concept of extension serves to make the mathemetical proofs
later in this blog post consistent, and enhances our understanding of layouts.

<div class="statement" id="def-layout-extension" markdown="1">

**Definition. (Layout extension)**

Let $L = (n_0, n_1, ..., n_{D-1}) : (s_0, s_1, ..., s_{D-1})$ be a
$D$-dimensional layout. $L$'s extension is the layout with all modes of $L$,
except for the last mode whose dimension is replaced with $+\infty$. More precisely:

$$
L_\text{ext} := (n_0, n_1, ..., n_{D-2}, +\infty) : (s_0, s_1, ..., s_{D-2}, s_{D-1})
$$

</div>

The multivariate function of $L_\text{ext}$ is defined naturally based on that
of $L$, except that the last coordinate can take any value, instead of being
restricted to $[0, n_{D-1})$:

$$
\begin{aligned}
g_{L_\text{ext}}
  &: [0, n_0) \times
     [0, n_1) \times \cdots
     [0, n_{D-2}) \times \mathbb{N}
  \to \mathbb{N} \\
g_{L_\text{ext}}(x_0, x_1, ..., x_{D-1})
  &= s_0 x_0 + s_1 x_1 + \cdots + s_{D-2} x_{D-2} + s_{D-1} x_{D-1}
\end{aligned}
$$

The singlevariate function $L_\text{ext}$ is therefore:

$$
L_\text{ext}(x)
  = L(x~\text{mod}~n_0 n_1 \cdots n_{D-1})
  + \text{cosize}(L) \cdot \left\lfloor \dfrac{x}{n_0 n_1 \cdots n_{D-1}} \right\rfloor
$$

## What function $f: \mathbb{N} \to \mathbb{N}$ can be admitted by a layout?

<div class="statement" id="function-to-layout">

Let $f: [0, M) \to \mathbb{N}$ be an arbitrary function. We present in this
section an algorithm with runtime $O(M^2 \log{M})$ that either finds a layout $L
= (n_0, n_1, ...  , n_{D-1}) : (s_0, s_1, ..., s_{D-1})$ such that $L(x) = f(x)$
for all $x \in [0, M)$, or reports that there is no such layout.

</div>

<details markdown="1">
<summary markdown="1">Here is the algorithm.
We also provide its <a href="#python-implementation">Python implementation</a>.
</summary>

### Algorithm

Without loss of generality, assume that $n_i > 1$ for all $i \in [0, D)$. That is,
we only look for layouts whose all dimensions is larger than $1$.
<details markdown="1">
<summary>Why can we assume so?</summary>

We first notice that if $n_i = 1$ for an index $i \in \{0, 1, ..., D-1\}$,
then for all $x \in \mathbb{N}$, the $i$-th coordinate of $x$ in $L$'s
coordinate space is:

$$
\left\lfloor \dfrac{x}{n_0 n_1 \cdots n_{i-1}} \right\rfloor~\text{mod}~1 = 0
$$

This means that $s_i$ never contributes to the value of $L(x)$, and hence can
take any value. For this reason, we call the dimensions where $n_i = 1$ trivial.
To avoid such trivial dimensions, we can assume that $\boxed{n_i > 1}$ for all
$i \in \{0, 1, \cdots, D-1\}$.

</details>

The gist of the algorithm is to guess the first mode $(n_0) : (s_0)$, and then
recurse. To this end, we start from the formula of the singlevariate function $L(x)$:

<div id="l-formula"></div>

$$
\begin{aligned}
L(x)
  &= \sum_{i=0}^{D-1} s_i \cdot \left(\left\lfloor \frac{x}{n_0 n_1 \cdots n_{i-1}} \right\rfloor~\text{mod}~n_i\right)
\end{aligned}
$$

From this formula, we necessarily have $L(0) = 0$. In other words, if $f(x) \neq
0$, there is no layout admitting $f$.

### Guessing $s_0$

Also from the formula, we can guess $s_0$ by letting $x = 1$. Thanks to the
assumption that $n_i > 1$ for all $i$'s, we have:

$$
\left\lfloor \dfrac{1}{n_0 n_1 \cdots n_{i-1}} \right\rfloor~\text{mod}~n_i
  = \begin{cases}
  1 & \text{if $i = 0$} \\
  0 & \text{if $i > 0$}
  \end{cases}
$$

Comparing this with [$L$'s formula](#l-formula), we have $\boxed{s_0 = L(1) =
f(1)}$.

### Guessing $n_0$

It is possible that the layout $L = (M) : (s_0)$ admits $f$. We can check
whether $f(k) = k s_0$ for all $kx \in [0, M)$, and if yes, we return here.

If no, then there exists a unique index $t$ such that $f(k) = k s_0$ for $k \in
\{0, 1, ..., t-1\}$ but $f(t) \neq t s_0$. We now know that $\boxed{n_0
\in [0, t)}$.

We will try for each value of $n_0$ *in the decreasing order* from $t-1, t-2,
..., 2$, and recurse on the first -- i.e., the largest -- value of $n_0$ that is
*consistent* with $f$.

There are some ground-laying work to ensure that the algorithm works.

<details markdown="1">
<summary>What does <i>consistent</i> mean?</summary>

Formally, we say that a number $n_0$ is consistent with respect to a function
$f: [0, M) \to \mathbb{N}$ and a stride $s_0$ if and only if:

$$
\boxed{
f(x) = f\mathopen{}\left( n_0 \cdot \lfloor x / n_0 \rfloor \right)
     + s_0 \cdot (x~\text{mod}~n_0),~~~\text{for all $x \in [0, M)$}
}
$$

Intuitively, consistency here refers to the event that the values of $f$, i.e.,
$\{f(0), f(1), ..., f(M-1)\}$, can be arranged into $n_0$ rows as follows:

```text
            0 = f(0)     | f(n_0)     | f(2*n_0)   | ...
          s_0 = f(1)     | f(n_0+1)   | f(2*n_0+1) | ...
          ...            | ...        | ...        | ...
(n_0-1) * s_0 = f(n_0-1) | f(2*n_0-1) | f(3*n_0-1) | ...
```
Here, all columns must have $n_0$ entries, except for the last column which
might have $M~\text{mod}~n_0$ entries.

Thus, $n_0$ being consistent with $f$ and $s_0$ means that there is
*potentially* a layout admitting $f$ whose first mode is $(n_0) : (s_0)$.


</details> <!-- What does consistent mean? -->

<details markdown="1">
<summary>Do we only need to recurse on the largest value of $n_0$? Short answer: <b>yes</b>.</summary>

We will prove two claims that give the positive answer to this question.

<div markdown="1" class="statement">

**Claim 1.**
Let $n_0$ be the largest consistent value for $f$ and $s_0$. Then, if $\hat{n}_0
< n_0$ is consistent, we necessarily have $\hat{n}_0~|~n_0$.

</div>

<details markdown="1">
<summary><b>Proof.</b></summary>
Assume that $\hat{n}_0~\nmid~n_0$, we will derive that $l = \text{lcm}(n_0,
\hat{n}_0)$ is also consistent to $f$ and $s_0$. This is a contradiction to the
maximality of $n_0$, since $l > n_0$ because $\hat{n}_0~\nmid~n_0$.

Take $k \in \mathbb{N}$ such that $\hat{n}_0~\nmid~k n_0$ and $t \in \mathbb{N}$
such that $t l + k n_0 < M$, we have:

$$
\left\lfloor \frac{tl + k n_0}{\hat{n}_0} \right\rfloor
= \underbrace{\frac{tl}{\hat{n}_0}}_{\in \mathbb{N}} + \left\lfloor \frac{k n_0}{\hat{n}_0} \right\rfloor
= \frac{tl}{\hat{n}_0} + \left\lfloor \frac{k n_0 - 1}{\hat{n}_0} \right\rfloor
= \left\lfloor \frac{tl + k n_0 - 1}{\hat{n}_0} \right\rfloor
$$

and:

$$
(tl + k n_0)~\text{mod}~\hat{n}_0 - 1 = (tl + k n_0 - 1)~\text{mod}~\hat{n}_0
$$

Hence, using the consistency of $\hat{n}_0$, we have:

$$
\begin{aligned}
f(tl + k n_0)
    &= f\mathopen{}\left( \hat{n}_0 \cdot \left\lfloor \frac{tl + k n_0}{\hat{n}_0} \right\rfloor \right)
     + s_0 \cdot ((t l + k n_0)~\text{mod}~\hat{n}_0) \\
    &= f\mathopen{}\left( \hat{n}_0 \cdot \left\lfloor \frac{tl + k n_0 - 1}{\hat{n}_0} \right\rfloor \right)
     + s_0 \cdot ((t l + k n_0 - 1)~\text{mod}~\hat{n}_0 + 1) \\
    &= f(t l + k n_0 - 1) + s_0
\end{aligned}
$$

Using this result for $k = 1, 2, ..., l / n_0 - 1$, we have $f(tl + r) = f(tl) +
s_0 r$ for all $r \in [0, l)$ and $t$ such that $tl + r < M$.

Finally, using the fact that all $x \in [0, M)$ can be written as $x = tl + r$
where $r \in [0, r)$, we derive the consistency of $l$, which is a contradiction. $\square$

</details>

<div markdown="1" class="statement">

**Claim 2.**
If a layout $\hat{L}$ whose first mode is $(\hat{n}_0) : (s_0)$ admits $f$, then
there exists a layout $L$ whose first mode is $(n_0) : (s_0)$ that also admits
$f$.

</div>

<details markdown="1">
<summary><b>Proof.</b></summary>

Let us write $\hat{L} = (\hat{n}_0, n_1, n_2, ...) : (s_0, s_1, s_2, ...)$.  We
will prove that $s_1 = s_0 \hat{n}_0$.

To prove this point, we use the hypothesis that $n_0$ is consistent with respect
to $f$ and $s_0$. This means that $f(x) = x s_0$ for all $x < n_0$. In
particular, for $x = \hat{n}_0 < n_0$:

$$
s_0 \hat{n}_0 = f(\hat{n}_0) = L(0, 1, 0, 0, ..., 0) = s_1
$$

Now that we have prove $s_1 = s_0 \hat{n}_0$, we see that the layout $\tilde{L}
= (\hat{n}_0 n_1, n_2, ...) : (s_0, s_2, ...)$ has the same singlevariate
function with $L$. This is because for any $x \in [0, M)$, we have:

$$
\begin{aligned}
  s_0 \cdot (x~\text{mod}~\hat{n}_0)
+ s_0 \hat{n}_0 \cdot \left( \left\lfloor \frac{x}{\hat{n}_0} \right\rfloor~\text{mod}~n_1 \right)
&= s_0 \cdot \left(
        x~\text{mod}~\hat{n}_0
        + \hat{n}_0 \cdot \left( \left\lfloor \frac{x}{\hat{n}_0}\right\rfloor~\text{mod}~n_1 \right)
    \right) \\
&= s_0 \cdot \left( x~\text{mod}~\hat{n}_0n_1 \right)
\end{aligned}
$$

Plus, the contributions for all mode $i$'s with $i \geq 2$ are the same between
$L$ and $\tilde{L}$. This gives $\tilde{L} = L$.

Now, if $\hat{n}_0 n_1 = n_0$, then we are done. If not, we repeat the process
above and end up with another layout with a strictly arger first size "$n_0$"
but the same first stride $s_0$. This process must converge, as the first stride
is increasing and upper-bounded. Also, the process can only converge with the
first stride being $n_0$, because otherwise, we can still repeat.

This completes the proof. $\square$

</details>

</details> <!-- Why does largest n_0 work? -->

### Recurse

To find $(n_1) : (s_1)$, we repeat the algorithm recursively on the function:

$$
g : \left[ 0, \left\lfloor M / n_0 \right\rfloor \right] \to \mathbb{N}~~~~~~~~~g(x) := f(n_0 x)
$$

Essentially, this means to restrict $f$ into the sub-domain where the $0$-th
coordinate is $0$.

### Runtime analysis

Our unoptimized implementation of the algorithm runs in $\boxed{O(M^2
\log{M})}$. The analysis is based on the assumption that $n_i \geq 2$ for all $i
\in [0, D)$. As such, the resulting layout $L$ has at most $O(\log{M})$ modes.
Furthermore, the unoptimized implementations of [mode](#guessing-s_0)
[guessing](#guessing-n_0) is $O(M^2)$.

<details markdown="1">
<summary>Let's not bore ourselves with the analysis, unless you raelly want to...</summary>

Let us analyze the complexity of the process above:

1. $O(1)$ Find $s_0 = f(1)$:

2. $O(M)$ Checking whether $(M): (s_0)$ is okay.

3. $O(M)$ For each $k \in \{t, t-1, ..., 2\}$:

    a. $O(M)$ Check if $n_0$ is consistent. If yes, take $n_0$ to be the
    *smallest* value, and repeat the algorithm.

    b. If no such value is found, then there is an inconsistency.

Thus, each value $(n_i, s_i)$ can be determined in $O(M^2)$, or an inconsistency
is found.

</details>

</details>


## Concatenation

<div markdown="1" class="statement" id="layout-def">

**Definition 4. (Concatenation)**

The concatenation of two layouts $L_1$ and $L_2$ -- denoted by $(L_1, L_2)$ -- is
the layout $L$ whose singlevariate function is:

$$
\begin{aligned}
L &: [0, \text{size}(L_1) \cdot \text{size}(L_2)) \to \mathbb{N} \\
L(x)
    &= L_1(x~\text{mod}~\text{size}(L_1))
    + L_2\mathopen{}\left(
        \left\lfloor \frac{x}{\text{size}(L_1)} \right\rfloor
    \right)
\end{aligned}
$$

</div>

With that definition, it is relatively easy to check that we have a closed
formula of layout concatentation. In particular, if:

$$
\begin{aligned}
L_1 &= (m_0, m_1, ..., m_{E-1}) : (t_0, t_1, ..., t_{E-1}) \\
L_2 &= (n_0, n_1, ..., n_{D-1}) : (s_0, s_1, ..., s_{D-1})
\end{aligned}
$$

Then their concatenation is simply obtained by concatenating their sizes and strides:

$$
(L_1, L_2) = (m_0, m_1, ..., m_{E-1}, n_0, n_1, ..., n_{D-1})
           : (t_0, t_1, ..., t_{E-1}, s_0, s_1, ..., s_{D-1})
$$

<details markdown="1">

<summary><b>Proof.</b></summary>

We use the following identity regrading floor functions:

<div markdown="1" class="statement">

For any positive integers $x, m, n$:

$$
\left\lfloor \dfrac{x}{mn} \right\rfloor
= \left\lfloor \dfrac{\left\lfloor x/m \right\rfloor}{n} \right\rfloor
$$

</div>

Using this identity, we have:

$$
\begin{aligned}
(L_1, L_2)(x)
  &= \underbrace{
        \sum_{i=1}^{E-1}
            t_i \cdot \mathopen{}\left(
                   \left\lfloor \frac{x}{m_0 m_1 \cdots m_{i-1}} \right\rfloor~\text{mod}~m_i
            \right)
     }_{L_1(x)} \\
  &+ \sum_{i=1}^{D-1}
     s_i \cdot \mathopen{}\left(
            \left\lfloor \frac{x}{m_0 m_1 \cdots m_{E-1} n_0 n_1 \cdots n_{i-1}} \right\rfloor~\text{mod}~n_i
     \right) \\
  &= L_1(x) + \sum_{i=1}^{D-1}
     s_i \cdot \mathopen{}\left(
            \left\lfloor \frac{x / (m_0 m_1 \cdots m_{E-1})}{n_0 n_1 \cdots n_{i-1}} \right\rfloor~\text{mod}~n_i
     \right) \\
  &= L_1(x) + \sum_{i=1}^{D-1}
     s_i \cdot \mathopen{}\left(
            \left\lfloor \frac{x / \text{size}(L_1)}{n_0 n_1 \cdots n_{i-1}} \right\rfloor~\text{mod}~n_i
     \right) \\
  &= L_1(x) + \sum_{i=1}^{D-1}
     s_i \cdot \mathopen{}\left(
            \left\lfloor \frac{\lfloor x / \text{size}(L_1) \rfloor}{n_0 n_1 \cdots n_{i-1}} \right\rfloor~\text{mod}~n_i
     \right) \\
  &= L_1(x) + L_2(x)
\end{aligned}
$$

Note that we have used the definition of size that  $m_0 m_1 \cdots m_{E-1} =
\text{size}(L_1)$, and then used the identity above to wrap $x /
\text{size}(L_1)$ in a floor function. $\square$

</details>

We also have the following summary regarding the size and cosize of layout
concatenations:

$$
\begin{aligned}
\text{size}((L_1, L_2)) &= \text{size}(L_1) \cdot \text{size}(L_2) \\
\text{cosize}((L_1, L_2)) &= \text{cosize}(L_1) + \text{cosize}(L_2)
\end{aligned}
$$

As simple as the concatenation operation is, it plays an important role in
defining two much more complex yet crucial operations on layouts:
[complementation](#complement) and [composition](#composition). We discuss
them in the next sections.

## Complement

<div class="statement" id="complement-def" markdown="1">

**Definition 5. (Complement)**

Let $A = N : S$ be a layout. Then, for an integer $M$, the *complement of $A$
with respect to $M$* -- denoted by $\text{Complement}(A, M)$ -- is the layout
$B$ that satisfies two conditions:
1. $B$'s singlevariate function is strictly increasing.
2. The concatenation layout $(A, B)$'s singlevariate function restricted on $[0, M)$
is a bijection.

</div>

Note that
[CuTe's original definition](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/02_layout_algebra.md#complement)
of complementation specifies the following conditions:

1. $\text{cosize}((A, B)) \leq M$.

2. $\text{size}(B) \geq \left\lceil \dfrac{M}{\text{size}(A)} \right\rceil$.

3. $B(i) \neq A(j)$ for all $i \in [1, \text{size}(B))$ and $j \in [0, \text{size}(A))$.

<details markdown="1">

<summary>
It is not hard to check that these conditions are equivalent to the
bijective condition in <a href="#complement-def">our definition</a>.
</summary>

<div markdown="1" class="statement">

**Claim.** The following two statements are equivalent:

1. The concatenation layout $(A, B)$'s singlevariate function restricted on $[0, M)$
is a bijection.

2. $\text{cosize}((A, B)) \leq M$,
$\text{size}(B) \geq \left\lceil \dfrac{M}{\text{size}(A)} \right\rceil$,
and $B(i) \neq A(j)$ for all $i \in [1, \text{size}(B))$ and
$j \in [0, \text{size}(A)$.

</div>

<details>

<summary><b>Proof.</b></summary>

$(1) \Longrightarrow (2):$ Assuming (1), then the injectivity of $(A, B)$ immediately
gives us $B(i) \neq A(j)$ for all $i \in [1, \text{size}(B))$ and $j \in [0, \text{size}(A))$.
Additionally, the domain of $(A, B)$ must contain $[0, M)$, so:

$$
\begin{aligned}
&\text{size}((A, B)) = \text{size}(A) \cdot \text{size}(B) \geq M \\
&\Longrightarrow \text{size}(B)
  \geq \dfrac{M}{\text{size(A)}} \\
&\Longrightarrow \text{size}(B)
  \geq \left\lceil \dfrac{M}{\text{size(A)}} \right\rceil
\end{aligned}
$$

Finally, the surjectivity of $(A, B): [0, M) \to [0, M)$ gives us
$\text{cosize}((A, B)) \geq M$.

$(2) \Longrightarrow (1):$ Conversely, assuming these three conditions, then
$B(i) \neq A(j)$ for all $i \in [1, \text{size}(B))$ and $j \in [0, \text{size}(A))$
means $(A, B): [0, M) \to [0, M)$ is injective. We need

<!--
In particular, at
$\hat{x} = \text{size}(A) \cdot \text{size}(B) - 1$:

$$
\begin{aligned}
M &> f_{(A, B)}(\hat{x}) \\
  &= A(\underbrace{\hat{x}~\text{mod}~\text{size}(A)}_{\text{}})
   + B\mathopen{}\left(
      \left\lfloor \frac{\hat{x}}{\text{size}(A)} \right\rfloor
      ~\text{mod}~\text{size}(B) \right)

\end{aligned}
$$
-->

TODO: finish this proof.



</details>

</details>

Not all layouts have a complement. In particular, bijection requirement in our
definition rules out all layout $A$ whose singlevariate function is not
injective. [Jay's note](https://research.colfax-intl.com/a-note-on-the-algebra-of-cute-layouts)
offers a method to check whether a layout has a complement layout.

The [function-to-layout Algorithm](#function-to-layout) offers a procedure to
find $\text{Complement}(A, M)$ for any layout $A$ and positive integer $M$, or
to tell that such complement does not exist.

Indeed, the idea is to determine $B$'s singlevariate function based on the given
conditions, and then use [function-to-layout Algorithm](#function-to-layout) to
find $B$, or to tell if there is no such $B$.

## Composition

<div class="statement" id="composition-def" markdown="1">

**Definition 6. (Composition)**

Let $A = N_a : S_a$ and $B = N_b : S_b$ be two layouts. Their composition $A
\circ $B is the layout such that $f_{A \circ B} \equiv f_A \circ f_B$.

</div>

As for complementation, not all pairs of layouts can be composed. To determine
two layouts' composition, we first determine the composite of their single
variate function, and then use the [function-to-layout
Algorithm](#function-to-layout) to find the composition or to conclude that such
composition does not exist.

It turns out that by using only arguments with function composition, we can
derive the left distribution of composition with respect to concatenation. This
is a different approach compared to
[CuTE's original documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute),
which uses the left distribution property to *define* composition.

<div markdown="1" class="statement">

**Proposition.**

Composition is left-distributing with respect to concatentation. That is, for
three layouts $A$, $B$, $C$, assuming all compositions are feasible, we have:

$$
A \circ (B, C) = (A \circ B, A \circ C)
$$

</div>

<details markdown="1">
<summary><b>Proof.</b></summary>

We simply use the definition of [concatenation](#concatenation) and
[composition](#composition).

TODO: write the proof.
</details>

## Logical division

<div class="statement" id="logical-division-def" markdown="1">

**Definition 7. (Logical division)**

Let $A = N_a : S_a$ and $B = N_b : S_b$ be two layouts. When all the intermediate
operations are possible, we define their logical division to be:

$$
\text{LogicalDivision}(A, B)
  := A \circ \text{Concat}(B, \text{Complement}(B, \text{size}(A)))
$$

</div>

It is not easy to build an intuition about logical division.
[Jay's note](https://research.colfax-intl.com/a-note-on-the-algebra-of-cute-layouts)
perhaps offers better intuitions for this purpose. However, we find it very
assuring to reason about the logical division via $A$ and $B$'s singlevariate
functions.

In particular, since $A$ and $B$ are layouts:

- Their singlevariate functions can be easily defined.

- Then, $C = \text{Complement}(B, \text{size}(A))$ is either well-defined or
does not exist.

- Then, $\text{Concat}(B, C)$ is well-defined.

- Finally, $A \circ \text{Concat}(B, C)$ is either well-defined or does not exist.

This way of reasoning does not only offer an assuring mathemtical definition of
logical division, but also gives an algorithm to evaluate two layout's logical
division. While we cannot assert that this is the way layouts are implemented in
CuTe, we implement logical division operation in Python to help with studying
the concepts.


## Python implementation

Here, we provide the Python implementation for the basic layout operations.
We have only tested the code on very basic cases, so if you use it, please
proceed carefully.

<details markdown="1">
<summary>Python implementation of basic CuTe layout operations.</summary>

```python
r"""Simple experiments with CuTe layout."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class Layout:
    r"""A CuTe layout."""

    N: npt.ArrayLike
    S: npt.ArrayLike

    def __post_init__(self):
        self._cum_prod = np.concatenate([[1], np.cumprod(self.N[:-1])])

    def __repr__(self) -> str:
        n_str = ",".join([str(n) for n in self.N])
        s_str = ",".join([str(s) for s in self.S])
        return f"[layout] ({n_str}) : ({s_str})"

    def size(self) -> int:
        r"""The layout's size is the product of all its dimensions."""
        return np.prod(self.N)

    def single2multi(self, x: int) -> tuple[int, ...]:
        r"""1-D coordinate to multi coordinate."""
        return tuple(((x // self._cum_prod) % self.N).tolist())

    def multi2single(self, *args) -> int:
        r"""Multi coordinate to 1-D coordinate."""
        return np.dot(self._cum_prod, np.array(args))

    def f_single(self, x: int) -> int:
        r"""Evaluate the layout's singlevariate function."""
        return np.array(self.single2multi(x)).dot(self.S)


def find_layout(f: npt.ArrayLike) -> Layout | None:
    r"""Returns a layout admitting `f` or `None` if no such layout exists."""

    if f[0] != 0:
        return None

    m = np.size(f)
    s_0 = f[1]

    # this is the layout (m) : (s_0)
    m_s_0 = np.arange(0, m * s_0, s_0)

    # check if (n_0) : (s_0) is a solution. if yes, return
    if np.all(m_s_0 == f):
        return Layout(N=np.array([m]), S=np.array([s_0]))

    # unique index t s.t.: f[i] = i * s_0 for i = 0, ..., t-1 but f[t] != i * t
    t = np.where(m_s_0 != f)[0][0]

    for n_0 in range(t, 0, -1):
        # check if f is consistent with (n_0) : (s_0) as the first mode
        tgt_sz = (m + n_0 - 1) // n_0 * n_0
        pad_sz = tgt_sz - m
        pad_f = np.pad(f, [(0, pad_sz)]).reshape(-1, n_0).transpose()

        # row diff for all columns, except for the last one
        row_d = pad_f[1:, :-1] - pad_f[:-1, :-1]

        # row diff in the last column. need to remove the padded zeros
        last_f = pad_f[: m % n_0, -1]
        last_d = last_f[1:] - last_f[:-1]

        n_0_consistent = np.all(row_d[:, :-1] == s_0) and np.all(last_d == s_0)
        if n_0_consistent:
            r = find_layout(pad_f[0, :])
            if r is None:
                return None
            return Layout(N=np.concatenate([[n_0], r.N]),
                          S=np.concatenate([[s_0], r.S]))

    return None


def test_find_layout():
    """Ensure that find_layout finds the correct layout."""

    def _test_case(n: npt.ArrayLike, s: npt.ArrayLike):
        layout_1 = Layout(N=n, S=s)
        f = [layout_1.f_single(x) for x in range(layout_1.size())]
        layout_2 = find_layout(f)
        assert np.all(layout_1.N == layout_2.N), (
            f"{layout_1.N=} but {layout_2.N=}.")
        assert np.all(layout_1.S == layout_2.S), (
            f"{layout_1.S=} but {layout_2.S=}.")

    _test_case([3, 5, 7], [4, 9, 8])
    _test_case([3, 2], [10, 13])
    _test_case([9], [4])


def test_layout_function():
    """Simple opreations."""
    layout = Layout(N=[3, 4], S=[1, 3])

    assert layout.single2multi(5) == (2, 1)
    assert layout.multi2single(2, 1) == 5
    assert layout.multi2single(2, 3) == 11

    for x in range(layout.size()):
        assert x == layout.multi2single(*layout.single2multi(x))
        assert x == layout.f_single(x)


test_find_layout()
test_layout_function()

```
</details>  <!-- Python implementation -->
