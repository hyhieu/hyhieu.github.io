---
layout: post
date: 2024-04-04
---

Quantizing an attention's KV cache
==================================

[[Hieu's personal blog index](./index)]

[work in progress]

In this post, we discuss a potential approach to quantize the KV cache of the
attention operation. When computing attention over a long sequence, we
typically store the states of the previous steps into a KV cache.

This KV cache is stored in its own memory, typically accompanied by its own
memory management logics. This post focuses on quantizing the content stored in
this memory.

## Attention
Given $q \in \mathbb{R}^{M \times D}$, $k, v \in \mathbb{R}^{N \times D}$, $W_o
\in \mathbb{R}^{H \times D}$, and a positive integer $H \in \mathbb{N}$,
typically called the *head dimension*, the attention operator computes:
$$
\begin{aligned}
\text{Attention}(Q, K, V)
  &:= \text{Softmax}\left(
      \frac{q \cdot k^\top}{\sqrt{H}}
    \right)
    \cdot v \cdot W_o
\end{aligned}
$$
where:
$$
\begin{aligned}
q &:= Q \cdot W_q \in \mathbb{R}^{M \times H} \\
k &:= K \cdot W_k \in \mathbb{R}^{N \times H} \\
v &:= V \cdot W_v \in \mathbb{R}^{N \times H} \\
\end{aligned}
$$
Here, $Q$, $K$, $V$ are called the *inputs* to the attention operator, while $W_q$,
$W_k$, $W_v$, and $W_o$ are called the *weight matrices* of the operator. The dimensions
of these weight matrices are:
$$
\begin{aligned}
W_q &\in \mathbb{R}^{D \times H} \\
W_k &\in \mathbb{R}^{D \times H} \\
W_v &\in \mathbb{R}^{D \times H} \\
W_o &\in \mathbb{R}^{H \times D}
\end{aligned}
$$

Note also that in practice, the output of a so-called multi-head attention
operator is the sum of the above outputs computed on different *heads*, i.e.,
different versions of $W_q$, $W_k$, $W_v$, and $W_o$.

Oftentimes, we store the computed $k$ and $v$ from the previous steps into a KV
cache. These caches are the target of our quantization.

## Invariant linear projections
Let us rewrite the above attention operator as:
$$
\begin{aligned}
\text{Attention}(Q, K, V)
  &= \text{Softmax}\left(
       \frac{q \cdot k^\top}{\sqrt{H}}
     \right)
     \cdot v \cdot W_o \\
  &= \text{softmax}\left(
       \frac{(Q W_q) \cdot (KW_k)^\top}{\sqrt{H}}
     \right)
     \cdot V W_v \cdot W_o \\
  &= \text{softmax}\left(
       \frac{Q W_q W_k^\top K^\top}{\sqrt{H}}
     \right)
     \cdot V W_v \cdot W_o
\end{aligned}
$$

Using the associativity of matrix products, we can rewrite the above as:
$$
\begin{aligned}
\text{Attention}(Q, K, V)
  &= \text{softmax}\left(
       \frac{Q \cdot (W_q W_k^\top) \cdot K^\top}{\sqrt{H}}
     \right)
     \cdot V \cdot (W_v W_o)
\end{aligned}
$$

We are now ready to introduce the idea of invariant linear projections.
Let $X$ and $Y$ be invertible matrices of shape $H \times H$, then:
$$
\begin{aligned}
\text{Attention}(Q, K, V)
  &= \text{softmax}\left(
       \frac{Q \cdot (W_q W_k^\top) \cdot K^\top}{\sqrt{H}}
     \right)
     \cdot V \cdot (W_v W_o) \\
  &= \text{softmax}\left(
       \frac{Q \cdot (W_q X X^{-1} W_k^\top) \cdot K^\top}{\sqrt{H}}
     \right)
     \cdot V \cdot (W_v Y Y^{-1} W_o) \\
  &= \text{softmax}\left(
       \frac{(Q \cdot W_q X) (X^{-1} W_k^\top \cdot K^\top)}{\sqrt{H}}
     \right)
     \cdot V \cdot (W_v Y Y^{-1} W_o) \\
  &= \text{softmax}\left(
       \frac{(Q \cdot W_q X) (K \cdot W_k (X^{-1})^\top)^\top}{\sqrt{H}}
     \right)
     \cdot ( V \cdot W_v Y) \cdot (Y^{-1} W_o) \\
\end{aligned}
$$

This means our attention equation is *invariant* if we replace the weight
matrices as follows:
$$
\begin{aligned}
W_q &\to W_q \cdot X  \\
W_k &\to W_k \cdot (X^{-1})^\top  \\
W_v &\to W_v \cdot Y  \\
W_o &\to Y^{-1} \cdot W_o
\end{aligned}
$$

We note that we can perform these replacements on the weight matrices alone, while
keeping whatever $Q$, $K$, and $V$ inputs passed into this attention operation.

The gist of the KV cache quantization procedure, which we will discuss in the
next section, is about finding the matrices $X$ and $Y$.

## Finding the quantizer matrices
We will find $X$ and $Y$ using gradient descent on a certain data distribution.
There are two challenges with this approach.  The first challenge is that
gradient descent does not guarantee that $X$ and $Y$ are invertible throughtout
the iterations. To ensure invertability, we will use the [Sherman-Morrison
identity](https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula).

The Sherman-Morrison identity states that if $A \in \mathbb{R}^{n
\times n}$ is invertible, then for
*any* two vectors $u, v \in \mathbb{n}$ such that $1 + v^\top A^{-1} u \neq 0$,
the matrix $A + uv^{\top}$ is also invertible, and its inverse is given by:

$$
\left( A + uv^\top \right)^{-1}
= A^{-1} - \dfrac{A^{-1} u ^\top A^{-1}}{1 + v^\top A^{-1} u}
$$

This identity will be our key tool to find the quantizer matrices $X$ and $Y$.
In particular, we can pick a value of $A$, such as $A = I_{n}$, the $n \times n$
identity matrix, and find $X$ that minimizes an objective function which allows
us to convert $K W_k(X^{-1})^\top$ from a high-bit format such as FP16 or BF16
into a low-bit format such as INT4 as losslessly as possible.

TODO(hieu): What does "lossless" mean here?

First, we generate $Q \in \mathbb{R}^{M \times N}$ and $K, V \in \mathbb{R}^{N
\times H}$ by running the model we want to quantize on some sample input data.
We suspect that similar to when we use data for calibration for other quantization
techniques, a small number of samples, such as 1000, would suffice. After running
the model through the data, we perform the transformations

## How about RoPE?

Thus far, we discussed quantizing the KV caches assuming that `q, k, v` are
projected using $W_q, W_k, W_v$ only. In reality, the $q$ and $k$ projections
are typically followed by a RoPE operation. This turns the original projections
into:
$$
\begin{aligned}
q_m &:= Q[m, :] \cdot W_q \cdot R_{m} \in \mathbb{R}^{M \times H} \\
k_n &:= K[n, :] \cdot W_k \cdot R_{n} \in \mathbb{R}^{N \times H}
\end{aligned}
$$
where $R_{j} \in \mathbb{R}^{H \times H}$ are the block-diagonal rotational
matrices defined by:
$$
R_{j} =
\begin{bmatrix}
  \cos{j \theta_1} & -\sin{j \theta_1} & 0 & 0 & \cdots & 0 & 0 \\
  \sin{j \theta_1} &  \cos{j \theta_1} & 0 & 0 & \cdots & 0 & 0 \\
  0 & 0 & \cos{j \theta_2} & -\sin{j \theta_2} & \cdots & 0 & 0 \\
  0 & 0 & \sin{j \theta_2} &  \cos{j \theta_2} & \cdots & 0 & 0 \\
  \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots  \\
  0 & 0 & 0 & 0 & \cdots & \cos{j \theta_{H/2}} & -\sin{j \theta_{H/2}} \\
  0 & 0 & 0 & 0 & \cdots & \sin{j \theta_{H/2}} &  \cos{j \theta_{H/2}}
\end{bmatrix}
\in \mathbb{R}^{H \times H}
$$

Let $X, Y$ be the invertible matrices that we use to quantize the KV caches.
Then, we can rewrite the above as:
$$
\begin{aligned}
  q_m k_n^\top
  &= \Big( Q_m \cdot W_q \cdot R_{m} \Big) \Big( K_n \cdot W_k \cdot R_{n} \Big)^\top \\
  &= Q_m W_q \cdot \underbrace{R_{m} R_n^\top}_{D_{m-n}} \cdot W_k^\top K_n \\
  &= Q_m \cdot W_q X \cdot \underbrace{R_{m} R_n^\top}_{D_{m-n}} \cdot X^{-1} W_k^\top \cdot K_n \\
\end{aligned}
$$

Let us look at the product $R_{m} R_n^\top$. Every diagonal block of $R_{m}
R_n^\top$ is a rotation matrix. We can write it as:
$$
\begin{aligned}
\begin{bmatrix}
  \cos{m \theta_1} & -\sin{m \theta_1} \\
  \sin{m \theta_1} &  \cos{m \theta_1} \\
\end{bmatrix}
\cdot
\begin{bmatrix}
  \cos{n \theta_1} & -\sin{n \theta_1} \\
  \sin{n \theta_1} &  \cos{n \theta_1} \\
\end{bmatrix}^\top
&=
\begin{bmatrix}
  \cos{m \theta_1} & -\sin{m \theta_1} \\
  \sin{m \theta_1} &  \cos{m \theta_1} \\
\end{bmatrix}
\cdot
\begin{bmatrix}
   \cos{n \theta_1} & \sin{n \theta_1} \\
  -\sin{n \theta_1} & \cos{n \theta_1}
\end{bmatrix} \\
&=
\begin{bmatrix}
  \cos{(m-n) \theta_1} & -\sin{(m-n) \theta_1} \\
  \sin{(m-n) \theta_1} &  \cos{(m-n) \theta_1}
\end{bmatrix}
\end{aligned}
$$