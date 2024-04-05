---
layout: post
date: 2024-04-04
---

Quantizing an attention's KV cache
==================================

[[Hieu's personal blog index](./index)]

[work in progress]

In this post, I will discuss the quantization of an attention operation's KV
cache.  When computing attention on a along sequence, we typically store the
states of the previous steps into a KV cache. This KV cache is stored in its
own memory, which can potentially come with its own memory management logics.
This post focuses on quantizing the content stored in that memory.

# Background

## Attention
Given $q \in \mathbb{R}^{M \times D}$, $k, v \in \mathbb{R}^{N \times D}$, $W_o
\in \mathbb{R}^{H \times D}$, and a positive integer $H \in \mathbb{N}$,
typically called the *head dimension*, the attention operator computes:
$$
\begin{aligned}
q &:= Q \cdot W_q \in \mathbb{R}^{M \times H} \\
k &:= K \cdot W_k \in \mathbb{R}^{N \times H} \\
v &:= V \cdot W_v \in \mathbb{R}^{N \times H} \\
\text{Attention}(Q, K, V)
  &:= \text{Softmax}\left(
      \frac{q \cdot k^\top}{\sqrt{H}}
    \right)
    \cdot v \cdot W_o
\end{aligned}
$$

Note that the dimensions of the QKV projection matrices are:
$$
\begin{aligned}
W_q &\in \mathbb{R}^{D \times H} \\
W_k &\in \mathbb{R}^{D \times H} \\
W_v &\in \mathbb{R}^{D \times H} \\
W_o &\in \mathbb{R}^{H \times D}
\end{aligned}
$$

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
       \frac{Q W_q W_k^\top K}{\sqrt{H}}
     \right)
     \cdot V W_v \cdot W_o
\end{aligned}
$$

Using the associativity of matrix products, we can rewrite the above as:
$$
\begin{aligned}
\text{Attention}(Q, K, V)
  &= \text{softmax}\left(
       \frac{Q \cdot (W_q W_k^\top) \cdot K}{\sqrt{H}}
     \right)
     \cdot V \cdot (W_v W_o)
\end{aligned}
$$

At this point, we introduce the very nice idea of invariant linear projections.
If $X$ and $Y$ are orthogonal matrices of shape $H \times H$, that is:
$$
X X^\top = X^\top X = Y^\top Y = Y Y^\top = I
$$
then:
$$
\begin{aligned}
\text{Attention}(Q, K, V)
  &= \text{softmax}\left(
       \frac{Q \cdot (W_q W_k^\top) \cdot K}{\sqrt{H}}
     \right)
     \cdot V \cdot (W_v W_o) \\
  &= \text{softmax}\left(
       \frac{Q \cdot (W_q X X^\top W_k^\top) \cdot K}{\sqrt{H}}
     \right)
     \cdot V \cdot (W_v Y Y^\top W_o)
\end{aligned}
$$

This means our attention equation is *invariant* if we replace:
$$
\begin{aligned}
W_q &\to W_q \cdot X  \\
W_k &\to W_k \cdot X  \\
W_v &\to W_v \cdot Y  \\
W_o &\to Y^\top \cdot W_o
\end{aligned}
$$
