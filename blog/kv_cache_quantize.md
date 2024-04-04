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
Given $q \in \mathbb{R}^{M \times D}$, $k, v \in \mathbb{R}^{N \times D}$, and a
positive integer $H \in \mathbb{N}$, typically called thea *head dimension*,
the attention operator computes:
$$
\begin{aligned}
q &:= Q \cdot W_q \in \mathbb{R}^{M \times H} \\
k &:= K \cdot W_k \in \mathbb{R}^{N \times H} \\
v &:= V \cdot W_v \in \mathbb{R}^{N \times H} \\
\text{Attention}(Q, K, V)
  &:= \text{softmax}\left(
      \frac{q \cdot k^\top}{\sqrt{H}}
    \right)
    \cdot v
\end{aligned}
$$
Oftentimes, we store the computed $k$ and $v$ from the previous steps into a KV
cache.
