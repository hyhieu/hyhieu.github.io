---
layout: post
date: 2024-04-04
---

Quantizing an attention's KV cache
==================================

[[Hieu's personal blog index](./index)]

[work in progress]

In this post, I will discuss an approach to quantize an attention operation's KV
cache.  When computing attention on a along sequence, we typically store the
states of the previous steps into a KV cache. This KV cache is stored in its own
memory, which can potentially come with its own memory management logics.  This
post focuses on quantizing the content stored in that memory.

# Background

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
Let $X$ and $Y$ be orthogonal matrices of shape $H \times H$, that is:
$$
X X^\top = X^\top X = Y^\top Y = Y Y^\top = I
$$
then:
$$
\begin{aligned}
\text{Attention}(Q, K, V)
  &= \text{softmax}\left(
       \frac{Q \cdot (W_q W_k^\top) \cdot K^\top}{\sqrt{H}}
     \right)
     \cdot V \cdot (W_v W_o) \\
  &= \text{softmax}\left(
       \frac{Q \cdot (W_q X X^\top W_k^\top) \cdot K^\top}{\sqrt{H}}
     \right)
     \cdot V \cdot (W_v Y Y^\top W_o)
\end{aligned}
$$

This means our attention equation is *invariant* if we replace the weight
matrices as follows:
$$
\begin{aligned}
W_q &\to W_q \cdot X  \\
W_k &\to W_k \cdot X  \\
W_v &\to W_v \cdot Y  \\
W_o &\to Y^\top \cdot W_o
\end{aligned}
$$

We note that we can perform these replacements on the weight matrices alone, while
keeping whatever $Q$, $K$, and $V$ inputs that we receive.
