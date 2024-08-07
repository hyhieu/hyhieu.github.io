---
layout: post
date: 2024-07-09
---

Understanding and Implementing Multi-head Latent Attention (MLA)
================================================================

[[Hieu's personal blog index](./index)]

Multi-head Latent Attention (MLA) is introduced in
[DeepSeek-V2](https://arxiv.org/abs/2405.04434) as a method to save the memory
cache size during inference. Despite its benefits, MLA is not well-understood
and existing open-sourced implementations are disastrous. In this post, we
discuss the MLA algorithm and propose a potential implementation.

## Multi-head Latent Attention (MLA)

At each step $t$ in a sequence, let $h_t \in \mathbb{R}^{1 \times d}$ be the
hidden state of the model at the step.

Note that we are using the row-major convention, which is also used in most ML
frameworks, in contrast to the column-major convention used in the DeepSeek-V2
paper.

### KV compression and decompression
Unlike existing attention approaches, the
MLA algorithm projects $h_t$ into a compressed state:

$$
c_t = h_t \cdot W_{\text{Down}KV} \in \mathbb{R}^{1 \times d_c}
$$

When the algorithm actually needs the KV caches, it projects the cached states back to
the original space *for each MLA head*:

$$
\begin{aligned}
k_{t, i}^{C} &= c_t \cdot W_{\text{Up}K, i} \in \mathbb{R}^{1 \times d} \\
v_{t, i}^{C} &= c_t \cdot W_{\text{Up}V, i} \in \mathbb{R}^{1 \times d}
\end{aligned}
$$

Elegantly, the above projections can be completely avoided during inference
time! To understand why it is the case, we need to study the attention equation.
For each query $q$ and head $i$-th, the attention equation can be written as
follows:

$$
\begin{aligned}
\text{Attention}(q, k_{t, i}, v_{t, i})
  &= \text{Softmax}\left(
    \frac{q \cdot W_{Q, i} \cdot k_{t, i}^\top}{\sqrt{d_h}}
  \right)
  \cdot
  v_{t, i} \cdot W_{V, i} \\
  &= \text{Softmax}\Big(
    \frac{q \cdot \overbrace{W_{Q, i} \cdot W_{\text{Up}K, i}^\top}^{\tilde{W}_{Q, i}} \cdot c_{t}^\top}{\sqrt{d_h}}
  \Big)
  \cdot
  c_{t} \cdot \underbrace{W_{\text{Up}V, i} \cdot W_{V, i}}_{\tilde{W}_{V, i}}
\end{aligned}
$$

This means that if we define:

$$
\begin{aligned}
\tilde{W}_{Q, i} &:= W_{Q, i} \cdot W_{\text{Up}K, i}^\top \\
\tilde{W}_{V, i} &:= W_{\text{Up}V, i} \cdot W_{V, i}
\end{aligned}
$$

Then we can compute the attention equation without ever realizing the
decompressed KV caches. In fact, the resulting attention equation is even
compatible with FlashAttention:

$$
\begin{aligned}
\text{Attention}(q, k_{t, i}, v_{t, i})
  &= \text{Softmax}\left(
    \frac{q \cdot \tilde{W}_{Q, i} \cdot k_{t, i}^\top}{\sqrt{d_h}}
  \right)
  \cdot
  v_{t, i} \cdot \tilde{W}_{V, i}
\end{aligned}
$$

Amazing, but not with a serious problem coming next!

### RoPE
Attentions need RoPE to encode positional information, i.e., to tell for
each token what is its index in its sequence. The problem is that RoPE depends on $t$:
RoPE works by inserting a matrix $R_t$ into the Softmax step, so that:

$$
W_{Q, i} \cdot W_{\text{Up}K, i}^\top
$$

becomes:
$$
W_{Q, i} \cdot R_t \cdot W_{\text{Up}K, i}^\top
$$

This prevents MLA from folding $W_{Q, i}$ and $W_{\text{Up}K, i}^\top$ into a
single $\tilde{W}_{Q, i}$ matrix. Look, there is no $t$ in the subscript of
$\tilde{W}_{Q, i}$.

The MLA solution is to create one extra head that carries the RoPE projections.
That is, for each head $i$-th:

$$
\begin{aligned}
q^{\text{rope}}_{t, i} &:= h_t \cdot W^\text{rope}_{Q, i} \cdot R_t \\
k^{\text{rope}}_t &:= h_t \cdot W^\text{rope}_{K} \cdot R_t
\end{aligned}
$$

Note that the RoPE operation happens in the MQA style, i.e., each query head has
its own projection with RoPE, but all key heads share the same projection.

These $q^{\text{rope}}_{t, i}$ and $k_t^{\text{rope}}$ are then appended to $q$ and
$k_{t, i}$ from the previous section, before the attention is executed. In equations,
for each head $i$-th:

$$
\begin{aligned}
q^\text{full}_{t, i}
  &:= \Big[ q_t, q^\text{rope}_{t, i} \Big] \\
k^\text{full}_{t, i}
  &:= \Big[ k_{t, i}, k^\text{rope}_{t, i} \Big] \\
\text{Attention}(q_{t}, k_{t, i}, v_{t, i})
  &:= \text{Attention}(q^\text{full}_{t}, k^\text{full}_{t, i}, v_{t, i})
\end{aligned}
$$

Overall, an MLA KV cache of for $L$ past tokens only needs to hold
$(d_c + d^\text{rope}_h) \cdot L$ numbers, where $d^\text{rope}_h$ is the
dimension of the RoPE projection.

Sounds good. But implementing this can cause a disaster: there is no fast
attention kernel supporting that MLA-MQA-concatenated style attention.
