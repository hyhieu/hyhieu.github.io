---
layout: post
date: 2023-11-27
---

# On CuTe layouts

[[Hieu's personal blog index](./index)]

[CuTe](https://github.com/NVIDIA/cutlass/tree/main/include/cute) is a
sub-library in NVIDIA's [CUTLASS](https://github.com/nvidia/cutlass).

CuTe provides tool that make GPU programming flexible and generalizable across
different generations of GPU architectures, at least
[Ampere](https://www.nvidia.com/en-us/data-center/ampere-architecture/) and
[Hopper](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/).
Thanks to CuTe, the open-source deep learning community have developed effiicent
tools for deep learning, among which perhaps the most well-known is
[Flash-Attention](https://github.com/Dao-AILab/flash-attention).

For some backgrounds, GPU programming is notoriously inflexible and completely
specific to target architectures. As such, the flexibility and generalization
provided by CuTe is no less than a miracle. Such miracle stems from the
ingenious design of its central concept:
*layout.* A layout maps an $\alpha$-dimensional coordinate into an integral
offset. As simple as that, layouts are accompanied by *operations* that allow
users to flexibly and efficiently write performant tensor programs.

In this blog post, I attempt to formalize the definition of CuTe layouts and
their accompanying operations.

## Layout
<details markdown='1'>
<summary>Definition of layout and its invariants.</summary>

**Definition 1. (Layout)** Let $\alpha$ be a positive integer. A layout $L = S :
D$ is a pair of two tuples, each with $\alpha$ positive integers:
$$
\begin{align*}
S &= (s_0, s_1, ..., s_{\alpha-1}) \\
D &= (d_0, d_1, ..., d_{\alpha-1})
\end{align*}
$$
The layout $L$ represents a multivariable function
$$
\begin{align*}
g_L :~~ &[0, s_0) \times [0, s_1) \times \cdots \times [0, s_{\alpha - 1}] \to \mathbb{N} \\
        & (x_0, x_1, \dots, x_{\alpha-1}) \mapsto d_0 x_0 + d_1 x_1 + \cdots + d_{\alpha-1} x_{\alpha-1}
\end{align*}
$$


</details>
