---
layout: post
date: 2023-11-27
---

# On CuTe layouts

**[work in progress. many citations missing, and many writings here could be wrong.]**

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

In this blog post, I attempt to formalize the definition of CuTe layout and its
accompanying operations.

## Layout

### Layout
<div id="layout-def"></div>

>**Definition 1. (Layout)** Let $D$ be a positive integer. A layout $L = N :
S$ is a pair of tuples, each with $D$ positive integers:
>$$
\begin{align*}
N &= (n_0, n_1, ..., n_{D-1}) \\
S &= (s_0, s_1, ..., s_{D-1})
\end{align*}
>$$
> The tuple $N$ is called the layout's *size,* while the tuple $S$ is called the
layout's *stride.* Additionally, each tuple $(n_i, s_i)$ for $i \in \{0, 1,
\dots, D-1\}$ is called a *mode* of $L$'s.

### Canonical function

#### Canonical multivariate function
A layout $L$ represents a multivariable function $g_L : [0, s_0) \times [0,
s_1) \times \cdots \times [0, s_{D - 1}) \subseteq \mathbb{N}^{D} \to
\mathbb{N}$, defined by:
$$
g_L(x_0, x_1, \dots, x_{\alpha-1}) := s_0 \cdot x_0 + s_1 \cdot x_1 + \cdots + s_{D-1} \cdot x_{D-1}
$$
We call $g_L$ the *canonical multivariate function* of $L$.

Throughout this note, for brevity, we will drop the word "canonical" when its
meaning is clear from the context.

#### Canonical singlevariate function
Other than the canonical multivariate function, we are also interested in the
*canonical singlevariate function* of a layout. This singlevariate function is
constructed from the layout's multivariate function via the natural isomorphism
between $[0, n_0 n_1 \cdots n_{D-1})$ and $[0, n_0) \times [0, n_1) \times
\cdots \times [0, n_{D - 1})$, which we can define via the canonical
multivariate function of the layout:
$$
\text{Multi}\to\text{Single}
    = (n_0, n_1, n_2, \dots, n_{D-1}) :
      (1, n_0, n_0 n_1, n_0 n_1 n_2, \dots, n_0 n_1 \cdots n_{D-2})
$$
Of course, $\text{Single}\to\text{Multi}$ is just the inverse of the above.

A self-contained formula for $\text{Multi}\to\text{Single}$ is:
$$
\begin{aligned}
\text{Multi}\to\text{Single}(x_0, x_1, \dots, x_{D-1})
  &:= x_0
    + n_0 \cdot x_1
    + n_0 n_1 \cdot x_2
    + n_0 n_1 n_2 \cdot x_3
    + \cdots
    + n_0 n_1 \cdots n_{D-2} \cdot x_{D-1} \\
\text{Single}\to\text{Multi}(x)
  &:= \left(
    x~\text{mod}~n_0,
    \left\lfloor \frac{x}{n_0} \right\rfloor~\text{mod}~n_1,
    \left\lfloor \frac{x}{n_0 n_1} \right\rfloor~\text{mod}~n_2,
    \dots,
    \left\lfloor \frac{x}{n_0 n_1 \cdots n_{D-2}} \right\rfloor~\text{mod}~n_{D-1}
  \right)
\end{aligned}
$$
<details markdown='1'>
<summary><b>Digression:</b> column-major vs. row-major.</summary>

The way we define the singlevariate function of a layout corresponds to how we
traverse the layout's $D$-dimensional coordinate space from left to right. This
traversal is sometimes called the *column-major* traversal. Column-major
traversal is used in MATLAB and Fortran. In contrast, most modern deep learning
framework like `numpy`, `torch`, and `jax` ise the row-major traversal. It is
possible to redefine the entire theory on layouts using row-major traversal, but
we choose to follow CuTe's original choice of being column-major.

</details>

#### The correspondence between layouts and canonical singlevariate functions
In general, multiple layouts might be associated to the same canonical function.
If two layouts have the same associated function, we say that they are
*equivalent*. This equivalence partitions the set of all layouts into equivalent
classes. In the next sections, when we discuss certain types of uniqueness for
layouts, we mean uniqueness upto the equivalence via a layout's canonical
singlevariate function.

<details markdown='1'>

<summary><b>Example:</b> multiple layouts associated to the same function.</summary>

The two layouts $A = (10) : (3)$ and $B = (2, 5) : (3, 6)$ share the same
function: $f_A(x) = f_B(x) = 3x$  for all $x \in \{0, 1, \dots 9 \}$

</details>


## Complement

<div id="complement-def"></div>

> **Definition 2. (Complement)**
Let $A = (N_a) : (D_a)$ be a layout.  For an integer $M$ that is divisible by
$\text{size}(A) = n_0 n_1 \cdots n_{\alpha-1}$, the *complement of $A$ with
respect to $M$*, denoted by $C(A, M)$, is the layout $B$ that satisfies two
conditions:
> 1. The associated layout function $f_B$ is strictly increasing.
> 2. The concatenation layout $(A, B)$ is a bijection $[0, M) \to [0, M)$.

There are some ground-laying work to ensure that [Definition 2](#complement-def) works.

<div id="complement-exist"></div>

> **Lemma 2.1.** Let $A$ be an $D$-dimensional layout, then the followings are equivalent:
>
> 1. Let $\sigma$ sorts $\{(n_0, s_0), (n_1, s_1), \dots, (n_{D-1}, s_{D-1})\}$
first by $d$ and then by $n$.
That is, $\sigma$ is the permutation of $\{0, 1, \dots, d-1\}$ such that for $0
\leq i < j \leq d-1$, we have $s_{\sigma(i)} \leq s_{\sigma(j)}$ and if
$s_{\sigma(i)} = s_{\sigma(j)}$ then $n_{\sigma(i)} \leq n_{\sigma(j)}$.
Then $n_{\sigma(i)} s_{\sigma(i)}~|~s_{\sigma(j)}$ for all $0 \leq i < j \leq D-1$.
>
> 2. $C(A, M)$ exists for *all* positive integers $M$ divisible by $\text{size}(A)$.





<details markdown='1'>
<br><br><br><br><br><br><br><br><br><br><br><br>

Consider the layout $A = (4) : (3)$ which maps $(0, 1, 2, 3) \mapsto (0, 3, 6, 9)$.
We will try to determine the complement $B := \text{Complement}(A, 24)$.

To that end, we want to find a layout $B = (m, n) : (e, f)$ such that the
concatenation $(A, B)$ maps $[0, 24) \mapsto [0, 24)$ bijectively.

Equivalently:
$$
\begin{aligned}
&~~~~~~~~~(4, m, n) : (3, e, f)~\text{maps}~[0, 24) \mapsto [0, 24) \\
&\Longrightarrow (4, 3, 2) : (3, 1, 12)~\text{maps}~[0, 24) \mapsto [0, 24) \\
\end{aligned}
$$

Therefore $B = (3, 2) : (1, 12)$.

In fact, more generally, we have $\text{complement}(A, 12k) = (3, 2k) : (1, 12)$ for
each positive integer $k$.

Maybe another more general rule is that
$\boxed{ \text{complement}\big( (n) : (d), knd \big) = (d, k) : (1, nd) }$.
</details>