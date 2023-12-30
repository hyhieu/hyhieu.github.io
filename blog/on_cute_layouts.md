---
layout: post
date: 2023-12-25
use_math: true
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

## Basic definitions and properties

### Layout

<blockquote markdown="1" id="layout-def" >

**Definition 1. (Layout)** Let $D$ be a positive integer. A layout $L = N :
S$ is a pair of tuples, each with $D$ positive integers:

$$
\begin{aligned}
N &= (n_0, n_1, ..., n_{D-1}) \\
S &= (s_0, s_1, ..., s_{D-1})
\end{aligned}
$$

The tuple $N$ is called the layout's *size,* while the tuple $S$ is called the
layout's *stride.* Additionally, each tuple $(n_i, s_i)$ for $i \in \{0, 1,
\dots, D-1\}$ is called a *mode* of $L$'s.

</blockquote>

### Canonical function

#### Canonical multivariate function
A layout $L$ represents a multivariable function $g_L : [0, n_0) \times [0,
n_1) \times \cdots \times [0, n_{D - 1}) \subseteq \mathbb{N}^{D} \to
\mathbb{N}$, defined by:

$$
g_L(x_0, x_1, \dots, x_{\alpha-1}) := n_0 \cdot x_0 + n_1 \cdot x_1 + \cdots + n_{D-1} \cdot x_{D-1}
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

It is not hard to check that the function $\text{Multi}\to\text{Single}$ as
defined above is bijective. Then, $\text{Single}\to\text{Multi}$ is just the
invert from the other direction.

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

<details markdown="1">
<summary><b>Digression:</b> column-major vs. row-major.</summary>

The way we define the singlevariate function of a layout corresponds to how we
traverse the layout's $D$-dimensional coordinate space from left to right. This
traversal is sometimes called the *column-major* traversal. Column-major
traversal is used in MATLAB and Fortran. In contrast, most modern deep learning
frameworks like `numpy`, `torch`, and `jax` use row-major traversal. It is
possible to redefine the entire theory on layouts using row-major traversal, but
we choose to follow CuTe's original choice of being column-major.

</details>

#### The correspondence between layouts and canonical singlevariate functions

<hr>

##### What function $f: \mathbb{N} \to \mathbb{N}$ can be represented by a layout?

Let $f: \{0, 1, ..., M\} \to \mathbb{N}$ be a function. We determine whether
there exists a layout $L = (n_0, n_1, \dots , n_{D-1}) : (s_0, s_1, \dots,
s_{D-1})$ such that $L(x) = f(x)$ for all $x \in \{0, 1, \dots, M\}$.

We first notice that if $n_i = 1$ for an index $i \in \{0, 1, \dots, D-1\}$,
then for all $x \in \mathbb{N}$, the $i$-th coordinate of $x$ in $L$'s
coordinate space is:

$$
\left\lfloor \dfrac{x}{n_0 n_1 \cdots n_{i-1}} \right\rfloor~\text{mod}~1 = 0
$$

This means that $s_i$ never contributes to the value of $L(x)$, and hence can
take any value. For this reason, we call the dimensions where $n_i = 1$ trivial.
To avoid such trivial dimensions, we can assume that $\boxed{n_i > 1}$ for all
$i \in \{0, 1, \cdots, D-1\}$.

We try to identify $L(x)$ with each $f(x)$. To this end, we write down the
formula for $L(x)$:

<div id="l-formula"></div>

$$
\begin{aligned}
L(x)
  &= \left(
    x~\text{mod}~n_0,
    \left\lfloor \frac{x}{n_0} \right\rfloor~\text{mod}~n_1,
    \left\lfloor \frac{x}{n_0 n_1} \right\rfloor~\text{mod}~n_2,
    \dots,
    \left\lfloor \frac{x}{n_0 n_1 \cdots n_{D-2}} \right\rfloor~\text{mod}~n_{D-1}
  \right)^\top \cdot (s_0, s_1, \dots, s_{D-1}) \\
  &= \sum_{i=0}^{D-1} s_i \cdot \left(\left\lfloor \frac{x}{n_0 n_1 \cdots n_{i-1}} \right\rfloor~\text{mod}~n_i\right)
\end{aligned}
$$

From this formula, we necessarily have $L(0) = 0$, so if $f(x) \neq 0$, there
exists no layout admitting $f$ as its singlevariate function.

Next, we compute $L(1)$. Thanks to the assumption that $n_i > 1$ for all $i$'s,
we have:

$$
\left\lfloor \dfrac{1}{n_0 n_1 \cdots n_{i-1}} \right\rfloor~\text{mod}~n_i
  = \begin{cases}
  1 & \text{if $i = 0$} \\
  0 & \text{if $i > 0$}
  \end{cases}
$$

Identifying this with [the formula for $L$](#l-formula), we have $\boxed{s_0 =
L(1) = f(1)}$. At this point, we can check if the layout $L_0 = (M) :
(s_0)$ satisfies $L_0(x) = f(x)$ for all $x \in [0, M)$.

If not, then there exists a unique index $t$ such that $f(k) = k \cdot f(1) = k
s_0$ for $k \in \{0, 1, \dots, t-1\}$ but $f(t) \neq t \cdot f(1)$. We now know
that $n_0 \leq t$.

Now we can try each value $k$ from $t$ downto $1$ to see if $n_0 = k, s_0 =
f(1)$ is consistent with the other values $f(x)$ for $x \in \{k, k+1, k+2,
\dots, M\}$.  Here, consistency means that $f(x + i k) = f(x) + i s_0$, for all
$x \in [0, M]$ and $i \in \mathbb{N}$ such that $x + i k \in [0, M]$. If no such
$k$ is found, we say that the function $f$ is *inconsistent*, i.e., there is no
layout admitting $f$ as its singlevariate function. Otherwise, we repeat the
process on the function to find $(n_1, s_1)$:

$$
g : \left[ 0, \left\lfloor M / n_0 \right\rfloor \right] \to \mathbb{N}~~~~~~~~~g(x) := f(n_0 x)
$$

Essentially, this means to restrict $f$ into the sub-domain where the $0$-th
coordinate is $0$.

Let us analyze the complexity of the process above:

1. $O(1)$ Find $s_0 = f(1)$:

2. $O(M)$ Checking whether $(M): (s_0)$ is okay.

<p id="step-3"></P>

3. $O(M)$ For each $k \in \{t, t-1, \dots, 1\}$:  $O(M)$

    3a. $O(M)$ Check if $f(x + ik) - f(x) = i s_0$ for all $x \in [0, M], i \in \mathbb{N}$ such that $x + ik \in [0, M]$

Thus, each value $(n_i, s_i)$ can be determined in $O(M^2), or an inconsistency
is found. Since there are $O(M)$ modes, the process above offers a deterministic
algorithm with complexity $\boxed{O(M^3)}$ to check whether there exists a layout
admitting *any* function $f: \mathbb{N} \to \mathbb{N}$ as its layout function.
If such a layout exists, the algorithm also determines the layout.

<mark markdown="1">**QUESTION:** In <a href="#step-3">Step 3</a> above,
how can we be sure that if any value
for $k$ is consistent, then the resulting layout must admit that $k$ as $n_0$? I can see
the argument for any other $l'$ such that $l~|~k$, but how about just any other $l$?
For instance, if we find an inconsistency later on by picking a value $k$, can we
tell $f$ is inconsistent?
</mark>







<hr>

##### Are layout representations unique?

In general, multiple layouts might represent the same singlevariate function.

<details markdown="1">
<summary><b>Example:</b> multiple layouts associated to the same function.</summary>

The two layouts $A = (10) : (3)$ and $B = (2, 5) : (3, 6)$ share the same
function: $f_A(x) = f_B(x) = 3x$  for all $x \in \{0, 1, \dots 9 \}$.

</details>

If two layouts have the same associated function, we say that they are
*equivalent*. This equivalence partitions the set of all layouts into equivalent
classes. In the next sections, when we discuss certain types of uniqueness for
layouts, we mean uniqueness upto the equivalence via a layout's canonical
singlevariate function.


## Complement

<blockquote id="complement-def" markdown="1">

**Definition 2. (Complement)**
Let $A = (N_a) : (D_a)$ be a layout.  For an integer $M$ that is divisible by
$\text{size}(A) = n_0 n_1 \cdots n_{\alpha-1}$, the *complement of $A$ with
respect to $M$*, denoted by $C(A, M)$, is the layout $B$ that satisfies two
conditions:
1. The associated layout function $f_B$ is strictly increasing.
2. The concatenation layout $(A, B)$ is a bijection $[0, M) \to [0, M)$.

</blockquote>

There are some ground-laying work to ensure that [Definition 2](#complement-def) works.

<blockquote id="complement-exist" markdown="1">

**Lemma 2.1.** Let $A$ be an $D$-dimensional layout, then the followings are equivalent:

1. Let $\sigma$ sorts $\{(n_0, s_0), (n_1, s_1), \dots, (n_{D-1}, s_{D-1})\}$
first by $d$ and then by $n$.
That is, $\sigma$ is the permutation of $\{0, 1, \dots, d-1\}$ such that for $0
\leq i < j \leq d-1$, we have $s_{\sigma(i)} \leq s_{\sigma(j)}$ and if
$s_{\sigma(i)} = s_{\sigma(j)}$ then $n_{\sigma(i)} \leq n_{\sigma(j)}$.
Then $n_{\sigma(i)} s_{\sigma(i)}~|~s_{\sigma(j)}$ for all $0 \leq i < j \leq D-1$.

2. $C(A, M)$ exists for *all* positive integers $M$ divisible by $\text{size}(A)$.

</blockquote>





<details markdown="1">
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