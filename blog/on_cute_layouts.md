---
layout: post
date: 2023-12-31
---

# On CuTe layouts

**[work in progress. many citations missing, and many writings here could be wrong.]**

[[Hieu's personal blog index](./index)]

[CuTe](https://github.com/NVIDIA/cutlass/tree/main/include/cute) is a
sub-library in NVIDIA's [CUTLASS](https://github.com/nvidia/cutlass).

CuTe makes GPU programming flexible and generalizable across different
generations of GPU architectures, at least
[Ampere](https://www.nvidia.com/en-us/data-center/ampere-architecture/) and
[Hopper](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/).
Thanks to CuTe, the open-source deep learning community have developed effiicent
tools for deep learning, among which perhaps the most well-known is
[Flash-Attention](https://github.com/Dao-AILab/flash-attention).

For some backgrounds, GPU programming is notoriously inflexible and completely
specific to target architectures. As such, the flexibility and generalization
provided by CuTe is no less than a miracle. Such miracle stems from the
ingenious design of its central concept:
*layout.* A layout maps an $D$-dimensional coordinate into an integral
offset. As simple as that, layouts are accompanied by *operations* that allow
users to flexibly and efficiently write performant tensor programs.

In this blog post, we attempt to formalize the definition of CuTe layout and its
accompanying operations: [complementation](#complemention) and
[composition](#composition).

Our idea is to think of layouts as a way to represent function $f: \mathbb{N}
\to \mathbb{N}$ that maps each integer to a multilinear function's output
defined by the layout.

We build a correspondence between layouts and these functions via an algorithm,
and then use this correspondence to construct layout operations such as
[complementation](#complemention) and [composition](#composition).

## Basic definitions and properties

### Layout

<div class="statement" markdown="1" id="layout-def">

**Definition 1.1. (Layout)** Let $D$ be a positive integer. A layout $L = N :
S$ is a pair of tuples, each with $D$ positive integers:

$$
\begin{aligned}
N &= (n_0, n_1, ..., n_{D-1}) \\
S &= (s_0, s_1, ..., s_{D-1})
\end{aligned}
$$

</div>

There are some terminologies associated with the definition of layout:

- The product of all elements in the tuple $N$ is called the layout's *size.*

- The tuple $S$ is called the layout's *stride.*

- The maximum offset that the layout can represent,
i.e., $\sum_{i=0}^{D-1}s_i (n_i - 1)$, is the layout's *cosize.*

- The pair $(n_i, s_i)$, sometimes written $(n_i) : (s_i)$, is called the
$i^\text{th}$ *mode* of $L$.

In this note, we are not very concerned with the goal of layout objects, much
rather with the mathemtical constructions of layouts and the operations on them.
However, the summary of layouts is that they are used to represent
*offsets* of elements in CuTe tensors (that is, how far is each element from the
tensor's first element in the GPU memory). In particular, the size tuple $N$
represents the shapes of a tensor, while the stride tuple $S$ represents the
strides in each mode of the shape.

In the next section, we will study the *canonical functions* of layouts, which
represent how to map certain coordinate representations to offsets in layouts.

\# TODO: maybe write about the extension of layouts (i.e., the last mode become $\infty$).

### Canonical functions

<div class="statement" markdown="1">

**Definition 2. (Canonical multivariate function)**
A layout $L$ represents a multivariable function $g_L : [0, n_0) \times [0,
n_1) \times \cdots \times [0, n_{D - 1}) \subseteq \mathbb{N}^{D} \to
\mathbb{N}$, defined by:

$$
g_L(x_0, x_1, ..., x_{D-1}) := s_0 \cdot x_0 + s_1 \cdot x_1 + \cdots + s_{D-1} \cdot x_{D-1}
$$

We call $g_L$ the *canonical multivariate function* of $L$.

</div>

When clear from context, we will drop the word "canonical" for brevity. We might
also use the overloaded notation:

$$
L(x_0, x_1, ..., x_{D-1}) := s_0 \cdot x_0 + s_1 \cdot x_1 + \cdots + s_{D-1} \cdot x_{D-1}
$$

Other than the canonical multivariate function, we are also interested in the
*canonical singlevariate function* of a layout. This singlevariate function is
constructed from the layout's multivariate function via the natural isomorphism
between $[0, n_0 n_1 \cdots n_{D-1})$ and $[0, n_0) \times [0, n_1) \times
\cdots \times [0, n_{D - 1})$, which we can define via the canonical
multivariate function of the layout:

$$
\text{Multi}\to\text{Single}
    = (n_0, n_1, n_2, ..., n_{D-1}) :
      (1, n_0, n_0 n_1, n_0 n_1 n_2, ..., n_0 n_1 \cdots n_{D-2})
$$

It is not hard to check that the function $\text{Multi}\to\text{Single}$ as
defined above is bijective. Then, $\text{Single}\to\text{Multi}$ is just the
invert from the other direction.

A self-contained formula for $\text{Multi}\to\text{Single}$ is:

$$
\begin{aligned}
\text{Multi}\to\text{Single}(x_0, x_1, ..., x_{D-1})
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
    ...,
    \left\lfloor \frac{x}{n_0 n_1 \cdots n_{D-2}} \right\rfloor~\text{mod}~n_{D-1}
  \right)
\end{aligned}
$$

Thus, we have the following definition:

<div class="statement" markdown="1">

**Definition 3. (Canonical singlevariate function)**
Let $L = (n_0, n_1, ..., n_{D-1}) : (s_0, s_1, ..., s_{D-1})$ be a layout.  Let
$M = n_0 n_1 \cdots n_{D-1}$ be $L$'s size.  The canonical singlevariate
function of $L$ is $f_L: [0, M) \to \mathbb{N}$ defined by:

$$
f_L(x)
  := \left( \text{Single}\to\text{Multi}(x) \right)^\top \cdot (s_0, s_1, ..., s_{D-1})
  = \sum_{i=0}^{D-1} s_i \cdot \left(\left\lfloor \frac{x}{n_0 n_1 \cdots n_{i-1}} \right\rfloor~\text{mod}~n_i\right)
$$

</div>

Similar to the case of canonical multivariate function, we also drop the terms
"canonical" when clear from context. We might also write $L(x)$ instead of
$f_L(x)$, again, when clear from context.

Additionally, since the original
[CuTe document](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/02_layout_operations.md)
simply uses the term "function" to refer to "canonical singlevariate function",
we will also call it the layout function, or maybe the layout's associated
function.

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
<br>

Clearly, given any layout $L$, we can easily construct its singlevariate
function. The reverse question is much less trivial: for which functions $f:
\mathbb{N} \to \mathbb{N}$ there is a layout whose singlevariate function is
$f$?

In the next section, we will discuss a more general question. For a function $f:
[0, M) \to \mathbb{N}$, we say that $L$ *admits* $f$ if $L(x) = f(x)$ for all $x
\in [0, M)$. Then, which functions $f: \mathbb{N} \to \mathbb{N}$ is admitted by
a layout?

### Basic operations

<div markdown="1" class="statement">

**Definition 1.2. (Concatenation)**

The concatenation of two layouts $L_1$ and $L_2$ -- denoted by $(L_1, L_2)$ is the layout
$L$L whose single variate function is:

$$
L(x)
    = \text{cosize}(L_1) \cdot L_1(x~\text{mod}~\text{size}(L_1))
    + L_2\mathopen{}\left(
        \left\lfloor \frac{x}{\text{size}(L_1)} \right\rfloor
        ~\text{mod}~\text{size}(L_2)
    \right)
$$

</div>

With that definition, it is relatively easy to check the closed form of layout concatentation.
In particular, if:

$$
\begin{aligned}
L_1 &= (m_0, m_1, ..., m_{D-1}) : (t_0, t_1, ..., t_{D-1}) \\
L_2 &= (n_0, n_1, ..., n_{D-1}) : (s_0, s_1, ..., s_{D-1})
\end{aligned}
$$

Then their concatenation is simply obtained by concatenating their sizes and strides:

$$
(L_1, L_2) = (m_0, m_1, ..., m_{D-1}, n_0, n_1, ..., n_{D-1})
           : (t_0, t_1, ..., t_{D-1}, s_0, s_1, ..., s_{D-1})
$$

## What function $f: \mathbb{N} \to \mathbb{N}$ can be admitted by a layout?

<div class="statement" id="function-to-layout">

Let $f: [0, M) \to \mathbb{N}$ be an arbitrary function. Then there is an
algorithm with runtime $O(M^2 \log{M})$ that finds a layout $L = (n_0, n_1, ...
, n_{D-1}) : (s_0, s_1, ..., s_{D-1})$ such that $L(x) = f(x)$ for all $x \in
[0, M)$, or reports that there is no such layout.

</div>

### Algorithm

Without loss of generality, assume that $n_i > 1$ for all $i \in [0, D)$.
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
<br>
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

### Python implementation

<details markdown="1">
<summary>Here's the algorithm implemented in Python.</summary>

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

    def __repr__(self) -> str:
        n_str = ",".join([str(n) for n in self.N])
        s_str = ",".join([str(s) for s in self.S])
        return f"[layout] ({n_str}) : ({s_str})"


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
        last_f = pad_f[: m%n_0, -1]
        last_d = last_f[1:] - last_f[:-1]

        n_0_consistent = np.all(row_d[:, :-1] == s_0) and np.all(last_d == s_0)
        if n_0_consistent:
            r = find_layout(pad_f[0, :])
            if r is None:
                return None
            return Layout(N=np.concatenate([[n_0], r.N]),
                          S=np.concatenate([[s_0], r.S]))

    return None


print(find_layout(np.array([0, 2, 4, 7, 9, 11])))  # (3,2) : (2,7)
```
</details>  <!-- Python implementation -->






<hr>

##### Are layout representations unique?

In general, multiple layouts might represent the same singlevariate function.

<details markdown="1">
<summary><b>Example:</b> multiple layouts associated to the same function.</summary>

The two layouts $A = (10) : (3)$ and $B = (2, 5) : (3, 6)$ share the same
function: $f_A(x) = f_B(x) = 3x$  for all $x \in \{0, 1, ... 9 \}$.

</details>

If two layouts have the same associated function, we say that they are
*equivalent*. This equivalence partitions the set of all layouts into equivalent
classes. In the next sections, when we discuss certain types of uniqueness for
layouts, we mean uniqueness upto the equivalence via a layout's canonical
singlevariate function.


## Complemention

<div class="statement" id="complement-def" markdown="1">

**Definition 4. (Complement)**
Let $A = N : S$ be a layout. Then, for an integer $M$, the *complement of $A$
with respect to $M$* -- denoted by $\text{Complement}(A, M)$ -- is the layout
$B$ that satisfies two conditions:
1. $B$'s singlevariate function is strictly increasing.
2. The concatenation layout $(A, B)$ is a bijection from $[0, M)$ to itself.

</div>

Note that
[CuTe's original definition](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/02_layout_operations.md#complement)
of complementation specifies the following conditions instead of (2).

1. $\text{size}(B) \geq \left\lfloor \dfrac{M}{\text{size(A)}} \right\rfloor$.

2. $\text{cosize}(B) \leq \left\lfloor \dfrac{M}{\text{cosize(A)}} \right\rfloor \cdot \text{cosize}(A)$.

It is not hard to check that together, these conditions are equivalent to (3) in
[our definition](#complement-def). In our (obviously biased) opinion, our
definition is more intuitive of what the complement operation does.


Not all layouts have a complement. In particular, we bijection requirement in
condition (2) rules out all layout $A$ whose singlevariate function is not
injective.

The [function-to-layout Algorithm](#function-to-layout) offers deterministic way
to find $\text{Complement}(A, M)$ for any layout $A$ and positive integer $M$,
or to tell that such complement does not exist.

Indeed, the idea is to determine $B$'s singlevariate function based on the given
conditions, and then use [function-to-layout Algorithm](#function-to-layout) to
find $B$, or to tell if there is no such $B$.

## Composition

<div class="statement" id="composition-def" markdown="1">

**Definition 5. (Composition)**

Let $A = N_a : S_a$ and $B = N_b : S_b$ be two layouts. Their composition $A
\circ $B is the layout such that $f_{A \circ B} \equiv f_A \circ f_B$.

</div>

As for complementation, not all pairs of layouts can be composed.  To determine
two layouts' composition, we first determine the composite of their single
variate function, and then use the [function-to-layout
Algorithm](#function-to-layout) to find the composition or to conclude that such
composition does not exist.
