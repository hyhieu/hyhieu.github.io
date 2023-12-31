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

In this blog post, I attempt to formalize the definition of CuTe layout and its
accompanying operations.

## Basic definitions and properties

### Layout

<div class="statement" markdown="1" id="layout-def">

**Definition. (Layout)** Let $D$ be a positive integer. A layout $L = N :
S$ is a pair of tuples, each with $D$ positive integers:

$$
\begin{aligned}
N &= (n_0, n_1, ..., n_{D-1}) \\
S &= (s_0, s_1, ..., s_{D-1})
\end{aligned}
$$

</div>

The tuple $N$ is called the layout's *size,* while the tuple $S$ is called the
layout's *stride.* Additionally, each tuple $(n_i, s_i)$ for $i \in \{0, 1,
..., D-1\}$ is called a *mode* of $L$'s.


### Canonical function

<div class="statement" markdown="1">

**Definition. (Canonical multivariate function)**
A layout $L$ represents a multivariable function $g_L : [0, n_0) \times [0,
n_1) \times \cdots \times [0, n_{D - 1}) \subseteq \mathbb{N}^{D} \to
\mathbb{N}$, defined by:

$$
g_L(x_0, x_1, ..., x_{D-1}) := n_0 \cdot x_0 + n_1 \cdot x_1 + \cdots + n_{D-1} \cdot x_{D-1}
$$

We call $g_L$ the *canonical multivariate function* of $L$.

</div>

Throughout this note, when clear from context, we will drop the word
"canonical" for brevity, and we might use the overloaded notation:

$$
L(x_0, x_1, ..., x_{D-1}) := n_0 \cdot x_0 + n_1 \cdot x_1 + \cdots + n_{D-1} \cdot x_{D-1}
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

**Definition. (Canonical singlevariate function)**
Let $L = (n_0, n_1, ..., n_{D-1}) : (s_0, s_1, ..., s_{D-1})$ be a layout.  Let
$M = n_0 n_1 \cdots n_{D-1}$ be $L$'s size.  The canonical singlevariate
function of L$$ is $f_L: [0, M) \to \mathbb{N}$ defined by:

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

### What function $f: \mathbb{N} \to \mathbb{N}$ can be admitted by a layout?

<div class="statement">

Let $f: [0, M) \to \mathbb{N}$ be an arbitrary function. Then there is an
algorithm with runtime $O(M^2 \log{M})$ that finds a layout $L = (n_0, n_1, ...
, n_{D-1}) : (s_0, s_1, ..., s_{D-1})$ such that $L(x) = f(x)$ for all $x \in
[0, M)$, or reports that there is no such layout.

</div>

#### Algorithm

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

The gist of the algorithm is to guess the first mode $(n_0) : (s_0)$, and then
recurse. To this end, we write down the formula for $L(x)$:

<div id="l-formula"></div>

$$
\begin{aligned}
L(x)
  &= \left(
    x~\text{mod}~n_0,
    \left\lfloor \frac{x}{n_0} \right\rfloor~\text{mod}~n_1,
    \left\lfloor \frac{x}{n_0 n_1} \right\rfloor~\text{mod}~n_2,
    ...,
    \left\lfloor \frac{x}{n_0 n_1 \cdots n_{D-2}} \right\rfloor~\text{mod}~n_{D-1}
  \right)^\top \cdot (s_0, s_1, ..., s_{D-1}) \\
  &= \sum_{i=0}^{D-1} s_i \cdot \left(\left\lfloor \frac{x}{n_0 n_1 \cdots n_{i-1}} \right\rfloor~\text{mod}~n_i\right)
\end{aligned}
$$

From this formula, we necessarily have $L(0) = 0$. In other words, if $f(x) \neq
0$, there is no layout admitting $f$.

#### Guessing $s_0$

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

#### Guessing $n_0$

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

Let $\hat{n}_0$ be the smallest positive value satisfying the condition above.
We prove that if there is $n_0 > \hat{n}_0$ which also satisfies the condition
above, then $\hat{n}_0~|~n_0$.

Let $n_0 = k \hat{n}_0 + r$ where $0 \leq r < \hat{n}_0$. Then for $x \in [0, M)$, we have:

$$
\begin{aligned}
x &= n_0 \cdot \left\lfloor \frac{x}{n_0} \right\rfloor + x~\text{mod}~n_0 \\
  &= n_0 \cdot \left\lfloor \frac{x / \hat{n}_0}{k + r / \hat{n}_0} \right\rfloor + x~\text{mod}~n_0 \\
\end{aligned}
$$

Indeed, apply the condition for $x = n_0$, we
have:

$$
f(n_0) = f(\hat{n}_0 \cdot \lfloor n_0 / \hat{n}_0 \rfloor)
       + s_0 \cdot (n_0~\text{mod}~\hat{n}_0)
$$


Here, consistency means that $f(x + i k) = f(x) + i s_0$, for all
$x \in [0, M]$ and $i \in \mathbb{N}$ such that $x + i k \in [0, M]$. If no such
$k$ is found, we say that the function $f$ is *inconsistent*, i.e., there is no
layout admitting $f$ as its singlevariate function. Otherwise, we repeat the
process on the function to find $(n_1, s_1)$:


$$
g : \left[ 0, \left\lfloor M / n_0 \right\rfloor \right] \to \mathbb{N}~~~~~~~~~g(x) := f(n_0 x)
$$

Essentially, this means to restrict $f$ into the sub-domain where the $0$-th
coordinate is $0$.

To prove the correctness of this algorithm, it remains to check that if there's
a layout admitting $f$, then there is a layout admitting $f$ whose first mode is
$(n_0) : (s_0)$ where $n_0$ is the *smallest value* found in (3a).

It is easy to check that if there is a layout admitting $f$ whose first model is
$(n^{'}_0): (s_0)$ where $n^{'}_0 > n_0$, then we must have $n_0~|~n^{'}_0$
(otherwise, using periodic argument, we can find $n^{'}_0 < n_0$ such that $f(x
+ i n^{'}_0) = f(x) + i s_0$).

Let $\hat{n}_0$

Now, suppose that a layout $L = (kn_0, n_1, ..., n_{D-1}) : (s_0, s_1, ...,
s_{D-1})$ admits $f$. We can see that the layout $L' = (n_0, k, n_1, ...,
n_{D-1}) : (s_0, s_0, s_1, ..., s_{D-1})$ has the same single variate function
as $L$, hence it also admits $f$.

This completes the proof that the smallest consistent value for $n_0$ suffices
for recursion.

<br>
</details> <!-- Why does largest n_0 work? -->

#### Runtime analysis

Our unoptimized implementation of the algorithm runs in $\boxed{O(M^2
\log{M})}$. The gist of the analysis is based on the assumption that $n_i \geq
2$ for all $i \in [0, D)$. As such, the resulting layout $L$ has at most
$O(\log{M})$ modes.  Furthermore, the unoptimized implementations of
[mode](#guessing-s_0) [guessing](#guessing-n_0) is $O(M^2)$.

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

#### Python implementation

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


## Complement

<div class="statement" id="complement-def" markdown="1">

**Definition 2. (Complement)**
Let $A = (N_a) : (D_a)$ be a layout.  For an integer $M$ that is divisible by
$\text{size}(A) = n_0 n_1 \cdots n_{D-1}$, the *complement of $A$ with
respect to $M$*, denoted by $C(A, M)$, is the layout $B$ that satisfies two
conditions:
1. The associated layout function $f_B$ is strictly increasing.
2. The concatenation layout $(A, B)$ is a bijection $[0, M) \to [0, M)$.

</div>

There are some ground-laying work to ensure that [Definition 2](#complement-def) works.

**Lemma 2.1.** Let $A$ be an $D$-dimensional layout, then the followings are equivalent:

1. Let $\sigma$ sorts $\{(n_0, s_0), (n_1, s_1), ..., (n_{D-1}, s_{D-1})\}$
first by $d$ and then by $n$.
That is, $\sigma$ is the permutation of $\{0, 1, ..., d-1\}$ such that for $0
\leq i < j \leq d-1$, we have $s_{\sigma(i)} \leq s_{\sigma(j)}$ and if
$s_{\sigma(i)} = s_{\sigma(j)}$ then $n_{\sigma(i)} \leq n_{\sigma(j)}$.
Then $n_{\sigma(i)} s_{\sigma(i)}~|~s_{\sigma(j)}$ for all $0 \leq i < j \leq D-1$.

2. $C(A, M)$ exists for *all* positive integers $M$ divisible by $\text{size}(A)$.





<details markdown="1">

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