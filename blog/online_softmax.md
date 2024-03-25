---
layout: post
date: 2024-03-06
---

Online Softmax
==============

[[Hieu's personal blog index](./index)]

In this post, I discuss the online softmax algorithm, which is used extensively
as a sub-routine of various memory-efficient attention algorithms.

# Problem formulation
We see a sequence of $N$ real numbers $s_1, s_2, \ldots, s_N \in \mathbb{R}$,
and $N$ vectors $V_1, V_2, \ldots, V_N \in \mathbb{R}^d$. These real numbers and
vectors are shown to us $K$ pairs of $(s_i, V_i)$ at a time. The goal is to
compute the quantity:

$$
\begin{align*}
O &= \dfrac{\sum_{i=1}^N \exp{(s_i)} \cdot V_i}{\sum_{i=1}^N \exp{(s_i)}}
\end{align*}
$$

One easy approach is to keep all the $(s_i, V_i)$ pairs in memory, and then
compute the output. However, this approach is not memory-efficient in many ways.
This is because we need to store $N$ pairs of $(s_i, V_i)$, which is $O(N)$
memory, which is a disaster when $N$ is large.

The online softmax algorithm aims to compute the output, while maintaining a
relatively minimal amount of memory.

# The algorithm
The gist of the online softmax algorithm is to store certain information about
the pairs $(s_i, V_i)$ that we see sequentially, and then use this information
to compute the output.

## A note on numerical stability

To account for the instability of taking $e$ to the power of large numbers, we
instead subtract the value $M = \max\{s_1, s_2, \ldots, s_N\}$ from each $s_i$
and compute the equivalent quantity:

$$
\begin{align*}
O &= \dfrac{\sum_{i=1}^N \exp{(s_i - M)} \cdot V_i}{\sum_{i=1}^N \exp{(s_i - M)}}
\end{align*}
$$

What we store and update while observing the $(s_i, V_i)$ pairs will closely
follow the equation above.

## Recursions

We define the following quantities:

$$
\begin{align*}
M_k &:= \max(s_1, s_2, \ldots, s_k) \\
S_k &:= \sum_{i=1}^k \exp(s_i - M_k) \\
O_k &:= \sum_{i=1}^k \exp(s_i - M_k) \cdot V_i
\end{align*}
$$

Then, the final answer to the problem is:

$$
\begin{align*}
O = \dfrac{\sum_{i=1}^N \exp(s_i) \cdot V_i}{\sum_{i=1}^N \exp(s_i)}
  = \dfrac{\sum_{i=1}^N \exp(s_i - M_N) \cdot V_i}{\sum_{i=1}^N \exp(s_i - M_N)}
  = \dfrac{O_N}{S_N}
\end{align*}
$$

As such, if we maintain $M_k, S_k, O_k$ for
$k = 1, 2, \ldots, N$, we can return $O_N / S_N$ at the end.  To this end, every time
we see a new pair $(s_k, V_k)$, we perform the updates to $M_k, S_k, O_k$ so that they
always compute the quantites above. For $M_k$ and $S_k$, the updates are:

$$
\begin{align*}
M_{k} &= \max(M_{k-1}, s_k) \\
S_{k} &= \exp(M_{k-1} - M_k) \cdot S_{k-1} + \exp(s_k - M_k) \\
\end{align*}
$$

The update rule for $O_k$ is slightly more involved. From the definition of $O_k$, we have:

$$
\begin{align*}
\exp(M_k) \cdot O_k
  &= \sum_{i=1}^k \exp(s_i) \cdot V_i \\
  &= \exp(s_k) \cdot V_k + \sum_{i=1}^{k-1} \exp(s_i) \cdot V_i \\
  &= \exp(s_k) \cdot V_k + \exp(M_{k-1}) \cdot O_{k-1}
\end{align*}
$$

Therefore, we have:

$$
\begin{align*}
O_k &= \exp(M_{k-1} - M_k) \cdot O_{k-1} + \exp(s_k - M_k) \cdot V_k
\end{align*}
$$


## What if we see two pairs: $(s_{k-1}, V_{k-1})$ and $(s_{k-2}, V_{k-2})$ at a time?

We see that most of the formula above can be reused:

$$
\begin{align*}
M_{k} &= \max(M_{k-2}, s_{k-1}, s_k) \\
S_{k} &= \exp(M_{k-2} - M_k) \cdot S_{k-2} + \exp(s_{k-1} - M_k) + \exp(s_k - M_k) \\
\end{align*}
$$

The key is to choose the identity regarding $O_k$ to recurse:

$$
\begin{align*}
\exp(M_k) \cdot O_k
  &= \exp(s_k) \cdot V_k + \exp(M_{k-1}) \cdot O_{k-1} \\
  &= \exp(s_k) \cdot V_k + \exp(s_{k-1}) \cdot V_{k-1} + \exp(M_{k-2}) \cdot O_{k-2} \\
\end{align*}
$$

Which leads to:

$$
\begin{align*}
O_k &= \exp(M_{k-2} - M_k) \cdot O_{k-2}
     + \exp(s_{k-1} - M_k) \cdot V_{k-1}
     + \exp(s_k - M_k) \cdot V_k
\end{align*}
$$

# How about $K$ pairs at a time?

In the general case, where we see $K$ pairs at a time, say $s_{k-i}, V_{k-i}$ for
$i = 1, 2, \ldots, K$, it turns out that the same recursions above simply extend to:

$$
\begin{align*}
M_{k} &= \max(M_{k-K}, s_{k-K+1}, \ldots, s_{k-1}) \\
S_{k} &= \exp(M_{k-K} - M_k) \cdot S_{k-K}
       + \sum_{i=1}^{K} \exp(s_{k-i} - M_k) \\
O_{k} &= \exp(M_{k-K} - M_k) \cdot O_{k-K}
       + \sum_{i=1}^{K} \exp(s_{k-i} - M_k) \cdot V_{k-i}
\end{align*}
$$

In many contexts that we use the online softmax algorithm, these updates can
even be vectorized to make the implementation more efficient.

# Python implementation

Finally, here is the Python implementation of the online softmax algorithm.
Bonus: it also supports a batch dimension, which can be considered the
vectorization of the described algorithm.

TODO(hieu): write the vectorize version.

<details markdown="1">  <!-- Python implementation -->

<summary>Code. You can run the tests
<a href="https://github.com/hyhieu/hyhieu.github.io/blob/master/blog/code/online_softmax.py">here</a>.</summary>

```python
def online_softmax(s: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Online softmax."""
    batch_size, n = s.shape

    M = np.copy(s[:, 0])  # batch_size
    S = np.ones(shape=[batch_size])  # batch_size
    O = np.copy(V[:, 0, :])  # batch_size, d

    for k in range(1, n):
        s_k = s[:, k]  # batch_size
        M_k = np.maximum(M, s[:, k])  # batch_size
        S_k = np.exp(M - M_k) * S + np.exp(s_k - M_k)  # batch_size
        O_k = np.exp(M - M_k)[:, None] * O + np.exp(s_k - M_k)[:, None] * V[:, k, :]  # batch_size, d
        M, S, O = M_k, S_k, O_k

    out = O / S[:, None]
    return out
```

</details>  <!-- Python implementation -->
