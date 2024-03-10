---
layout: post
date: 2024-03-06
---

Online Softmax
==============

[[Hieu's personal blog index](./index)]

In this post, I discuss the online softmax algorithm. This algorithm is used
extensively as a sub-routine of various memory-efficient attention algorithms.
Memory-efficient attention algorithms are is not the concern of this blog post.
Instead, we are only interested in the setting of the online softmax algorithm.

# Problem formulation
You see a sequence of $N$ real numbers: $s_1, s_2, \ldots, s_N \in \mathbb{R}$,
and $N$ vectors $V_1, V_2, \ldots, V_N \in \mathbb{R}^d$. These real numbers and
vectors are given $K$ pairs of $(s_i, V_i)$ at a time. The goal is to compute
the quantity:

$$
\begin{align*}
O &= \dfrac{\sum_{i=1}^N \exp{(s_i)} \cdot V_i}{\sum_{i=1}^N \exp{(s_i)}}
\end{align*}
$$

Of course, one easy approach is to keep all the $(s_i, V_i)$ pairs in memory,
and then compute the output. However, this approach is not memory-efficient. The
online softmax algorithm aims to compute the output, while maintaining a
relatively minimal amount of memory.

# The algorithm
The gist of the online softmax algorithm is to store certain information about
the sequence of $(s_i, V_i)$ pairs, and then use this information to compute the
output.

**A small note for numerical stability.**
To account for the instability of taking $e$ to the power of large numbers, we
instead subtract the value $M = \max\{s_1, s_2, \ldots, s_N\}$ from each $s_i$
and compute the equivalent quantity:

$$
\begin{align*}
O &= \dfrac{\sum_{i=1}^N \exp{(s_i - M)} \cdot V_i}{\sum_{i=1}^N \exp{(s_i - M)}}
\end{align*}
$$

What we store and update through the algorithm will closely follow the equation
above.

**Recursions.** We define the following quantities:

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
we see a new pair $(s_k, V_k)$, we perform the following updates:

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


**What if we see two pairs: $(s_k, V_k)$ and $(s_{k-1}, V_{k-1})$ at the same time?**
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

# Python implementation

Finally, here is the Python implementation of the online softmax algorithm.
Bonus: it also supports a batch dimension, which can be considered the
vectorization of the described algorithm.

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
