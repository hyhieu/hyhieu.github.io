r"""Online softmax."""

import numpy as np


def reference_softmax(s: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Reference softmax."""
    max_s = np.max(s, axis=1, keepdims=True)
    s -= max_s
    e = np.exp(s)
    p = e / np.sum(e, axis=1, keepdims=True)
    o = np.einsum("bn,bnd->bd", p, V)

    return o


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


def test_online_softmax(batch_size: int, n: int, d: int):
    """Test for online softmax."""
    np.random.seed(0)
    np.set_printoptions(precision=3, suppress=True, linewidth=200)

    V = np.random.randn(batch_size, n, d)
    s = np.random.randn(batch_size, n)

    ref = reference_softmax(np.copy(s), np.copy(V))
    out = online_softmax(np.copy(s), np.copy(V))

    diff = np.abs(out - ref).max()
    if diff < 1e-6:
        print("\033[92mPASSED\033[0m")
    else:
        print("\033[91mFAILED\033[0m")


def main():
    r"""Entry point."""

    test_online_softmax(12, 5, 10)
    test_online_softmax(11, 8, 21)
    test_online_softmax(31, 16, 5)


if __name__ == "__main__":
    main()
