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

    def __post_init__(self):
        self._cum_prod = np.concatenate([[1], np.cumprod(self.N[:-1])])

    def __repr__(self) -> str:
        n_str = ",".join([str(n) for n in self.N])
        s_str = ",".join([str(s) for s in self.S])
        return f"[layout] ({n_str}) : ({s_str})"

    def size(self) -> int:
        r"""The layout's size is the product of all its dimensions."""
        return np.prod(self.N)

    def single2multi(self, x: int) -> tuple[int, ...]:
        r"""1-D coordinate to multi coordinate."""
        return tuple(((x // self._cum_prod) % self.N).tolist())

    def multi2single(self, *args) -> int:
        r"""Multi coordinate to 1-D coordinate."""
        return np.dot(self._cum_prod, np.array(args))

    def f_single(self, x: int) -> int:
        r"""Evaluate the layout's singlevariate function."""
        return np.array(self.single2multi(x)).dot(self.S)


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
        # TODO(hieu): continue here

        # check if f is consistent with (n_0) : (s_0) as the first mode
        tgt_sz = (m + n_0 - 1) // n_0 * n_0
        pad_sz = tgt_sz - m
        pad_f = np.pad(f, [(0, pad_sz)]).reshape(-1, n_0).transpose()

        # row diff for all columns, except for the last one
        row_d = pad_f[1:, :-1] - pad_f[:-1, :-1]

        # row diff in the last column. need to remove the padded zeros
        last_f = pad_f[: m % n_0, -1]
        last_d = last_f[1:] - last_f[:-1]

        n_0_consistent = np.all(row_d[:, :-1] == s_0) and np.all(last_d == s_0)
        if n_0_consistent:
            r = find_layout(pad_f[0, :])
            if r is None:
                return None
            return Layout(N=np.concatenate([[n_0], r.N]),
                          S=np.concatenate([[s_0], r.S]))

    return None


# print(find_layout(np.array([0, 2, 4, 7, 9, 11])))  # (3,2) : (2,7)


def test_layout_function():
    """Simple opreations."""
    layout = Layout(N=[3, 4], S=[1, 3])

    assert layout.single2multi(5) == (2, 1)
    assert layout.multi2single(2, 1) == 5
    assert layout.multi2single(2, 3) == 11

    for x in range(layout.size()):
        assert x == layout.multi2single(*layout.single2multi(x))
        assert x == layout.f_single(x)


test_layout_function()
