"""Hard (OpenIFS) or smooth (RULE 5) forms of the switches in the IFS ports.

Shared by CLOUDSC (:mod:`.microphysics.ifs_cloudsc`) and CUMASTRN
(:mod:`.convection._ifs_cumastrn`).  ``IfsSmooth(hard=True)`` returns the
reference's exact min/max/comparison/select; ``hard=False`` the C1 forms of
:mod:`legoesm.core.smooth`: min/max through the C1 positive part (exact
outside a window of width ``w``), comparisons through the C1 cubic
smoothstep centred on the threshold, ``on``/``off`` one-sided on
``[thr, thr + w]`` (exact at and below ``thr``), ``sel`` a linear blend.
"""
from __future__ import annotations

import jax.numpy as jnp

from legoesm.core import smooth as _sm


class IfsSmooth:
    """Hard (oracle) or smooth forms.  ``w`` is the transition width."""

    def __init__(self, hard: bool):
        self.hard = hard

    def pos(self, x, w):
        return jnp.maximum(x, 0.0) if self.hard else _sm.positive_part_c1(x, 1.0 / w)

    def max(self, a, b, w):
        return jnp.maximum(a, b) if self.hard else b + _sm.positive_part_c1(a - b, 1.0 / w)

    def min(self, a, b, w):
        return jnp.minimum(a, b) if self.hard else a - _sm.positive_part_c1(a - b, 1.0 / w)

    def abs(self, x, w):
        return jnp.abs(x) if self.hard else self.pos(x, w) + self.pos(-x, w)

    def _step(self, d, w, strict):
        """weight of (d > 0) [strict] or (d >= 0)."""
        if self.hard:
            return (d > 0) if strict else (d >= 0)
        return _sm.smoothstep_cubic(d / w + 0.5)

    def gt(self, a, b, w):
        return self._f(self._step(a - b, w, True), a)

    def ge(self, a, b, w):
        return self._f(self._step(a - b, w, False), a)

    def lt(self, a, b, w):
        return self._f(self._step(b - a, w, True), a)

    def le(self, a, b, w):
        return self._f(self._step(b - a, w, False), a)

    def on(self, x, thr, w, strict=True):
        """weight of (x > thr) [strict] or (x >= thr), one-sided (VC47): smooth only on
        [thr, thr + w], exactly 0 for x <= thr (so a gated substitute never reaches an output)."""
        if self.hard:
            return self._f((x > thr) if strict else (x >= thr), x)
        return _sm.smoothstep_cubic((x - thr) / w)

    def off(self, x, thr, w, strict=True):
        """weight of (x < thr) [strict] or (x <= thr), one-sided on [thr, thr + w] (VC47):
        exactly 1 for x <= thr."""
        if self.hard:
            return self._f((x < thr) if strict else (x <= thr), x)
        return 1.0 - _sm.smoothstep_cubic((x - thr) / w)

    def neg1(self, x, w):
        """weight of (x < 0), one-sided: smooth only on [-w, 0], exactly 0 for x >= 0."""
        if self.hard:
            return (x < 0).astype(jnp.result_type(x))
        return _sm.smoothstep_cubic(-x / w)

    @staticmethod
    def _f(m, like):
        return m.astype(jnp.result_type(like)) if m.dtype == jnp.bool_ else m

    def sel(self, m, x, y):
        if self.hard:
            return jnp.where(m > 0.5, x, y)
        return m * x + (1.0 - m) * y
