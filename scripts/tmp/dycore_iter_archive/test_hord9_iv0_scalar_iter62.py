"""iter62 (codex/oracle): the scalar PPM path `_ppm_1d` must use FV3's
iv=0 (positive-definite) limiter for hord=9.

FV3 tp_core.F90:610 — `if(iord==9 .or. iord==13) call pert_ppm(...,0)` —
so for the SCALAR / mass / vorticity transport (`fv_tp_2d` → `_ppm_1d`),
iord/hord=9 uses pert_ppm with iv=0.  iv=1 (`_pert_ppm`) is FV3's
boundary-only limiter and the separate momentum (ytp_v/xtp_u) path.

The `_ppm_1d` hord==9 branch previously dispatched to `_pert_ppm` (iv=1),
a latent mislabel (unexercised because live scalar callers use the hord=12
default, which is already iv=0).  These tests pin the corrected dispatch:
hord=9 must now be bit-identical to hord=12 (both iv=0), and that identity
is non-trivial because iv=0 and iv=1 genuinely differ.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from legoesm.core.fv_tp_2d import _ppm_1d, _pert_ppm, _pert_ppm_iv0  # noqa: E402


def test_iv0_and_iv1_actually_differ():
    """Guard: the two limiters must diverge on a discriminating input, so
    the hord9==hord12 identity below is a meaningful iv=0 selection, not a
    no-op.  Pick a positive cell mean whose parabola dips negative
    (iv=0 clips it; iv=1, seeing same-sign bl·br, flattens differently)."""
    q = jnp.array([[0.2]])
    bl = jnp.array([[-0.9]])   # left edge well below q
    br = jnp.array([[-0.9]])   # right edge well below q → parabola min < 0
    bl0, br0 = _pert_ppm_iv0(q, bl, br)
    bl1, br1 = _pert_ppm(bl, br)
    assert not (np.allclose(bl0, bl1) and np.allclose(br0, br1)), (
        "iv=0 and iv=1 limiters coincide on the discriminating input — "
        "the hord9/hord12 equality test would be vacuous")


def _scalar_field(n, M, seed):
    rng = np.random.default_rng(seed)
    # halo=2 in the sweep axis (axis=1): shape (6, n+4, M).  Positive mean
    # with sharp features so the iv=0 positive-definite clip engages.
    base = 1.0 + 0.8 * np.sin(np.linspace(0, 6.0, n + 4))[None, :, None]
    noise = 0.6 * rng.standard_normal((6, n + 4, M))
    return jnp.asarray(np.clip(base + noise, 0.01, None))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_hord9_matches_hord12_iv0(seed):
    n, M = 8, 3
    q = _scalar_field(n, M, seed)
    bl9, br9, _ = _ppm_1d(q, n, hord=9)
    bl12, br12, _ = _ppm_1d(q, n, hord=12)
    assert np.allclose(np.asarray(bl9), np.asarray(bl12), rtol=1e-12, atol=1e-12), (
        "hord=9 scalar limiter no longer matches the iv=0 hord=12 default "
        "(FV3 iord=9 ⇒ pert_ppm iv=0)")
    assert np.allclose(np.asarray(br9), np.asarray(br12), rtol=1e-12, atol=1e-12)
