"""dyn_zad's written association (dynzad.f90:112-126, :134-136) under jit."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.ocean.vertical import nemo_advective_vertical_momentum_advection


def _nemo_dynzad(u, w_half, h, area):
    """NEMO's statements in numpy (one rounding per written operation)."""
    nk = u.shape[-1]
    coef = (0.25 * (1.0 / area)) / h
    out = np.zeros_like(u)
    prev = np.zeros(u.shape[:-1])
    for jk in range(nk - 1):
        zz = 2.0 * w_half[..., jk + 1]                 # zWfi + zWf
        cur = zz * (u[..., jk] - u[..., jk + 1])
        out[..., jk] = -(coef[..., jk] * (prev + cur))
        prev = cur
    out[..., nk - 1] = -(coef[..., nk - 1] * (prev + 0.0))
    return out


def _fields(seed=7, shape=(24, 31, 9)):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=shape) * 0.3
    w = np.concatenate([np.zeros(shape[:-1] + (1,)),
                        rng.normal(size=shape[:-1] + (shape[-1] - 1,)) * 1e6,
                        np.zeros(shape[:-1] + (1,))], axis=-1)
    h = rng.uniform(5.0, 400.0, size=shape)
    area = rng.uniform(1e9, 4e10, size=shape[:-1] + (1,))
    return u, w, h, area


def _card(written, u, w, h, area):
    f = jax.jit(lambda *a: nemo_advective_vertical_momentum_advection(
        *a, bottom_face_mask_mode="nemo_faithful",
        nemo_written_association=written))
    return np.asarray(f(*(jnp.asarray(x) for x in (u, w, h, area))))


def test_written_association_is_nemos_bit_for_bit_under_jit():
    u, w, h, area = _fields()
    ref = _nemo_dynzad(u, w, h, area)
    assert np.array_equal(_card(True, u, w, h, area), ref)


def test_the_default_association_is_not_nemos():
    # non-vacuity: the pre-round-7 form moves the last bit somewhere
    u, w, h, area = _fields()
    ref = _nemo_dynzad(u, w, h, area)
    got = _card(False, u, w, h, area)
    assert np.allclose(got, ref, rtol=1e-12, atol=0.0)
    assert np.count_nonzero(got != ref) > 0
