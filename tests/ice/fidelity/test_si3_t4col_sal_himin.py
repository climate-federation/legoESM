"""C8: nn_icesal=2 drainage/flushing and salinity bounds act only where h_i > rn_sal_himin = 0.1 m.

NEMO gates the whole update (icethd_sal.F90:216, namelist_ice_cfg:126); main drained and
clipped every column.  Oracle = NEMO 5.0.1 SI3 ORCA1 tier-4 record (fixture c8_sal_himin:
8 columns with h_i <= 0.1 m changed by main, 4 thicker controls).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.ice import bitz_lipscomb as bl
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG as K

from si3_t4col_fixture import col, dt, forcing, load, within, within_flux

jax.config.update("jax_enable_x64", True)
FX = load("c8_sal_himin")


def _pre(h=None):
    n = len(FX["cell"])
    i = lambda k: jnp.asarray(col(FX, f"sal:in:{k}"))  # noqa: E731
    pre = bl.SI3ColumnArrays(i("a_i"), i("h_i") if h is None else h, jnp.zeros(n), i("t_su"),
                             jnp.zeros((n, 3)), jnp.zeros((n, 3)), i("s_i"), i("sz_i"), jnp.zeros(n))
    return pre, forcing(n, t_bottom=i("t_bo"), sss=i("sss"))


def test_fixture_thin_columns_fail_on_main():
    thin = FX["is_fail_on_main"]
    assert thin.sum() == 8 and np.all(col(FX, "sal:in:h_i")[thin] <= 0.1)
    assert not within(FX, "sal", "s_i", col(FX, "sal:main:s_i"))[thin].any()


def test_gated_salinity_and_salt_fluxes_match_nemo():
    pre, fz = _pre()
    S, Sz, fx = bl._sal_step(pre, fz, dt(FX), K, gate_width=0.0)
    assert within(FX, "sal", "s_i", S).all() and within(FX, "sal", "sz_i", Sz).all()
    for f in ("sfx_bri", "sfx_res"):
        acc = col(FX, f"sal:in:{f}") + np.asarray(getattr(fx, f))
        assert within_flux(FX, "sal", f, acc).all(), f


def test_salt_budget_closes():
    pre, fz = _pre()
    S, _, fx = bl._sal_step(pre, fz, dt(FX), K)
    ds = K.rho_ice * pre.concentration * pre.h_ice * (S - pre.S_bulk) / dt(FX)
    np.testing.assert_allclose(np.asarray(ds), -np.asarray(fx.sfx_bri + fx.sfx_res), rtol=0, atol=1e-15)


def test_default_gate_is_exact_outside_band_and_differentiable_inside():
    pre, fz = _pre()
    hard = bl._sal_step(pre, fz, dt(FX), K, gate_width=0.0)[0]
    soft = bl._sal_step(pre, fz, dt(FX), K)[0]
    h = np.asarray(pre.h_ice)
    out = (h <= 0.1) | (h >= 0.1 + bl._SAL_GATE_WIDTH_M)
    assert np.array_equal(np.asarray(soft)[out], np.asarray(hard)[out])
    k = int(np.flatnonzero(~FX["is_fail_on_main"])[0])          # a draining control column

    def f(hk):
        p, z = _pre(pre.h_ice.at[k].set(hk))
        return bl._sal_step(p, z, dt(FX), K)[0][k]

    for hk in (0.1 + 2e-4, 0.1 + 5e-4, 0.1 + 8e-4):
        g = jax.grad(f)(hk)
        fd = (f(hk + 1e-7) - f(hk - 1e-7)) / 2e-7
        assert bool(jnp.isfinite(g)) and float(jnp.abs(g)) > 0
        np.testing.assert_allclose(float(g), float(fd), rtol=1e-4)


def test_default_gate_band_value_pinned_and_bad_width_rejected():
    from legoesm.core.smooth import smoothstep_cubic
    pre, fz = _pre()
    k = int(np.flatnonzero(~FX["is_fail_on_main"])[0])
    full = bl._sal_step(pre, fz, dt(FX), K, gate_width=0.0)[0][k] - pre.S_bulk[k]
    for frac in (0.25, 0.5, 0.75):
        p, z = _pre(pre.h_ice.at[k].set(0.1 + frac * bl._SAL_GATE_WIDTH_M))
        got = bl._sal_step(p, z, dt(FX), K)[0][k] - p.S_bulk[k]
        np.testing.assert_allclose(float(got), float(smoothstep_cubic(frac)) * float(full), rtol=1e-12)
    import pytest
    for bad in (-1e-3, float("nan")):
        with pytest.raises(ValueError, match="gate_width"):
            bl._sal_step(pre, fz, dt(FX), K, gate_width=bad)


def test_band_values_against_hand_numbers_incl_active_bound():
    """Independent numbers (review): dt = 14400 s, drainage target 5 psu over 1.73e6 s, rn_simin 0.1.
    h = 0.1005 m sits at the band centre, where the C1 gate is exactly 0.5."""
    h = jnp.asarray([0.1005, 0.1005, 0.2, 0.05])
    S = jnp.asarray([8.0, 0.05, 8.0, 8.0])                  # drains; below rn_simin; thick; thin
    n = 4
    pre = bl.SI3ColumnArrays(jnp.ones(n), h, jnp.zeros(n), jnp.full(n, 250.0), jnp.zeros((n, 3)),
                             jnp.zeros((n, 3)), S, jnp.stack([S] * 3, -1), jnp.zeros(n))
    fz = forcing(n, t_bottom=jnp.full(n, 271.2), sss=jnp.full(n, 34.0))
    got = np.asarray(bl._sal_step(pre, fz, 14400.0, K)[0])
    full_drain = 3.0 * 14400.0 / 1730000.0                   # 0.024971098265895952
    np.testing.assert_allclose(got, [8.0 - 0.5 * full_drain, 0.05 + 0.5 * 0.05, 8.0 - full_drain, 8.0],
                               rtol=1e-14)
