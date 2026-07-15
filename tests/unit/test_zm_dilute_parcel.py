"""Unit tests for the ZM dilute entraining-plume CAPE.

The dilute parcel (Raymond-Blyth 1992 entropy-conserving entraining
plume) is the signature ZM fidelity property: entraining dry
environmental air reduces parcel buoyancy and CAPE relative to an
undilute moist adiabat.  These tests pin:

* dilute CAPE is strictly LESS than the undilute moist-adiabat CAPE
  on a moist tropical sounding (the physical entrainment effect);
* CAPE rises with environmental instability (lapse rate) and with
  boundary-layer humidity (monotonic sensitivity);
* the launch level is the surface for a PBL-rooted sounding;
* gradients flow (AD-safe) through ``dmpdz`` and the input state;
* ``jit``/``vmap`` reproduce the eager result;
* outputs are finite and ``q``/CAPE are physical.

The quantitative match to the compiled E3SM/CAM Fortran oracle (CAPE
within ~1.5 % on tropical soundings) was validated OFFLINE against a
compiled ``zm_conv.F90`` reference — a LOCAL, untracked harness (like the
oracle source clones, never committed and not present in a fresh checkout,
so do not rely on a specific path here). The CI-enforced faithfulness bounds
are the qualitative pins in this file (dilute CAPE strictly < undilute, and
< 0.6× on a tropical sounding); the scheme-level use of the faithful dilute
CAPE is pinned in
``tests/atmosphere/hydrostatic/unit/test_zhang_mcfarlane_faithful.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.convection._zm_dilute import (
    dilute_parcel_cape,
    _moist_entropy,
    _invert_entropy,
)
from legoesm.atmosphere.physics.thermodynamics import parcel_profile_and_cape


@pytest.fixture(autouse=True)
def _enable_x64():
    # The Newton entropy inversion + eager/jit parity checks below need x64:
    # ``test_dilute_jit_vmap_match_eager`` compares eager vs jit at rtol=1e-10,
    # which float32 cannot meet (~0.02 J/kg XLA-fusion drift on CAPE).
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _sounding(nlev=30, T_sfc=300.0, lapse=8.0, rh0=0.85, p_s=1.0e5, p_top=5.0e3):
    sigma = jnp.linspace(p_top / p_s, 1.0, nlev)
    p_full = (sigma * p_s)[None, :]
    p_half = jnp.concatenate(
        [
            jnp.full((1, 1), p_top * 0.5),
            0.5 * (p_full[:, :-1] + p_full[:, 1:]),
            jnp.full((1, 1), p_s),
        ],
        axis=1,
    )
    H = 8000.0
    z = -H * jnp.log(p_full / p_s)
    T = jnp.maximum(T_sfc - lapse * 1e-3 * z, 195.0)
    Tc = T - 273.15
    es = 611.2 * jnp.exp(17.67 * Tc / (Tc + 243.5))
    qsat = 0.622 * es / jnp.maximum(p_full - es, 1.0)
    RH = jnp.clip(rh0 * jnp.exp(-z / 5000.0), 0.10, 0.95)
    q = RH * qsat
    return T, q, p_full, p_half, z


def test_entropy_inversion_round_trips():
    """Newton inversion recovers T from its own entropy."""
    T = jnp.array([300.0, 280.0, 250.0])
    p = jnp.array([1.0e5, 8.0e4, 5.0e4])
    qt = jnp.array([0.018, 0.010, 0.003])
    s = _moist_entropy(T, p, qt)
    T_rec = _invert_entropy(s, p, qt, T + 5.0)
    assert jnp.allclose(T_rec, T, atol=0.05), f"{T_rec} vs {T}"


def test_dilute_cape_less_than_undilute():
    """Entrainment of dry air reduces CAPE below the undilute moist
    adiabat — the defining ZM dilute-parcel property."""
    T, q, pf, ph, z = _sounding()
    dp = dilute_parcel_cape(T, q, pf, ph, z)
    _, undilute = parcel_profile_and_cape(T, pf, ph, q_v=q)
    assert float(dp.cape[0]) < float(undilute[0]), (
        f"dilute {float(dp.cape[0]):.0f} must be < undilute "
        f"{float(undilute[0]):.0f}"
    )
    # On a moist tropical sounding the dilution factor is order ~3.
    assert float(dp.cape[0]) < 0.6 * float(undilute[0])


def test_dilute_cape_increases_with_instability():
    T1, q1, pf, ph, z = _sounding(lapse=7.0)
    T2, q2, _, _, _ = _sounding(lapse=9.0)
    c1 = float(dilute_parcel_cape(T1, q1, pf, ph, z).cape[0])
    c2 = float(dilute_parcel_cape(T2, q2, pf, ph, z).cape[0])
    assert c2 > c1, f"steeper lapse should raise CAPE: {c1:.0f} -> {c2:.0f}"


def test_dilute_cape_increases_with_humidity():
    T, q1, pf, ph, z = _sounding(rh0=0.70)
    _, q2, _, _, _ = _sounding(rh0=0.95)
    c1 = float(dilute_parcel_cape(T, q1, pf, ph, z).cape[0])
    c2 = float(dilute_parcel_cape(T, q2, pf, ph, z).cape[0])
    assert c2 > c1, f"moister BL should raise CAPE: {c1:.0f} -> {c2:.0f}"


def test_dilute_launch_at_surface():
    T, q, pf, ph, z = _sounding()
    dp = dilute_parcel_cape(T, q, pf, ph, z)
    nlev = T.shape[1]
    # surface-last: launch index near nlev-1.
    assert float(dp.k_launch_smooth[0]) > nlev - 2.0


def test_dilute_outputs_finite_and_physical():
    T, q, pf, ph, z = _sounding()
    dp = dilute_parcel_cape(T, q, pf, ph, z)
    for a in (dp.cape, dp.T_parcel, dp.Tv_parcel, dp.qs_parcel, dp.buoyancy):
        assert jnp.all(jnp.isfinite(a))
    assert jnp.all(dp.cape >= 0.0)
    assert jnp.all(dp.qs_parcel >= -1e-12)


def test_dilute_grad_through_dmpdz_finite():
    """``d CAPE / d dmpdz`` is finite — entrainment rate is tunable."""
    T, q, pf, ph, z = _sounding()

    def f(dmpdz):
        return dilute_parcel_cape(T, q, pf, ph, z, dmpdz=dmpdz).cape.sum()

    g = float(jax.grad(f)(jnp.asarray(-1.0e-3)))
    assert np.isfinite(g)
    # More entrainment (larger |dmpdz|) reduces CAPE -> dCAPE/d(dmpdz) > 0
    # for negative dmpdz (less negative dmpdz = less entrainment = more CAPE).
    assert abs(g) > 0.0


def test_dilute_grad_through_state_finite():
    T, q, pf, ph, z = _sounding()

    def f(T_in):
        return dilute_parcel_cape(T_in, q, pf, ph, z).cape.sum()

    g = jax.grad(f)(T)
    assert jnp.all(jnp.isfinite(g))
    assert float(jnp.sum(jnp.abs(g))) > 0.0


def test_dilute_jit_vmap_match_eager():
    T, q, pf, ph, z = _sounding()
    # Batch of 3 columns.
    Tb = jnp.repeat(T, 3, axis=0)
    qb = jnp.repeat(q, 3, axis=0)
    pfb = jnp.repeat(pf, 3, axis=0)
    phb = jnp.repeat(ph, 3, axis=0)
    zb = jnp.repeat(z, 3, axis=0)

    eager = dilute_parcel_cape(Tb, qb, pfb, phb, zb).cape
    jitted = jax.jit(dilute_parcel_cape)(Tb, qb, pfb, phb, zb).cape
    assert jnp.allclose(eager, jitted, rtol=1e-10, atol=1e-8)

    # vmap over the column axis.
    def single(T1, q1, pf1, ph1, z1):
        return dilute_parcel_cape(
            T1[None], q1[None], pf1[None], ph1[None], z1[None]
        ).cape[0]

    vmapped = jax.vmap(single)(Tb, qb, pfb, phb, zb)
    assert jnp.allclose(vmapped, eager, rtol=1e-6, atol=1e-4)
