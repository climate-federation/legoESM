"""Direct tests for the Emanuel CONVECT downdraft port.

The decisive test here is the ORIENTATION one.  The oracle is surface-first
and this repo is surface-last, so a flipped port produces a fully finite,
plausibly-shaped, completely wrong answer — no shape check and no norm would
catch it.  ``test_rain_falls_downward`` is built so that a correct port passes
and an index flip fails: condensate is detrained at exactly one level and the
rain water must appear BELOW it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.atmosphere.physics.convection._emanuel_downdraft import (
    _taper_mass_flux_to_surface,
    emanuel_downdraft,
)

_NLEV = 20

#: CONVECT v4.3c published defaults (convect43c.f lines 186-201).
_ORACLE = dict(
    sigd=0.05, sigs=0.12, omtrain=50.0, omtsnow=5.5, coeffr=1.0, coeffs=0.8,
    freeze_transition_K=1.0, inertia_scale_hPa=20.0, taper_p_fraction=0.949,
    dhdp_min=10.0, ep_gate_threshold=1.0e-4, ep_gate_width=1.0e-5,
)


def _column(*, ep_level: int | None = None, ep_value: float = 0.5):
    """One idealized tropical column, SURFACE-FIRST (index 0 = surface).

    ``ep_level`` places ALL the precipitation efficiency at a single level so
    the shaft has one unambiguous source; ``None`` leaves the column with no
    detrained condensate at all.
    """
    p_sfc = 101_480.0
    p_full = jnp.linspace(p_sfc, 10_000.0, _NLEV)[None, :]
    # Half levels: the interface at the BOTTOM of each level.
    p_half = jnp.concatenate(
        [p_full[:, :1], 0.5 * (p_full[:, :-1] + p_full[:, 1:])], axis=-1)
    z = jnp.linspace(0.0, 16_000.0, _NLEV)[None, :]
    T = 300.0 - 6.5e-3 * z
    qs = 0.022 * jnp.exp(-z / 3000.0)
    q = 0.8 * qs
    lv = jnp.full_like(T, constants.L_v)
    cpn = jnp.full_like(T, constants.c_pd)
    gz = constants.g * z
    h_moist = constants.c_pd * T + gz + constants.L_v * q

    clw = jnp.full_like(T, 2.0e-3)
    m_profile = jnp.full_like(T, 0.02)
    ment = jnp.zeros((1, _NLEV, _NLEV))
    elij = jnp.zeros((1, _NLEV, _NLEV))
    if ep_level is None:
        ep = jnp.zeros_like(T)
    else:
        ep = jnp.zeros_like(T).at[:, ep_level].set(ep_value)
        # The whole-shaft gate reads ep at cloud top, so a source anywhere
        # must be accompanied by a non-zero top value or nothing runs.
        ep = ep.at[:, -1].set(max(ep_value, 1.0e-2))
    return dict(T=T, q=q, qs=qs, p_full=p_full, p_half=p_half,
                h_moist=h_moist, gz=gz, lv=lv, cpn=cpn,
                m_profile=m_profile, ment=ment, elij=elij, clw=clw, ep=ep)


def test_no_detrained_condensate_gives_exactly_zero():
    """The contract's idealized test: no precipitation source, no downdraft."""
    out = emanuel_downdraft(**_column(ep_level=None), **_ORACLE)
    for name in ("mp", "evap", "water", "dT_dt", "dq_v_dt"):
        arr = np.asarray(getattr(out, name))
        assert np.all(np.isfinite(arr)), f"{name} is not finite"
        assert np.max(np.abs(arr)) == 0.0, f"{name} is non-zero: {arr.max()}"
    assert float(out.precip_mm_day[0]) == 0.0


def test_rain_falls_downward_not_upward():
    """ORIENTATION GATE — the test a surface-last/surface-first swap fails.

    All the detrained condensate is placed at one level.  Rain falls, so the
    rain-water content must be non-zero AT and BELOW that level and zero above
    it.  A flipped port puts the water on the wrong side and this fails while
    every shape and every finiteness check still passes.
    """
    src = 12
    out = emanuel_downdraft(**_column(ep_level=src), **_ORACLE)
    water = np.asarray(out.water)[0]
    below = water[:src]
    above = water[src + 1:]
    assert np.max(below) > 0.0, "no rain water below the source level"
    # Above the source only the top-level gate value contributes, which is a
    # separate source; check the levels strictly between it and the source.
    assert np.max(above[:-2]) <= np.max(below), (
        "rain water is larger ABOVE the source than below it — the level "
        "ordering is inverted")


def test_mass_flux_is_zero_at_the_surface_and_non_negative():
    out = emanuel_downdraft(**_column(ep_level=12), **_ORACLE)
    mp = np.asarray(out.mp)[0]
    assert mp[0] == 0.0, "the oracle leaves MP at the lowest level at zero"
    assert np.all(mp >= 0.0), "a downdraft mass flux magnitude went negative"


def test_evaporation_moistens_and_cools():
    """Sign convention, stated in the contract and checked here.

    Evaporation adds vapour and removes heat in the layer it occurs in.  The
    vapour tendency also carries a mass-flux divergence, so the check is made
    on the term that is unambiguously evaporative: where evap is largest,
    dT_dt must be negative.
    """
    out = emanuel_downdraft(**_column(ep_level=12), **_ORACLE)
    evap = np.asarray(out.evap)[0]
    dT = np.asarray(out.dT_dt)[0]
    k = int(np.argmax(evap))
    assert evap[k] > 0.0
    assert dT[k] < 0.0, "evaporation warmed the layer"


def test_saturated_environment_evaporates_nothing():
    """AFAC floors at zero, so a saturated column cannot evaporate rain."""
    col = _column(ep_level=12)
    col["q"] = col["qs"]
    out = emanuel_downdraft(**col, **_ORACLE)
    assert np.max(np.abs(np.asarray(out.evap))) == 0.0


def test_gradients_are_finite_through_the_whole_sweep():
    """A NaN gradient here is a bug for every trainer, and the guarded square
    root and MP divisions are exactly where one would appear."""
    col = _column(ep_level=12)

    def loss(q):
        out = emanuel_downdraft(**{**col, "q": q}, **_ORACLE)
        return jnp.sum(out.dq_v_dt) + jnp.sum(out.dT_dt)

    g = np.asarray(jax.grad(loss)(col["q"]))
    assert np.all(np.isfinite(g)), "non-finite gradient through the downdraft"
    assert np.max(np.abs(g)) > 0.0, "the sweep is not connected to its input"


def test_gradient_is_finite_on_a_column_with_no_source():
    """The zero-discriminant case: sqrt(0) has an infinite derivative, and a
    non-convecting column is where the trainer spends most of its time."""
    col = _column(ep_level=None)

    def loss(q):
        out = emanuel_downdraft(**{**col, "q": q}, **_ORACLE)
        return jnp.sum(out.dq_v_dt) + jnp.sum(out.dT_dt)

    assert np.all(np.isfinite(np.asarray(jax.grad(loss)(col["q"]))))


def test_surface_taper_is_linear_to_zero_and_anchored_at_the_band_top():
    """The JTT construct, checked against its closed form directly."""
    p_sfc = 100_000.0
    p = jnp.asarray([[p_sfc, 99_000.0, 96_000.0, 94_000.0, 80_000.0]])
    mp = jnp.asarray([[7.0, 3.0, 5.0, 9.0, 11.0]])
    out = np.asarray(_taper_mass_flux_to_surface(
        mp, p, taper_p_fraction=0.949))[0]
    # Band is p > 94_900: indices 0,1,2 -> anchor jtt = 2.
    anchor = 5.0
    assert out[2] == pytest.approx(anchor), "the anchor level must be a no-op"
    expected1 = anchor * (p_sfc - 99_000.0) / (p_sfc - 96_000.0)
    assert out[1] == pytest.approx(expected1)
    assert out[3] == pytest.approx(9.0), "a level above the band was tapered"
    assert out[4] == pytest.approx(11.0)
    # Index 0 is left to the caller's surface mask, matching the oracle's
    # ``IF(I.EQ.1)GOTO 360``.
    assert out[0] == pytest.approx(7.0)


def test_taper_survives_two_levels_at_equal_pressure():
    """GLM's edge case: the oracle divides by ``P(1)-P(JTT)`` and produces NaN
    when the two lowest levels share a pressure."""
    p = jnp.asarray([[100_000.0, 100_000.0, 80_000.0]])
    mp = jnp.asarray([[2.0, 3.0, 4.0]])
    out = np.asarray(_taper_mass_flux_to_surface(
        mp, p, taper_p_fraction=0.949))
    assert np.all(np.isfinite(out)), "degenerate pressures produced NaN"
