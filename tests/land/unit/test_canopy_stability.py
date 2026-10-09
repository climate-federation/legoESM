"""Unit tests for canopy/stability.py boundary-layer + MOST parameters.

Covers the DifferBESS aa6e8b9 boundary-layer corrections:
- ``compute_boundary_layer_resistance`` reads ``cv`` / ``d_leaf`` from the
  caller (no longer hard-coded 0.01 / 0.04).
- The CLM5-aligned config defaults (cv = 0.0135) and the per-PFT leaf-width
  table are present.
- MOST runs finite/positive after the kB^-1 = 0 (z0h = z0m) change.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants

from legoesm.land.canopy.config import CanopyConfig, PFT_LEAF_WIDTH
from legoesm.land.canopy.stability import (
    compute_boundary_layer_resistance,
    monin_obukhov_stability,
)


def test_boundary_layer_resistance_matches_forced_convection_formula():
    uav = jnp.array([2.0]); LAI = jnp.array([3.0]); fSun = jnp.array([0.6])
    cv = jnp.array([0.0135]); d_leaf = jnp.array([0.025])
    Rb_Sun, Rb_Sh = compute_boundary_layer_resistance(uav, LAI, fSun, cv, d_leaf)
    rb = 1.0 / (cv * jnp.sqrt(uav / d_leaf))
    assert jnp.allclose(Rb_Sun, rb / (LAI * fSun), rtol=1e-6)
    assert jnp.allclose(Rb_Sh, rb / (LAI * (1.0 - fSun)), rtol=1e-6)


def test_boundary_layer_resistance_depends_on_cv_and_dleaf():
    """Proves cv/d_leaf are live parameters, not the old hard-coded literals."""
    uav = jnp.array([2.0]); LAI = jnp.array([3.0]); fSun = jnp.array([0.5])
    Rb_a, _ = compute_boundary_layer_resistance(
        uav, LAI, fSun, jnp.array([0.0135]), jnp.array([0.025]))
    # Old BESS values (cv=0.01, d_leaf=0.04) must give a different Rb.
    Rb_b, _ = compute_boundary_layer_resistance(
        uav, LAI, fSun, jnp.array([0.01]), jnp.array([0.04]))
    assert not jnp.allclose(Rb_a, Rb_b)
    # Larger cv -> smaller resistance.
    Rb_hi, _ = compute_boundary_layer_resistance(
        uav, LAI, fSun, jnp.array([0.02]), jnp.array([0.025]))
    assert float(Rb_hi[0]) < float(Rb_a[0])


def test_canopy_config_clm5_boundary_layer_defaults():
    cfg = CanopyConfig()
    assert cfg.cv == 0.0135
    # PFT leaf-width table: small needles vs broad leaves (Schuepp 1993).
    assert PFT_LEAF_WIDTH["ENF"] == 0.01
    assert PFT_LEAF_WIDTH["EBF"] == 0.04
    assert PFT_LEAF_WIDTH["DBF"] == 0.025


def test_most_finite_after_kb_inv_zero():
    """MOST returns finite, positive resistances after z0h = z0m (kB^-1 = 0)."""
    one = jnp.array([1.0])
    ustar, rah, raw, uav, _zeta = monin_obukhov_stability(
        ur=3.0 * one, Ta=300.0 * one, Tv_atm=300.5 * one, Tc=301.0 * one,
        q_atm=0.010 * one, q_c=0.011 * one, zldis=10.0 * one, z0m=0.5 * one,
        zeta_cap_width=CanopyConfig().zeta_cap_smoothing_width)
    for v in (ustar, rah, raw, uav):
        assert bool(jnp.all(jnp.isfinite(v)))
    assert float(ustar[0]) > 0.0
    assert float(rah[0]) > 0.0


def test_resistances_are_continuous_through_neutral_stability():
    """Sweeping the canopy temperature through neutral must not make the
    friction velocity or the aerodynamic resistance jump: a jump there leaves
    the canopy Newton solve with no root at sunset (CLM5's 0.01 |zeta| floor
    jumped rah ~8% at neutral)."""
    import jax
    from legoesm.land.canopy.stability import monin_obukhov_stability
    Tc = jnp.linspace(299.0, 301.5, 2501)
    out = jax.vmap(lambda t: monin_obukhov_stability(
        jnp.asarray(3.0), jnp.asarray(300.0), jnp.asarray(300.5), t,
        jnp.asarray(0.010), jnp.asarray(0.011), jnp.asarray(30.0),
        jnp.asarray(0.1),
        zeta_cap_width=CanopyConfig().zeta_cap_smoothing_width))(Tc)
    zeta = out[4]
    assert bool(jnp.any(zeta > 0)) and bool(jnp.any(zeta < 0))   # sweep crosses neutral
    for name, v in (("ustar", out[0]), ("rah", out[1])):
        step = jnp.abs(jnp.diff(v))
        assert float(step.max()) < 20.0 * float(jnp.median(step)) + 1e-12, name


def test_dense_canopy_night_column_from_amip_now_has_a_root():
    """A tropical-forest night column captured from a real AMIP run.  Under
    CLM5's stable forms its canopy-air energy balance had no root (sensible
    heat fell as the air-surface temperature difference grew) and the solve
    stalled at the damping ceiling; with the Beljaars-Holtslag stable side it
    converges."""
    from legoesm.land.canopy.config import CanopyConfig
    from legoesm.land.canopy.solver import CanopyForcingBundle, solve_canopy_closure
    col = dict(LAI=4.951489473647515, SZA=90.0, La=404.5686950683594, epsf=0.97,
               epss=0.96, fSun=0.00019156204786448017, APAR_Sun=0.0, APAR_Sh=0.0,
               Vcmax25_Sun=1.1131688309543069, Vcmax25_Sh=245.97412737341875,
               Vcmax25_C4Sun=0.0, Vcmax25_C4Sh=0.0, ASW_Sun=0.0, ASW_Sh=0.0,
               ASW_Soil=0.0, Ts_bc=297.7799184556097, Ca=415.0,
               Ps=99471.85028998576, Ta=297.6650924946381, lam=constants.L_v,
               Cp=constants.c_pd, rhoa=1.1641655345665334, Tv_atm=300.9427827761013,
               q_atm=0.018117642632922024, m=8.15500771186095,
               b0=0.0090611196798455, alf=0.3, TgC=25.0, fC4=0.0,
               fStress_soil=0.9999694356760194, ur=2.283843242333378, CI=0.75,
               z0m=2.3727493400205093, displa=21.196560770849885,
               z0=46.924054171054976, cv=0.0135, d_leaf=0.025,
               r_soil_surface=461.70504797067184, fwet=0.0)
    b = CanopyForcingBundle(**{k: jnp.asarray(v) for k, v in col.items()})
    x0 = jnp.asarray([297.6650924946381, 297.6650924946381, 290.5, 290.5,
                      297.6650924946381, 0.01884268315515326])
    x, _n, converged = solve_canopy_closure(x0, b, CanopyConfig())
    assert bool(converged)
    assert float(jnp.max(jnp.abs(x[:2] - x[4]))) < 5.0


def test_stable_zeta_cap_is_smooth_bounded_and_near_identity_below():
    import jax
    import numpy as np
    from legoesm.land.canopy.stability import (
        _ZETA_MAX_STABLE as ZMAX, _cap_stable_zeta)
    W = CanopyConfig().zeta_cap_smoothing_width
    cap = lambda z: float(_cap_stable_zeta(jnp.asarray(z), W))
    assert abs(cap(0.1) - 0.1) < W * np.exp(-(ZMAX - 0.1) / W)
    assert abs(cap(ZMAX) - (ZMAX - W * np.log(2.0))) < 1e-6
    assert ZMAX - 1e-6 < cap(50.0) <= ZMAX
    assert ZMAX - W * np.log(2.0) < cap(0.6) < ZMAX
    # the neutral floor survives the cap (softplus leakage must not push the
    # stable branch negative, into the unstable forms)
    for z in (0.0, 1e-6, 2e-6, 1e-4):
        assert cap(z) >= 1e-6
    assert abs(cap(1e-3) - 1e-3) < 1e-5
    # derivative continuous through the cap (the hard clip jumped 1 -> 0 here)
    d = jax.grad(lambda z: _cap_stable_zeta(z, W))
    lo, hi = float(d(jnp.asarray(ZMAX - 1e-9))), float(d(jnp.asarray(ZMAX + 1e-9)))
    assert abs(lo - 0.5) < 1e-6 and abs(hi - 0.5) < 1e-6


# Captured production AMIP column (2026-09-28, 40962-cell mesh): stable air at
# sunset over a short C4 canopy; 55 iterations without converging under the
# hard stable-zeta clip, 3 with the smooth cap.
_ZCAP_X0 = [297.2393324175292, 293.68368599247435, 247.52055596565017,
            373.49999999999994, 294.92259259851295, 0.005930197400070915]
_ZCAP_COL = dict(
    LAI=0.6471436963540378, SZA=88.6781908115728, La=301.63646125793457,
    epsf=0.97, epss=0.96, fSun=0.0712566808433442, APAR_Sun=198.74736326526045,
    APAR_Sh=9.593855714286633, Vcmax25_Sun=0.0, Vcmax25_Sh=0.0,
    Vcmax25_C4Sun=0.7914995066910798, Vcmax25_C4Sh=9.363296419511164,
    ASW_Sun=60.46099740968655, ASW_Sh=26.749205460570906,
    ASW_Soil=13.209926935248177, Ts_bc=293.7946313501537, Ca=415.0,
    Ps=99697.46758811206, Ta=295.31354669894745, lam=constants.L_v,
    Cp=constants.c_pd, rhoa=1.1786080175081368, Tv_atm=296.06634429267206,
    q_atm=0.004194271465023573, m=1.894397743108224, b0=0.01894397743108224,
    alf=0.3, TgC=25.0, fC4=1.0, fStress_soil=0.017304065895086675,
    ur=2.946329212745357, CI=0.75, z0m=0.026840399618048596,
    displa=0.18735207624146172, z0=64.45224515471612, cv=0.0135, d_leaf=0.025,
    r_soil_surface=1331.4987749450343, fwet=0.0)


def test_stable_column_converges_only_with_the_smooth_zeta_cap():
    from legoesm.land.canopy.solver import CanopyForcingBundle, solve_canopy_closure
    b = CanopyForcingBundle(**{k: jnp.asarray(v) for k, v in _ZCAP_COL.items()})
    x0 = jnp.asarray(_ZCAP_X0)
    _, n, conv = solve_canopy_closure(x0, b, CanopyConfig())
    assert bool(conv) and int(n) < 10
    _, _, conv_hard = solve_canopy_closure(          # ~hard clip
        x0, b, CanopyConfig(zeta_cap_smoothing_width=1e-9))
    assert not bool(conv_hard)


def test_zeta_cap_width_is_validated_and_reaches_the_solve():
    import pytest
    from legoesm.land.canopy.solver import CanopyForcingBundle, solve_canopy_closure
    for bad in (0.0, 1.0):
        with pytest.raises(ValueError, match="zeta_cap_smoothing_width"):
            CanopyConfig(zeta_cap_smoothing_width=bad).validate()
    b = CanopyForcingBundle(**{k: jnp.asarray(v) for k, v in _ZCAP_COL.items()})
    with pytest.raises(ValueError, match="zeta_cap_smoothing_width"):
        solve_canopy_closure(jnp.asarray(_ZCAP_X0), b,
                             CanopyConfig(zeta_cap_smoothing_width=0.0))


def test_most_refuses_zero_iterations():
    import pytest
    one = jnp.array([1.0])
    with pytest.raises(ValueError, match="n_iters"):
        monin_obukhov_stability(
            ur=3.0 * one, Ta=300.0 * one, Tv_atm=300.5 * one, Tc=301.0 * one,
            q_atm=0.010 * one, q_c=0.011 * one, zldis=10.0 * one, z0m=0.5 * one,
            n_iters=0, zeta_cap_width=CanopyConfig().zeta_cap_smoothing_width)


def test_most_n_iters_reaches_the_canopy_solve():
    """The canopy residual must run the configured number of MOST iterations:
    on the captured stable column a 1-iteration loop gives a different root."""
    from legoesm.land.canopy.solver import CanopyForcingBundle, solve_canopy_closure
    import pytest
    b = CanopyForcingBundle(**{k: jnp.asarray(v) for k, v in _ZCAP_COL.items()})
    x10, _, ok10 = solve_canopy_closure(jnp.asarray(_ZCAP_X0), b, CanopyConfig())
    x1, _, ok1 = solve_canopy_closure(jnp.asarray(_ZCAP_X0), b, CanopyConfig(most_n_iters=1))
    assert bool(ok10) and bool(ok1)
    assert float(jnp.max(jnp.abs(x10 - x1))) > 1e-6
    with pytest.raises(ValueError, match="most_n_iters"):
        CanopyConfig(most_n_iters=0).validate()
    assert CanopyConfig().most_n_iters == 10
