"""Soft energy-balance LE cap + minimum cuticular conductance floor.

Two robustness features keep the canopy leaf-temperature Newton solve from
diverging under hot / dry / high-VPD forcing (observed at sparse dry FLUXNET
sites such as US-Ton savanna):

* ``apply_le_cap`` -- a smooth (softplus) bound on the latent-heat flux that still
  admits nocturnal dew and a modest daytime LE > Rn, but stops the runaway that
  pushes the leaf energy balance to NaN.  Default ``"soft"``; ``"off"`` reproduces
  the un-capped behaviour; ``"hard"`` is the legacy ``clip(LE, 0, max(Rn,0))``.
* ``_GS_MIN_MOL`` -- a minimum cuticular conductance.  Under full water stress with
  the legacy/default ``CanopyConfig.stress_b0=True`` the stress factor scales the
  Ball-Berry slope AND intercept to zero, which without a floor gives ``gs = 0``, a
  singular Newton Jacobian, and NaN fluxes.  (With ``stress_b0=False`` b0 stays > 0
  and the floor is inactive.)

These tests pin the cap's bounds + smoothness + differentiability, the dispatch
hardening, and the dry-site (full-stress) NaN regression at the leaf-balance and
full-canopy levels.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.surface_scheme import TwoLeafCanopyConfig
from legoesm.land.surface_scheme.two_leaf_canopy import (
    compute_two_leaf_canopy_fluxes,
)
from legoesm.land.canopy.energy_balance import (
    apply_le_cap,
    leaf_energy_balance_bt,
)


# --------------------------------------------------------------------------- #
# apply_le_cap                                                                 #
# --------------------------------------------------------------------------- #

def test_soft_cap_passes_normal_le_through():
    # LE well inside [-SOFT_LOW, max(Rn,0)+slack] is essentially unchanged.
    out = apply_le_cap(jnp.array(120.0), jnp.array(400.0), "soft")
    assert abs(float(out) - 120.0) < 1.0


def test_soft_cap_bounds_runaway_le():
    # A huge LE is bounded to roughly max(Rn,0) + day-slack (~80 W/m2).
    out = float(apply_le_cap(jnp.array(5000.0), jnp.array(300.0), "soft"))
    assert out < 300.0 + 80.0 + 5.0          # below cap_hi + softplus margin
    assert out > 300.0                        # but allowed to exceed Rn somewhat


def test_soft_cap_passes_dew_through():
    # Upper-only bound: negative LE (dew/condensation) passes through unchanged
    # (it is bounded by the humidity gradient, not by the cap).
    out = float(apply_le_cap(jnp.array(-50.0), jnp.array(0.0), "soft"))
    assert abs(out - (-50.0)) < 1.0


def test_soft_cap_zero_flux_stays_near_zero():
    # A zero raw flux must NOT become spurious evaporation (no lower-bound
    # offset); holds across day/night Rn (regression for codex finding 1).
    for rn in (-50.0, 0.0, 100.0, 500.0):
        out = float(apply_le_cap(jnp.array(0.0), jnp.array(rn), "soft"))
        assert abs(out) < 1.0


def test_soft_cap_day_slack_exceeds_night_slack():
    # The upper bound is wider by day (Rn large) than at night (Rn <= 0).
    hi_day = float(apply_le_cap(jnp.array(5000.0), jnp.array(600.0), "soft"))
    hi_night = float(apply_le_cap(jnp.array(5000.0), jnp.array(-50.0), "soft"))
    assert hi_day > hi_night


def test_soft_cap_is_differentiable_at_the_bound():
    # softplus => finite gradient everywhere, including where the cap engages.
    g = jax.grad(lambda le: apply_le_cap(le, jnp.array(300.0), "soft"))
    for le in (-50.0, 0.0, 300.0, 380.0, 1000.0):
        assert jnp.isfinite(g(jnp.array(le)))


def test_off_mode_is_identity():
    LE = jnp.array([2000.0, -50.0, 100.0])
    assert jnp.allclose(apply_le_cap(LE, jnp.array([300.0, 0.0, 400.0]), "off"), LE)


def test_hard_mode_clips_to_available_energy():
    out = apply_le_cap(jnp.array([2000.0, -50.0, 100.0]),
                       jnp.array([300.0, 0.0, 400.0]), "hard")
    assert jnp.allclose(out, jnp.array([300.0, 0.0, 100.0]))


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="le_cap_mode"):
        apply_le_cap(jnp.array(100.0), jnp.array(300.0), "bogus")


# --------------------------------------------------------------------------- #
# minimum cuticular conductance floor (full water stress)                      #
# --------------------------------------------------------------------------- #

_LEAF = dict(
    ASW=jnp.array(250.0), ALW=jnp.array(-40.0), Tf=jnp.array(305.0),
    Ps=jnp.array(101325.0), Ca=jnp.array(400.0), Tc=jnp.array(303.0),
    q_f=jnp.array(0.030), q_c=jnp.array(0.008), RH_c=jnp.array(0.3),
    VPD_c=jnp.array(3000.0), lam=jnp.array(2.5e6), Cp=jnp.array(1005.0),
    rhoa=jnp.array(1.2), Rb=jnp.array(40.0),
)


def test_leaf_balance_finite_at_full_water_stress():
    # Legacy stress_b0=True: full stress drives the Ball-Berry slope AND intercept
    # to zero (explicit m=b0=0 here).  The conductance floor must keep gs>0 so
    # LE/Tf are finite, not NaN.  (stress_b0=False instead keeps b0>0 directly.)
    Rn, LE, H, Tf_new, gs, Ci = leaf_energy_balance_bt(
        An=jnp.array(0.0), m=jnp.array(0.0), b0=jnp.array(0.0), **_LEAF)
    for v in (LE, H, Tf_new, gs, Ci):
        assert jnp.all(jnp.isfinite(v))
    assert float(gs) > 0.0                 # floored, not exactly zero


def test_leaf_balance_grad_finite_at_full_water_stress():
    # The gradient wrt the (zero) stomatal slope is finite at full stress.
    def le_of_m(m):
        _, LE, *_ = leaf_energy_balance_bt(
            An=jnp.array(0.0), m=m, b0=jnp.array(0.0), **_LEAF)
        return LE
    assert jnp.isfinite(jax.grad(le_of_m)(jnp.array(0.0)))


def _dry_forcing(ncol):
    return AtmToSurface(
        T_lowest=jnp.full(ncol, 313.0), q_lowest=jnp.full(ncol, 0.003),
        u_lowest=jnp.full(ncol, 2.0), v_lowest=jnp.zeros(ncol),
        p_lowest=jnp.full(ncol, 98000.0), p_surface=jnp.full(ncol, 101325.0),
        rho_lowest=jnp.full(ncol, 1.10), sw_down=jnp.full(ncol, 950.0),
        lw_down=jnp.full(ncol, 360.0), cos_zenith=jnp.full(ncol, 0.9),
        precip_total=jnp.zeros(ncol), precip_snow=jnp.zeros(ncol),
        co2_ppmv=jnp.full(ncol, 410.0), has_radiation=True,
        has_precipitation=False,
    )


def test_full_canopy_finite_at_zero_soil_moisture():
    # Regression: hot/dry/high-VPD forcing with w_frac_rz=0 (bone-dry soil) used
    # to diverge to NaN in the leaf-temperature Newton solve.  The conductance
    # floor + soft LE cap keep the full canopy closure finite.
    ncol = 2
    T_soil = jnp.full(ncol, 312.0)
    out = compute_two_leaf_canopy_fluxes(
        T_soil_top=T_soil, forcing=_dry_forcing(ncol),
        canopy_config=TwoLeafCanopyConfig(max_iters=30),
        land_config=MultiLayerLandConfig(), canopy_params=None,
        w_frac_rz=jnp.zeros(ncol), wind_speed=jnp.full(ncol, 2.0),
        wind_dir_x=jnp.ones(ncol), wind_dir_y=jnp.zeros(ncol),
        soil_thermal_fn=lambda G, dt_: T_soil, dt=1800.0,
    )
    for v in (out.lhflx, out.shflx, out.gpp, out.Tf_Sun, out.Tf_Sh,
              out.T_surface, out.T_canopy_air):
        assert jnp.all(jnp.isfinite(v))
