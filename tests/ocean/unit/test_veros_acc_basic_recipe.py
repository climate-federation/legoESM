"""Tests for the legoESM-Veros acc_basic transfer recipe.

THE TRANSFER TEST: acc_basic is the analytic TKE-only twin of acc -- SAME grid /
dt / topography / forcing, but enable_eke=False (constant K_gm=K_iso=1000) and
enable_Prandtl_tke=False (constant Prandtl=10). This file verifies:

  1. The recipe maps the two config deltas onto EXISTING canonical options
     (GMRediConfig.eke=None; TKEConfig.prandtl_mode="constant", Prandtl_tke0=10),
     and SHARES every other config + builder with the acc recipe verbatim.
  2. The recipe constructs, validates, and steps once finite (with the full
     faithful stack: ab2 + rigid_lid + dt_mom_ratio=9 + additive friction +
     explicit_ab2 Coriolis).
  3. It is frozen-state probe-compatible.
  4. REGRESSION: the shared-helper refactor of veros_acc_recipe.build_acc_state
     did not perturb the acc recipe (the ACC recipe still seeds an EKE field).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.field import Field
from legoesm.ocean.fidelity.veros_acc_basic_recipe import (
    ACC_BASIC_GM_REDI_CONFIG, ACC_BASIC_TKE_CONFIG,
    build_acc_basic_model_config, build_acc_basic_recipe,
)
from legoesm.ocean.fidelity.veros_acc_recipe import (
    ACC_GM_REDI_CONFIG, ACC_TKE_CONFIG, NX, NY, NZ, build_acc_recipe,
)


# ---------------------------------------------------------------------------
# Config-delta mapping (the heart of the transfer test)
# ---------------------------------------------------------------------------


def test_eke_off_maps_to_constant_gm_redi():
    """enable_eke=False -> Veros K_gm=K_gm_0=1000 and K_iso=K_iso_0=1000 constant
    (eke.py:68-77). legoESM mapping: GMRediConfig.eke=None with kappa_GM=
    kappa_Redi=1000."""
    gm = ACC_BASIC_GM_REDI_CONFIG
    assert gm.eke is None, "acc_basic has enable_eke=False -> no prognostic EKE"
    assert gm.kappa_GM == 1000.0    # Veros K_gm_0
    assert gm.kappa_Redi == 1000.0  # Veros K_iso_0
    # The slope/taper numerics are EKE-independent in Veros -> shared with acc.
    assert gm.S_max == 0.01                      # iso_slopec
    assert gm.taper_width_frac == 0.5            # iso_dslope / iso_slopec
    assert gm.K_iso_steep == 500.0               # acc_basic.py:42
    assert gm.slope_density == "neutral"
    assert gm.implicit_K33 is True


def test_prandtl_off_maps_to_constant_prandtl():
    """enable_Prandtl_tke=False -> Veros constant Prandtl = Prandtl_tke0 = 10
    (tke.py:90). legoESM mapping: prandtl_mode='constant', Prandtl_tke0=10."""
    tke = ACC_BASIC_TKE_CONFIG
    assert tke.prandtl_mode == "constant"
    assert tke.Prandtl_tke0 == 10.0
    # Prognostic carried TKE (Veros enable_tke), adiabatic N^2 for convection.
    assert tke.prognostic is True
    assert tke.n2_mode == "adiabatic"
    # EKE off -> no eke_diss_iw recycling; but Veros's no-idemix branch
    # STILL adds K_diss_bot to the TKE forc (tke.py:178) — routed via the
    # existing source seam. (K_diss_gm + K_diss_h - P_diss_skew of
    # tke.py:176 have no legoESM TKE routing yet; documented gap in the
    # recipe.)
    assert tke.source_eke_diss is False
    assert tke.source_bottom_drag_diss is True
    # Veros vertical-metric slots (the TKE metric-consistency fix).
    assert tke.veros_dz_slots is True


def test_tke_shared_params_match_acc():
    """The non-Prandtl TKE knobs are IDENTICAL to acc (same channel physics)."""
    b, a = ACC_BASIC_TKE_CONFIG, ACC_TKE_CONFIG
    for f in ("c_k", "c_eps", "alpha_tke", "mxl_min", "tke_mxl_choice",
              "kappaM_min", "kappaM_max", "kappaH_min", "enable_kappaH_profile"):
        assert getattr(b, f) == getattr(a, f), f


# ---------------------------------------------------------------------------
# Recipe construction / sharing
# ---------------------------------------------------------------------------


def test_recipe_shares_grid_z_topography_with_acc():
    """Transfer integrity: grid, z-coord, land mask, IC, wind, restoring are
    SHARED with acc (same builders) -> bit-identical."""
    basic = build_acc_basic_recipe(with_surface_forcing=True)
    acc = build_acc_recipe(with_surface_forcing=True)
    # Grid centres + land footprint identical.
    np.testing.assert_array_equal(
        np.asarray(basic.grid.lat), np.asarray(acc.grid.lat))
    np.testing.assert_array_equal(
        np.asarray(basic.grid.lon), np.asarray(acc.grid.lon))
    np.testing.assert_array_equal(
        np.asarray(basic.land_mask), np.asarray(acc.land_mask))
    np.testing.assert_allclose(
        np.asarray(basic.z_coord.dz_ref), np.asarray(acc.z_coord.dz_ref))
    # Initial T/S identical (same linear stratification IC).
    np.testing.assert_allclose(
        np.asarray(basic.initial_state.T.data),
        np.asarray(acc.initial_state.T.data))
    # Wind stress identical.
    np.testing.assert_allclose(
        np.asarray(basic.wind_forcing.tau_x),
        np.asarray(acc.wind_forcing.tau_x))


def test_recipe_shares_dycore_options_with_acc():
    """Every dycore numerics option is the SAME as acc (the transfer point)."""
    b = build_acc_basic_model_config(with_surface_forcing=True)
    a = build_acc_recipe(with_surface_forcing=True).model_config
    for f in ("eos", "lateral_viscosity_operator", "momentum_advection",
              "momentum_flux_scheme", "vertical_momentum_scheme",
              "tracer_advection", "implicit_vertical_mixing", "K_v",
              "A_h_lat_scaling", "A_h_cos_power", "bottom_drag_r", "A_h"):
        assert getattr(b, f) == getattr(a, f), f


def test_acc_basic_recipe_no_eke_field_seeded():
    """eke=None -> the initial state must NOT carry an EKE field (the only
    seed-branch difference vs acc, which DOES seed eke)."""
    basic = build_acc_basic_recipe(with_surface_forcing=True)
    assert basic.initial_state.eke is None
    assert basic.initial_state.eke_diss is None
    # But the prognostic-TKE field IS seeded (Veros enable_tke).
    assert basic.initial_state.tke is not None
    assert basic.initial_state.tke.data.shape == (
        basic.grid.n_lat, basic.grid.n_lon, NZ - 1)


def test_acc_recipe_still_seeds_eke_field_regression():
    """REGRESSION (shared-helper refactor safety): the acc recipe MUST still
    seed its prognostic EKE field after build_acc_state was parametrized."""
    acc = build_acc_recipe(with_surface_forcing=True)
    assert acc.initial_state.eke is not None, (
        "refactor regression: acc recipe lost its EKE-field seed")
    assert acc.model_config.gm_redi is ACC_GM_REDI_CONFIG
    assert acc.physics_config.vertical_mixing.tke is ACC_TKE_CONFIG


def test_recipe_resolution_and_constants():
    recipe = build_acc_basic_recipe()
    assert recipe.z_coord.n_levels == NZ
    assert recipe.initial_state.T.data.shape == (recipe.grid.n_lat, recipe.grid.n_lon, NZ)
    from legoesm.ocean.constants_config import VEROS_CONSTANTS_CONFIG
    assert recipe.model_config.g == VEROS_CONSTANTS_CONFIG.g
    assert recipe.model_config.rho_0 == VEROS_CONSTANTS_CONFIG.rho_0
    assert recipe.model_config.constants is VEROS_CONSTANTS_CONFIG


# ---------------------------------------------------------------------------
# Validation + single finite step (faithful stack)
# ---------------------------------------------------------------------------


def _seed_ab2_rigid_lid(state, model, grid):
    """Seed the AB2 increment carry + rigid-lid streamfunction carry to zero
    (the free-run driver pattern from run_acc_freerun._run_legoesm)."""
    if state.T_incr_prev is None:
        _z = lambda d: Field(data=jnp.zeros_like(d.data), name=d.name + "_incr_prev",
                             dims=d.dims, units=d.units)
        state = state._replace(
            T_incr_prev=_z(state.T), S_incr_prev=_z(state.S),
            u_incr_prev=_z(state.u), v_incr_prev=_z(state.v))
    if state.psi is None:
        rl = model._ensure_rigid_lid_data(state)
        zV = jnp.zeros((grid.n_lat + 1, grid.n_lon + 1), dtype=state.u.data.dtype)
        zI = jnp.zeros((rl.nisle,), dtype=state.u.data.dtype)
        state = state._replace(psi=zV, dpsi=zV, dpsi_prev=zV, dpsin=zI, dpsin_prev=zI)
    return state


def test_recipe_validates_and_steps_once_finite():
    """Full faithful stack: ab2 + rigid_lid + dt_mom_ratio=9 + additive friction
    + explicit_ab2 Coriolis. Must validate and produce a finite first step."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    recipe = build_acc_basic_recipe(with_surface_forcing=True)
    cfg = recipe.model_config._replace(
        outer_integrator="ab2",
        barotropic_solver="rigid_lid",
        dt_mom_ratio=9.0,
        momentum_friction_additive=True,
        coriolis_scheme="explicit_ab2",
    )
    LatLonCGridOceanModel._validate_config(cfg)  # must not raise
    model = LatLonCGridOceanModel(recipe.grid, recipe.z_coord, cfg)
    # f*dt_mom < 0.5 at the interior max latitude (41 deg): explicit_ab2 OK.
    model.check_coriolis_stability(43200.0)

    state = _seed_ab2_rigid_lid(recipe.initial_state, model, recipe.grid)
    after = model.step(state, dt=43200.0, surface_forcing=recipe.wind_forcing)
    jax.block_until_ready(after.u.data)
    for fld in ("u", "v", "T", "S"):
        arr = np.asarray(getattr(after, fld).data)
        assert np.all(np.isfinite(arr)), f"non-finite {fld} after one step"
    # rigid lid: eta stays ~0.
    assert float(np.max(np.abs(np.asarray(after.eta.data)))) < 1e-6


def test_frozen_state_probe_compatible():
    """The frozen-state tendency probe (with_surface_forcing=False) must run on
    the acc_basic recipe and produce finite tendencies."""
    from legoesm.ocean.fidelity.tendency_probe import probe_latlon_cgrid

    recipe = build_acc_basic_recipe()  # no forcing -> probe config
    probe = probe_latlon_cgrid(
        recipe.initial_state, recipe.grid, recipe.z_coord,
        recipe.model_config, dt=4800.0)
    for name in probe._fields:
        arr = np.asarray(getattr(probe, name))
        assert np.all(np.isfinite(arr)), f"probe field {name} not finite"


def test_acc_basic_free_run_ships_faithful_stepping_composition():
    """acc_basic's own config builder is a copy of the acc config and missed
    the PR #429 composition defaults — pin the same bundle here (free-run
    path), with the frozen-probe path keeping legoESM defaults."""
    from legoesm.ocean.fidelity.veros_acc_basic_recipe import (
        build_acc_basic_model_config,
    )
    free = build_acc_basic_model_config(with_surface_forcing=True)
    assert free.outer_integrator == "ab2"
    assert free.dt_mom_ratio == 9.0           # DT_TRACER_S / DT_MOM_S
    assert free.barotropic_solver == "rigid_lid"
    assert free.coriolis_scheme == "explicit_ab2"
    assert free.ab2_scope == "advective"
    assert free.momentum_friction_additive is True

    probe = build_acc_basic_model_config(with_surface_forcing=False)
    assert probe.outer_integrator == "forward_euler"
    assert probe.coriolis_scheme == "matsuno_split"
    assert probe.ab2_scope == "total"
