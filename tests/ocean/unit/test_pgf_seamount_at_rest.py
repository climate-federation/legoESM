"""Seamount-at-rest PGF balance gate.

A stratified ocean AT REST over a Gaussian ridge, with buoyancy a function of
the model's OWN physical cell-center depth (``compute_centroid_depth``), is in
exact discrete hydrostatic balance: one model step must NOT generate flow.  Any
``max|du|`` is a spurious pressure-gradient-force (PGF) error from the sloping /
stepped bathymetry — the canonical Beckmann-Haidvogel seamount test.

This gate pins:
  * ``pgf_scheme="smc03"`` (Shchepetkin-McWilliams density-Jacobian) is
    well-balanced on BOTH the partial-cell coord (machine-zero) and pure z-star
    (≈1e-5, ~13000x better than the uncorrected legacy path).
  * the SELF-TEST: the legacy z-star path (no PGF correction) IS large at rest,
    so the gate is provably non-vacuous (it would catch a regression that drops
    the smc03 z-star wiring).

The IC must use the physical centroid depth, NOT ``z_full_ref`` — using the
reference depth fabricates a density gradient at the partial bottom cell that
looks like a PGF error (the bug that produced an earlier false "smc03 is buggy"
verdict).
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.grids.latlon import create_beta_plane_cgrid_geometry
from legoesm.ocean.eos import LinearEOSConfig
from legoesm.ocean.fidelity.oceananigans_recipe import oceananigans_canonical_ocean_config
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.vertical import (
    create_ocean_z_star, create_partial_cell_coordinate, compute_centroid_depth,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

NX, NY, NZ = 32, 4, 16
LX = 2.0e6
H = 2.0e3
H0 = 250.0          # ridge height [m] → partial / compressed columns over the bump
WIDTH = 2.0e4
N2 = 1e-4
LATC = -45.0
ALPHA_T = 2.0e-4
T_REF_C = 10.0
DX_M = LX / NX
F0 = 2.0 * constants.Omega * np.sin(np.radians(LATC))
DT = 150.0


def _build(partial: bool, pgf_scheme: str, orient: str = "x"):
    grid = create_beta_plane_cgrid_geometry(
        NY, NX, dx_m=DX_M, dy_m=DX_M, f0=F0, beta=0.0,
        y_origin_m=-NY * DX_M / 2, x_origin_m=-LX / 2, cartesian_pseudo_lat=True)
    z_star = create_ocean_z_star(n_levels=NZ, H_max=H)
    if orient == "y":
        # Meridional (y) bathymetry slope — exercises density_jacobian_pgf_smc03_y
        # well-balancedness independently of the x-operator. A monotone linear
        # ramp across the (few) rows is a clean slope at this coarse NY.
        ramp = np.linspace(0.0, H0, NY)
        Hb = (H - ramp)[:, None] * np.ones((1, NX))
    else:
        xc = -LX / 2 + (np.arange(NX) + 0.5) * DX_M
        hill = H0 * np.exp(-xc ** 2 / (2.0 * WIDTH ** 2))
        Hb = (H - hill)[None, :] * np.ones((NY, 1))
    z = create_partial_cell_coordinate(z_star, jnp.asarray(Hb)) if partial else z_star
    cfg = oceananigans_canonical_ocean_config(
        eos_linear=LinearEOSConfig(alpha_T=ALPHA_T, beta_S=0.0),
        g=constants.g, rho_0=1000.0, A_h=0.0, K_h=0.0,
        momentum_advection="weno5", barotropic_solver="implicit_cn",
        coriolis_scheme="explicit_ab2", bottom_drag_r=0.0,
        tracer_advection="weno5", weno_smoothness="split")
    cfg = cfg._replace(barotropic=cfg.barotropic._replace(barotropic_implicit_theta_eta=1.0,
                       barotropic_implicit_theta_pgf=1.0), pgf_scheme=pgf_scheme)
    wall = jnp.ones((NY, NX), dtype=jnp.asarray(grid.cos_lat).dtype)
    state = rest_state_latlon_cgrid_ocean(
        grid, z, land_mask_override=wall, H_bathy_override=jnp.asarray(Hb),
        T_water_init_C=T_REF_C, T_deep=T_REF_C)
    # CONSISTENT rest IC: buoyancy linear in the model's OWN physical cell-center
    # depth → exact discrete hydrostatic balance.
    H_bathy = jnp.asarray(Hb)
    z_phys = -np.asarray(compute_centroid_depth(jnp.zeros_like(H_bathy), H_bathy, z))
    mask = np.asarray(state.land_mask.data)
    T = (T_REF_C + (N2 * z_phys) / (constants.g * ALPHA_T)) * mask[:, :, None]
    state = state._replace(T=state.T.replace(data=jnp.asarray(T)))
    model = LatLonCGridOceanModel(grid, z, cfg)
    state = model.seed_scan_carry(state, DT)
    return state, model


def _rest_step_max_du(partial: bool, pgf_scheme: str, orient: str = "x") -> float:
    state, model = _build(partial, pgf_scheme, orient=orient)
    s1 = model.step(state, DT, surface_forcing=None)
    du = np.abs(np.asarray(s1.u.data) - np.asarray(state.u.data))
    dv = np.abs(np.asarray(s1.v.data) - np.asarray(state.v.data))
    return float(max(du.max(), dv.max()))


def test_smc03_partial_cell_rest_balance():
    """smc03 on partial cells: machine-zero spurious flow from rest."""
    assert _rest_step_max_du(True, "smc03") < 1e-10


def test_smc03_zstar_rest_balance():
    """smc03 generalized to z-star: well-balanced (~1e-7 measured), not the raw
    terrain-following error (~2e-2). Gate at 1e-5 — 2+ orders above the measured
    residual (EOS-curvature / dz_ref-vs-compressed rho' detail) but 3 orders
    below the uncorrected error, so a real regression still trips it."""
    assert _rest_step_max_du(False, "smc03") < 1e-5


def test_smc03_zstar_meridional_slope_balance():
    """Independently pin density_jacobian_pgf_smc03_y on z-star: a meridional
    (y) bathymetry slope at rest must not generate flow either (the x-ridge
    test above only exercises smc03_x well-balancedness)."""
    assert _rest_step_max_du(False, "smc03", orient="y") < 1e-5
    # non-vacuous: uncorrected z-star over the same y-slope IS large.
    assert _rest_step_max_du(False, "adcroft", orient="y") > 1e-2


def test_zstar_uncorrected_pgf_is_large_selftest():
    """Non-vacuous self-test: the legacy z-star path (no PGF correction) spins up
    a LARGE spurious flow from rest — so the smc03 gate above is provably
    catching a real defect, not passing trivially."""
    legacy = _rest_step_max_du(False, "adcroft")   # adcroft on z-star == no correction
    assert legacy > 1e-2
    # and smc03 must be at least 1000x better than the uncorrected path
    assert _rest_step_max_du(False, "smc03") < legacy / 1000.0


def test_unknown_pgf_scheme_raises():
    """Dispatch hardening: a typo'd pgf_scheme fails loudly at construction,
    never silently falls through to the uncorrected gradient (which on z-star
    is the full terrain-following PGF error)."""
    with pytest.raises(ValueError, match="pgf_scheme must be one of"):
        _build(True, "smc3_typo")
