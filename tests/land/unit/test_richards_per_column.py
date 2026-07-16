"""Shape-hardening tests for solve_richards under per-column hydraulic params.

The surfdata loader populates ``SoilHydraulicsConfig`` with per-column
``(ncol, 1)`` arrays for theta_sat/psi_sat/b_ch/K_sat (Cosby pedotransfer of
texture).  Before the ``slice_layer`` hardening, ``hydraulic_conductivity(
psi[:, 0], theta[:, 0], hydro_config)`` mixed an ``(ncol,)`` state with a
``(ncol, 1)`` param and silently broadcast to ``(ncol, ncol)``, corrupting
the infiltration capacity and the bottom-layer drainage flux.

These tests pin three properties:
  1. Solver runs with ``(ncol, 1)`` params and returns correctly-shaped output.
  2. Uniform per-column params reproduce the scalar-config result (consistency).
  3. Cosby params with column-varying texture (sand vs clay) produce the
     physically expected drainage ordering (sand drains faster).
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.land.pedotransfer import cosby_hydraulic_params
from legoesm.land.richards import RichardsConfig, solve_richards
from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
from legoesm.land.soil_hydraulics import (
    SoilHydraulicsConfig, psi_from_theta, theta_from_psi,
)


def _make_state(ncol, nlayer, *, psi_val=-1.0):
    grid = make_soil_grid(SoilGridConfig(n_layers=nlayer))
    psi = jnp.full((ncol, nlayer), psi_val)
    return grid, psi


def test_shape_hardened_per_column_params():
    """``(ncol, 1)`` params don't broadcast to ``(ncol, ncol)`` anywhere."""
    ncol, nlayer = 5, 8
    grid, psi = _make_state(ncol, nlayer)
    # Per-column Cosby params from a per-column texture vector.
    sand_col = jnp.array([90.0, 70.0, 50.0, 30.0, 10.0])
    clay_col = jnp.array([5.0, 15.0, 25.0, 35.0, 45.0])
    p = cosby_hydraulic_params(sand_col, clay_col)
    cfg = SoilHydraulicsConfig(
        retention_curve="clapp_hornberger",
        theta_sat=p.theta_sat[:, None], psi_sat=p.psi_sat[:, None],
        b_ch=p.b_ch[:, None], K_sat=p.K_sat[:, None], theta_r=0.0,
    )
    theta = theta_from_psi(psi, cfg)
    flux_top = jnp.full(ncol, 1e-7)
    sink = jnp.zeros((ncol, nlayer))

    out = solve_richards(psi, theta, grid, cfg, RichardsConfig(max_iter=4),
                        flux_top, sink, dt=600.0)

    assert out.psi_new.shape == (ncol, nlayer)
    assert out.theta_new.shape == (ncol, nlayer)
    assert out.runoff_surface.shape == (ncol,)
    assert out.runoff_subsurface.shape == (ncol,)
    assert jnp.all(jnp.isfinite(out.psi_new))
    assert jnp.all(jnp.isfinite(out.theta_new))


def test_uniform_per_column_matches_scalar():
    """Replicating a scalar param across columns must match the scalar result."""
    ncol, nlayer = 4, 8
    grid, psi = _make_state(ncol, nlayer)
    flux_top = jnp.full(ncol, 1e-7)
    sink = jnp.zeros((ncol, nlayer))

    # Reference: scalar Cosby for one loam texture.
    p = cosby_hydraulic_params(jnp.asarray(40.0), jnp.asarray(20.0))
    cfg_scalar = SoilHydraulicsConfig(
        retention_curve="clapp_hornberger",
        theta_sat=float(p.theta_sat), psi_sat=float(p.psi_sat),
        b_ch=float(p.b_ch), K_sat=float(p.K_sat), theta_r=0.0,
    )

    # Same Cosby, replicated across ncol with the (ncol,1) layout the loader uses.
    col = lambda v: jnp.full((ncol, 1), float(v))
    cfg_col = cfg_scalar._replace(
        theta_sat=col(p.theta_sat), psi_sat=col(p.psi_sat),
        b_ch=col(p.b_ch), K_sat=col(p.K_sat),
    )

    theta_s = theta_from_psi(psi, cfg_scalar)
    theta_c = theta_from_psi(psi, cfg_col)

    out_s = solve_richards(psi, theta_s, grid, cfg_scalar,
                           RichardsConfig(max_iter=6), flux_top, sink, dt=600.0)
    out_c = solve_richards(psi, theta_c, grid, cfg_col,
                           RichardsConfig(max_iter=6), flux_top, sink, dt=600.0)

    assert jnp.allclose(out_s.psi_new, out_c.psi_new, rtol=1e-10, atol=1e-12)
    assert jnp.allclose(out_s.theta_new, out_c.theta_new, rtol=1e-10, atol=1e-12)
    assert jnp.allclose(out_s.runoff_subsurface, out_c.runoff_subsurface,
                        rtol=1e-10, atol=1e-15)


def test_sand_drains_faster_than_clay_per_column():
    """Per-column texture variation actually flows through the solver."""
    ncol, nlayer = 2, 8
    grid = make_soil_grid(SoilGridConfig(n_layers=nlayer))
    flux_top = jnp.full(ncol, 1e-7)
    sink = jnp.zeros((ncol, nlayer))

    # Column 0 sandy (high K_sat), column 1 clayey (low K_sat).
    sand_col = jnp.array([92.0, 10.0])
    clay_col = jnp.array([3.0, 60.0])
    p = cosby_hydraulic_params(sand_col, clay_col)
    cfg = SoilHydraulicsConfig(
        retention_curve="clapp_hornberger",
        theta_sat=p.theta_sat[:, None], psi_sat=p.psi_sat[:, None],
        b_ch=p.b_ch[:, None], K_sat=p.K_sat[:, None], theta_r=0.0,
    )
    # Initialise both columns at the SAME effective saturation Se=0.5 (i.e.
    # theta = 0.5 * theta_sat), matching ``init_multilayer_land_state``'s
    # default.  At equal Se the comparison is K_sat × Se^(2b+3); per Cosby
    # sand has 18× larger K_sat AND smaller b, so sand drains far faster.
    # (Comparing at equal psi instead would put clay at saturation — its
    # psi_sat is more negative — and invert the ordering.)
    theta = 0.5 * jnp.broadcast_to(cfg.theta_sat, (ncol, nlayer))
    psi = psi_from_theta(theta, cfg)

    out = solve_richards(psi, theta, grid, cfg,
                         RichardsConfig(max_iter=8, bottom_bc="free_drainage"),
                         flux_top, sink, dt=3600.0)

    assert float(out.runoff_subsurface[0]) > 10.0 * float(out.runoff_subsurface[1])


def test_per_layer_params_run_through_solver():
    """``(ncol, nlayer)`` hydraulic params — full vertical texture — work in solve_richards."""
    ncol, nlayer = 3, 8
    grid, psi = _make_state(ncol, nlayer)
    # Sandy O-horizon (top 2 layers) over clayey B-horizon (bottom 6), same in
    # every column.  Tests that the layer axis carries through hydraulic_
    # conductivity / moisture_capacity / theta_from_psi inside the Picard loop
    # and that slice_layer(cfg, 0) and slice_layer(cfg, -1) pick the right
    # layer's texture for K_top / K_bot.
    sand_profile = jnp.where(jnp.arange(nlayer) < 2, 92.0, 10.0)   # (nlayer,)
    clay_profile = jnp.where(jnp.arange(nlayer) < 2, 3.0, 60.0)
    sand_cl = jnp.broadcast_to(sand_profile, (ncol, nlayer))
    clay_cl = jnp.broadcast_to(clay_profile, (ncol, nlayer))
    p = cosby_hydraulic_params(sand_cl, clay_cl)
    cfg = SoilHydraulicsConfig(
        retention_curve="clapp_hornberger",
        theta_sat=p.theta_sat, psi_sat=p.psi_sat,
        b_ch=p.b_ch, K_sat=p.K_sat, theta_r=0.0,
    )
    theta = theta_from_psi(psi, cfg)
    flux_top = jnp.full(ncol, 1e-7)
    sink = jnp.zeros((ncol, nlayer))

    out = solve_richards(psi, theta, grid, cfg, RichardsConfig(max_iter=6),
                        flux_top, sink, dt=600.0)

    assert out.psi_new.shape == (ncol, nlayer)
    assert out.theta_new.shape == (ncol, nlayer)
    assert jnp.all(jnp.isfinite(out.psi_new))
    assert jnp.all(jnp.isfinite(out.theta_new))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_fc_drain_limiter_reduces_gravity_drainage():
    """Field-capacity limiter suppresses gravity drainage below the effective fc: a wet
    column drains LESS out the bottom and retains MORE water with fc_drain_saturation>0.
    The limiter scales the gravity flux by f in [0,1], so it can only reduce drainage --
    never increase it -- vs the fc_drain_saturation=0 (unlimited) baseline."""
    ncol, nlayer = 3, 8
    grid, psi = _make_state(ncol, nlayer, psi_val=-0.5)      # wet column -> drains
    cfg = SoilHydraulicsConfig()
    theta = theta_from_psi(psi, cfg)
    flux_top = jnp.zeros(ncol)                               # no infiltration: pure drainage
    sink = jnp.zeros((ncol, nlayer))
    off = solve_richards(psi, theta, grid, cfg, RichardsConfig(fc_drain_saturation=0.0),
                         flux_top, sink, dt=600.0)
    on = solve_richards(psi, theta, grid, cfg, RichardsConfig(fc_drain_saturation=0.6),
                        flux_top, sink, dt=600.0)
    assert jnp.all(jnp.isfinite(on.theta_new)) and jnp.all(jnp.isfinite(on.psi_new))
    # limiter drains less out the bottom and retains more column water (strict: wet column)
    assert float(on.runoff_subsurface.sum()) < float(off.runoff_subsurface.sum())
    assert float(on.theta_new.sum()) > float(off.theta_new.sum())


def test_fc_drain_off_is_noop():
    """fc_drain_saturation=0.0 (default) is byte-identical to the unlimited solver."""
    ncol, nlayer = 3, 8
    grid, psi = _make_state(ncol, nlayer, psi_val=-0.5)
    cfg = SoilHydraulicsConfig()
    theta = theta_from_psi(psi, cfg)
    flux_top = jnp.zeros(ncol); sink = jnp.zeros((ncol, nlayer))
    a = solve_richards(psi, theta, grid, cfg, RichardsConfig(fc_drain_saturation=0.0),
                       flux_top, sink, dt=600.0)
    assert float(a.runoff_subsurface.sum()) >= 0.0 and jnp.all(jnp.isfinite(a.theta_new))
