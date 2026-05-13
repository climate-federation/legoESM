"""Regression test for the iter-42..48 fp64 diagnostic-dtype contract.

Every conservation / energy / moisture diagnostic that takes
prognostic state arrays as input must return fp64 when JAX x64 is
enabled.  A regression that drops back to fp32 (e.g. removing an
``astype(_conservation_accumulator())`` call from a helper) would
re-introduce the fp32-field noise iter-42..48 eliminated.

Each test calls the helper on a small fp32-input fixture and asserts
the returned scalar (or array) is fp64.
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest


def test_column_water_vapor_returns_fp64():
    """iter-48: diagnostics/column_integrals.column_water_vapor."""
    from legoesm.diagnostics.column_integrals import column_water_vapor
    q_v = jnp.ones((6, 8, 8, 10), dtype=jnp.float32) * 1e-2
    p_s = jnp.ones((6, 8, 8), dtype=jnp.float32) * 1e5
    dsigma = jnp.ones(10, dtype=jnp.float32) / 10
    out = column_water_vapor(q_v, p_s, dsigma)
    assert out.dtype == jnp.float64, (
        f"column_water_vapor dropped fp64 promotion: dtype={out.dtype}"
    )


def test_column_moist_static_energy_returns_fp64():
    """iter-47: diagnostics/energy_budget.column_moist_static_energy."""
    from legoesm.diagnostics.energy_budget import column_moist_static_energy
    T = jnp.ones((4, 4, 10), dtype=jnp.float32) * 280.0
    q_v = jnp.ones((4, 4, 10), dtype=jnp.float32) * 5e-3
    u = jnp.zeros((4, 4, 10), dtype=jnp.float32)
    v = jnp.zeros((4, 4, 10), dtype=jnp.float32)
    phis = jnp.zeros((4, 4), dtype=jnp.float32)
    p_s = jnp.ones((4, 4), dtype=jnp.float32) * 1e5
    dsigma = jnp.ones(10, dtype=jnp.float32) / 10
    sigma_full = jnp.linspace(0.05, 0.95, 10, dtype=jnp.float32)
    out = column_moist_static_energy(
        T, q_v, u, v, phis, p_s, dsigma, sigma_full,
    )
    assert out.dtype == jnp.float64


def test_column_dry_static_energy_returns_fp64():
    """iter-47: diagnostics/energy_budget.column_dry_static_energy."""
    from legoesm.diagnostics.energy_budget import column_dry_static_energy
    T = jnp.ones((4, 4, 10), dtype=jnp.float32) * 280.0
    phis = jnp.zeros((4, 4), dtype=jnp.float32)
    p_s = jnp.ones((4, 4), dtype=jnp.float32) * 1e5
    dsigma = jnp.ones(10, dtype=jnp.float32) / 10
    sigma_full = jnp.linspace(0.05, 0.95, 10, dtype=jnp.float32)
    out = column_dry_static_energy(T, phis, p_s, dsigma, sigma_full)
    assert out.dtype == jnp.float64


def test_compute_global_moisture_returns_fp64():
    """iter-44: core/conservation.compute_global_moisture."""
    from legoesm.core.conservation import compute_global_moisture
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    grid = create_cubed_sphere(8)
    q_v = jnp.ones((6, 8, 8, 10), dtype=jnp.float32) * 5e-3
    p_s = jnp.ones((6, 8, 8), dtype=jnp.float32) * 1e5
    dsigma = jnp.ones(10, dtype=jnp.float32) / 10
    out = compute_global_moisture(q_v, p_s, dsigma, grid)
    assert out.dtype == jnp.float64


def test_compute_atmospheric_angular_momentum_returns_fp64():
    """iter-46: diagnostics/angular_momentum AAM helper."""
    from legoesm.diagnostics.angular_momentum import (
        compute_atmospheric_angular_momentum,
    )
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    grid = create_cubed_sphere(8)

    class _HC:
        dz = jnp.ones(10) * 100.0
        rho_ref = jnp.ones(10) * 1.0

    hc = _HC()
    u = jnp.ones((6, 8, 8, 10), dtype=jnp.float32) * 10.0
    rho_full = jnp.ones((6, 8, 8, 10), dtype=jnp.float32) * 1.0
    aam_column, aam_total = compute_atmospheric_angular_momentum(
        u, rho_full, grid, hc,
    )
    assert aam_column.dtype == jnp.float64, (
        f"AAM column dropped fp64: {aam_column.dtype}"
    )
    # aam_total is Python float; just assert finite + non-zero.
    assert jnp.isfinite(aam_total)


def test_compute_total_energy_pe_returns_fp64():
    """iter-45: diagnostics/total_energy_pe.compute_total_energy_pe."""
    from legoesm.diagnostics import compute_total_energy_pe
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.core.state import FV3HydrostaticState
    from legoesm.core.field import Field
    from legoesm.grids.vertical import standard_hybrid_levels

    grid = create_cubed_sphere(8)
    coord = standard_hybrid_levels(10)
    n = grid.n
    nlev = 10
    state = FV3HydrostaticState(
        u_d=Field(data=jnp.zeros((6, n + 1, n + 1, nlev), dtype=jnp.float32),
                  name="u_d", dims=("face", "i", "j", "k")),
        v_d=Field(data=jnp.zeros((6, n + 1, n + 1, nlev), dtype=jnp.float32),
                  name="v_d", dims=("face", "i", "j", "k")),
        T=Field(data=jnp.full((6, n, n, nlev), 280.0, dtype=jnp.float32),
                name="T", dims=("face", "i", "j", "k")),
        p_s=Field(data=jnp.full((6, n, n), 1e5, dtype=jnp.float32),
                  name="p_s", dims=("face", "i", "j")),
        phis=Field(data=jnp.zeros((6, n, n), dtype=jnp.float32),
                   name="phis", dims=("face", "i", "j")),
    )
    te_column, te_total = compute_total_energy_pe(state, grid, coord)
    assert te_column.dtype == jnp.float64, (
        f"PE total-energy column dropped fp64: {te_column.dtype}"
    )
    # te_total is Python float; assert finite + non-zero.
    assert te_total > 0


def test_compute_total_energy_nh_returns_fp64():
    """iter-45: diagnostics/total_energy_nh.compute_total_energy_nh."""
    from legoesm.diagnostics import compute_total_energy_nh
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from tests.test_cases.dcmip2025 import dcmip25_tc1_init

    grid = create_cubed_sphere(8)
    state, hcoord, _tmetric = dcmip25_tc1_init(grid, n_levels=8)
    te_column, te_total = compute_total_energy_nh(state, grid, hcoord)
    assert te_column.dtype == jnp.float64, (
        f"NH total-energy column dropped fp64: {te_column.dtype}"
    )
    # te_total is a Python float; sanity-check non-zero positive
    # (atmosphere has positive internal+gravitational+kinetic energy).
    assert te_total > 0
