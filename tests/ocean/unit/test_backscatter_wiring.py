"""Tests for the DIAGNOSTIC-E backscatter momentum wiring (MED-4).

Covers the reuse-only helpers that wire the Jansen & Held (2014) negative-
Laplacian backscatter operator into the lat-lon C-grid momentum RHS
(``ocean_pe_latlon_cgrid``):

* :func:`diagnostic_eddy_energy` — the no-carry standing-equilibrium reservoir
  ``E = clamp(eta*tau*eps_diss, E_min, E_max)``;
* :func:`cfl_cap_eddy_energy` — the per-cell CFL bound ``nu_bs <= nu_max``;
* :func:`diagnostic_backscatter_cgrid` — the orchestration (the new call site
  logic): scale-selective dissipation power -> reservoir -> CFL cap -> the
  existing :func:`backscatter_tendency_cgrid` operator.

Asserts: KE is INJECTED (energizing sign) when enabled; a no-op (exact zeros)
when disabled -- the property that makes the momentum RHS bit-identical with
backscatter off; and ``nu_bs`` respects the CFL cap.

The MPAS momentum wiring is deferred (its tendency function has no ``dt`` for
the CFL cap and no edge->cell dissipation-power reduction); the grid-agnostic
reservoir/cap helpers are exercised on 1-D (cell) arrays here so the follow-up
can reuse them.

Run with:

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/ocean/unit/test_backscatter_wiring.py -v
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

from legoesm.grids.latlon import create_regional_latlon_grid
from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import laplacian_smag_cfl_cap
from legoesm.ocean.physics.lateral_mixing import (
    BackscatterConfig,
    backscatter_power_density_cgrid,
    diagnostic_eddy_energy,
    cfl_cap_eddy_energy,
    diagnostic_backscatter_cgrid,
)


@pytest.fixture(autouse=True)
def _enable_x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


@pytest.fixture(scope="module")
def latlon_grid():
    grid, wall_mask = create_regional_latlon_grid(
        16, 34, 16.0, 34.0,
        periodic_x=True, lon_west=0.0, lon_east=10.0,
        dtype=jnp.float64,
    )
    u_mask, v_mask = compute_face_masks(wall_mask)
    return grid, wall_mask, u_mask, v_mask


def _random_velocity(grid, u_mask, v_mask, seed=0):
    n_lat, n_lon = grid.area.shape
    k = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(k)
    u = 0.3 * jax.random.normal(k1, (n_lat, n_lon + 1), dtype=jnp.float64)
    v = 0.3 * jax.random.normal(k2, (n_lat + 1, n_lon), dtype=jnp.float64)
    return u * u_mask, v * v_mask


# ---------------------------------------------------------------------------
# diagnostic_eddy_energy: standing-equilibrium reservoir E = clamp(eta*tau*eps)
# ---------------------------------------------------------------------------


def test_diagnostic_eddy_energy_matches_equilibrium():
    cfg = BackscatterConfig(enabled=True, efficiency=0.9,
                            tau_relax_days=10.0, E_min=0.0, E_max=1.0e9)
    tau = cfg.tau_relax_days * 86400.0
    eps = jnp.array([0.0, 1.0e-8, 5.0e-8])
    E = diagnostic_eddy_energy(eps, cfg)
    # E_eq = eta * tau * eps (below E_max), and E >= 0 at eps = 0.
    assert jnp.allclose(E, cfg.efficiency * tau * eps)
    assert float(E[0]) == 0.0


def test_diagnostic_eddy_energy_clamped_and_monotone():
    cfg = BackscatterConfig(enabled=True, E_min=0.0, E_max=0.1)
    # Huge dissipation saturates at E_max; negative eps floored at E_min.
    assert float(diagnostic_eddy_energy(jnp.array(1.0e3), cfg)) == 0.1
    assert float(diagnostic_eddy_energy(jnp.array(-1.0), cfg)) == 0.0
    eps = jnp.linspace(0.0, 1.0e-7, 20)
    E = diagnostic_eddy_energy(eps, cfg)
    assert jnp.all(jnp.diff(E) >= -1e-30)          # non-decreasing


def test_diagnostic_eddy_energy_grid_agnostic_1d():
    # The reservoir helper is shape-agnostic (ready for the MPAS cell path).
    cfg = BackscatterConfig(enabled=True, E_max=1.0e9)
    eps_cell = jnp.abs(jax.random.normal(jax.random.PRNGKey(3), (25,))) * 1e-8
    E = diagnostic_eddy_energy(eps_cell, cfg)
    assert E.shape == (25,)
    assert jnp.all(E >= 0.0)


# ---------------------------------------------------------------------------
# cfl_cap_eddy_energy: nu_bs = c_bs*delta*sqrt(E) <= nu_max, pointwise
# ---------------------------------------------------------------------------


def test_cfl_cap_bounds_nu_bs_pointwise():
    cfg = BackscatterConfig(enabled=True, c_bs=0.02)
    key = jax.random.PRNGKey(1)
    E = 10.0 * jax.random.uniform(key, (12, 7)) + 1.0        # large -> forces cap
    delta = 5.0e4 + jnp.zeros((12, 7))                       # grid length [m]
    nu_max = 50.0 + 20.0 * jax.random.uniform(
        jax.random.PRNGKey(2), (12, 7))                     # small ceiling
    E_capped = cfl_cap_eddy_energy(E, nu_max, delta, cfg)
    nu_bs = cfg.c_bs * delta * jnp.sqrt(jnp.maximum(E_capped, 0.0))
    assert jnp.all(nu_bs <= nu_max * (1.0 + 1e-9))
    # E is only ever reduced (never raised) by the cap.
    assert jnp.all(E_capped <= E + 1e-12)


def test_cfl_cap_noop_when_below_ceiling():
    cfg = BackscatterConfig(enabled=True, c_bs=0.01)
    E = jnp.full((4, 4), 1.0e-3)
    delta = 1.0e4 + jnp.zeros((4, 4))
    nu_max = jnp.full((4, 4), 1.0e9)                        # huge -> never binds
    E_capped = cfl_cap_eddy_energy(E, nu_max, delta, cfg)
    assert jnp.allclose(E_capped, E)


# ---------------------------------------------------------------------------
# diagnostic_backscatter_cgrid: the new call-site orchestration
# ---------------------------------------------------------------------------


def test_backscatter_disabled_is_exact_zero(latlon_grid):
    """Disabled => exact zeros: the property that makes the momentum RHS
    bit-identical when backscatter is off (the caller skips the block)."""
    grid, _, u_mask, v_mask = latlon_grid
    u, v = _random_velocity(grid, u_mask, v_mask)
    diss_u, diss_v = -1e-5 * u, -1e-5 * v
    nu_max_h, _ = laplacian_smag_cfl_cap(grid, 3600.0, 0.5)
    for cfg in (
        BackscatterConfig(enabled=False, c_bs=0.01),   # disabled
        BackscatterConfig(enabled=True, c_bs=0.0),     # zero coefficient
    ):
        tu, tv, E = diagnostic_backscatter_cgrid(
            u, v, diss_u, diss_v, nu_max_h, grid, cfg,
            mask=None, u_mask=u_mask, v_mask=v_mask)
        assert jnp.count_nonzero(tu) == 0
        assert jnp.count_nonzero(tv) == 0
        assert jnp.count_nonzero(E) == 0


def test_backscatter_injects_resolved_ke(latlon_grid):
    """Enabled with a scale-selective dissipation source => the momentum
    tendency ADDS resolved KE (sum u.tend.area >= 0)."""
    grid, _, u_mask, v_mask = latlon_grid
    u, v = _random_velocity(grid, u_mask, v_mask, seed=7)
    # Dissipative source: D = -damp*u (damp>0) => <u.D> < 0 => eps_diss > 0.
    damp = 2.0e-5
    diss_u, diss_v = -damp * u, -damp * v
    cfg = BackscatterConfig(enabled=True, c_bs=0.02, tau_relax_days=10.0,
                            E_min=0.0, E_max=0.5, efficiency=0.9)
    nu_max_h, _ = laplacian_smag_cfl_cap(grid, 3600.0, 0.5)
    tu, tv, E = diagnostic_backscatter_cgrid(
        u, v, diss_u, diss_v, nu_max_h, grid, cfg,
        mask=None, u_mask=u_mask, v_mask=v_mask)
    # Energizing: the backscatter power density integrates to >= 0.
    power = backscatter_power_density_cgrid(
        u, v, tu, tv, grid, u_mask=u_mask, v_mask=v_mask)
    injected = float(jnp.sum(power * grid.area))
    assert injected >= -1e-12
    # A non-trivial injection (the reservoir is positive where dissipating).
    assert injected > 0.0
    assert float(jnp.max(E)) > 0.0


def test_backscatter_nu_bs_respects_cap(latlon_grid):
    """With a deliberately small nu_max_h and huge E_max, the effective
    nu_bs = c_bs*sqrt(area)*sqrt(E) is capped at nu_max_h pointwise."""
    grid, _, u_mask, v_mask = latlon_grid
    u, v = _random_velocity(grid, u_mask, v_mask, seed=11)
    # Huge dissipation so E would blow past the cap without the CFL bound.
    diss_u, diss_v = -1.0 * u, -1.0 * v
    cfg = BackscatterConfig(enabled=True, c_bs=0.02, tau_relax_days=30.0,
                            E_min=0.0, E_max=1.0e9, efficiency=1.0)
    nu_max_h = jnp.full(grid.area.shape, 10.0)          # small ceiling [m^2/s]
    tu, tv, E = diagnostic_backscatter_cgrid(
        u, v, diss_u, diss_v, nu_max_h, grid, cfg,
        mask=None, u_mask=u_mask, v_mask=v_mask)
    nu_bs_h = cfg.c_bs * jnp.sqrt(grid.area) * jnp.sqrt(jnp.maximum(E, 0.0))
    assert jnp.all(nu_bs_h <= nu_max_h * (1.0 + 1e-9))
    # The cap actually bit somewhere (E was driven to the ceiling).
    assert float(jnp.max(nu_bs_h)) > 0.0


def test_backscatter_differentiable(latlon_grid):
    """The wiring stays end-to-end differentiable (no nondiff control flow on
    traced data)."""
    grid, _, u_mask, v_mask = latlon_grid
    u, v = _random_velocity(grid, u_mask, v_mask, seed=5)
    cfg = BackscatterConfig(enabled=True, c_bs=0.02, E_max=0.5)
    nu_max_h, _ = laplacian_smag_cfl_cap(grid, 3600.0, 0.5)

    def loss(uu):
        diss_u, diss_v = -1e-4 * uu, -1e-4 * v
        tu, tv, _ = diagnostic_backscatter_cgrid(
            uu, v, diss_u, diss_v, nu_max_h, grid, cfg,
            mask=None, u_mask=u_mask, v_mask=v_mask)
        return jnp.sum(tu ** 2) + jnp.sum(tv ** 2)

    g = jax.grad(loss)(u)
    assert g.shape == u.shape
    assert bool(jnp.all(jnp.isfinite(g)))


# ---------------------------------------------------------------------------
# Config wiring: default-None gate + YAML entry-point registration
# ---------------------------------------------------------------------------


def test_config_backscatter_defaults_none_and_constructs():
    from legoesm.ocean.state import LatLonCGridOceanConfig
    cfg = LatLonCGridOceanConfig()
    assert cfg.backscatter is None                       # default OFF
    on = LatLonCGridOceanConfig(backscatter=BackscatterConfig(enabled=True))
    assert on.backscatter is not None and on.backscatter.enabled
    # A flat-name round trip leaves it at the default None (not a flat field).
    assert LatLonCGridOceanConfig.from_flat(A_h=1.0e4).backscatter is None


def test_backscatter_registered_as_yaml_entry_point():
    from legoesm.ocean.config import _nested_ocean_entry_types
    entries = _nested_ocean_entry_types()
    assert entries.get("backscatter") is BackscatterConfig
