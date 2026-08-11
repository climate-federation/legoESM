"""Fox-Kemper MLE on the MPAS Voronoi mesh — numerical + invariant tests.

Mirrors ``test_mle.py`` (C-grid) for the Voronoi port ``mle_mpas.py``:
exact tracer conservation, restratification signature, the equatorial
finiteness guard (rn_lat=20 floor), the convection gate, partial cells,
and jit-stability.  See docs/ocean/experiments/mle_mpas_port_plan.md.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.physics.lateral_mixing.mle import MLEConfig, mle_coefficient
from legoesm.ocean.physics.lateral_mixing.mle_mpas import mle_tracer_tendency_mpas
from legoesm.ocean.vertical import (
    create_ocean_z_star,
    create_partial_cell_coordinate,
    compute_layer_thickness,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def mesh():
    """Level-3 icosahedral Voronoi mesh (642 cells) — spans the full globe
    incl. the equator (for the f-floor test)."""
    return create_voronoi_mesh(subdivision_level=3)


@pytest.fixture(scope="module")
def z_coord():
    """6-level z-star, surface-refined so a real mixed layer resolves."""
    return create_ocean_z_star(
        n_levels=6, H_max=600.0, dz_surface=15.0, dz_deep=200.0
    )


@pytest.fixture(scope="module")
def cfg():
    return MLEConfig(ce=0.06)


# ============================================================================
# Helpers
# ============================================================================

def _mixed_layer_front(mesh, z_coord):
    """A near-surface mixed layer with a horizontal buoyancy front.

    Top 2 levels well-mixed (same T per column); a sharp T drop below ->
    a ~30 m mixed layer.  T varies with longitude -> a real ML-mean
    buoyancy gradient across cells (the MLE driver).  S uniform.
    """
    nlev = z_coord.dz_ref.shape[0]
    lon = np.asarray(mesh.lonCell)                  # [rad] 0..2pi
    # Surface T: warm pool near lon=0, cold near lon=pi -> light/dense front.
    T_surf = 18.0 + 6.0 * np.cos(lon)               # (nCells,) 12..24 C
    T = np.empty((mesh.nCells, nlev))
    for k in range(nlev):
        if k <= 1:
            T[:, k] = T_surf                         # mixed top
        else:
            T[:, k] = T_surf - 4.0 * (k - 1)         # stratified below
    S = np.full((mesh.nCells, nlev), 35.0)
    return jnp.asarray(T), jnp.asarray(S)


def _cell_volume(mesh, z_coord, eta, H_bathy):
    """True live volume areaCell·h_k (partial-cell aware) — the model's mass."""
    h_k = compute_layer_thickness(eta, H_bathy, z_coord)
    return np.asarray(mesh.areaCell)[:, None] * np.asarray(h_k)


# ============================================================================
# Tests
# ============================================================================

def test_exact_tracer_conservation(mesh, z_coord, cfg):
    """Sum(dT·vol) and Sum(dS·vol) vanish to roundoff — the must-pass."""
    T, S = _mixed_layer_front(mesh, z_coord)
    eta = jnp.zeros((mesh.nCells,))
    H = jnp.full((mesh.nCells,), 600.0)
    dT, dS = mle_tracer_tendency_mpas(T, S, eta, H, mesh, z_coord, cfg, eos="wright")
    vol = _cell_volume(mesh, z_coord, eta, H)
    # Tendency must be non-trivial (a real front), then exactly conservative.
    # PHYSICAL nontriviality floor (codex MLE-rhop r1 P1): the old in-situ
    # feed left a ~1e-22 roundoff residue that satisfied a bare > 0 —
    # demand a genuinely active MLE transport so an in-situ revert fails
    # conservation tests too, not only the restratification test.
    assert float(jnp.max(jnp.abs(dT))) > 1e-9
    netT = float(jnp.sum(dT * vol))
    netS = float(jnp.sum(dS * vol))
    scaleT = float(jnp.sum(jnp.abs(dT) * vol)) + 1e-30
    scaleS = float(jnp.sum(jnp.abs(dS) * vol)) + 1e-30
    assert abs(netT) / scaleT < 1e-11, (netT, scaleT)
    assert abs(netS) / scaleS < 1e-11, (netS, scaleS)


def test_restratification_reduces_ml_temperature_variance(mesh, z_coord, cfg):
    """One forward-Euler MLE step slumps the front: with uniform salinity,
    buoyancy ∝ temperature, so restratification reduces the horizontal
    variance of the mixed-layer (top-2-level mean) temperature."""
    T, S = _mixed_layer_front(mesh, z_coord)
    eta = jnp.zeros((mesh.nCells,))
    H = jnp.full((mesh.nCells,), 600.0)

    def _ml_T_var(Tf):
        return float(jnp.var(jnp.mean(Tf[:, :2], axis=1)))   # ML-mean T spread

    v0 = _ml_T_var(T)
    dT, dS = mle_tracer_tendency_mpas(T, S, eta, H, mesh, z_coord, cfg, eos="wright")
    dt = 3600.0
    v1 = _ml_T_var(T + dt * dT)
    assert v1 < v0, (v0, v1)


def test_partial_cell_conservation(mesh, z_coord, cfg):
    """Exact tracer conservation against the TRUE model mass areaCell·h_k on a
    PARTIAL-CELL coordinate (varied bathymetry, real bottom steps) — the case
    that breaks if dz_live uses dz_ref·jacobian instead of compute_layer_thickness."""
    lon = np.asarray(mesh.lonCell)
    lat = np.asarray(mesh.latCell)
    # Varied bathy 120..600 m -> partial bottom cells + shallow columns.
    H_np = 360.0 + 240.0 * np.cos(lat) * np.cos(lon)
    H = jnp.asarray(np.clip(H_np, 120.0, 600.0))
    z_pc = create_partial_cell_coordinate(z_coord, H)
    T, S = _mixed_layer_front(mesh, z_coord)
    eta = jnp.zeros((mesh.nCells,))
    dT, dS = mle_tracer_tendency_mpas(T, S, eta, H, mesh, z_pc, cfg, eos="wright")
    assert bool(jnp.all(jnp.isfinite(dT))) and bool(jnp.all(jnp.isfinite(dS)))
    # PHYSICAL nontriviality floor (codex MLE-rhop r1 P1): the old in-situ
    # feed left a ~1e-22 roundoff residue that satisfied a bare > 0 —
    # demand a genuinely active MLE transport so an in-situ revert fails
    # conservation tests too, not only the restratification test.
    assert float(jnp.max(jnp.abs(dT))) > 1e-9
    vol = _cell_volume(mesh, z_pc, eta, H)         # areaCell · true h_k
    for d in (dT, dS):
        net = float(jnp.sum(d * vol))
        scale = float(jnp.sum(jnp.abs(d) * vol)) + 1e-30
        assert abs(net) / scale < 1e-11, (net, scale)


def test_finite_at_equator(mesh, z_coord, cfg):
    """The constant rc_f (rn_lat=20 floor) keeps the streamfunction finite at
    the equator — every output is finite even for near-zero-f cells."""
    lat = np.asarray(mesh.latCell)
    assert np.any(np.abs(lat) < np.deg2rad(5.0)), "mesh has no near-equator cells"
    T, S = _mixed_layer_front(mesh, z_coord)
    eta = jnp.zeros((mesh.nCells,))
    H = jnp.full((mesh.nCells,), 600.0)
    dT, dS = mle_tracer_tendency_mpas(T, S, eta, H, mesh, z_coord, cfg, eos="wright")
    assert bool(jnp.all(jnp.isfinite(dT)))
    assert bool(jnp.all(jnp.isfinite(dS)))


def test_convection_gate_zeros_unstable_column(mesh, z_coord):
    """With nn_conv=1 and a fully statically-UNSTABLE stratification (dense
    water over light everywhere), every column's ML-N^2 < 0 so MLE is gated
    off -> zero tendency.  Toggling the gate off makes it non-zero."""
    nlev = z_coord.dz_ref.shape[0]
    # Unstable: T INCREASES with depth (light over dense) -> dense-over-light
    # is rho decreasing downward.  Make rho decrease with depth: warmer (lighter)
    # at depth.  Use cold surface, warm deep.
    lon = np.asarray(mesh.lonCell)
    T = np.empty((mesh.nCells, nlev))
    for k in range(nlev):
        T[:, k] = (8.0 + 3.0 * np.cos(lon)) + 2.0 * k   # warmer (lighter) downward
    T = jnp.asarray(T)
    S = jnp.full((mesh.nCells, nlev), 35.0)
    eta = jnp.zeros((mesh.nCells,))
    H = jnp.full((mesh.nCells,), 600.0)

    cfg_on = MLEConfig(ce=0.06, no_mle_in_convection=True)
    cfg_off = MLEConfig(ce=0.06, no_mle_in_convection=False)
    dT_on, _ = mle_tracer_tendency_mpas(T, S, eta, H, mesh, z_coord, cfg_on, eos="wright")
    dT_off, _ = mle_tracer_tendency_mpas(T, S, eta, H, mesh, z_coord, cfg_off, eos="wright")
    assert float(jnp.max(jnp.abs(dT_on))) == 0.0, "gate did not suppress MLE"
    assert float(jnp.max(jnp.abs(dT_off))) > 0.0, "ungated case should be non-zero"


def test_jit_stable(mesh, z_coord, cfg):
    """The tendency jits and reproduces the eager result.  Tolerances are loose
    on purpose: the bolus divergence is a near-cancelling signed sum of O(1e-3)
    edge transports whose net is O(1e-8), so XLA's reassociation of the reduction
    under jit perturbs the cancelled result at the ~1e-10 level — far below the
    signal and physically irrelevant (the eager path cancels to the same answer
    to roundoff).  We check the jit output stays finite, conservative, and equal
    to eager within that cancellation floor."""
    T, S = _mixed_layer_front(mesh, z_coord)
    eta = jnp.zeros((mesh.nCells,))
    H = jnp.full((mesh.nCells,), 600.0)
    fn = jax.jit(lambda T, S, eta, H: mle_tracer_tendency_mpas(
        T, S, eta, H, mesh, z_coord, cfg, eos="wright"))
    dT_j, dS_j = fn(T, S, eta, H)
    dT_e, dS_e = mle_tracer_tendency_mpas(T, S, eta, H, mesh, z_coord, cfg, eos="wright")
    assert bool(jnp.all(jnp.isfinite(dT_j))) and bool(jnp.all(jnp.isfinite(dS_j)))
    # Norm-relative (NOT element-wise allclose): a few maximally-cancelled cells
    # carry O(1e-2) jit-vs-eager relative error, but the L2 norm — dominated by
    # the bulk of the field — agrees to the cancellation floor.
    relT = float(jnp.linalg.norm(dT_j - dT_e) / (jnp.linalg.norm(dT_e) + 1e-30))
    relS = float(jnp.linalg.norm(dS_j - dS_e) / (jnp.linalg.norm(dS_e) + 1e-30))
    assert relT < 1e-2 and relS < 1e-2, (relT, relS)
    # The jit path is still EXACTLY conservative (the meaningful invariant).
    vol = _cell_volume(mesh, z_coord, eta, H)
    for d in (dT_j, dS_j):
        net = float(jnp.sum(d * vol))
        scale = float(jnp.sum(jnp.abs(d) * vol)) + 1e-30
        assert abs(net) / scale < 1e-11


def test_coefficient_uses_shared_core(cfg):
    """rc_f is the shared mle.py constant (no re-derived coefficient): positive
    and finite at the ORCA1 default."""
    rc_f = mle_coefficient(cfg.ce, cfg.lat_ref_deg)
    assert rc_f > 0.0 and np.isfinite(rc_f)


def test_physics_contract_present():
    import legoesm.ocean.physics.lateral_mixing.mle_mpas as m
    c = m.__physics_contract__
    assert c["conserves"] == ["tracer"]
    assert c["outputs"] == {"dT_dt": "degC/s", "dS_dt": "PSU/s"}
    assert c["differentiable"] is True


def test_uniform_salinity_untouched_by_thermal_front(mesh, z_coord, cfg):
    """REGRESSION (2026-08-11): S is uniform in the front fixture, so dS must
    vanish identically while dT carries the restratification.  The
    horizontal-only flux divergence pumped uniform tracers at
    transport-convergence cells (globally conservative, locally corrupting:
    -54 psu river-plume extremes on the structured lane); the vertical
    continuity branch (NEMO zw_mle, Voronoi form) closes the overturning
    cell."""
    T, S = _mixed_layer_front(mesh, z_coord)
    eta = jnp.zeros((mesh.nCells,))
    H = jnp.full((mesh.nCells,), 600.0)
    dT, dS = mle_tracer_tendency_mpas(T, S, eta, H, mesh, z_coord, cfg, eos="wright")
    scale = float(jnp.max(jnp.abs(dT)))
    assert scale > 1e-9, "front produced no tendency; test would be vacuous"
    assert float(jnp.max(jnp.abs(dS))) < 1e-9 * scale, (
        "uniform S gained a tendency: bolus transport not divergence-free "
        "per cell (missing/broken vertical branch)")
