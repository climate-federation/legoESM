"""Tests for the implicit Crank-Nicolson barotropic solver on MPAS.

Mirrors ``tests/ocean/unit/test_barotropic_noise_invariant.py`` for the
MPAS Voronoi-mesh path.  The lat-lon C-grid suffers from a Coriolis-
averaging null-mode chequerboard; the analogous TRiSK rotational null
branch on hexagonal C-grids (Thuburn 2008; Ringler+ 2010 §6) was
empirically confirmed in the 5-yr ``global_overturning_mpas_baseline``
restarts (σ_grid/rms ≈ 4, growing monotonically).  The implicit CN
solver eliminates both modes by construction.

Run with:
    JAX_ENABLE_X64=1 python -m pytest \\
        tests/ocean/unit/test_barotropic_implicit_mpas.py -v
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

os.environ.setdefault("JAX_ENABLE_X64", "1")
jax.config.update("jax_enable_x64", True)

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init_mpas import rest_state_mpas_ocean
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
from legoesm.ocean.dynamics.barotropic_implicit_mpas import (
    barotropic_implicit_mpas,
)
from legoesm.core.operators_voronoi import (
    vector_laplacian_del2,
    edge_thickness as _edge_avg,
)
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.surface_forcing.config import (
    PrescribedForcingConfig,
    SurfaceForcingConfig,
)
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig


# ----- Fixtures -----------------------------------------------------------


@pytest.fixture(scope="module")
def mesh():
    return create_voronoi_mesh(subdivision_level=2)


@pytest.fixture(scope="module")
def z_coord():
    return create_ocean_z_star(n_levels=5, H_max=4000.0)


@pytest.fixture(scope="module")
def state(mesh, z_coord):
    return rest_state_mpas_ocean(
        mesh, z_coord,
        T_surface=10.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0, land_lat_threshold=85.0,
    )


def _make_physics(tau_max: float = 0.05):
    return OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(
            scheme="prescribed",
            prescribed=PrescribedForcingConfig(
                wind_profile="cosine_latitude", tau_max=tau_max,
            ),
        ),
        lateral_mixing=LateralMixingConfig(scheme="none"),
    )


# ----- Crit-1-style metric on u_bar grid noise ----------------------------


def _u_bar_grid_metric(u_3d, eta, H_bathy, z_coord, mesh, mask, polar_lat_deg=70.0):
    """Return σ(grid-scale u_bar) / rms(u_bar) on interior edges.

    Mirrors the diagnostic used in
    ``scripts/global_overturning/diagnose_mpas_baro_noise.py``.
    Hex Voronoi has no native zonal/meridional Laplacian; we use the
    TRiSK ``vector_laplacian_del2`` and rescale by mean(dvEdge)² so
    the metric has the same scale as u_bar itself.
    """
    from legoesm.ocean.vertical import compute_layer_thickness

    h_k = compute_layer_thickness(eta, H_bathy, z_coord, min_water_column_m=0.5)
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    h_e_k = 0.5 * (h_k[c1] + h_k[c2])
    H_total = jnp.maximum(eta + H_bathy, 0.5)
    H_e = _edge_avg(H_total, mesh)
    Hu = jnp.sum(u_3d * h_e_k, axis=1)
    u_bar = Hu / jnp.maximum(H_e, 1.0e-10)

    u_bar_np = np.asarray(u_bar, dtype=np.float64)
    lap = np.asarray(vector_laplacian_del2(u_bar, mesh), dtype=np.float64)
    dv = np.asarray(mesh.dvEdge)
    lap_scaled = lap * np.mean(dv) ** 2

    lat_cell = np.asarray(mesh.latCell) * 180.0 / np.pi
    polar_cell = np.abs(lat_cell) > polar_lat_deg
    c1_n = np.asarray(c1)
    c2_n = np.asarray(c2)
    polar_edge = polar_cell[c1_n] | polar_cell[c2_n]
    mask_n = np.asarray(mask)
    ocean_edge = (mask_n[c1_n] > 0.5) & (mask_n[c2_n] > 0.5)
    interior = ocean_edge & ~polar_edge

    if not interior.any():
        return float("nan"), float("nan"), float("nan")

    rms_u = float(np.sqrt(np.mean(u_bar_np[interior] ** 2)))
    sigma_grid = float(np.std(lap_scaled[interior]))
    ratio = sigma_grid / max(rms_u, 1.0e-30)
    return rms_u, sigma_grid, ratio


# ----- Tests --------------------------------------------------------------


def test_implicit_solver_default_is_explicit_substep():
    """Default config keeps the legacy explicit substep so existing
    experiments are bit-stable."""
    cfg = MPASOceanConfig()
    assert cfg.barotropic_solver == "explicit_substep"


def test_invalid_barotropic_solver_rejected(mesh, z_coord):
    """Config validation rejects unknown solver names."""
    cfg = MPASOceanConfig(barotropic_solver="bogus")
    with pytest.raises(ValueError, match="barotropic_solver"):
        MPASOceanModel(mesh, z_coord, cfg)


def test_implicit_solver_rest_state_stable(state, mesh, z_coord):
    """Rest state remains exactly at rest (eta=0, u=0) under the
    implicit solver."""
    cfg = MPASOceanConfig(
        barotropic_solver="implicit_cn",
        A_h=1.0e3, A_v=1.0e-3, K_v=1.0e-4,
        n_barotropic_substeps=5,  # unused by implicit but kept for shape
    )
    model = MPASOceanModel(mesh, z_coord, cfg)
    state_new = model.step(state, dt=300.0)
    assert bool(jnp.all(jnp.isfinite(state_new.eta.data)))
    assert bool(jnp.all(jnp.isfinite(state_new.u.data)))
    # Rest state has zero forcing and zero gradient → stays put modulo
    # PCG residual.  Tolerance 1e-8 absorbs PCG default tol·||rhs||.
    assert float(jnp.max(jnp.abs(state_new.eta.data))) < 1.0e-8
    assert float(jnp.max(jnp.abs(state_new.u.data))) < 1.0e-8


def test_implicit_solver_finite_under_wind(state, mesh, z_coord):
    """Implicit solver produces finite state after a 5-day cosine-
    latitude wind spinup."""
    physics = _make_physics(tau_max=0.05)
    cfg = MPASOceanConfig(
        barotropic_solver="implicit_cn",
        A_h=5.0e5, A_v=1.0e-3, K_v=1.0e-4,
        bottom_drag_r=1.1e-3,
        physics=physics,
    )
    model = MPASOceanModel(mesh, z_coord, cfg)
    s = state
    dt = 600.0
    for _ in range(int(5.0 * 86400.0 / dt)):
        s = model.step(s, dt=dt)
    assert bool(jnp.all(jnp.isfinite(s.eta.data)))
    assert bool(jnp.all(jnp.isfinite(s.u.data)))


def test_implicit_solver_conserves_mass(state, mesh, z_coord):
    """Per-step ``Σ η · A_c`` drift should be at PCG-tolerance level
    over 50 steps from rest with no freshwater."""
    cfg = MPASOceanConfig(
        barotropic_solver="implicit_cn",
        A_h=1.0e3, A_v=1.0e-3, K_v=1.0e-4,
        bottom_drag_r=1.1e-3,
        physics=_make_physics(tau_max=0.05),
        barotropic_implicit_pcg_tol=1.0e-12,
    )
    model = MPASOceanModel(mesh, z_coord, cfg)

    area = np.asarray(mesh.areaCell)
    mask = np.asarray(state.land_mask.data)
    V0 = float(np.sum(np.asarray(state.eta.data) * area * mask))
    abs_eta_sum = float(
        np.sum(np.abs(np.asarray(state.eta.data)) * area * mask)
    )

    s = state
    for _ in range(50):
        s = model.step(s, dt=600.0)
        abs_eta_sum = max(
            abs_eta_sum,
            float(np.sum(np.abs(np.asarray(s.eta.data)) * area * mask)),
        )

    V1 = float(np.sum(np.asarray(s.eta.data) * area * mask))
    rel_drift = abs(V1 - V0) / max(abs_eta_sum, 1.0e-30)
    # Wind forcing causes |Σ η·A| ≠ 0 but the change must be at the
    # PCG-residual level — well below 1e-6 with tol=1e-12.
    assert rel_drift < 5.0e-7, (
        f"Mass drift after 50 steps: {rel_drift:.3e} (relative to "
        f"max |Σ |η|·A| = {abs_eta_sum:.3e})"
    )


def test_implicit_solver_grad_smoke(state, mesh, z_coord):
    """``jax.grad`` through the implicit MPAS solver returns finite,
    non-zero values."""
    cfg = MPASOceanConfig(
        barotropic_solver="implicit_cn",
        A_h=1.0e3, A_v=1.0e-3, K_v=1.0e-4,
        barotropic_implicit_pcg_maxiter=100,
    )
    model = MPASOceanModel(mesh, z_coord, cfg)

    rng = np.random.default_rng(0)
    eta0 = jnp.asarray(
        0.001 * rng.standard_normal(state.eta.data.shape),
        dtype=jnp.float64,
    )

    def loss(eta):
        s = state._replace(eta=state.eta.replace(data=eta))
        s_new = model.step(s, dt=300.0)
        return jnp.sum(s_new.eta.data ** 2)

    g = jax.grad(loss)(eta0)
    assert bool(jnp.all(jnp.isfinite(g))), "jax.grad produced NaN/Inf"
    assert float(jnp.linalg.norm(g)) > 0.0, \
        "jax.grad returned exactly zero gradient"


def test_implicit_solver_cleaner_than_explicit_substep(state, mesh, z_coord):
    """Both solvers under identical wind spinup — assert implicit
    produces strictly lower σ_grid/rms ratio on time-mean u_bar.

    This catches regressions that would silently revert the implicit-CN
    benefit on MPAS.  The ico4 5-yr baseline run shows σ/rms ≈ 4 with
    growth, vs an implicit solver that should keep this ratio bounded.
    At level 2 (≈3000 km cells) the spread is much smaller than at
    ico4, so the test only requires implicit < explicit (any
    improvement) rather than the 3× ratio used in the lat-lon analog.
    Strong noise on MPAS develops at finer resolution and longer
    integration than is feasible for a unit test (see
    scripts/global_overturning/diagnose_mpas_baro_noise.py for the
    resolution-of-record check).
    """
    physics = _make_physics(tau_max=0.05)
    common = dict(
        A_h=5.0e5, A_v=1.0e-3, K_v=1.0e-4,
        bottom_drag_r=1.1e-3,
        physics=physics,
    )
    dt = 600.0
    n_steps = int(10.0 * 86400.0 / dt)  # 10 days
    n_avg = n_steps // 3                 # average last third

    ratios = {}
    for solver_name in ("implicit_cn", "explicit_substep"):
        cfg = MPASOceanConfig(
            barotropic_solver=solver_name,
            n_barotropic_substeps=30,
            **common,
        )
        model = MPASOceanModel(mesh, z_coord, cfg)
        s = state
        u_sum = np.zeros_like(np.asarray(state.u.data), dtype=np.float64)
        eta_sum = np.zeros_like(np.asarray(state.eta.data), dtype=np.float64)
        for i in range(n_steps):
            s = model.step(s, dt=dt)
            if i >= n_steps - n_avg:
                u_sum += np.asarray(s.u.data)
                eta_sum += np.asarray(s.eta.data)
        u_mean = jnp.asarray(u_sum / n_avg, dtype=jnp.float64)
        eta_mean = jnp.asarray(eta_sum / n_avg, dtype=jnp.float64)
        rms_u, sigma_g, ratio = _u_bar_grid_metric(
            u_mean, eta_mean, state.H_bathy.data, z_coord,
            mesh, state.land_mask.data,
        )
        ratios[solver_name] = ratio

    r_imp = ratios["implicit_cn"]
    r_exp = ratios["explicit_substep"]
    assert r_imp < r_exp, (
        f"implicit_cn σ_grid/rms = {r_imp:.3e} should be < "
        f"explicit = {r_exp:.3e} (any improvement is sufficient at "
        f"level-2 resolution; the 5-yr ico4 run shows the order-of-"
        f"magnitude difference)"
    )


def test_implicit_solver_cn_function_directly(state, mesh, z_coord):
    """Direct call to ``barotropic_implicit_mpas`` from rest-state +
    weak F_slow_eta returns finite (eta, u_bar, Hu_avg)."""
    cfg = MPASOceanConfig(barotropic_solver="implicit_cn")
    rng = np.random.default_rng(0)
    F_eta = jnp.asarray(
        1.0e-7 * rng.standard_normal(state.eta.data.shape),
        dtype=jnp.float64,
    )
    eta_new, u_bar_new, Hu_avg = barotropic_implicit_mpas(
        state, mesh, z_coord, cfg, dt=300.0,
        F_slow_eta=F_eta,
    )
    assert eta_new.shape == state.eta.data.shape
    assert u_bar_new.shape == (mesh.nEdges,)
    assert Hu_avg.shape == (mesh.nEdges,)
    assert bool(jnp.all(jnp.isfinite(eta_new)))
    assert bool(jnp.all(jnp.isfinite(u_bar_new)))
    assert bool(jnp.all(jnp.isfinite(Hu_avg)))
