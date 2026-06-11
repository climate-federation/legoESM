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

from legoesm.core.field import Field
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
        T_water_init_C=10.0, T_deep=2.0, S_uniform=35.0,
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
    ``scripts/run/global_overturning/diagnose_mpas_baro_noise.py``.
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
    scripts/run/global_overturning/diagnose_mpas_baro_noise.py for the
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


def test_implicit_solver_strong_depth_contrast(state, mesh, z_coord):
    """P0 of the MPAS topography plan
    (``docs/ocean_experiments/realistic_geometry_mpas_plan.md``):
    confirm the implicit-CN Helmholtz operator handles realistic
    bathymetry depth contrast (~600x, 10m shelves to 6000m abyss
    on ETOPO ico4) without losing PCG convergence or mass
    conservation.

    A scalar-mean linearization of the Helmholtz stiffness would
    misrepresent the gravity-wave timescale on shelves and could
    badly condition the elliptic solve.  The current implementation
    builds ``H_e = edge_thickness(eta + H_bathy)`` per edge from
    per-cell ``H_bathy``, so the stiffness matrix is cell-local;
    this test exercises that path on a synthetic 600x-contrast
    bathymetry and asserts both finiteness and mass closure.
    """
    nCells = state.H_bathy.data.shape[0]
    rng = np.random.default_rng(42)
    log_H = rng.uniform(np.log(10.0), np.log(6000.0), size=nCells)
    H_synth = jnp.asarray(np.exp(log_H), dtype=state.H_bathy.data.dtype)
    H_synth = jnp.where(state.land_mask.data > 0.5, H_synth, 4000.0)
    H_field = Field(
        data=H_synth, name="H_bathy", dims=("nCells",), units="m",
    )
    state_topo = state._replace(H_bathy=H_field)

    cfg = MPASOceanConfig(
        barotropic_solver="implicit_cn",
        A_h=1.0e3, A_v=1.0e-3, K_v=1.0e-4,
        bottom_drag_r=1.1e-3,
        physics=_make_physics(tau_max=0.05),
        barotropic_implicit_pcg_tol=1.0e-12,
        barotropic_implicit_pcg_maxiter=400,
        min_water_column_m=1.0,
    )
    model = MPASOceanModel(mesh, z_coord, cfg)

    area = np.asarray(mesh.areaCell)
    mask = np.asarray(state_topo.land_mask.data)
    V0 = float(np.sum(np.asarray(state_topo.eta.data) * area * mask))
    abs_eta_sum = 1.0e-30

    s = state_topo
    for k in range(20):
        s = model.step(s, dt=600.0)
        assert bool(jnp.all(jnp.isfinite(s.eta.data))), (
            f"eta non-finite at step {k} under 600x depth contrast"
        )
        assert bool(jnp.all(jnp.isfinite(s.u.data))), (
            f"u non-finite at step {k} under 600x depth contrast"
        )
        abs_eta_sum = max(
            abs_eta_sum,
            float(np.sum(np.abs(np.asarray(s.eta.data)) * area * mask)),
        )

    V1 = float(np.sum(np.asarray(s.eta.data) * area * mask))
    rel_drift = abs(V1 - V0) / abs_eta_sum
    assert rel_drift < 5.0e-7, (
        f"Mass drift under 600x depth contrast: {rel_drift:.3e} "
        f"(reference max |Sum|eta|*A| = {abs_eta_sum:.3e})"
    )


def test_implicit_helmholtz_operator_uses_cell_local_H(mesh):
    """Direct probe of the implicit Helmholtz operator
    ``A(eta) = eta - coeff * div(H_e * grad eta)``.

    A scalar-mean linearization would scale the dispersive (div-grad)
    term identically for any uniform H.  The cell-local
    implementation reads ``H_e = edge_thickness(H_total)`` per edge,
    so doubling H must double the magnitude of ``(A - I)(eta)`` for
    the same eta perturbation.  This is the unambiguous gate that the
    Helmholtz operator does not collapse H to a global mean.
    """
    from legoesm.ocean.dynamics.barotropic_implicit_mpas import (
        _make_helmholtz,
    )
    from legoesm.core.operators_voronoi import edge_thickness

    nCells = mesh.nCells
    rng = np.random.default_rng(0)
    eta_pert = jnp.asarray(rng.standard_normal(nCells), dtype=jnp.float64)
    mask = jnp.ones((nCells,), dtype=jnp.float64)
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    edge_mask = mask[c1] * mask[c2]

    coeff = jnp.asarray(1.0e6, dtype=jnp.float64)

    def dispersive_response(H_value: float) -> jnp.ndarray:
        H_total = jnp.full((nCells,), H_value, dtype=jnp.float64)
        H_e = edge_thickness(H_total, mesh)
        A_op = _make_helmholtz(H_e, coeff, mesh, mask, edge_mask)
        return A_op(eta_pert) - eta_pert  # = -coeff * div(H_e * grad eta)

    r1 = dispersive_response(1000.0)
    r2 = dispersive_response(2000.0)

    norm1 = float(jnp.linalg.norm(r1))
    norm2 = float(jnp.linalg.norm(r2))
    ratio = norm2 / max(norm1, 1.0e-30)

    # Doubling H must double ||r||.  Tolerance allows for floating-
    # point noise but excludes any scalar-mean degeneracy (which would
    # give ratio = 1.0).
    assert abs(ratio - 2.0) < 1.0e-10, (
        f"||(A-I) eta|| should scale linearly in uniform H. "
        f"||r||(H=1000)={norm1:.3e}, ||r||(H=2000)={norm2:.3e}, "
        f"ratio={ratio:.6f} (expected 2.0)."
    )

    # Also verify spatial sensitivity to a non-uniform H (impulse on
    # one cell): the response field should differ from the uniform-H
    # response in the neighborhood of the impulse, not just globally.
    H_uniform = jnp.full((nCells,), 1000.0, dtype=jnp.float64)
    H_perturbed = H_uniform.at[0].set(6000.0)
    H_e_uniform = edge_thickness(H_uniform, mesh)
    H_e_perturbed = edge_thickness(H_perturbed, mesh)

    A_uniform = _make_helmholtz(H_e_uniform, coeff, mesh, mask, edge_mask)
    A_perturbed = _make_helmholtz(H_e_perturbed, coeff, mesh, mask, edge_mask)

    diff = A_perturbed(eta_pert) - A_uniform(eta_pert)
    diff_np = np.asarray(diff)
    # The local impulse on cell 0's bathymetry should produce a
    # response concentrated near cell 0.  The cell-0 entry must be
    # the largest in magnitude.
    assert int(np.argmax(np.abs(diff_np))) == 0, (
        "Perturbing H_bathy on cell 0 should change A(eta) most at "
        "cell 0; if it instead changes A globally, the operator is "
        "averaging H across cells."
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


def test_implicit_solver_baro_u_viscosity_damps_u_bar(state, mesh, z_coord):
    """``barotropic_u_viscosity`` actually damps u_bar inside the
    implicit-CN solver — required by the partial-cell ETOPO stability
    fix (project_mpas_etopo_instability.md).  Without this hook the
    rotational u_bar null mode at topographic step edges grows
    e-folding ~5 days and NaNs the model around day 24.

    Test plan: drive a small nonzero u_bar via F_slow_u, then compare
    the corrector u_bar magnitudes between viscosity=0 and viscosity=1e7.
    The viscosity must reduce ‖u_bar‖_∞ measurably (not merely round-off)
    so we know the operator is wired in (regression guard).
    """
    rng = np.random.default_rng(42)
    # Random F_slow_u with grid-scale structure to excite the Laplacian
    F_u = jnp.asarray(
        1.0e-5 * rng.standard_normal(mesh.nEdges), dtype=jnp.float64,
    )

    cfg_off = MPASOceanConfig(
        barotropic_solver="implicit_cn", barotropic_u_viscosity=0.0,
    )
    cfg_on = MPASOceanConfig(
        barotropic_solver="implicit_cn", barotropic_u_viscosity=1.0e7,
    )

    _, u_bar_off, _ = barotropic_implicit_mpas(
        state, mesh, z_coord, cfg_off, dt=300.0, F_slow_u=F_u,
    )
    _, u_bar_on, _ = barotropic_implicit_mpas(
        state, mesh, z_coord, cfg_on, dt=300.0, F_slow_u=F_u,
    )

    max_off = float(jnp.max(jnp.abs(u_bar_off)))
    max_on = float(jnp.max(jnp.abs(u_bar_on)))
    assert max_off > 0.0, "F_slow_u should drive a nonzero u_bar"
    assert max_on < max_off, (
        f"barotropic_u_viscosity must reduce ‖u_bar‖_∞; got "
        f"viscosity=0: {max_off:.3e}, viscosity=1e7: {max_on:.3e}"
    )
    # Sanity: damping shouldn't be catastrophic for a one-step kick
    assert max_on > 0.0, "viscosity should not zero out u_bar"


def test_implicit_solver_equatorial_visc_boost_strongest_at_equator(
    state, mesh, z_coord,
):
    """``equatorial_visc_boost`` must apply MORE damping at edges near
    the equator than at high latitudes.  Targets the equatorial f→0
    u_baro mode (project_mpas_etopo_instability.md §"equatorial mode").

    Test: drive the same F_slow_u with two configs — uniform A_baro
    (boost=0) and boosted (boost=10).  Compare the per-edge damping by
    checking that the boost reduces u_bar AT EQUATORIAL EDGES more than
    AT POLAR EDGES.
    """
    rng = np.random.default_rng(123)
    F_u = jnp.asarray(
        1.0e-5 * rng.standard_normal(mesh.nEdges), dtype=jnp.float64,
    )
    cfg_uniform = MPASOceanConfig(
        barotropic_solver="implicit_cn",
        barotropic_u_viscosity=1.0e6,
        equatorial_visc_boost=0.0,
    )
    cfg_boosted = MPASOceanConfig(
        barotropic_solver="implicit_cn",
        barotropic_u_viscosity=1.0e6,
        equatorial_visc_boost=10.0,
    )
    _, u_uniform, _ = barotropic_implicit_mpas(
        state, mesh, z_coord, cfg_uniform, dt=300.0, F_slow_u=F_u,
    )
    _, u_boosted, _ = barotropic_implicit_mpas(
        state, mesh, z_coord, cfg_boosted, dt=300.0, F_slow_u=F_u,
    )

    lat_e = np.asarray(mesh.latEdge)
    is_eq = np.abs(lat_e) < np.deg2rad(15.0)
    is_polar = np.abs(lat_e) > np.deg2rad(60.0)
    assert is_eq.sum() > 0 and is_polar.sum() > 0, (
        "Test mesh must have both equatorial and polar edges"
    )

    eq_uniform = float(np.mean(np.abs(np.asarray(u_uniform)[is_eq])))
    eq_boosted = float(np.mean(np.abs(np.asarray(u_boosted)[is_eq])))
    pol_uniform = float(np.mean(np.abs(np.asarray(u_uniform)[is_polar])))
    pol_boosted = float(np.mean(np.abs(np.asarray(u_boosted)[is_polar])))

    eq_reduction = (eq_uniform - eq_boosted) / max(eq_uniform, 1e-30)
    pol_reduction = (pol_uniform - pol_boosted) / max(pol_uniform, 1e-30)
    assert eq_reduction > pol_reduction, (
        f"Equatorial damping reduction ({eq_reduction:.3f}) should "
        f"exceed polar reduction ({pol_reduction:.3f}) when "
        f"equatorial_visc_boost > 0"
    )
    assert eq_reduction > 0, (
        f"Equatorial reduction must be positive; got {eq_reduction:.3f}"
    )
