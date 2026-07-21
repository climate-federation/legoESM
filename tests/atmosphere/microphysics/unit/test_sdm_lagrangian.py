"""Unit tests for the advected-Lagrangian SDM LES coupling."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.thermo import saturation_vapor_pressure
from legoesm.atmosphere.dynamics.les import spectral_les_plane as sl
from legoesm.atmosphere.dynamics.les.spectral_les_moist import (
    LagrangianSDMSegmentDiagnostics,
    make_lagrangian_sdm_les_step,
    make_lagrangian_sdm_step_segment,
    make_anelastic_reference,
    step_lagrangian_sdm_les,
    update_lagrangian_sdm_segment_diagnostics,
)
from legoesm.atmosphere.physics.microphysics.sdm import (
    SDMConfig,
    SuperDropletState,
    advect_step,
    coalescence_cells_step,
    condensation_coupling_step,
    diagnose_liquid_mixing_ratios,
    initialize_lagrangian_sdm,
    make_lagrangian_sdm_state,
    particle_cell_indices,
    represented_water_mass,
    sample_lognormal_radius,
    total_water_mass,
)

_PREF = 4.0 / 3.0 * np.pi * constants.rho_water


def _grid(nx=4, ny=4, nz=4, L=4.0, n_tracers=3):
    cfg = sl.SpectralLESConfig(
        nx=nx, ny=ny, nz=nz, Lx=L, Ly=L, Lz=L,
        z0=0.01, buoyancy=False, moist=False, n_tracers=n_tracers,
        spectral_filter=False, time_scheme="ab2")
    return sl.make_grid(cfg)


def _ref(g, theta0=300.0, qv0=0.0):
    return make_anelastic_reference(
        np.asarray(g.z_c), np.asarray(g.z_f), 101500.0,
        np.full(g.cfg.nz, theta0), np.full(g.cfg.nz, qv0))


def _lag_state(g, radii, xi=1.0e6, xyz=None):
    n = len(radii)
    if xyz is None:
        xyz = (np.full(n, 0.5), np.full(n, 0.5), np.full(n, 1.5))
    x, y, z = [jnp.asarray(a, dtype=jnp.float64) for a in xyz]
    o = jnp.ones((n,), dtype=jnp.float64)
    droplets = SuperDropletState(
        multiplicity=o * xi,
        radius=jnp.asarray(radii, dtype=jnp.float64),
        solute_mass=o * 0.0,
        active=o,
    )
    return make_lagrangian_sdm_state(droplets, x, y, z, jax.random.PRNGKey(0), g)


def _lag_state_slots(g, radii, xi, active, xyz, key=0):
    n = len(radii)
    x, y, z = [jnp.asarray(a, dtype=jnp.float64) for a in xyz]
    droplets = SuperDropletState(
        multiplicity=jnp.asarray(xi, dtype=jnp.float64),
        radius=jnp.asarray(radii, dtype=jnp.float64),
        solute_mass=jnp.zeros((n,), dtype=jnp.float64),
        active=jnp.asarray(active, dtype=jnp.float64),
    )
    return make_lagrangian_sdm_state(droplets, x, y, z, jax.random.PRNGKey(key), g)


def _cell_sums(state, g, values):
    _, _, _, cell_id = particle_cell_indices(state, g)
    n_cells = g.cfg.ny * g.cfg.nx * g.cfg.nz
    return jnp.zeros((n_cells,), dtype=values.dtype).at[cell_id].add(values)


def _assert_tree_allclose(actual, expected, *, rtol=1e-12, atol=1e-14):
    actual_leaves = jax.tree.leaves(actual)
    expected_leaves = jax.tree.leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
        actual_np = np.asarray(actual_leaf)
        expected_np = np.asarray(expected_leaf)
        if actual_np.dtype.kind in "biu":
            np.testing.assert_array_equal(actual_np, expected_np)
        else:
            np.testing.assert_allclose(
                actual_np, expected_np, rtol=rtol, atol=atol)


def test_advection_moves_with_resolved_flow_and_wraps_periodic():
    g = _grid(L=4.0)
    cfg = SDMConfig(terminal_velocity="rogers_yau")
    state = _lag_state(g, [0.0, 0.0], xyz=(
        np.array([0.5, 3.8]), np.array([0.5, 0.5]), np.array([1.5, 1.5])))
    u = jnp.ones((g.cfg.ny, g.cfg.nx, g.cfg.nz), dtype=jnp.float64)
    v = jnp.zeros_like(u)
    w = jnp.zeros((g.cfg.ny, g.cfg.nx, g.cfg.nz + 1), dtype=jnp.float64)

    out, dp = advect_step(state, u, v, w, g, 0.5, cfg, rho=1.0, p=9.0e4, T=283.0)

    np.testing.assert_allclose(np.asarray(out.x), np.array([1.0, 0.3]), atol=1e-12)
    np.testing.assert_allclose(np.asarray(out.y), np.asarray(state.y), atol=1e-12)
    np.testing.assert_allclose(np.asarray(out.z), np.asarray(state.z), atol=1e-12)
    assert float(jnp.sum(out.droplets.active)) == 2.0
    assert float(jnp.sum(dp)) == 0.0


def test_sedimentation_crossing_accumulates_surface_precipitation():
    g = _grid(L=4.0)
    cfg = SDMConfig(terminal_velocity="rogers_yau")
    R, xi = 1.0e-4, 5.0e5
    state = _lag_state(g, [R], xi=xi, xyz=(
        np.array([0.5]), np.array([0.5]), np.array([0.1])))
    zero = jnp.zeros((g.cfg.ny, g.cfg.nx, g.cfg.nz), dtype=jnp.float64)
    w = jnp.zeros((g.cfg.ny, g.cfg.nx, g.cfg.nz + 1), dtype=jnp.float64)

    out, dp = advect_step(state, zero, zero, w, g, 1.0, cfg, 1.0, 9.0e4, 283.0)

    expected_mass = xi * _PREF * R**3
    area = g.dx * g.dy
    assert float(out.droplets.active[0]) == 0.0
    assert float(out.z[0]) == 0.0
    assert float(jnp.sum(dp) * area) == pytest.approx(expected_mass, rel=1e-12)
    assert float(jnp.sum(out.surface_precip) * area) == pytest.approx(
        expected_mass, rel=1e-12)


def test_top_wall_clamps_without_deactivating():
    g = _grid(L=4.0)
    cfg = SDMConfig(terminal_velocity="rogers_yau")
    state = _lag_state(g, [0.0], xyz=(
        np.array([0.5]), np.array([0.5]), np.array([3.8])))
    zero = jnp.zeros((g.cfg.ny, g.cfg.nx, g.cfg.nz), dtype=jnp.float64)
    w = jnp.ones((g.cfg.ny, g.cfg.nx, g.cfg.nz + 1), dtype=jnp.float64)

    out, _ = advect_step(state, zero, zero, w, g, 1.0, cfg, 1.0, 9.0e4, 283.0)

    assert float(out.z[0]) == pytest.approx(g.cfg.Lz)
    assert float(out.droplets.active[0]) == 1.0


def test_cellwise_collision_conserves_represented_water():
    g = _grid(nx=2, ny=2, nz=2, L=2.0)
    cfg = SDMConfig(collision_kernel="golovin", golovin_b=1.5e3)
    n_sd = 128
    state = _lag_state(g, np.full(n_sd, 1.4e-5), xi=1.0e6, xyz=(
        np.full(n_sd, 0.5), np.full(n_sd, 0.5), np.full(n_sd, 0.5)))
    m0 = float(jnp.sum(represented_water_mass(state.droplets)))

    out = coalescence_cells_step(state, g, rho=1.0, p=9.0e4, T=283.0,
                                 dt=1.0, cfg=cfg)

    m1 = float(jnp.sum(represented_water_mass(out.droplets)))
    assert m1 == pytest.approx(m0, rel=1e-9)
    assert float(jnp.sum(out.droplets.active * out.droplets.multiplicity)) <= float(
        jnp.sum(state.droplets.active * state.droplets.multiplicity))


def test_cellwise_collision_two_active_padded_cell_collides():
    g = _grid(nx=2, ny=2, nz=2, L=2.0)
    cfg = SDMConfig(collision_kernel="golovin", golovin_b=1.0e16)
    n_sd = 256
    R = 2.0e-5
    active = np.zeros(n_sd)
    active[:2] = 1.0
    xyz = (np.full(n_sd, 0.5), np.full(n_sd, 0.5), np.full(n_sd, 0.5))
    state = _lag_state_slots(
        g, np.full(n_sd, R), np.ones(n_sd), active, xyz)
    m0 = float(jnp.sum(represented_water_mass(state.droplets)))

    out = coalescence_cells_step(
        state, g, rho=1.0, p=9.0e4, T=283.0, dt=1.0, cfg=cfg)

    m1 = float(jnp.sum(represented_water_mass(out.droplets)))
    num1 = float(jnp.sum(out.droplets.active * out.droplets.multiplicity))
    assert m1 == pytest.approx(m0, rel=1e-12, abs=1e-20)
    assert num1 == pytest.approx(1.0)
    assert float(jnp.max(out.droplets.radius)) == pytest.approx(
        np.cbrt(2.0) * R, rel=1e-12)


def test_cellwise_collision_isolates_cells_and_pairs_after_odd_cell():
    g = _grid(nx=2, ny=2, nz=2, L=2.0)
    cfg = SDMConfig(collision_kernel="golovin", golovin_b=1.0e16)
    n_sd = 16
    active = np.zeros(n_sd)
    active[:3] = 1.0
    x = np.full(n_sd, 0.5)
    x[1:3] = 1.5
    xyz = (x, np.full(n_sd, 0.5), np.full(n_sd, 0.5))
    radii = np.full(n_sd, 2.0e-5)
    radii[0] = 1.0e-5
    state = _lag_state_slots(g, radii, np.ones(n_sd), active, xyz)
    mass0 = _cell_sums(state, g, represented_water_mass(state.droplets))

    out = coalescence_cells_step(
        state, g, rho=1.0, p=9.0e4, T=283.0, dt=1.0, cfg=cfg)

    mass1 = _cell_sums(out, g, represented_water_mass(out.droplets))
    num1 = _cell_sums(out, g, out.droplets.active * out.droplets.multiplicity)
    np.testing.assert_allclose(np.asarray(mass1), np.asarray(mass0),
                               rtol=1e-12, atol=1e-20)
    assert float(num1[0]) == pytest.approx(1.0)
    assert float(num1[2]) == pytest.approx(1.0)
    assert float(jnp.sum(num1)) == pytest.approx(2.0)


def test_sorted_cell_collision_isolates_cells_conserves_and_certain_collision():
    g = _grid(nx=2, ny=2, nz=2, L=2.0)
    cfg = SDMConfig(collision_kernel="golovin", golovin_b=1.0e16)
    n_sd = 16
    active = np.zeros(n_sd)
    active[:4] = 1.0
    x = np.full(n_sd, 0.5)
    x[2:4] = 1.5
    xyz = (x, np.full(n_sd, 0.5), np.full(n_sd, 0.5))
    radii = np.full(n_sd, 2.0e-5)
    radii[2:4] = 3.0e-5
    state = _lag_state_slots(g, radii, np.ones(n_sd), active, xyz)
    mass0 = _cell_sums(state, g, represented_water_mass(state.droplets))

    out = coalescence_cells_step(
        state, g, rho=1.0, p=9.0e4, T=283.0, dt=1.0, cfg=cfg)

    mass1 = _cell_sums(out, g, represented_water_mass(out.droplets))
    num1 = _cell_sums(out, g, out.droplets.active * out.droplets.multiplicity)
    np.testing.assert_allclose(np.asarray(mass1), np.asarray(mass0),
                               rtol=1e-12, atol=1e-20)
    assert float(num1[0]) == pytest.approx(1.0)
    assert float(num1[2]) == pytest.approx(1.0)
    assert float(jnp.max(out.droplets.radius)) == pytest.approx(
        np.cbrt(2.0) * 3.0e-5, rel=1e-12)


def test_condensation_two_way_coupling_conserves_water_and_energy():
    g = _grid(nx=2, ny=2, nz=2, L=2.0)
    ref = _ref(g, theta0=300.0, qv0=0.0)
    cfg = SDMConfig(
        include_curvature=False, include_solute=False,
        condensation_integrator="euler", n_substeps_condensation=1)
    state = _lag_state(g, [1.0e-5], xi=1.0e8, xyz=(
        np.array([0.5]), np.array([0.5]), np.array([0.5])))
    theta = jnp.full((g.cfg.ny, g.cfg.nx, g.cfg.nz), 300.0, dtype=jnp.float64)
    tracers = jnp.zeros((g.cfg.ny, g.cfg.nx, g.cfg.nz, 3), dtype=jnp.float64)
    tracers = tracers.at[..., 0].set(0.03)
    water0 = total_water_mass(state, tracers, g, ref.rho_c)
    T0 = theta * ref.exner_c[None, None, :]

    out, theta1, tracers1, dq_liq = condensation_coupling_step(
        state, theta, tracers, g, ref.exner_c, ref.p_c, ref.rho_c, 0.2, cfg)

    water1 = total_water_mass(out, tracers1, g, ref.rho_c)
    T1 = theta1 * ref.exner_c[None, None, :]
    assert float(water1) == pytest.approx(float(water0), rel=2e-12, abs=1e-14)
    np.testing.assert_allclose(
        np.asarray(T1 - T0),
        np.asarray((constants.L_v / constants.c_pd) * dq_liq),
        rtol=1e-12,
        atol=1e-12,
    )
    assert float(jnp.max(dq_liq)) > 0.0
    assert float(jnp.min(tracers1[..., 0])) < 0.03


def test_particle_binning_diagnoses_cloud_and_rain_slots():
    g = _grid(nx=2, ny=2, nz=2, L=2.0)
    state = _lag_state(g, [1.0e-5, 1.0e-4], xi=1.0e6, xyz=(
        np.array([0.5, 0.5]), np.array([0.5, 0.5]), np.array([0.5, 0.5])))
    qc, qr = diagnose_liquid_mixing_ratios(state, g, rho=1.0, r_rain=4.0e-5)
    V = g.dx * g.dy * g.dz
    assert float(qc[0, 0, 0]) == pytest.approx(1.0e6 * _PREF * (1.0e-5)**3 / V)
    assert float(qr[0, 0, 0]) == pytest.approx(1.0e6 * _PREF * (1.0e-4)**3 / V)


def test_lagrangian_diagnostic_excludes_haze_from_cloud_smear_metric():
    g = _grid(nx=8, ny=8, nz=4, L=8.0)
    cfg = SDMConfig()
    xs, ys, zs = np.meshgrid(
        (np.arange(g.cfg.nx) + 0.5) * g.dx,
        (np.arange(g.cfg.ny) + 0.5) * g.dy,
        np.asarray(g.z_c),
        indexing="xy",
    )
    xyz_haze = (xs.ravel(), ys.ravel(), zs.ravel())
    xyz = (
        np.concatenate([xyz_haze[0], [0.5]]),
        np.concatenate([xyz_haze[1], [0.5]]),
        np.concatenate([xyz_haze[2], [float(g.z_c[0])]]),
    )
    n_haze = xyz_haze[0].size
    state = _lag_state_slots(
        g,
        np.concatenate([np.full(n_haze, 1.0e-6), [3.0e-6]]),
        np.ones(n_haze + 1) * 1.0e8,
        np.ones(n_haze + 1),
        xyz,
    )

    qc_legacy, _ = diagnose_liquid_mixing_ratios(
        state, g, rho=1.0, r_rain=cfg.r_rain, r_cloud=0.0)
    qc_fixed, _ = diagnose_liquid_mixing_ratios(
        state, g, rho=1.0, r_rain=cfg.r_rain, r_cloud=cfg.r_cloud)

    def intermediate_fraction(qc):
        peak = jnp.max(qc)
        return jnp.mean(((qc > 0.01 * peak) & (qc < 0.5 * peak)).astype(jnp.float64))

    assert float(intermediate_fraction(qc_legacy)) > 0.95
    assert float(intermediate_fraction(qc_fixed)) < 0.05
    assert float(jnp.count_nonzero(qc_fixed)) == 1.0
    cell_air = g.dx * g.dy * g.dz
    assert float(jnp.sum(qc_fixed)) == pytest.approx(
        1.0e8 * _PREF * (3.0e-6)**3 / cell_air, rel=1e-12)


def test_cell_stratified_initializer_gives_per_cell_floor_and_exact_number():
    g = _grid(nx=4, ny=2, nz=2, L=8.0)
    n_per_cell = 8
    n_cells = g.cfg.ny * g.cfg.nx * g.cfg.nz
    number_profile = jnp.array([1.0e6, 2.0e6], dtype=jnp.float64)
    radius_profile = jnp.array([1.0e-6, 2.0e-6], dtype=jnp.float64)

    state = initialize_lagrangian_sdm(
        jax.random.PRNGKey(23),
        g,
        n_sd=n_per_cell * n_cells,
        number_concentration=number_profile,
        radius=radius_profile,
        dtype=jnp.float64,
        spatial_sampling="cell_stratified",
    )

    ix, iy, iz, _ = particle_cell_indices(state, g)
    counts = _cell_sums(state, g, state.droplets.active)
    represented_number = _cell_sums(
        state, g, state.droplets.active * state.droplets.multiplicity)
    expected_number = np.broadcast_to(
        np.asarray(number_profile)[None, None, :] * g.dx * g.dy * g.dz,
        (g.cfg.ny, g.cfg.nx, g.cfg.nz),
    ).reshape(-1)

    np.testing.assert_array_equal(np.asarray(counts), np.full(n_cells, n_per_cell))
    np.testing.assert_allclose(
        np.asarray(represented_number), expected_number, rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(state.droplets.radius), np.asarray(radius_profile)[np.asarray(iz)])
    assert int(jnp.min(ix)) >= 0 and int(jnp.max(ix)) < g.cfg.nx
    assert int(jnp.min(iy)) >= 0 and int(jnp.max(iy)) < g.cfg.ny


def test_cell_stratified_initializer_accepts_per_particle_aerosol_spectrum():
    g = _grid(nx=4, ny=2, nz=2, L=8.0)
    n_per_cell = 4
    n_cells = g.cfg.ny * g.cfg.nx * g.cfg.nz
    n_sd = n_per_cell * n_cells
    radius = jnp.linspace(2.0e-8, 2.0e-7, n_sd, dtype=jnp.float64)
    solute = jnp.linspace(1.0e-20, 1.0e-18, n_sd, dtype=jnp.float64)

    state = initialize_lagrangian_sdm(
        jax.random.PRNGKey(24),
        g,
        n_sd=n_sd,
        number_concentration=1.0e6,
        radius=radius,
        solute_mass=solute,
        dtype=jnp.float64,
        spatial_sampling="cell_stratified",
    )

    counts = _cell_sums(state, g, state.droplets.active)
    represented_number = _cell_sums(
        state, g, state.droplets.active * state.droplets.multiplicity)
    expected_number = np.full(n_cells, 1.0e6 * g.dx * g.dy * g.dz)

    np.testing.assert_array_equal(np.asarray(counts), np.full(n_cells, n_per_cell))
    np.testing.assert_allclose(np.asarray(state.droplets.radius), np.asarray(radius))
    np.testing.assert_allclose(
        np.asarray(state.droplets.solute_mass), np.asarray(solute))
    np.testing.assert_allclose(
        np.asarray(represented_number), expected_number, rtol=1e-12)


def test_cic_diagnostic_conserves_and_spreads_single_particle():
    g = _grid(nx=2, ny=2, nz=2, L=2.0)
    state = _lag_state(
        g,
        [1.0e-5],
        xi=1.0e6,
        xyz=(np.array([1.0]), np.array([1.0]), np.array([1.0])),
    )

    qc_nearest, _ = diagnose_liquid_mixing_ratios(
        state, g, rho=1.0, r_rain=4.0e-5, assignment="nearest")
    qc_cic, _ = diagnose_liquid_mixing_ratios(
        state, g, rho=1.0, r_rain=4.0e-5, assignment="cic")

    represented = 1.0e6 * _PREF * (1.0e-5) ** 3
    cell_air = g.dx * g.dy * g.dz
    assert float(jnp.sum(qc_nearest) * cell_air) == pytest.approx(
        represented, rel=1e-12)
    assert float(jnp.sum(qc_cic) * cell_air) == pytest.approx(
        represented, rel=1e-12)
    assert int(jnp.count_nonzero(qc_nearest)) == 1
    assert int(jnp.count_nonzero(qc_cic)) == 8
    assert float(jnp.max(qc_cic)) == pytest.approx(
        float(jnp.max(qc_nearest)) / 8.0, rel=1e-12)


def test_supersaturated_lagrangian_sdm_aerosol_forms_cloud_and_conserves_water():
    g = _grid(nx=2, ny=2, nz=2, L=200.0, n_tracers=3)
    ref = _ref(g, theta0=290.0, qv0=0.0)
    n_sd = 64 * g.cfg.nx * g.cfg.ny * g.cfg.nz
    r_dry = sample_lognormal_radius(
        jax.random.PRNGKey(5), n_sd, 5.0e-8, 2.0,
        r_min=1.0e-8, r_max=5.0e-7, dtype=jnp.float64)
    solute_mass = 4.0 / 3.0 * jnp.pi * 1770.0 * r_dry**3
    sdm = initialize_lagrangian_sdm(
        jax.random.PRNGKey(6),
        g,
        n_sd=n_sd,
        number_concentration=1.0e8,
        radius=r_dry,
        solute_mass=solute_mass,
        dtype=jnp.float64,
        spatial_sampling="cell_stratified",
    )

    T0 = jnp.full((g.cfg.nz,), 290.0, dtype=jnp.float64)
    e = 1.02 * saturation_vapor_pressure(T0)
    qv = constants.epsilon * e / (ref.p_c - e)
    theta = jnp.broadcast_to((T0 / ref.exner_c)[None, None, :],
                             (g.cfg.ny, g.cfg.nx, g.cfg.nz))
    tracers = jnp.zeros((g.cfg.ny, g.cfg.nx, g.cfg.nz, 3), dtype=jnp.float64)
    tracers = tracers.at[..., 0].set(qv[None, None, :])
    cfg = SDMConfig(condensation_integrator="be",
                    lagrangian_diagnostic_assignment="cic")
    water0 = total_water_mass(sdm, tracers, g, ref.rho_c)

    for _ in range(20):
        sdm, theta, tracers, _ = condensation_coupling_step(
            sdm, theta, tracers, g, ref.exner_c, ref.p_c, ref.rho_c,
            2.0, cfg)

    qc, qr = diagnose_liquid_mixing_ratios(
        sdm, g, ref.rho_c, cfg.r_rain, cfg.r_cloud,
        assignment=cfg.lagrangian_diagnostic_assignment)
    water1 = total_water_mass(sdm, tracers, g, ref.rho_c)
    lwp = jnp.mean(jnp.sum(qc * ref.rho_c[None, None, :], axis=-1) * g.dz)
    cloud_cover = jnp.mean(jnp.any(qc > 1.0e-5, axis=-1))

    assert float(water1) == pytest.approx(float(water0), rel=1e-12, abs=1e-8)
    assert float(lwp * 1.0e3) == pytest.approx(20.0, rel=0.25)
    assert float(cloud_cover) == pytest.approx(1.0)
    assert float(jnp.max(qr)) == pytest.approx(0.0)
    assert float(jnp.mean(sdm.droplets.radius >= cfg.r_cloud)) > 0.95


def test_stratified_cic_cloud_patch_is_compact_and_not_speckled():
    g = _grid(nx=6, ny=6, nz=3, L=6.0)
    source_cells = [(ix, iy, 1) for iy in (2, 3) for ix in (2, 3)]
    offsets_xy = (0.125, 0.375, 0.625, 0.875)
    offsets_z = (0.25, 0.75)
    xyz = [[], [], []]
    for ix, iy, iz in source_cells:
        for ox in offsets_xy:
            for oy in offsets_xy:
                for oz in offsets_z:
                    xyz[0].append((ix + ox) * g.dx)
                    xyz[1].append((iy + oy) * g.dy)
                    xyz[2].append((iz + oz) * g.dz)
    n_sd = len(xyz[0])
    state = _lag_state_slots(
        g,
        np.full(n_sd, 1.0e-5),
        np.ones(n_sd) * 1.0e6,
        np.ones(n_sd),
        tuple(np.asarray(a) for a in xyz),
    )

    counts = np.asarray(_cell_sums(state, g, state.droplets.active)).reshape(
        g.cfg.ny, g.cfg.nx, g.cfg.nz)
    qc, _ = diagnose_liquid_mixing_ratios(
        state, g, rho=1.0, r_rain=4.0e-5, assignment="cic")
    qc_np = np.asarray(qc)
    peak = float(qc_np.max())
    nonzero_fraction = float(np.count_nonzero(qc_np) / qc_np.size)
    intermediate_fraction = float(np.mean((qc_np > 0.01 * peak) & (qc_np < 0.5 * peak)))
    source_mask = np.zeros_like(counts, dtype=bool)
    for ix, iy, iz in source_cells:
        source_mask[iy, ix, iz] = True

    np.testing.assert_array_equal(counts[source_mask], np.full(len(source_cells), 32))
    assert int(np.count_nonzero(counts[~source_mask])) == 0
    assert 0.2 < nonzero_fraction < 0.6
    assert intermediate_fraction > 0.1
    assert float(jnp.sum(qc) * g.dx * g.dy * g.dz) == pytest.approx(
        float(jnp.sum(represented_water_mass(state.droplets))), rel=1e-12)


def test_tiny_spectral_les_lagrangian_sdm_smoke_conserves_water():
    g = _grid(nx=4, ny=4, nz=4, L=200.0, n_tracers=3)
    ref = _ref(g, theta0=300.0, qv0=0.02)
    cfg = SDMConfig(
        include_curvature=False, include_solute=False,
        condensation_integrator="euler", n_substeps_condensation=1,
        terminal_velocity="rogers_yau", collision_kernel="golovin")
    ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
    z = jnp.zeros((ny, nx, nz), dtype=jnp.float64)
    theta = jnp.full((ny, nx, nz), 300.0, dtype=jnp.float64)
    tracers = jnp.zeros((ny, nx, nz, 3), dtype=jnp.float64).at[..., 0].set(0.02)
    les = sl.SpectralLESState(
        u=z, v=z, w=jnp.zeros((ny, nx, nz + 1), dtype=jnp.float64),
        rhs_u_prev=z, rhs_v_prev=z,
        rhs_w_prev=jnp.zeros((ny, nx, nz + 1), dtype=jnp.float64),
        theta=theta, rhs_theta_prev=jnp.zeros_like(theta),
        tracers=tracers, rhs_tracers_prev=jnp.zeros_like(tracers))
    sdm = initialize_lagrangian_sdm(
        jax.random.PRNGKey(7), g, n_sd=24,
        number_concentration=1.0e5, radius=1.0e-6, dtype=jnp.float64)
    water0 = total_water_mass(sdm, les.tracers, g, ref.rho_c)

    for n in range(3):
        les, sdm, _, diag = step_lagrangian_sdm_les(
            les, sdm, g, ref, 0.1, cfg, u_geo=(0.0, 0.0), f_cor=0.0,
            first=(n == 0), do_condensation=True, do_coalescence=True)
        assert bool(jnp.all(jnp.isfinite(les.theta)))
        assert bool(jnp.all(jnp.isfinite(les.tracers)))
        assert bool(jnp.all(jnp.isfinite(sdm.droplets.radius)))
        assert float(diag["n_active"]) == 24.0

    water1 = total_water_mass(sdm, les.tracers, g, ref.rho_c)
    assert float(water1) == pytest.approx(float(water0), rel=2e-10, abs=1e-10)
    assert float(jnp.max(les.tracers[..., 1] + les.tracers[..., 2])) >= 0.0


def test_lagrangian_sdm_deterministic_collision_step_has_finite_nonzero_grad():
    g = _grid(nx=2, ny=2, nz=2, L=20.0, n_tracers=3)
    ref = _ref(g, theta0=300.0, qv0=0.02)
    cfg = SDMConfig(
        include_curvature=False, include_solute=False,
        condensation_integrator="euler", n_substeps_condensation=1,
        terminal_velocity="rogers_yau", collision_kernel="golovin",
        golovin_b=1.0e8, collision_mode="deterministic")
    ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
    z = jnp.zeros((ny, nx, nz), dtype=jnp.float64)
    theta = jnp.full((ny, nx, nz), 300.0, dtype=jnp.float64)
    tracers = jnp.zeros((ny, nx, nz, 3), dtype=jnp.float64).at[..., 0].set(0.02)
    les = sl.SpectralLESState(
        u=z, v=z, w=jnp.zeros((ny, nx, nz + 1), dtype=jnp.float64),
        rhs_u_prev=z, rhs_v_prev=z,
        rhs_w_prev=jnp.zeros((ny, nx, nz + 1), dtype=jnp.float64),
        theta=theta, rhs_theta_prev=jnp.zeros_like(theta),
        tracers=tracers, rhs_tracers_prev=jnp.zeros_like(tracers))
    step = make_lagrangian_sdm_les_step(
        g, ref, cfg, u_geo=(0.0, 0.0), f_cor=0.0,
        do_condensation=True, do_coalescence=True)

    def total_rain_after_step(r_cloud):
        droplets = SuperDropletState(
            multiplicity=jnp.array([1.0e6, 1.0e6], dtype=jnp.float64),
            radius=jnp.stack([
                r_cloud,
                jnp.asarray(5.0e-5, dtype=jnp.float64),
            ]),
            solute_mass=jnp.zeros((2,), dtype=jnp.float64),
            active=jnp.ones((2,), dtype=jnp.float64),
        )
        sdm = make_lagrangian_sdm_state(
            droplets,
            jnp.array([5.0, 5.0], dtype=jnp.float64),
            jnp.array([5.0, 5.0], dtype=jnp.float64),
            jnp.array([5.0, 5.0], dtype=jnp.float64),
            jax.random.PRNGKey(3),
            g,
        )
        les1, _sdm1, _us, diag = step(
            les, sdm, jnp.asarray(0.1, dtype=jnp.float64), first=True)
        return jnp.sum(les1.tracers[..., 2]) + 0.0 * jnp.sum(diag["q_r"])

    grad = jax.grad(total_rain_after_step)(jnp.asarray(2.0e-5, dtype=jnp.float64))
    assert bool(jnp.isfinite(grad))
    assert abs(float(grad)) > 0.0


def test_lagrangian_sdm_les_step_factory_jits_static_objects():
    g = _grid(nx=2, ny=2, nz=2, L=20.0, n_tracers=3)
    ref = _ref(g, theta0=300.0, qv0=0.02)
    cfg = SDMConfig(
        include_curvature=False, include_solute=False,
        condensation_integrator="euler", n_substeps_condensation=1,
        terminal_velocity="rogers_yau", collision_kernel="golovin")
    ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
    z = jnp.zeros((ny, nx, nz), dtype=jnp.float64)
    theta = jnp.full((ny, nx, nz), 300.0, dtype=jnp.float64)
    tracers = jnp.zeros((ny, nx, nz, 3), dtype=jnp.float64).at[..., 0].set(0.02)
    les = sl.SpectralLESState(
        u=z, v=z, w=jnp.zeros((ny, nx, nz + 1), dtype=jnp.float64),
        rhs_u_prev=z, rhs_v_prev=z,
        rhs_w_prev=jnp.zeros((ny, nx, nz + 1), dtype=jnp.float64),
        theta=theta, rhs_theta_prev=jnp.zeros_like(theta),
        tracers=tracers, rhs_tracers_prev=jnp.zeros_like(tracers))
    sdm = initialize_lagrangian_sdm(
        jax.random.PRNGKey(11), g, n_sd=8,
        number_concentration=1.0e5, radius=1.0e-6, dtype=jnp.float64)
    step = make_lagrangian_sdm_les_step(
        g, ref, cfg, u_geo=(0.0, 0.0), f_cor=0.0,
        do_condensation=False, do_coalescence=True)

    les1, sdm1, _, diag = step(les, sdm, jnp.asarray(0.1, dtype=jnp.float64),
                               first=True)

    assert bool(jnp.all(jnp.isfinite(les1.theta)))
    assert bool(jnp.all(jnp.isfinite(les1.tracers)))
    assert bool(jnp.all(jnp.isfinite(sdm1.droplets.radius)))
    assert float(diag["n_active"]) == 8.0


def test_lagrangian_sdm_segment_matches_python_step_loop():
    g = _grid(nx=2, ny=2, nz=2, L=20.0, n_tracers=3)
    ref = _ref(g, theta0=300.0, qv0=0.02)
    cfg = SDMConfig(
        include_curvature=False, include_solute=False,
        condensation_integrator="euler", n_substeps_condensation=1,
        terminal_velocity="rogers_yau", collision_kernel="golovin",
        collision_mode="stochastic")
    ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
    z = jnp.zeros((ny, nx, nz), dtype=jnp.float64)
    theta = jnp.full((ny, nx, nz), 300.0, dtype=jnp.float64)
    tracers = jnp.zeros((ny, nx, nz, 3), dtype=jnp.float64).at[..., 0].set(0.02)
    les0 = sl.SpectralLESState(
        u=z, v=z, w=jnp.zeros((ny, nx, nz + 1), dtype=jnp.float64),
        rhs_u_prev=z, rhs_v_prev=z,
        rhs_w_prev=jnp.zeros((ny, nx, nz + 1), dtype=jnp.float64),
        theta=theta, rhs_theta_prev=jnp.zeros_like(theta),
        tracers=tracers, rhs_tracers_prev=jnp.zeros_like(tracers))
    sdm0 = initialize_lagrangian_sdm(
        jax.random.PRNGKey(17), g, n_sd=32,
        number_concentration=1.0e5, radius=1.0e-6, dtype=jnp.float64)

    def step_raw(les, sdm, dt, first=False):
        return step_lagrangian_sdm_les(
            les, sdm, g, ref, dt, cfg, u_geo=(0.0, 0.0), f_cor=0.0,
            first=first, do_condensation=True, do_coalescence=True)

    step = jax.jit(step_raw, static_argnames=("first",))
    run_segment = make_lagrangian_sdm_step_segment(step_raw, segment_steps=3)
    dt = jnp.asarray(0.1, dtype=jnp.float64)

    def zero_diag():
        zero = jnp.asarray(0.0, dtype=jnp.float64)
        return LagrangianSDMSegmentDiagnostics(zero, zero, zero, zero)

    les_loop, sdm_loop, diag_loop = les0, sdm0, zero_diag()
    for n in range(5):
        les_loop, sdm_loop, us, diag = step(
            les_loop, sdm_loop, dt, first=(n == 0))
        diag_loop = update_lagrangian_sdm_segment_diagnostics(
            diag_loop, us, diag)

    les_seg, sdm_seg, us, diag = step(les0, sdm0, dt, first=True)
    diag_seg = update_lagrangian_sdm_segment_diagnostics(
        zero_diag(), us, diag)
    les_seg, sdm_seg, diag_seg = run_segment(
        les_seg, sdm_seg, dt, jnp.asarray(3, dtype=jnp.int32), diag_seg)
    les_seg, sdm_seg, diag_seg = run_segment(
        les_seg, sdm_seg, dt, jnp.asarray(1, dtype=jnp.int32), diag_seg)

    _assert_tree_allclose(les_seg, les_loop)
    _assert_tree_allclose(sdm_seg, sdm_loop)
    _assert_tree_allclose(diag_seg, diag_loop)


def test_run_bomex_lagrangian_sdm_driver_smoke(tmp_path):
    repo = Path(__file__).resolve().parents[4]
    case_dir = Path(os.environ.get(
        "LEGOESM_GSAM_ROOT",
        "/home/gentine/Documents/Code/gSAM/gsam1.8.7/gSAM1.8.7",
    )) / "CASES" / "BOMEX"
    if not (case_dir / "snd").exists():
        pytest.skip(f"BOMEX gSAM case deck not found at {case_dir}")
    py = repo / ".venv" / "bin" / "python"
    if not py.exists():
        py = Path(sys.executable)
    env = os.environ.copy()
    env.update({"JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "1"})
    cmd = [
        str(py), "scripts/run/run_bomex_les.py",
        "--case-dir", str(case_dir),
        "--lagrangian-sdm",
        "--nx", "4", "--ny", "4", "--nz", "4",
        "--Lx", "200", "--Ly", "200", "--Lz", "200",
        "--hours", "0.001", "--dt", "1.0",
        "--n-sd", "16", "--collision-mode", "deterministic",
        "--print-every", "1", "--record-frames", "0",
        "--output", str(tmp_path / "bomex_lag_sdm"),
    ]
    proc = subprocess.run(
        cmd, cwd=repo, env=env, text=True, capture_output=True, timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[DONE-LAGRANGIAN-SDM]" in proc.stdout
    out = np.load(tmp_path / "bomex_lag_sdm" / "bomex_lagrangian_sdm_final.npz")
    assert np.isfinite(out["water_final"])
    assert float(out["max_sdm_water_error"]) < 1.0e-6
    assert float(out["mean_particle_displacement"]) > 0.0
