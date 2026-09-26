"""MPAS/Voronoi OCEAN SPMD step (``voronoi_spmd_ocean``) vs the serial step.

The serial reference is ``MPASOceanModel._step_impl`` on the SAME reordered +
padded global mesh, so the only thing under test is the sharding: ppermute halo
fill, in-step stage-frontier refreshes (cells / edges / both / vertices),
owned-masked psum reductions (eta-floor redistribution, conservation fixer,
implicit PCG), and the forcing-leaf routing (traced per-cell leaves ride the
entry pack; concrete sponge references are localised once).

Non-vacuity: ``test_halo_nostage_breaks_parity`` runs the SAME comparison with
the halo staging switched off (``LEGOESM_MPAS_HALO_NOSTAGE=1`` — a documented
measurement-only knob that skips the ppermute rounds) and requires the parity
to FAIL, so a regression that silently stops exchanging halos cannot pass.

Needs 4 CPU devices: ``XLA_FLAGS=--xla_force_host_platform_device_count=4``;
x64 (the re-association floor is ~1e-13 relative there).
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

N_DEV = 4


def _need_devices(n):
    if len(jax.devices()) < n:
        pytest.skip(f"need {n} devices (XLA_FLAGS=--xla_force_host_platform_device_count={n})")


def _build(config_kwargs, *, nlev=4, n_dev=N_DEV, level=2, partial_cells=False):
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
    from legoesm.ocean.freshwater import FreshwaterForcing
    from legoesm.ocean.init_mpas import rest_state_mpas_ocean
    from legoesm.ocean.mpas_config import MPASOceanConfig
    from legoesm.ocean.sponge import SpongeForcing
    from legoesm.ocean.state import OceanSurfaceForcing
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding
    from legoesm.parallel.voronoi_spmd_ocean import mask_padded_cells

    mesh0 = create_voronoi_mesh(level)
    n_real = mesh0.nCells
    mesh = reorder_voronoi_for_sharding(mesh0, n_dev, edge_order="owner")
    z = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    config = MPASOceanConfig(**config_kwargs)
    state = rest_state_mpas_ocean(mesh, z, H_max=4000.0)
    state = mask_padded_cells(state, n_real)
    rng = np.random.default_rng(7)
    m = np.asarray(state.land_mask.data)
    if partial_cells:
        # variable bathymetry -> the coordinate carries PER-CELL statics
        # (h_partial, bottom_level, is_active): the case job 27326448 hit
        from legoesm.ocean.vertical import create_partial_cell_coordinate
        H = np.asarray(state.H_bathy.data) * (0.4 + 0.6 * rng.random(mesh.nCells)) * m
        state = state._replace(H_bathy=state.H_bathy.replace(data=jnp.asarray(H)))
        z = create_partial_cell_coordinate(z, jnp.asarray(H))
    model = MPASOceanModel(mesh, z, config)
    em = m[np.asarray(mesh.cellsOnEdge[0])] * m[np.asarray(mesh.cellsOnEdge[1])]
    state = state._replace(
        u=state.u.replace(data=jnp.asarray(
            0.05 * rng.standard_normal(state.u.data.shape) * em[:, None])),
        T=state.T.replace(data=state.T.data + jnp.asarray(
            0.5 * rng.standard_normal(state.T.data.shape) * m[:, None])),
        S=state.S.replace(data=state.S.data + jnp.asarray(
            0.05 * rng.standard_normal(state.S.data.shape) * m[:, None])),
        eta=state.eta.replace(data=jnp.asarray(
            0.02 * rng.standard_normal(state.eta.data.shape) * m)),
    )
    nC = mesh.nCells
    fw = FreshwaterForcing(
        precip=jnp.asarray(1e-5 * rng.random(nC) * m),
        evap=jnp.asarray(1e-5 * rng.random(nC) * m),
        runoff=jnp.asarray(1e-6 * rng.random(nC) * m),
        ice_fw=jnp.zeros(nC),
        restoring=jnp.zeros(nC),
    )
    sf = OceanSurfaceForcing(
        sw_down=jnp.asarray(200.0 * rng.random(nC) * m),
        q_net=jnp.asarray(50.0 * rng.standard_normal(nC) * m),
        tau_x=jnp.asarray(0.1 * rng.standard_normal(nC) * m),
        tau_y=jnp.asarray(0.1 * rng.standard_normal(nC) * m),
    )
    lat = np.degrees(np.asarray(mesh.latCell))
    gamma = np.where(np.abs(lat) > 60.0, 1.0 / (5 * 86400.0), 0.0) * m
    sp = SpongeForcing(
        gamma=jnp.asarray(gamma),
        T_ref=np.asarray(state.T.data),          # concrete -> localised path
        S_ref=np.asarray(state.S.data),
    )
    return mesh, n_real, model, state, (fw, sf, sp)


def _run_serial(model, state, forcing, dt, n):
    fw, sf, sp = forcing

    @jax.jit
    def _one(st):
        return model._step_impl(st, dt, freshwater=fw, surface_forcing=sf, sponge=sp)

    for _ in range(n):
        state = _one(state)
    return state


def _run_spmd(model, mesh, n_real, state, forcing, dt, n, tracer_advection,
              n_dev=N_DEV):
    from legoesm.parallel.voronoi_spmd_ocean import (
        build_mpas_ocean_spmd_layout,
        gather_state_mpas_ocean_spmd,
        make_sharded_mpas_ocean_step,
        shard_state_mpas_ocean_spmd,
    )
    fw, sf, sp = forcing
    layout = build_mpas_ocean_spmd_layout(
        mesh, n_dev, n_cells_real=n_real, tracer_advection=tracer_advection,
        nlev=state.T.data.shape[1])
    step = make_sharded_mpas_ocean_step(model, layout)
    st = shard_state_mpas_ocean_spmd(state, layout)
    fw_s = jax.tree.map(lambda x: jax.device_put(x, layout.cell_sharding), fw)
    sf_s = jax.tree.map(lambda x: jax.device_put(x, layout.cell_sharding), sf)
    sp_s = sp._replace(gamma=jax.device_put(sp.gamma, layout.cell_sharding))

    @jax.jit
    def _one(s, fw_, sf_, gamma_):
        return step(s, dt, freshwater=fw_, surface_forcing=sf_,
                    sponge=sp_s._replace(gamma=gamma_))

    for _ in range(n):
        st = _one(st, fw_s, sf_s, sp_s.gamma)
    out = gather_state_mpas_ocean_spmd(st, layout, to_host=True)
    return out, layout


def _compare(ref, got, n_real, mesh):
    m = np.asarray(ref.land_mask.data)[:n_real] > 0.5
    em = (np.asarray(ref.land_mask.data)[np.asarray(mesh.cellsOnEdge[0])]
          * np.asarray(ref.land_mask.data)[np.asarray(mesh.cellsOnEdge[1])]) > 0.5
    worst = {}
    for name in ("T", "S", "eta", "w"):
        a = np.asarray(getattr(ref, name).data)[:n_real][m]
        b = np.asarray(getattr(got, name).data)[:n_real][m]
        worst[name] = (float(np.max(np.abs(a - b))), float(np.max(np.abs(a))))
    a = np.asarray(ref.u.data)[em]
    b = np.asarray(got.u.data)[em]
    worst["u"] = (float(np.max(np.abs(a - b))), float(np.max(np.abs(a))))
    assert all(np.isfinite([v for pair in worst.values() for v in pair])), worst
    return worst


# Absolute agreement floors per field (x64), from the 2026-09-08 measurement
# on this fixture (3 steps): SPMD-vs-serial(jit) T 3.3e-10, S 8.5e-10,
# eta 7e-17 (explicit) / 2e-11 (implicit_cn: fixed-M distributed PCG vs stock
# CG, both at 1e-10 tolerance), w 1.9e-11, u 1.7e-12; IDENTICAL at 2 and 4
# devices, so it does not scale with partition boundaries.  The CONTROL
# serial-jit-vs-serial-EAGER (no sharding at all) differs by T 4.9e-9,
# S 3.8e-9, eta 1.5e-6, u 1.1e-6 — the model's own XLA re-association floor,
# an order of magnitude ABOVE the SPMD gap.  A halo defect lands at
# O(1e-3..1) (the nostage control below); the vertex lane has its own
# global-id probe (test_vertex_lane_restores_global_ids).
_ATOL = {"T": 2e-9, "S": 5e-9, "eta": 1e-10, "w": 1e-10, "u": 1e-10}


_CASES = {
    "explicit_fixer": dict(use_conservation_fixer=True, n_barotropic_substeps=10,
                           normalize_freshwater=True),
    "partial_cells_fixer": dict(use_conservation_fixer=True, n_barotropic_substeps=10,
                                _partial_cells=True),
    "implicit_superbee_del4": dict(
        barotropic_solver="implicit_cn", tracer_advection="superbee",
        K_zeta_bih=1.0e13, n_barotropic_substeps=10),
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_spmd_matches_serial(case):
    _need_devices(N_DEV)
    from legoesm.grids.halo import set_halo_backend, set_spmd_mesh
    kw = dict(_CASES[case])
    partial = bool(kw.pop("_partial_cells", False))
    mesh, n_real, model, state, forcing = _build(kw, partial_cells=partial)
    dt = 300.0
    ref = _run_serial(model, state, forcing, dt, 3)
    try:
        got, layout = _run_spmd(model, mesh, n_real, state, forcing, dt, 3,
                                kw.get("tracer_advection", "upwind"))
    finally:
        set_halo_backend("local")
        set_spmd_mesh(None)
    worst = _compare(ref, got, n_real, mesh)
    bad = {k: v for k, v in worst.items() if v[0] > _ATOL[k]}
    assert not bad, (bad, worst)
    # the pads stayed land and the owned rows really came from 4 devices
    assert layout.n_devices == N_DEV
    assert np.all(np.asarray(got.land_mask.data)[n_real:] == 0)


def test_spmd_matches_serial_3_devices_uneven_vertex_blocks():
    """3 devices: 162 cells split 54/54/54 (no cell pads) while the 320
    vertices do NOT divide — the vertex lane must follow the mesh's incident-
    cell ownership, not an equal split.  del4 (vertex consumer) on."""
    _need_devices(3)
    from legoesm.grids.halo import set_halo_backend, set_spmd_mesh
    kw = _CASES["implicit_superbee_del4"]
    mesh, n_real, model, state, forcing = _build(kw, n_dev=3)
    dt = 300.0
    ref = _run_serial(model, state, forcing, dt, 3)
    try:
        got, layout = _run_spmd(model, mesh, n_real, state, forcing, dt, 3,
                                "superbee", n_dev=3)
    finally:
        set_halo_backend("local")
        set_spmd_mesh(None)
    assert layout.n_cells % 3 == 0 and mesh.nVertices % 3 != 0
    worst = _compare(ref, got, n_real, mesh)
    bad = {k: v for k, v in worst.items() if v[0] > _ATOL[k]}
    assert not bad, (bad, worst)


def test_vertex_lane_restores_global_ids():
    """Direct gate for the vertex ppermute schedule: owned vertices seeded with
    their global id, everything else with -1; after the lane runs every LOCAL
    vertex slot on every device carries its global id (padding slots stay -1).
    A dropped / mis-addressed vertex exchange leaves -1s or wrong ids."""
    _need_devices(N_DEV)
    from legoesm.grids.halo import set_halo_backend, set_spmd_mesh
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding
    from legoesm.parallel.voronoi_spmd_ocean import (
        build_mpas_ocean_spmd_layout, spmd_vertex_refresh_probe,
    )
    mesh = reorder_voronoi_for_sharding(create_voronoi_mesh(2), N_DEV, edge_order="owner")
    try:
        layout = build_mpas_ocean_spmd_layout(mesh, N_DEV, nlev=1)
        assert layout.vppermute_perms, "fixture has no vertex exchange"
        seed = np.full((N_DEV, layout.max_lv), -1.0)
        for d in range(N_DEV):
            lv = layout.local_vertices[d]
            owned = layout.vertex_owner[lv] == d
            seed[d, :len(lv)][owned] = lv[owned]
        # every device must be missing something before the lane runs
        assert all(np.any(seed[d, :len(layout.local_vertices[d])] < 0) for d in range(N_DEV))
        out = spmd_vertex_refresh_probe(layout, seed)
    finally:
        set_halo_backend("local")
        set_spmd_mesh(None)
    for d in range(N_DEV):
        lv = layout.local_vertices[d]
        np.testing.assert_array_equal(out[d, :len(lv)], lv.astype(float))


def test_halo_nostage_breaks_parity(monkeypatch):
    """Non-vacuity: with the halo staging skipped (the ppermute rounds never
    run) the SAME comparison must FAIL by orders of magnitude — otherwise the
    parity tests could pass without exchanging halos."""
    _need_devices(N_DEV)
    from legoesm.grids.halo import set_halo_backend, set_spmd_mesh
    kw = _CASES["explicit_fixer"]
    mesh, n_real, model, state, forcing = _build(kw)
    dt = 300.0
    ref = _run_serial(model, state, forcing, dt, 3)
    monkeypatch.setenv("LEGOESM_MPAS_HALO_NOSTAGE", "1")
    try:
        got, _ = _run_spmd(model, mesh, n_real, state, forcing, dt, 3, "upwind")
    finally:
        set_halo_backend("local")
        set_spmd_mesh(None)
    worst = _compare(ref, got, n_real, mesh)
    assert max(v[0] / _ATOL[k] for k, v in worst.items()) > 1e3, worst


def test_spmd_matches_serial_ico4_eight_steps():
    """Larger mesh (2562 cells), 8 steps, del4 + superbee + implicit: a
    partition with more seams and a longer window than the ico2 fixture, so a
    ring-3 stale-halo consumer that the small mesh cannot excite would show up
    as an error GROWING with steps (GLM review D1).

    Measured 2026-09-08 (x64): SPMD-vs-serial(jit) T 1.4e-9, S 5.0e-9,
    eta 7e-11, w 4.5e-11, u 1.7e-11 — the SAME at 2 and 4 devices; the
    serial-jit-vs-serial-EAGER control on this fixture is T 4.0e-7, S 6.9e-9,
    eta 1.5e-4, u 6.9e-6, i.e. the SPMD gap sits below the model's own
    re-association floor.  Tolerances = ~3x the measured SPMD gap."""
    _need_devices(N_DEV)
    from legoesm.grids.halo import set_halo_backend, set_spmd_mesh
    kw = _CASES["implicit_superbee_del4"]
    mesh, n_real, model, state, forcing = _build(kw, level=4)
    dt = 300.0
    ref = _run_serial(model, state, forcing, dt, 8)
    try:
        got, _ = _run_spmd(model, mesh, n_real, state, forcing, dt, 8, "superbee")
    finally:
        set_halo_backend("local")
        set_spmd_mesh(None)
    worst = _compare(ref, got, n_real, mesh)
    atol = {"T": 5e-9, "S": 2e-8, "eta": 3e-10, "w": 2e-10, "u": 1e-10}
    bad = {k: v for k, v in worst.items() if v[0] > atol[k]}
    assert not bad, (bad, worst)


def test_refusals():
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.ocean.mpas_config import MPASOceanConfig
    from legoesm.parallel.voronoi_spmd_ocean import (
        build_mpas_ocean_spmd_layout,
        refuse_unsupported_spmd_config,
    )
    with pytest.raises(NotImplementedError, match="use_baroclinic_rho_ref"):
        refuse_unsupported_spmd_config(MPASOceanConfig(use_baroclinic_rho_ref=True))
    # unprepared (unpadded) mesh is refused before any device work
    mesh = create_voronoi_mesh(1)
    if mesh.nCells % 3:
        with pytest.raises(ValueError, match="not\\s+padded"):
            build_mpas_ocean_spmd_layout(mesh, 3)


def test_cfl_check_ignores_padded_edges():
    """Padded ghost edges (dcEdge=1 m) must not trip the barotropic CFL check."""
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
    from legoesm.ocean.mpas_config import MPASOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding
    mesh0 = create_voronoi_mesh(2)
    # 480 edges: divisible by 4 (no edge pads) but not by 7 -> ghost edges exist
    mesh = reorder_voronoi_for_sharding(mesh0, 7, edge_order="owner")
    assert mesh.nEdges > mesh0.nEdges, "fixture must actually pad edges"
    z = create_ocean_z_star(n_levels=3)
    cfl_pad = MPASOceanModel(mesh, z, MPASOceanConfig()).check_barotropic_cfl(300.0)
    cfl_ref = MPASOceanModel(mesh0, z, MPASOceanConfig()).check_barotropic_cfl(300.0)
    assert np.isclose(cfl_pad, cfl_ref, rtol=1e-12)
