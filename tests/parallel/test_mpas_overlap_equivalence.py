"""The interior/rim halo-overlap step must match the ordinary sharded step.

``LEGOESM_MPAS_HALO_OVERLAP=1`` computes the tendency for interior owned
entities while the ppermute halo fill is in flight, then recomputes the rim on
a submesh once the halo lands. This asserts that switching it on changes NO
result: overlap-on and overlap-off sharded steps, from the same initial state,
agree to floating-point re-association after a few SSP-RK3 steps. Forces the
ppermute strategy (the overlap requires it) and uses the production
hyperdiffusion config (nu_del4>0, so the rim closure runs at its wider radius).

Needs 4 CPU devices: run with
``XLA_FLAGS=--xla_force_host_platform_device_count=4``.
"""
import os

import numpy as np
import pytest

pytest.importorskip("jax")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)


def _build_and_run(overlap: bool, n_steps=3):
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig, MPASPrimitiveEquationModel,
    )
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.parallel.mesh import (
        create_voronoi_device_mesh, replicate_pytree,
    )
    from legoesm.parallel.sharded_dynamics import make_voronoi_sharded_step
    from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas

    devices = 4
    mesh = create_voronoi_mesh(subdivision_level=5)   # 10242 cells
    mesh = reorder_voronoi_for_sharding(mesh, devices)
    sigma = create_sigma_coordinate(8)
    cfg = MPASPrimitiveEquationConfig(
        nu_del4=1e16, nu_del4_ps=1e16, fix_mass=True,
        pv_scheme='energy', time_integrator='ssp_rk3')
    dev_config = create_voronoi_device_mesh(
        nCells=mesh.nCells, nEdges=mesh.nEdges,
        nVertices=mesh.nVertices, n_devices=devices)
    mesh_rep = replicate_pytree(mesh, dev_config)
    model = MPASPrimitiveEquationModel(mesh_rep, sigma, cfg)

    prev = os.environ.get("LEGOESM_MPAS_HALO_OVERLAP")
    os.environ["LEGOESM_MPAS_HALO_OVERLAP"] = "1" if overlap else "0"
    try:
        # Force ppermute so the overlap engages (auto could pick allgather on
        # a small per-device shard).
        step_fn = make_voronoi_sharded_step(model, dev_config,
                                            halo_strategy="ppermute")
        state = baroclinic_wave_init_mpas(mesh, sigma, perturbed=True)
        dt = 200.0
        for _ in range(n_steps):
            state = step_fn(state, dt)
    finally:
        if prev is None:
            os.environ.pop("LEGOESM_MPAS_HALO_OVERLAP", None)
        else:
            os.environ["LEGOESM_MPAS_HALO_OVERLAP"] = prev
    return state


def test_overlap_matches_serial_sharded():
    if len(jax.devices("cpu")) < 4:
        pytest.skip("need 4 CPU devices "
                    "(XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    off = _build_and_run(overlap=False)
    on = _build_and_run(overlap=True)
    # Same program up to stencil-sum re-association on the rim submesh.
    tol = 1e-9
    for name in ("u", "T", "p_s"):
        a = np.asarray(getattr(off, name).data)
        b = np.asarray(getattr(on, name).data)
        denom = np.max(np.abs(a)) + 1e-30
        err = np.max(np.abs(a - b)) / denom
        assert err < tol, f"{name}: overlap vs serial rel err {err:.2e} > {tol}"
