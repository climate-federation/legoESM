"""SPMD equivalence gate for the lat-lon C-grid ocean step on a TRIPOLE grid.

The tripole twin of ``test_latlon_ocean_spmd_step.py`` (regular grid): N steps of
``make_sharded_ocean_step`` on a synthetic tripole (active bipolar north fold)
across 4 (CPU) devices must match the single-device step to the same
floating-point re-association tolerance.  This is the CORRECTNESS GATE for the
tripole north fold under SPMD (the data-dependent ``north_fold_mask`` /
``apply_north_fold`` conversion of the ~15 inline operator fold overwrites + the
staggered-v north boundary): the single shard_map trace runs on every band, so
the serial ``if fold_is_local`` overwrite is uniformly False — the fold is
selected on the north band (``axis_index("lat")==N-1``).  A missed fold path
(an operator still serial-only, or the v top-row reconstructed as a pole wall
instead of the fold partner) shows up here as an O(1e-3+) mismatch at the
northern band, far above the re-association floor.

eORCA025 ¼° (n_lat=1207, the faithful high-res OMIP cell) is a tripole; this
gate is what unlocks running it multi-GPU.

Run: ``XLA_FLAGS=--xla_force_host_platform_device_count=4 \
      JAX_ENABLE_X64=1 pytest tests/parallel/test_latlon_ocean_spmd_tripole.py``
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.tripole import create_synthetic_tripole
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel


def _perturbed_state(grid, z_coord):
    """Rest state + small u/v/T/eta perturbations so the step exercises every
    term (advection / Coriolis / PGF / fold), not the trivial rest point."""
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0)
    rng = np.random.default_rng(0)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    nlev = z_coord.n_levels
    u = 0.02 * rng.standard_normal((n_lat, n_lon + 1, nlev))
    v = 0.02 * rng.standard_normal((n_lat + 1, n_lon, nlev))
    eta = 0.005 * rng.standard_normal((n_lat, n_lon))
    T = (5.0 + 15.0 * np.exp(np.linspace(0, -4, nlev))[None, None, :]
         + 0.05 * rng.standard_normal((n_lat, n_lon, nlev)))
    return state._replace(
        u=state.u.replace(data=jnp.asarray(u)),
        v=state.v.replace(data=jnp.asarray(v)),
        eta=state.eta.replace(data=jnp.asarray(eta)),
        T=state.T.replace(data=jnp.asarray(T)))


def _have_sharded_step():
    try:
        from legoesm.ocean.dynamics.sharded_ocean_step import (  # noqa: F401
            make_sharded_ocean_step,
        )
        from legoesm.parallel.mesh import create_latlon_mesh  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(jax.device_count() < 4,
                    reason="needs >=4 devices (XLA_FLAGS host device count)")
@pytest.mark.skipif(not _have_sharded_step(),
                    reason="sharded_ocean_step module not present")
@pytest.mark.parametrize("barotropic_solver", [
    "explicit_substep",
    # implicit_cn under SPMD is a SEPARATE gap (NOT the north fold): its
    # Helmholtz PCG solver (solve_helmholtz_freesurface fori_loop + A_op halo
    # ppermute + global_rel_residual reductions in barotropic_common) was never
    # SPMD-wired — the regular-grid gate only ever ran explicit_substep — and it
    # SIGABRTs in the in-A_op halo ppermute under shard_map.  The tripole north
    # fold itself is PROVEN by the explicit_substep case (the SPMD-validated
    # solver).  Skipped (a SIGABRT aborts the process, so xfail can't catch it);
    # tracked as the implicit_cn-SPMD follow-up.
    pytest.param("implicit_cn", marks=pytest.mark.skip(
        reason="implicit_cn barotropic solver not yet SPMD-wired (separate "
               "gap from the north fold; explicit_substep is the SPMD gate)")),
])
def test_latlon_ocean_spmd_tripole_matches_single_device(barotropic_solver):
    # explicit_substep is the SPMD-proven solver (the regular-grid gate uses it)
    # -> isolates the tripole north fold; implicit_cn is the OMIP production
    # solver (the cold-start-stable one) -> a separate SPMD follow-up (skipped).
    from legoesm.parallel.mesh import create_latlon_mesh
    from legoesm.ocean.dynamics.sharded_ocean_step import (
        make_sharded_ocean_step,
        shard_state_latlon,
        gather_state_latlon,
    )

    n_lat, n_lon, nlev = 48, 96, 10          # n_lat % 4 == 0 (band-divisible)
    grid = create_synthetic_tripole(n_lat, n_lon)
    assert grid.fold.is_active, "synthetic tripole must carry an active fold"
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    cfg = LatLonCGridOceanConfig(
        barotropic_solver=barotropic_solver,
        A_h=1000.0, K_h=500.0, A_v=1e-3, K_v=1e-5,
        n_barotropic_substeps=10,
    )
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    state0 = _perturbed_state(grid, z_coord)
    dt, n_steps = 600.0, 3

    # single-device reference
    s = state0
    for _ in range(n_steps):
        s = model.step(s, dt)

    model._ensure_vertex_mask(state0)        # prime the build-once vmask cache

    dev = create_latlon_mesh(n_devices=4)
    step = make_sharded_ocean_step(model, dev.mesh)
    ss = shard_state_latlon(state0, dev.mesh)
    for _ in range(n_steps):
        ss = step(ss, dt)
    ss = gather_state_latlon(ss, dev.mesh)

    # Same FP re-association floor as the regular-grid gate (the sharded
    # split-explicit barotropic re-associates its reductions across bands);
    # ~1e-4 is physically negligible yet catches a real missing-fold regression
    # (those give O(1e-3+) at the northern band). See test_latlon_ocean_spmd_step.
    _ATOL, _RTOL = 2.0e-4, 1.0e-3
    for nm in ("u", "v", "eta", "T", "S"):
        a = np.asarray(getattr(s, nm).data)
        b = np.asarray(getattr(ss, nm).data)
        np.testing.assert_allclose(b, a, atol=_ATOL, rtol=_RTOL,
                                   err_msg=f"SPMD tripole {nm} mismatch")
