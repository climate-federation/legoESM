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


def _smooth_surface_forcing(grid):
    """A smooth, NONZERO cell-centred ``OceanSurfaceForcing`` (tau_x + q_net) so
    the SPMD step exercises the in-core external-forcing path (wind-stress
    momentum + surface heat), not just the rest/perturbation dynamics.

    Both channels are ``(n_lat, n_lon)`` T-point fields (the
    ``omip2_applicator`` layout) — they shard ``P("lat")`` exactly like a cell
    state field and are interpolated to the u/v faces INSIDE the step.
    """
    from legoesm.ocean.state import OceanSurfaceForcing

    lat = np.asarray(grid.lat_T)                      # (n_lat, n_lon) [rad]
    lon = np.asarray(grid.lon_T)
    # Idealised zonal wind stress (trade/westerly banding) + a smooth heat flux.
    tau_x = (0.1 * np.cos(3.0 * lat)).astype(np.float64)     # [Pa]
    q_net = (40.0 * np.cos(lat) * np.cos(lon)).astype(np.float64)  # [W/m^2]
    return OceanSurfaceForcing(
        tau_x=jnp.asarray(tau_x),
        tau_y=jnp.zeros_like(jnp.asarray(tau_x)),
        q_net=jnp.asarray(q_net),
    )


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
@pytest.mark.parametrize("barotropic_solver",
                         ["explicit_substep", "implicit_cn"])
@pytest.mark.parametrize("with_forcing", [False, True],
                         ids=["unforced", "forced"])
def test_latlon_ocean_spmd_tripole_matches_single_device(
        barotropic_solver, with_forcing):
    # explicit_substep is the SPMD-proven solver (the regular-grid gate uses it);
    # implicit_cn is the OMIP production solver (cold-start-stable).  Under SPMD
    # implicit_cn routes to the FIXED-iteration distributed PCG (static scan
    # schedule) — barotropic_implicit_latlon_cgrid keys _use_pcg on the armed
    # spmd halo backend — instead of the jax.scipy.cg while_loop (whose
    # collectives SIGABRT under shard_map).  Both must match the single-device
    # step on a tripole (north fold + barotropic solve).
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
    # from_flat: barotropic_solver / n_barotropic_substeps moved into the
    # nested barotropic group (#501); the flat-name shim routes them.
    cfg = LatLonCGridOceanConfig.from_flat(
        barotropic_solver=barotropic_solver,
        A_h=1000.0, K_h=500.0, A_v=1e-3, K_v=1e-5,
        n_barotropic_substeps=10,
    )
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    state0 = _perturbed_state(grid, z_coord)
    dt, n_steps = 600.0, 3

    # NONZERO surface forcing (wind stress + heat) when ``with_forcing`` — the
    # SAME global pytree is passed to BOTH the serial and the SPMD step, so the
    # equivalence gate now also proves the in-core external-forcing path
    # (interp_cell_to_{u,v}face wind stress + surface heat) under SPMD: the
    # cell-shaped forcing is sharded P("lat") and the tau face interpolation
    # routes through the armed SPMD band halo.  ``None`` reproduces the unforced
    # step byte-for-byte.
    sf = _smooth_surface_forcing(grid) if with_forcing else None

    # single-device reference
    s = state0
    for _ in range(n_steps):
        s = model.step(s, dt, surface_forcing=sf)

    model._ensure_vertex_mask(state0)        # prime the build-once vmask cache

    dev = create_latlon_mesh(n_devices=4)
    step = make_sharded_ocean_step(model, dev.mesh)
    ss = shard_state_latlon(state0, dev.mesh)
    for _ in range(n_steps):
        ss = step(ss, dt, surface_forcing=sf)
    ss = gather_state_latlon(ss, dev.mesh)

    # FP re-association floor (NOT a bug margin): the sharded reductions are
    # batch_psum_spmd across bands (provably global — a missing/local reduction
    # gives O(1) error, a missing band-cut halo O(1e-2)), so the only
    # serial-vs-SPMD difference is reduction ORDER.  explicit_substep holds the
    # strict 2e-4 floor (it proves the north fold + the operators/halos/single
    # reductions).  implicit_cn adds the FIXED-iteration PCG, whose per-iteration
    # global dot products propagate the reduction-order round-off through the
    # Krylov iterates -> a slightly larger floor (measured eta max-abs 3.4e-4
    # = 0.34 mm SSH, domain-wide, first crossing 2e-4 at a band cut); 5e-4 is
    # physically negligible yet still catches a real missing-halo/reduction
    # regression.  Follow-up: a multi-step boundedness sweep for extra confidence.
    _ATOL = 5.0e-4 if barotropic_solver == "implicit_cn" else 2.0e-4
    _RTOL = 1.0e-3
    for nm in ("u", "v", "eta", "T", "S"):
        a = np.asarray(getattr(s, nm).data)
        b = np.asarray(getattr(ss, nm).data)
        np.testing.assert_allclose(b, a, atol=_ATOL, rtol=_RTOL,
                                   err_msg=f"SPMD tripole {nm} mismatch")
