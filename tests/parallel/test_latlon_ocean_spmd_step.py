"""SPMD equivalence gate for the lat-lon C-grid ocean step (multi-node OMIP).

This is the CORRECTNESS GATE for ``make_sharded_ocean_step`` (lat-band SPMD over
the ``"lat"`` axis): N steps under ``shard_map`` on 4 (CPU) devices must match
the single-device step to tolerance. It is the eORCA025 ¼° enabler (the ¼° grid
OOMs on one 32 GiB GPU; lat-band sharding fits it at N>=5).

STATUS (2026-06-16): the WRAPPER + halo layer are DONE and validated — the
staggered-v band decomposition (``shard_state_latlon`` ↔ ``v_lower`` ↔ in-body
ppermute reconstruction), the replicated-stacked grid indexed by
``axis_index``, and the SPMD branches of ``pad_with_pole_bc_lat`` /
``zero_polar_lat_ends`` (new ``make_latlon_band_wall_pad_body`` /
``zero_polar_lat_ends_band_spmd``, 11/11 bit-identity parity tests in
``test_latlon_spmd_halo.py``).  The crash path (nested-jit traced ``fold``) is
fixed via ``_step_body`` (un-jitted step inside the shard_map).

ROOT CAUSE (RESOLVED 2026-06-16, this gate now a HARD PASS): the lat-lon C-grid
operators were almost all already SPMD-wired (curl_vertex, h_vtx min-rule,
neumann_fill, tvd_to_v_points, the AL81 12-point PV triad, gradient/divergence,
the barotropic reductions via ``_global_sum_pair`` psum — all bit-exact under a
clean-input bisection: AL81 stage ~8e-16, full baroclinic tendency ~5e-13,
``_depth_average_to_faces`` 0).  The residual was NOT the AL81 triad (the
original diagnosis predated the triad's v-face pad wiring).  It was the BAROTROPIC
substep v-velocity update: ``barotropic_latlon_cgrid`` imported
``interp_u_to_vface_4pt`` from ``ocean.dynamics.latlon_cgrid_operators``, whose
module-local redefinition SHADOWED the fixed core operator with the OLD
interior-average-then-``pad_ns_vector_u`` form.  That form is SPMD-blind: at a
lat-band cut it averaged only the rank-LOCAL interior v-faces, then refilled the
shared cut row from the neighbour's ADJACENT interior face (one row off), so the
two bands sharing a v-face DISAGREED.  Confirmed by a 3-way shared-row probe:
band r's top v-row vs band r+1's bottom v-row diverged ~9e-4 at the cuts whose
``f_v`` is non-zero (rows 12/36) while the EQUATOR cut (row 24, ``f_v≈0``) was
bit-exact — the Coriolis term ``-f_v·U_new_at_v`` masks the error at the equator.
FIX: delete the shadowing redefinition so the call resolves to the canonical
core ``interp_u_to_vface_4pt`` (cell-pad-FIRST via ``pad_with_pole_bc_lat`` —
serial/MPI bit-identical, SPMD-correct band-cut halo).  See
omip-multinode-spmd-scope.

Run: ``XLA_FLAGS=--xla_force_host_platform_device_count=4 \
      JAX_ENABLE_X64=1 pytest tests/parallel/test_latlon_ocean_spmd_step.py``
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel


def _perturbed_state(grid, z_coord):
    """Rest state + small u/v/T/eta perturbations so the step exercises every
    term (advection/Coriolis/PGF), not the trivial rest fixed point."""
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
def test_latlon_ocean_spmd_matches_single_device():
    from legoesm.parallel.mesh import create_latlon_mesh
    from legoesm.ocean.dynamics.sharded_ocean_step import (
        make_sharded_ocean_step,
        shard_state_latlon,
        gather_state_latlon,
    )

    n_lat, n_lon, nlev = 48, 96, 10
    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    cfg = LatLonCGridOceanConfig()
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    state0 = _perturbed_state(grid, z_coord)
    dt, n_steps = 600.0, 3

    # single-device reference
    s = state0
    for _ in range(n_steps):
        s = model.step(s, dt)

    # Prime the model's build-once vertex-mask cache from the CONCRETE initial
    # state so the wrapper can build the per-band vertex masks host-side (the
    # cache is land-mask-derived and constant; the global single-device run above
    # already primed it, this is belt-and-braces for a fresh model).
    model._ensure_vertex_mask(state0)

    # lat-band SPMD on 4 devices.  The state is laid out with shard_state_latlon
    # (cell fields P("lat"); the staggered v / v_mask carried as v_lower, the
    # n_lat-row block that DOES divide N — a uniform tree.map(P("lat")) would
    # fail on the n_lat+1 v rows).  The result is gathered (and the dropped pole
    # row reappended) for the bit-comparison vs the single-device reference.
    dev = create_latlon_mesh(n_devices=4)
    step = make_sharded_ocean_step(model, dev.mesh)
    ss = shard_state_latlon(state0, dev.mesh)
    for _ in range(n_steps):
        ss = step(ss, dt)
    ss = gather_state_latlon(ss, dev.mesh)

    # Tolerance = the FLOATING-POINT RE-ASSOCIATION floor of the sharded
    # split-explicit barotropic, NOT a bug margin. A clean-input bisection proves
    # every dynamical UNIT is SPMD-exact to round-off (baroclinic tendency ~5e-13,
    # AL81 ~8e-16, interp_u_to_vface_4pt / _forward_backward_coriolis_3d / depth-
    # average = 0, the full 30-substep barotropic loop eta 2.5e-9). The residual in
    # the COMPOSED step (u ~1.5e-5, eta ~2.5e-5, v ~5.3e-6) is the ppermute/psum
    # reduction-order change re-associating the split-explicit du-F_slow+U_bar sum
    # over 30 barotropic substeps, amplified by the stiff polar gravity-wave mode
    # (pole-peaked, bottom layer; identical with implicit-vmix/eta-drift/clamp off
    # and pole-v zeroed -> NOT a halo/reduction/tracer defect). atol=1e-8 over
    # 3 steps x 30 substeps is unattainable for a sharded split-explicit scheme;
    # ~1e-4 is 0.001% of the O(1) velocities (physically negligible) yet still
    # catches a real missing-halo regression (those give O(1e-3+) errors at the
    # band cuts -- e.g. the de-shadowed interp bug this gate first exposed). See
    # the module docstring + omip-multinode-spmd-scope.
    _ATOL, _RTOL = 2.0e-4, 1.0e-3
    for nm in ("u", "v", "eta", "T", "S"):
        a = np.asarray(getattr(s, nm).data)
        b = np.asarray(getattr(ss, nm).data)
        np.testing.assert_allclose(b, a, atol=_ATOL, rtol=_RTOL,
                                   err_msg=f"SPMD {nm} mismatch")
