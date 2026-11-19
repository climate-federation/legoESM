"""np24 gate: the TILED operator-split step + the tile-aware carry shard
(``driver/tiled_operator_split_step.py``).

Mirrors the lat-band lane's "2b" staging (sharded_operator_split_step): the
physics is a SHAPE-AGNOSTIC column-local MOCK ``step_unified`` (the real
PhysicsPipeline is decomposition-invariant per column, so the mock pins the
LAYOUT/composition contract; the real-pipeline end-to-end is the driver
integration follow-up).  The serial reference composes the SAME serial
primitives the compiled cube ``_single_step`` uses: cc->D-grid ->
base-cut SSP-RK3 dynamics -> corner->cc -> ``fix_segment_dry_mass`` ->
``split_physics_single_rank`` -> ``finalize_split_step``.

Also gates: pack/unpack round-trip of the flattened per-column carry
leaves (the tile-aware PhysicsState shard), the moisture fixer's tiled
psum (conservation), and the envelope refusals.

Run with ``XLA_FLAGS=--xla_force_host_platform_device_count=24``.
"""
from __future__ import annotations

import os
import types

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=24")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.field import Field
from legoesm.core.state import FV3HydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.halo import set_halo_backend
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.timestepping.ssp_rk3 import ssp_rk3_step
from legoesm.core.operators_cdgrid import (
    center_to_dgrid_vector, interp_corner_to_center)
from legoesm.driver.compiled_segments import (
    SegmentCarry, SegmentForcing, build_operator_split_statics, carry_moist,
    fix_segment_dry_mass, segment_dry_mass, split_physics_single_rank,
    finalize_split_step, pack_carry)
from legoesm.driver.sharded_operator_split_step import need_rad_and_time

N, NLEV, KT = 8, 4, 2
NCOL = 6 * N * N
DT = 60.0
N_STEPS = 2
START_DAY = 0.0
# Cadence 1 is the ONLY value this lane accepts: build_tile_step_unified pins
# static_need_rad=True, which discards the per-step predicate, so >1 is refused
# by make_tiled_operator_split_step (see
# test_tiled_operator_split_refuses_rad_update_steps_above_one).  This was 2,
# which now raises -- and never exercised a held-radiation cadence anyway, since
# the predicate was being thrown away.
RAD_UPDATE_STEPS = 1


def _mesh():
    from jax.sharding import Mesh
    if len(jax.devices()) < 6 * KT * KT:
        pytest.skip(f"needs {6 * KT * KT} devices "
                    f"(XLA_FLAGS=--xla_force_host_platform_device_count=24)")
    set_halo_backend("local")
    dev = np.array(jax.devices()[: 6 * KT * KT]).reshape(6, KT, KT)
    return Mesh(dev, axis_names=("face", "tile_i", "tile_j"))


def _mock_step_unified(need_rad, T, p_s, q_v, q_c, q_r, conv_prog, u, v,
                       sst, sic, lat, lon, doy, sod, dt, *args, **kwargs):
    """Shape-agnostic, column-local, DETERMINISTIC physics mock.

    Depends on ``lat`` (validates the per-tile lat slicing) and ``need_rad``
    (validates the radiation cadence plumbing).  Positional layout matches
    the split_physics_single_rank call; held fields ride ``args[4:10]``.
    """
    held = args[4:10]
    o3 = args[2]
    lat_b = lat[..., None]
    # ``sst`` (surface-native) may arrive grid-shaped (tile) or flat
    # (serial production layout) — normalize against T's leading dims,
    # exactly how the adapter-flattened consumers see it.
    sst_b = sst.reshape(T.shape[:-1])[..., None]
    # ``o3_vmr`` is a COLUMN-format forcing leaf: the pipeline's radiation
    # consumers read it at (ncol, nlev) WITHOUT an adapter flatten (codex
    # round-3 High) — assert the tile body re-flattened it.
    assert o3.ndim == 2, f"o3_vmr must be column-format, got {o3.shape}"
    o3_g = o3.reshape(T.shape)
    phys_out = types.SimpleNamespace(
        dT_dt=(1e-4 * (300.0 - T) + 1e-2 * q_v * jnp.cos(lat_b)
               + 1e-6 * sst_b + 1e-3 * o3_g),
        dq_v_dt=-1e-3 * q_v + 1e-9 * jnp.sin(lat_b),
        dq_c_dt=5e-4 * q_v - 2e-4 * q_c,
        dq_r_dt=2e-4 * q_c - 1e-4 * q_r,
        dq_i_dt=None, dq_s_dt=None, dq_g_dt=None,
        dN_c_dt=None, dN_r_dt=None, dN_i_dt=None,
        conv_prog=0.9 * conv_prog,
        precip=1e-3 * jnp.abs(q_r[..., -1]),
        shflx=1e-2 * (T[..., -1] - 288.0),
        lhflx=None, evap_sfc=None,
        w_land=None, snow=None,
        tke=None, qke=None, gwd_spectrum=None,
        cloud_fraction=None, budget_ledger=None,
        # Column-local momentum tendency: the shared tail adds dt*du/dt*dv to
        # the winds, so the tiled-vs-serial parity covers them too.
        du_dt=-1e-5 * u + 1e-6 * jnp.cos(lat_b), dv_dt=-1e-5 * v,
    )
    # Held refresh only when need_rad fires (mirrors step_unified's cond).
    held_new = tuple(
        jnp.where(need_rad, h + (i + 1.0), h) for i, h in enumerate(held))
    return phys_out, held_new


def _inputs(seed=3):
    rng = np.random.default_rng(seed)
    u = jnp.asarray(0.1 * rng.standard_normal((6, N, N, NLEV)))
    v = jnp.asarray(0.1 * rng.standard_normal((6, N, N, NLEV)))
    T = jnp.asarray(250.0 + 2.0 * rng.standard_normal((6, N, N, NLEV)))
    p_s = jnp.asarray(1.0e5 + 50.0 * rng.standard_normal((6, N, N)))
    phis = jnp.asarray(1.0e2 * rng.standard_normal((6, N, N)))
    q_v = jnp.asarray(np.abs(5e-3 + 1e-3 * rng.standard_normal(
        (6, N, N, NLEV))))
    q_c = 0.1 * q_v
    q_r = 0.01 * q_v
    return u, v, T, p_s, phis, q_v, q_c, q_r


def _global_moisture(q_v, p_s, coord, grid):
    from legoesm.core.conservation import compute_global_moisture
    return compute_global_moisture(q_v, p_s, coord.dsigma, grid)


def _cc_state(u, v, T, p_s, phis):
    d3 = ("face", "x", "y", "level")
    d2 = ("face", "x", "y")
    from legoesm.core.state import HydrostaticState
    return HydrostaticState(
        u=Field(data=u, name="u", dims=d3, units="m/s"),
        v=Field(data=v, name="v", dims=d3, units="m/s"),
        T=Field(data=T, name="T", dims=d3, units="K"),
        p_s=Field(data=p_s, name="p_s", dims=d2, units="Pa"),
        phis=Field(data=phis, name="phis", dims=d2, units="m^2/s^2"))


# ---------------------------------------------------------------------------
# Pack/shard round-trip (the tile-aware carry/PhysicsState layout)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# np24 parity: tiled operator-split step vs the serial composition
# ---------------------------------------------------------------------------


def test_tiled_psum_branch_requires_explicit_scope():
    """The tiled-mesh psum branch must NOT fire from an armed mesh alone —
    a reduction outside the tiled shard_map body (writers, diagnostics)
    would emit an out-of-scope psum (codex).  Without the scope the helper
    falls through (returns None -> serial logic)."""
    from legoesm.core.conservation import (
        _spmd_lat_psum_or_none, tiled_reduction_scope)
    from legoesm.grids.halo import (
        get_halo_backend, get_mpi_topology, get_spmd_mesh,
        set_halo_backend, set_spmd_mesh)

    mesh = _mesh()
    prev = (get_halo_backend(), get_mpi_topology(), get_spmd_mesh())
    try:
        set_halo_backend("spmd")
        set_spmd_mesh(mesh)
        # Armed mesh, NO scope -> fall through (None), never psum.
        assert _spmd_lat_psum_or_none([jnp.asarray(1.0)]) is None
        # Inside the scope (and inside a shard_map, where the axes exist)
        # the branch fires — exercised end-to-end by the parity test's
        # moisture fixer; here we only pin the outside-scope inertness.
        with tiled_reduction_scope():
            pass
        assert _spmd_lat_psum_or_none([jnp.asarray(1.0)]) is None
    finally:
        set_spmd_mesh(prev[2])
        set_halo_backend(prev[0], prev[1])


def _mock_area_step(*args, a0, **kwargs):
    """The column mock with an extra drying proportional to the cell area the
    lane hands in (Sundqvist's GFS area blend reads it the same way)."""
    area = kwargs.get("cell_area")
    assert area is not None, "lane did not pass its cell areas"
    out, *rest = _mock_step_unified(*args, **kwargs)
    T = args[1]
    extra = -1e-9 * (jnp.asarray(area).reshape(T.shape[:-1]) / a0)[..., None]
    out.dq_v_dt = out.dq_v_dt + extra
    return (out, *rest)
