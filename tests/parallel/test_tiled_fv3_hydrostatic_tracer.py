"""np24 gate: the tiled advective TRACER tendency stage — the cube-MOIST np>6
unlock (increment 1 of docs/scaling/cube_moist_tiled_step_design.md).

``make_tiled_fv3_tracer_advection_stage_2d`` advects one cc tracer ``q`` by the
SAME advective horizontal transport the thermodynamic stage applies to
temperature (``-(u . grad q)`` via the shared in-stage scalar halo + centred cc
gradients) plus a passed per-column-local ``vert_adv_q``.  Reference = the
PRODUCTION serial ``advective_tracer_tendency`` (the one place the FV3 cube moist
transport numerics live; base case ``hyperdiff_coeff=0``) with a zero
vertical_fn so the reference is the pure horizontal part, plus the same
``vert_adv_q`` added back — exactly what the stage returns.

Gated by BIT-IDENTITY (FMA-robust relative tol; the in-stage ppermute reorders
the global pad's contiguous arithmetic -> O(1e-13) ULP drift).  NOT a wall-clock
measurement (np24 on Ginsburg CPU shard_map anti-scales; the future-HW win is the
capability).  24 host CPU devices (kt=2) / 54 (kt=3):
``XLA_FLAGS=--xla_force_host_platform_device_count=54``.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.halo import set_halo_backend
from legoesm.core.operators_cdgrid import dgrid_to_center_vector
from legoesm.atmosphere.dynamics.tracer_transport import advective_tracer_tendency
from legoesm.parallel.tiled_production_cdgrid import (
    make_tiled_fv3_tracer_advection_stage_2d,
)

N = 24                                              # nl=12 (kt=2), nl=8 (kt=3)
NLEV = 6


def _inputs(n, nlev, seed):
    """Random but finite winds + a positive tracer field (a mixing ratio); the
    gate tests numerical equivalence, not physics.  vert_adv_q is a random finite
    cc field (an upstream cc-local stage output in the model)."""
    rng = np.random.default_rng(seed)
    u_d = jnp.asarray(rng.standard_normal((6, n + 1, n + 1, nlev)))
    v_d = jnp.asarray(rng.standard_normal((6, n + 1, n + 1, nlev)))
    q = jnp.asarray(1.0e-2 + 1.0e-3 * rng.standard_normal((6, n, n, nlev)))
    vert_adv_q = jnp.asarray(rng.standard_normal((6, n, n, nlev)))   # 1/s
    return u_d, v_d, q, vert_adv_q


def _global_tracer(u_d, v_d, q, vert_adv_q, cdgrid):
    """Global base dq/dt = serial advective_tracer_tendency horizontal part
    (hyperdiff=0, zero vertical_fn) + the passed vert_adv_q."""
    grid = cdgrid.base
    u_cell, v_cell = dgrid_to_center_vector(u_d, v_d)
    horiz = advective_tracer_tendency(
        q[..., None], u_cell, v_cell, grid,
        lambda q1: jnp.zeros_like(q1), hyperdiff_coeff=0.0)[..., 0]
    return np.asarray(horiz + vert_adv_q)


@pytest.fixture(scope="module")
def cdg():
    set_halo_backend("local")
    g = create_cubed_sphere_cdgrid(create_cubed_sphere(N))
    assert g.base.duogrid is None, "base cut is non-duogrid"
    assert g.base.halo_interp_offsets is not None, "needs interp offsets"
    return g


def _rel_cc(dq_t, dq_g):
    diff = float(np.max(np.abs(np.asarray(dq_t) - dq_g)))
    return diff / (float(np.max(np.abs(dq_g))) + 1e-300)


@pytest.mark.parametrize("KT", [2, 3])
def test_tiled_tracer_matches_global(cdg, KT):
    ndev = 6 * KT * KT
    if len(jax.devices()) < ndev:
        pytest.skip(
            f"kt={KT} needs {ndev} host devices "
            f"(--xla_force_host_platform_device_count={ndev})")
    u_d, v_d, q, vert_adv_q = _inputs(N, NLEV, 50 + KT)
    dq_g = _global_tracer(u_d, v_d, q, vert_adv_q, cdg)

    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    dev = np.array(jax.devices()[:ndev]).reshape(6, KT, KT)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    stage = make_tiled_fv3_tracer_advection_stage_2d(mesh, cdg, N, KT, NLEV)

    fw = NamedSharding(mesh, P("face", None, None, None))
    dq_t = stage(
        jax.device_put(u_d, fw), jax.device_put(v_d, fw),
        jax.device_put(q, fw), jax.device_put(vert_adv_q, fw))

    rel = _rel_cc(dq_t, dq_g)
    assert rel < 1e-10, f"dq_dt rel {rel:.3e} (kt={KT})"


def test_tracer_horiz_matches_thermo_T_path(cdg):
    """The tracer horizontal advect MUST be the SAME operator the thermo stage
    applies to T: with vert_adv_q=0 the tiled tracer dq/dt equals the serial
    -(u . grad q) — i.e. the shared _tiled_scalar_horiz_advect is faithful to the
    production gradient_{x,y}_3d path (host-composed, no 24 devices needed)."""
    from legoesm.core.operators_3d import gradient_x_3d, gradient_y_3d
    u_d, v_d, q, _ = _inputs(N, NLEV, 99)
    grid = cdg.base
    u_cell, v_cell = dgrid_to_center_vector(u_d, v_d)
    horiz_ref = -(u_cell * gradient_x_3d(q, grid) + v_cell * gradient_y_3d(q, grid))
    horiz_serial = advective_tracer_tendency(
        q[..., None], u_cell, v_cell, grid,
        lambda q1: jnp.zeros_like(q1), hyperdiff_coeff=0.0)[..., 0]
    rel = float(np.max(np.abs(np.asarray(horiz_ref) - np.asarray(horiz_serial))))
    rel /= float(np.max(np.abs(np.asarray(horiz_serial)))) + 1e-300
    assert rel < 1e-12, f"serial tracer horiz != -(u.grad q): {rel:.3e}"


def test_tracer_stage_rejects_bad(cdg):
    """Fail-loud guards mirror the thermo/momentum factories: n indivisible by kt
    and a duogrid base must raise at factory time."""
    if len(jax.devices()) < 6:
        pytest.skip("guard test builds a (6,1,1) mesh -> needs 6 host devices")
    from jax.sharding import Mesh
    dev = np.array(jax.devices()[:6]).reshape(6, 1, 1)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    with pytest.raises(ValueError, match="not divisible"):
        make_tiled_fv3_tracer_advection_stage_2d(mesh, cdg, N, 5, NLEV)
