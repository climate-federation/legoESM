"""3D PE dycore velocity transforms sub-face-tiled: dgrid_to_cgrid (D->C) +
dgrid_to_center_vector (D->cc).

Both are PURELY LOCAL (within-face: adjacent-corner avg + the cosa_u face-normal
projection for D->C; 4-pt box for D->cc) — NO halo, NO cross-face rotation — so
the staggered ``(nl+1)`` tile slice carries every needed corner.  The 3D
fv3_hydrostatic_tendencies uses THESE (not the SW fv3_d2cc/fv3_cc2c vector-halo
path).  Bit-identity vs the global ops, 2D + 4D, host-body + np24 shard_map.

54 host CPU devices: ``XLA_FLAGS=--xla_force_host_platform_device_count=54``.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.operators_cdgrid import (
    dgrid_to_cgrid, dgrid_to_center_vector)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.parallel.tiled_production_cdgrid import (
    dgrid_to_cgrid_tile_2d, dgrid_to_center_vector_tile_2d,
    make_tiled_dgrid_to_cgrid_stage_2d,
    make_tiled_dgrid_to_center_vector_stage_2d,
)


def _cube(kt, nl):
    n = kt * nl
    return create_cubed_sphere_cdgrid(create_cubed_sphere(n=n, use_duogrid=True)), n


def _corner_winds(n, nlev, seed):
    rng = np.random.default_rng(seed)
    shp = (6, n + 1, n + 1) if nlev is None else (6, n + 1, n + 1, nlev)
    return jnp.asarray(rng.standard_normal(shp)), jnp.asarray(rng.standard_normal(shp))


@pytest.mark.parametrize("nlev", [None, 4])
def test_dgrid_to_cgrid_host_body(nlev):
    kt, nl = 3, 6
    cd, n = _cube(kt, nl)
    u_d, v_d = _corner_winds(n, nlev, 71)
    ug, vg = (np.asarray(x) for x in dgrid_to_cgrid(u_d, v_d, cd))  # (n+1,n)/(n,n+1)
    worst = 0.0
    for ti in range(kt):
        for tj in range(kt):
            uc, vc = dgrid_to_cgrid_tile_2d(
                u_d, v_d, cd.cosa_u, ti * nl, tj * nl, nl)
            uc, vc = np.asarray(uc), np.asarray(vc)
            g_uc = ug[:, ti * nl: ti * nl + nl + 1, tj * nl: tj * nl + nl]
            g_vc = vg[:, ti * nl: ti * nl + nl, tj * nl: tj * nl + nl + 1]
            worst = max(worst, float(np.max(np.abs(uc - g_uc))),
                        float(np.max(np.abs(vc - g_vc))))
    assert worst < 1e-12, f"dgrid_to_cgrid tiled vs global (nlev={nlev}): {worst:.3e}"


@pytest.mark.parametrize("nlev", [None, 4])
def test_dgrid_to_center_vector_host_body(nlev):
    kt, nl = 3, 6
    cd, n = _cube(kt, nl)
    u_d, v_d = _corner_winds(n, nlev, 72)
    ug, vg = (np.asarray(x) for x in dgrid_to_center_vector(u_d, v_d))  # (n,n) cc
    worst = 0.0
    for ti in range(kt):
        for tj in range(kt):
            uc, vc = dgrid_to_center_vector_tile_2d(u_d, v_d, ti * nl, tj * nl, nl)
            uc, vc = np.asarray(uc), np.asarray(vc)
            g_uc = ug[:, ti * nl:(ti + 1) * nl, tj * nl:(tj + 1) * nl]
            g_vc = vg[:, ti * nl:(ti + 1) * nl, tj * nl:(tj + 1) * nl]
            worst = max(worst, float(np.max(np.abs(uc - g_uc))),
                        float(np.max(np.abs(vc - g_vc))))
    assert worst < 1e-12, f"dgrid_to_center_vector tiled vs global (nlev={nlev}): {worst:.3e}"


@pytest.mark.parametrize("nlev", [None, 4])
def test_dgrid_to_cgrid_shard_map_np24(nlev):
    kt, nl = 2, 6
    ndev = 6 * kt * kt
    if len(jax.devices()) < ndev:
        pytest.skip(f"needs --xla_force_host_platform_device_count={ndev}")
    from jax.sharding import Mesh

    cd, n = _cube(kt, nl)
    u_d, v_d = _corner_winds(n, nlev, 73)
    ug, vg = (np.asarray(x) for x in dgrid_to_cgrid(u_d, v_d, cd))
    dev = np.array(jax.devices()[:ndev]).reshape(6, kt, kt)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    stage = make_tiled_dgrid_to_cgrid_stage_2d(mesh, cd, n, kt, nlev=nlev)
    uc, vc = (np.asarray(x) for x in stage(u_d, v_d))
    bu, bv = nl + 1, nl + 1
    worst = 0.0
    for f in range(6):
        for ti in range(kt):
            for tj in range(kt):
                t_uc = uc[f, ti * bu:(ti + 1) * bu, tj * nl:(tj + 1) * nl]
                g_uc = ug[f, ti * nl: ti * nl + nl + 1, tj * nl: tj * nl + nl]
                t_vc = vc[f, ti * nl:(ti + 1) * nl, tj * bv:(tj + 1) * bv]
                g_vc = vg[f, ti * nl: ti * nl + nl, tj * nl: tj * nl + nl + 1]
                worst = max(worst, float(np.max(np.abs(t_uc - g_uc))),
                            float(np.max(np.abs(t_vc - g_vc))))
    assert worst < 1e-12, f"dgrid_to_cgrid np24 vs global (nlev={nlev}): {worst:.3e}"


@pytest.mark.parametrize("nlev", [None, 4])
def test_dgrid_to_center_vector_shard_map_np24(nlev):
    kt, nl = 2, 6
    ndev = 6 * kt * kt
    if len(jax.devices()) < ndev:
        pytest.skip(f"needs --xla_force_host_platform_device_count={ndev}")
    from jax.sharding import Mesh

    cd, n = _cube(kt, nl)
    u_d, v_d = _corner_winds(n, nlev, 74)
    ug, vg = (np.asarray(x) for x in dgrid_to_center_vector(u_d, v_d))
    dev = np.array(jax.devices()[:ndev]).reshape(6, kt, kt)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    stage = make_tiled_dgrid_to_center_vector_stage_2d(mesh, n, kt, nlev=nlev)
    uc, vc = (np.asarray(x) for x in stage(u_d, v_d))
    worst = 0.0
    for f in range(6):
        for ti in range(kt):
            for tj in range(kt):
                t_uc = uc[f, ti * nl:(ti + 1) * nl, tj * nl:(tj + 1) * nl]
                g_uc = ug[f, ti * nl:(ti + 1) * nl, tj * nl:(tj + 1) * nl]
                t_vc = vc[f, ti * nl:(ti + 1) * nl, tj * nl:(tj + 1) * nl]
                g_vc = vg[f, ti * nl:(ti + 1) * nl, tj * nl:(tj + 1) * nl]
                worst = max(worst, float(np.max(np.abs(t_uc - g_uc))),
                            float(np.max(np.abs(t_vc - g_vc))))
    assert worst < 1e-12, f"dgrid_to_center_vector np24 vs global (nlev={nlev}): {worst:.3e}"
