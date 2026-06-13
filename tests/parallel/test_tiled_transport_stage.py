"""U3: the PPM i-sweep transport tiled under a REAL shard_map == the global
sweep (task #3 cube tiled np>6 stage).

Proves the approach-C tiling (U2/U2b: global pre-pad → per-tile slice →
_ppm_transport_1d(external_halo=4, rd_prepadded=True)) holds INSIDE a
``(6, kt)`` ``(face, tile_i)`` shard_map (6*kt host devices) — the step
where the np>6 perf actually materialises.  i-sweep only, synthetic
field/courant (the cross-face/staggered FIELD faithfulness is U2b; this
pins the SHARD_MAP WIRING).  Run with
``--xla_force_host_platform_device_count=24``.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.fv3_sw_core import _ppm_transport_1d
from legoesm.parallel.tiled_transport import (
    make_tiled_transport_sweep_stage,
    make_tiled_transport_sweep_stage_2d,
    transport_sweep_tile,
    transport_sweep_tile_2d,
    transport_jsweep_tile_2d,
)


def test_transport_sweep_tile_body_host():
    """Non-shard CI coverage (codex U3 MEDIUM): the per-tile body
    ``transport_sweep_tile`` (the stage's exact slice + _ppm) reassembled
    over tiles == the global PPM sweep, WITHOUT a multi-device mesh — so the
    stage's slice logic is exercised even when the shard_map test below
    skips (below 6*kt host devices)."""
    kt, h3, n, m = 3, 4, 18, 5
    nl = n // kt
    rng = np.random.default_rng(11)
    field = jnp.asarray(rng.standard_normal((6, n, m)))
    courant = jnp.asarray(rng.standard_normal((6, n + 1, m)))
    rd = jnp.asarray(np.abs(rng.standard_normal((6, n, m))) + 0.1)
    global_flux = np.asarray(
        _ppm_transport_1d(field, courant, rd, 1, external_halo=0))
    vp_g = jnp.pad(field, [(0, 0), (h3, h3), (0, 0)], mode="edge")
    rd_g = jnp.pad(rd, [(0, 0), (1, 1), (0, 0)], mode="edge")
    # a = t*nl is a Python int here (host); axis_index int32 in the stage.
    tiles = [np.asarray(transport_sweep_tile(vp_g, courant, rd_g, t * nl,
                                             nl, h3)) for t in range(kt)]
    for t in range(kt - 1):
        np.testing.assert_allclose(
            tiles[t][:, nl, :], tiles[t + 1][:, 0, :], atol=1e-12, rtol=1e-12,
            err_msg=f"host body shared interface mismatch t={t}")
    reassembled = np.concatenate(
        [tiles[t][:, :nl, :] for t in range(kt)]
        + [tiles[-1][:, nl:nl + 1, :]], axis=1)
    np.testing.assert_allclose(
        reassembled, global_flux, atol=1e-12, rtol=1e-12,
        err_msg="transport_sweep_tile body reassembled != global PPM sweep")


@pytest.mark.parametrize("kt", [2, 4])
def test_tiled_transport_sweep_in_shardmap(kt):
    ndev = 6 * kt
    if len(jax.devices()) < ndev:
        pytest.skip(f"needs --xla_force_host_platform_device_count={ndev}")
    from jax.sharding import Mesh

    h3 = 4
    n = 24
    nl = n // kt
    m = 5
    rng = np.random.default_rng(7)
    field = jnp.asarray(rng.standard_normal((6, n, m)))
    courant = jnp.asarray(rng.standard_normal((6, n + 1, m)))
    rd = jnp.asarray(np.abs(rng.standard_normal((6, n, m))) + 0.1)

    global_flux = np.asarray(
        _ppm_transport_1d(field, courant, rd, 1, external_halo=0))  # (6,n+1,m)

    # Global pre-pad EXACTLY as _ppm_transport_1d(external_halo=0) does —
    # these are the FACE-REPLICATED stage inputs.
    vp_g = jnp.pad(field, [(0, 0), (h3, h3), (0, 0)], mode="edge")
    rd_g = jnp.pad(rd, [(0, 0), (1, 1), (0, 0)], mode="edge")

    dev = np.array(jax.devices()[:ndev]).reshape(6, kt)
    mesh = Mesh(dev, axis_names=("face", "tile_i"))
    stage = make_tiled_transport_sweep_stage(mesh, n, kt, h3=h3)
    flux_tiled = np.asarray(stage(vp_g, courant, rd_g))  # (6, kt*(nl+1), m)

    fb = flux_tiled.reshape(6, kt, nl + 1, m)
    # Shared-interface consistency (tile t's [nl] == tile t+1's [0]).
    for t in range(kt - 1):
        np.testing.assert_allclose(
            fb[:, t, nl, :], fb[:, t + 1, 0, :], atol=1e-12, rtol=1e-12,
            err_msg=f"shard_map shared interface mismatch (kt={kt} t={t})")
    # Reassemble (lower tile owns the shared interface) → global sweep.
    reassembled = np.concatenate(
        [fb[:, t, :nl, :] for t in range(kt)]
        + [fb[:, -1, nl:nl + 1, :]], axis=1)              # (6, n+1, m)
    np.testing.assert_allclose(
        reassembled, global_flux, atol=1e-12, rtol=1e-12,
        err_msg=f"tiled-in-shardmap PPM sweep != global (kt={kt})")


# =====================================================================
# U3b: full 2-D (6, kt, kt) tiling — cross axis tiled + the j-sweep.
# =====================================================================

def test_transport_sweep_tile_2d_body_host():
    """Host CI coverage: the 2-D-tiled i-sweep body reassembled over
    (tile_i, tile_j) == the global i-sweep (cross axis sliced WITHOUT a halo
    — the i-sweep is independent per j-row)."""
    kt, h3, n = 3, 4, 18
    nl = n // kt
    rng = np.random.default_rng(13)
    field = jnp.asarray(rng.standard_normal((6, n, n)))
    courant = jnp.asarray(rng.standard_normal((6, n + 1, n)))
    rd = jnp.asarray(np.abs(rng.standard_normal((6, n, n))) + 0.1)
    global_flux = np.asarray(
        _ppm_transport_1d(field, courant, rd, 1, external_halo=0))  # (6,n+1,n)
    vp_g = jnp.pad(field, [(0, 0), (h3, h3), (0, 0)], mode="edge")
    rd_g = jnp.pad(rd, [(0, 0), (1, 1), (0, 0)], mode="edge")
    cols = []
    for tj in range(kt):
        ti_tiles = [np.asarray(transport_sweep_tile_2d(
            vp_g, courant, rd_g, ti * nl, tj * nl, nl, h3)) for ti in range(kt)]
        col = np.concatenate(
            [ti_tiles[ti][:, :nl, :] for ti in range(kt)]
            + [ti_tiles[-1][:, nl:nl + 1, :]], axis=1)        # (6, n+1, nl)
        cols.append(col)
    reassembled = np.concatenate(cols, axis=2)                  # (6, n+1, n)
    np.testing.assert_allclose(
        reassembled, global_flux, atol=1e-12, rtol=1e-12,
        err_msg="2-D i-sweep body reassembled != global PPM sweep")


def test_transport_jsweep_tile_2d_body_host():
    """Host CI coverage: the 2-D-tiled j-sweep (ytp_v, axis=2) body
    reassembled == the global j-sweep."""
    kt, h3, n = 3, 4, 18
    nl = n // kt
    rng = np.random.default_rng(17)
    field = jnp.asarray(rng.standard_normal((6, n, n)))
    courant = jnp.asarray(rng.standard_normal((6, n, n + 1)))
    rd = jnp.asarray(np.abs(rng.standard_normal((6, n, n))) + 0.1)
    global_flux = np.asarray(
        _ppm_transport_1d(field, courant, rd, 2, external_halo=0))  # (6,n,n+1)
    vp_g = jnp.pad(field, [(0, 0), (0, 0), (h3, h3)], mode="edge")
    rd_g = jnp.pad(rd, [(0, 0), (0, 0), (1, 1)], mode="edge")
    rows = []
    for ti in range(kt):
        tj_tiles = [np.asarray(transport_jsweep_tile_2d(
            vp_g, courant, rd_g, ti * nl, tj * nl, nl, h3)) for tj in range(kt)]
        row = np.concatenate(
            [tj_tiles[tj][:, :, :nl] for tj in range(kt)]
            + [tj_tiles[-1][:, :, nl:nl + 1]], axis=2)         # (6, nl, n+1)
        rows.append(row)
    reassembled = np.concatenate(rows, axis=1)                  # (6, n, n+1)
    np.testing.assert_allclose(
        reassembled, global_flux, atol=1e-12, rtol=1e-12,
        err_msg="2-D j-sweep body reassembled != global PPM sweep")


@pytest.mark.parametrize("sweep", ["i", "j"])
def test_tiled_transport_sweep_2d_in_shardmap(sweep):
    kt = 2
    ndev = 6 * kt * kt
    if len(jax.devices()) < ndev:
        pytest.skip(f"needs --xla_force_host_platform_device_count={ndev}")
    from jax.sharding import Mesh

    h3, n = 4, 24
    nl = n // kt
    rng = np.random.default_rng(23)
    field = jnp.asarray(rng.standard_normal((6, n, n)))
    rd = jnp.asarray(np.abs(rng.standard_normal((6, n, n))) + 0.1)
    if sweep == "i":
        courant = jnp.asarray(rng.standard_normal((6, n + 1, n)))
        global_flux = np.asarray(
            _ppm_transport_1d(field, courant, rd, 1, external_halo=0))
        vp_g = jnp.pad(field, [(0, 0), (h3, h3), (0, 0)], mode="edge")
        rd_g = jnp.pad(rd, [(0, 0), (1, 1), (0, 0)], mode="edge")
    else:
        courant = jnp.asarray(rng.standard_normal((6, n, n + 1)))
        global_flux = np.asarray(
            _ppm_transport_1d(field, courant, rd, 2, external_halo=0))
        vp_g = jnp.pad(field, [(0, 0), (0, 0), (h3, h3)], mode="edge")
        rd_g = jnp.pad(rd, [(0, 0), (0, 0), (1, 1)], mode="edge")

    dev = np.array(jax.devices()[:ndev]).reshape(6, kt, kt)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    stage = make_tiled_transport_sweep_stage_2d(mesh, n, kt, h3=h3, sweep=sweep)
    flux = np.asarray(stage(vp_g, courant, rd_g))

    if sweep == "i":
        fb = flux.reshape(6, kt, nl + 1, kt, nl)   # (f, ti, i, tj, j)
        cols = []
        for tj in range(kt):
            col = np.concatenate(
                [fb[:, ti, :nl, tj, :] for ti in range(kt)]
                + [fb[:, kt - 1, nl:nl + 1, tj, :]], axis=1)   # (6, n+1, nl)
            cols.append(col)
        reassembled = np.concatenate(cols, axis=2)              # (6, n+1, n)
    else:
        fb = flux.reshape(6, kt, nl, kt, nl + 1)   # (f, ti, i, tj, j)
        rows = []
        for ti in range(kt):
            row = np.concatenate(
                [fb[:, ti, :, tj, :nl] for tj in range(kt)]
                + [fb[:, ti, :, kt - 1, nl:nl + 1]], axis=2)    # (6, nl, n+1)
            rows.append(row)
        reassembled = np.concatenate(rows, axis=1)              # (6, n, n+1)
    np.testing.assert_allclose(
        reassembled, global_flux, atol=1e-12, rtol=1e-12,
        err_msg=f"2-D tiled-in-shardmap {sweep}-sweep != global")
