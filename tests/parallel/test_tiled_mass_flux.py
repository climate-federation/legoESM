"""3D PE vertical mass flux (hybrid coord) sub-face-tiled.

compute_mass_flux_hybrid takes the ALREADY-3D horizontal divergence div_3d +
p_s and integrates along the LEVEL axis only (dp_from_hybrid per-column, cumsum
over levels, frac_B from coord.B_half) — horizontally pointwise, NO stencil — so
it tiles EXACTLY (slice the cc horizontal tile; the vertical work runs local per
tile since nlev is replicated).  Bit-identity vs the global op for BOTH outputs
(mass_flux + D_total_p), host-body + np24.

24 host CPU devices: ``XLA_FLAGS=--xla_force_host_platform_device_count=24``.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.vertical import make_hybrid_levels, compute_mass_flux_hybrid
from legoesm.parallel.tiled_production_cdgrid import (
    compute_mass_flux_hybrid_tile_2d,
    make_tiled_compute_mass_flux_hybrid_stage_2d,
)


def _inputs(n, nlev, seed):
    rng = np.random.default_rng(seed)
    div_3d = jnp.asarray(1.0e-6 * rng.standard_normal((6, n, n, nlev)))
    p_s = jnp.asarray(1.0e5 + 1.0e3 * rng.standard_normal((6, n, n)))
    return div_3d, p_s


def _reassemble_cc(get_tile, kt):
    return np.concatenate(
        [np.concatenate([np.asarray(get_tile(ti, tj)) for tj in range(kt)], axis=2)
         for ti in range(kt)], axis=1)


def test_mass_flux_hybrid_host_body_tiling():
    kt, nl, nlev = 3, 6, 8
    n = kt * nl
    coord = make_hybrid_levels(nlev)
    div_3d, p_s = _inputs(n, nlev, 71)
    mf_g, dt_g = (np.asarray(x) for x in compute_mass_flux_hybrid(div_3d, p_s, coord))

    def get_mf(ti, tj):
        return compute_mass_flux_hybrid_tile_2d(
            div_3d, p_s, coord, ti * nl, tj * nl, nl)[0]

    def get_dt(ti, tj):
        return compute_mass_flux_hybrid_tile_2d(
            div_3d, p_s, coord, ti * nl, tj * nl, nl)[1]

    mf_t = _reassemble_cc(get_mf, kt)
    dt_t = _reassemble_cc(get_dt, kt)
    assert mf_t.shape == (6, n, n, nlev + 1)
    assert dt_t.shape == (6, n, n, 1)
    # EXACT equality: the op is pure mul/cumsum/pad per column (no transcendentals,
    # no cross-column reduction), so horizontal tiling cannot change any FP op —
    # "bit-identity" must hold exactly, not merely to a tolerance.
    np.testing.assert_array_equal(
        mf_t, mf_g, err_msg="tiled mass_flux host-body != global (not bit-identical)")
    np.testing.assert_array_equal(
        dt_t, dt_g, err_msg="tiled D_total_p host-body != global (not bit-identical)")


def test_mass_flux_hybrid_shard_map_np24():
    kt, nl, nlev = 2, 6, 8
    ndev = 6 * kt * kt
    if len(jax.devices()) < ndev:
        pytest.skip(f"needs --xla_force_host_platform_device_count={ndev}")
    from jax.sharding import Mesh

    n = kt * nl
    coord = make_hybrid_levels(nlev)
    div_3d, p_s = _inputs(n, nlev, 72)
    mf_g, dt_g = (np.asarray(x) for x in compute_mass_flux_hybrid(div_3d, p_s, coord))
    dev = np.array(jax.devices()[:ndev]).reshape(6, kt, kt)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    stage = make_tiled_compute_mass_flux_hybrid_stage_2d(mesh, coord, n, kt)
    mf_s, dt_s = stage(div_3d, p_s)
    mf_s = np.asarray(mf_s).reshape(6, kt, nl, kt, nl, nlev + 1)
    dt_s = np.asarray(dt_s).reshape(6, kt, nl, kt, nl, 1)
    mf_t = _reassemble_cc(lambda ti, tj: mf_s[:, ti, :, tj, :, :], kt)
    dt_t = _reassemble_cc(lambda ti, tj: dt_s[:, ti, :, tj, :, :], kt)
    assert mf_t.shape == (6, n, n, nlev + 1)
    assert dt_t.shape == (6, n, n, 1)
    # EXACT equality (see host-body test): per-column op, tiling changes no FP op.
    np.testing.assert_array_equal(
        mf_t, mf_g, err_msg="tiled mass_flux np24 != global (not bit-identical)")
    np.testing.assert_array_equal(
        dt_t, dt_g, err_msg="tiled D_total_p np24 != global (not bit-identical)")
