"""COMPOSED 3D-PE vertical transport stage sub-face-tiled.

Fuses the three cc-local vertical ops (compute_mass_flux_hybrid ->
vertical_advection_hybrid (of a field) + compute_omega_hybrid) into ONE
shard_map: the cc tile is sliced ONCE and the ops chain locally with no
inter-op gather/reshard.  Bit-identical (EXACT) to running each global op then
slicing — all per-column, no halo.  Validates all FOUR outputs (mass_flux,
tend, omega, D_total_p), host-composition + np24.

24 host CPU devices: ``XLA_FLAGS=--xla_force_host_platform_device_count=24``.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.vertical import (
    make_hybrid_levels, compute_mass_flux_hybrid,
    vertical_advection_hybrid, compute_omega_hybrid,
)
from legoesm.parallel.tiled_production_cdgrid import (
    make_tiled_vertical_pe_stage_2d,
)


def _inputs(n, nlev, seed):
    rng = np.random.default_rng(seed)
    div_3d = jnp.asarray(1.0e-6 * rng.standard_normal((6, n, n, nlev)))
    field = jnp.asarray(250.0 + 30.0 * rng.standard_normal((6, n, n, nlev)))
    p_s = jnp.asarray(1.0e5 + 1.0e3 * rng.standard_normal((6, n, n)))
    dp_s_dt = jnp.asarray(1.0 * rng.standard_normal((6, n, n)))
    return div_3d, field, p_s, dp_s_dt


def _global_bundle(div_3d, field, p_s, dp_s_dt, coord):
    mf, dt = compute_mass_flux_hybrid(div_3d, p_s, coord)
    tend = vertical_advection_hybrid(field, mf, p_s, coord)
    omega = compute_omega_hybrid(mf, p_s, dp_s_dt, coord)
    return [np.asarray(x) for x in (mf, tend, omega, dt)]


def _reassemble_cc(get_tile, kt):
    return np.concatenate(
        [np.concatenate([np.asarray(get_tile(ti, tj)) for tj in range(kt)], axis=2)
         for ti in range(kt)], axis=1)


def test_vertical_pe_compose_host_body():
    """Composition exactness WITHOUT 24 devices: slice-once-then-chain (host)
    must equal the global composition for every cc tile.  Runs in any lane so
    the slice-once equivalence is covered even when np24 is skipped."""
    kt, nl, nlev = 3, 6, 8
    n = kt * nl
    coord = make_hybrid_levels(nlev)
    div_3d, field, p_s, dp_s_dt = _inputs(n, nlev, 53)
    mf_g, tend_g, omega_g, dt_g = _global_bundle(div_3d, field, p_s, dp_s_dt, coord)

    def _chain(ti, tj, which):
        a_i, a_j = ti * nl, tj * nl

        def _s2(arr):
            return arr[:, a_i:a_i + nl, a_j:a_j + nl]

        mf, dt = compute_mass_flux_hybrid(_s2(div_3d), _s2(p_s), coord)
        if which == "mf":
            return mf
        if which == "dt":
            return dt
        if which == "tend":
            return vertical_advection_hybrid(_s2(field), mf, _s2(p_s), coord)
        return compute_omega_hybrid(mf, _s2(p_s), _s2(dp_s_dt), coord)

    for which, g in (("mf", mf_g), ("tend", tend_g), ("omega", omega_g), ("dt", dt_g)):
        t = _reassemble_cc(lambda ti, tj: _chain(ti, tj, which), kt)
        np.testing.assert_array_equal(
            t, g, err_msg=f"composed {which} host-body != global (slice-once chain)")


def test_vertical_pe_stage_shape_guard():
    """The stage shape guard fires (non-vacuous) on a non-cc horizontal shape."""
    kt, nlev = 2, 8
    n = kt * 6
    coord = make_hybrid_levels(nlev)
    # Build a 1-device mesh shim only enough to construct the stage closure; the
    # guard runs in pure Python before _body, so no real devices are needed.
    from jax.sharding import Mesh
    if len(jax.devices()) < 6 * kt * kt:
        pytest.skip(f"needs --xla_force_host_platform_device_count={6 * kt * kt}")
    dev = np.array(jax.devices()[:6 * kt * kt]).reshape(6, kt, kt)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    stage = make_tiled_vertical_pe_stage_2d(mesh, coord, n, kt)
    bad = jnp.zeros((6, n + 1, n, nlev))   # wrong horizontal extent
    p_s = jnp.zeros((6, n + 1, n))
    with pytest.raises(ValueError, match="must be cc"):
        stage(bad, bad, p_s, p_s)


def test_vertical_pe_stage_np24():
    kt, nl, nlev = 2, 6, 8
    ndev = 6 * kt * kt
    if len(jax.devices()) < ndev:
        pytest.skip(f"needs --xla_force_host_platform_device_count={ndev}")
    from jax.sharding import Mesh

    n = kt * nl
    coord = make_hybrid_levels(nlev)
    div_3d, field, p_s, dp_s_dt = _inputs(n, nlev, 51)
    mf_g, tend_g, omega_g, dt_g = _global_bundle(div_3d, field, p_s, dp_s_dt, coord)

    dev = np.array(jax.devices()[:ndev]).reshape(6, kt, kt)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    stage = make_tiled_vertical_pe_stage_2d(mesh, coord, n, kt)
    mf_s, tend_s, omega_s, dt_s = stage(div_3d, field, p_s, dp_s_dt)

    def _re(arr, last):
        a = np.asarray(arr).reshape(6, kt, nl, kt, nl, last)
        return _reassemble_cc(lambda ti, tj: a[:, ti, :, tj, :, :], kt)

    mf_t = _re(mf_s, nlev + 1)
    tend_t = _re(tend_s, nlev)
    omega_t = _re(omega_s, nlev)
    dt_t = _re(dt_s, 1)
    assert mf_t.shape == (6, n, n, nlev + 1)
    assert dt_t.shape == (6, n, n, 1)
    np.testing.assert_array_equal(mf_t, mf_g, err_msg="composed mass_flux != global")
    np.testing.assert_array_equal(tend_t, tend_g, err_msg="composed vertical_advection != global")
    np.testing.assert_array_equal(omega_t, omega_g, err_msg="composed omega != global")
    np.testing.assert_array_equal(dt_t, dt_g, err_msg="composed D_total_p != global")
