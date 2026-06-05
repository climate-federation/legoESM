"""Serial-vs-SPMD parity + differentiability for the Duo-Grid cubed-sphere halo (B1).

The Duo-Grid Lagrange (kinked->extended) edge remap + corner fill is applied as a
*local, per-face* post-step after the cross-panel collective, so the ``shard_map``
(SPMD) halo must reproduce the serial local ``pad_halo_4d(duogrid=dg)`` halo
bit-for-bit AND carry ``jax.grad`` gradients identically (the shard_map all_gather
collective is AD-transparent; the per-face remap differentiates the same way
serial does).

Existing coverage (``test_cubesphere_exchange.py``) already checks the MULTI-field
packed duogrid path and the full-dycore SPMD step (``test_cubed_sphere_spmd_step``).
This file closes two remaining gaps:
  * the SINGLE-field ``packed_pad_halo_4d(f, duogrid=dg)`` path (which previously
    early-returned without the duogrid remap), and
  * a ``jax.grad`` AD-safety check through the SPMD duo-grid halo (no prior test
    differentiates through the shard_map duogrid exchange).

Run with 6 simulated CPU devices::

    XLA_FLAGS=--xla_force_host_platform_device_count=6 JAX_ENABLE_X64=1 \
        .venv/bin/python -m pytest tests/parallel/test_duogrid_spmd_parity.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _make_6way_mesh():
    devices = jax.devices("cpu")
    if len(devices) < 6:
        pytest.skip("Need >=6 CPU devices "
                    "(set XLA_FLAGS=--xla_force_host_platform_device_count=6)")
    return jax.sharding.Mesh(
        np.array(devices[:6]).reshape(6), axis_names=("face",))


@pytest.fixture
def mesh6():
    return _make_6way_mesh()


@pytest.fixture
def duogrid():
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    grid = create_cubed_sphere(8, use_duogrid=True)
    assert grid.duogrid is not None, "duogrid not active on grid"
    return grid.duogrid


def test_single_field_packed_duogrid_matches_unpacked(mesh6, duogrid):
    """``packed_pad_halo_4d(ONE field, duogrid=dg)`` == serial
    ``pad_halo_4d(duogrid=dg)``.

    Guards the single-field duogrid-skip fix: the multi-field packed path always
    applied the kinked->extended remap, but the 1-field early-return returned a
    nearest-copy halo while the unpacked reference applied the remap.
    """
    from legoesm.grids.halo import pad_halo_4d
    from legoesm.parallel.cubesphere_exchange import packed_pad_halo_4d

    f = jax.random.normal(jax.random.PRNGKey(1), (6, 8, 8, 3))
    ref = np.array(pad_halo_4d(f, duogrid=duogrid))
    (out,) = packed_pad_halo_4d(f, mesh=mesh6, duogrid=duogrid)
    np.testing.assert_allclose(np.array(out), ref, rtol=1e-6, atol=1e-10)


def test_single_field_packed_duogrid_h2_matches_unpacked(mesh6, duogrid):
    """Same single-field parity at halo=2 (the 2-strip allgather kernel)."""
    from legoesm.grids.halo import pad_halo_4d
    from legoesm.parallel.cubesphere_exchange import packed_pad_halo_4d

    f = jax.random.normal(jax.random.PRNGKey(7), (6, 8, 8, 2))
    ref = np.array(pad_halo_4d(f, halo=2, duogrid=duogrid))
    (out,) = packed_pad_halo_4d(f, mesh=mesh6, duogrid=duogrid, halo=2)
    np.testing.assert_allclose(np.array(out), ref, rtol=1e-6, atol=1e-10)


def test_packed_duogrid_grad_matches_serial_grad(mesh6, duogrid):
    """``jax.grad`` through the SPMD duo-grid halo is finite and equals the serial
    duo-grid-halo gradient (AD-safety of the shard_map collective + per-face remap).
    """
    from legoesm.grids.halo import pad_halo_4d
    from legoesm.parallel.cubesphere_exchange import packed_pad_halo_4d

    f = jax.random.normal(jax.random.PRNGKey(2), (6, 8, 8, 2))

    def loss_serial(x):
        return jnp.sum(pad_halo_4d(x, duogrid=duogrid) ** 2)

    def loss_spmd(x):
        (out,) = packed_pad_halo_4d(x, mesh=mesh6, duogrid=duogrid)
        return jnp.sum(out ** 2)

    g_serial = np.array(jax.grad(loss_serial)(f))
    g_spmd = np.array(jax.grad(loss_spmd)(f))
    assert np.all(np.isfinite(g_spmd))
    np.testing.assert_allclose(g_spmd, g_serial, rtol=1e-6, atol=1e-9)


def test_multi_field_packed_duogrid_grad_matches_serial(mesh6, duogrid):
    """Same AD-safety for the multi-field packed path (the production transport
    case): grad through two fields exchanged in one collective matches serial."""
    from legoesm.grids.halo import pad_halo_4d
    from legoesm.parallel.cubesphere_exchange import packed_pad_halo_4d

    f1 = jax.random.normal(jax.random.PRNGKey(3), (6, 8, 8, 2))
    f2 = jax.random.normal(jax.random.PRNGKey(4), (6, 8, 8, 2))

    def loss_serial(a, b):
        pa = pad_halo_4d(a, duogrid=duogrid)
        pb = pad_halo_4d(b, duogrid=duogrid)
        return jnp.sum(pa ** 2) + jnp.sum(pb ** 2)

    def loss_spmd(a, b):
        ra, rb = packed_pad_halo_4d(a, b, mesh=mesh6, duogrid=duogrid)
        return jnp.sum(ra ** 2) + jnp.sum(rb ** 2)

    ga_s, gb_s = jax.grad(loss_serial, argnums=(0, 1))(f1, f2)
    ga_p, gb_p = jax.grad(loss_spmd, argnums=(0, 1))(f1, f2)
    for gp, gs in ((ga_p, ga_s), (gb_p, gb_s)):
        gp, gs = np.array(gp), np.array(gs)
        assert np.all(np.isfinite(gp))
        np.testing.assert_allclose(gp, gs, rtol=1e-6, atol=1e-9)
