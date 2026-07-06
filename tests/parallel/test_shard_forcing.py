"""Unit tests for ``compiled_segments.shard_forcing`` (cs_spmd step 3).

The grid-shaped ``SegmentForcing`` leaves must be COMMITTED to the state's
device sharding under SPMD (face-first cubed-sphere), by FIELD NAME — never
by shape (a leading-6 ``ghg_vmr`` must stay replicated).  Single-device,
``device_config=None`` and mpi4jax-distributed paths are byte-identical
no-ops.

Runs on 6 virtual CPU devices (XLA host device count, set before jax import).
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=6")

import jax.numpy as jnp
import numpy as np
from legoesm.driver.compiled_segments import (
    GRID_SHAPED_FORCING_FIELDS,
    pack_forcing,
    shard_forcing,
)
from legoesm.parallel.mesh import create_device_mesh


def _forcing(n=4, nlev=3, with_overrides=False):
    shape2d = (6, n, n)
    kw = {}
    if with_overrides:
        kw = dict(
            sfc_albedo_override=jnp.full(shape2d, 0.1),
            sfc_T_override=jnp.full(shape2d, 288.0),
        )
    return pack_forcing(
        sst=jnp.full(shape2d, 290.0),
        sic=jnp.zeros(shape2d),
        day_of_year=15.0,
        seconds_of_day=3600.0,
        solar_weights=jnp.ones(shape2d),
        s_0=1361.0,
        o3_vmr=jnp.full((6, n, n, nlev), 1e-6),
        aerosol_od=jnp.zeros((6, n, n, nlev)),
        # A 6-species GHG vector: the leading-6 shape that a shape-based
        # heuristic would wrongly face-shard.
        ghg_vmr=jnp.asarray([400e-6, 1.8e-6, 0.3e-6, 0.2e-9, 0.5e-9, 1e-9]),
        **kw,
    )


def _is_face_sharded(leaf, mesh) -> bool:
    s = leaf.sharding
    from jax.sharding import NamedSharding
    return (isinstance(s, NamedSharding) and s.mesh is mesh
            and s.spec and s.spec[0] == "face")


def test_none_device_config_is_identity():
    f = _forcing()
    out = shard_forcing(f, None)
    assert out is f


def test_single_device_is_identity():
    f = _forcing()
    dc = create_device_mesh(n_devices=1)
    out = shard_forcing(f, dc)
    assert out is f


def test_mpi_distributed_is_identity():
    f = _forcing()

    class _DC:
        is_distributed = True
        face_sharding = object()

    out = shard_forcing(f, _DC())
    assert out is f


def test_grid_leaves_face_sharded_scalars_untouched():
    dc = create_device_mesh(n_devices=6)
    assert dc.mesh is not None, "needs 6 virtual devices (XLA_FLAGS)"
    f = _forcing(with_overrides=True)
    out = shard_forcing(f, dc)

    # Grid-shaped leaves committed face-first on the build mesh.
    for name in ("sst", "sic", "solar_weights", "o3_vmr", "aerosol_od",
                 "aerosol_lw_od", "sfc_albedo_override", "sfc_T_override"):
        leaf = getattr(out, name)
        assert leaf is not None
        assert _is_face_sharded(leaf, dc.mesh), f"{name} not face-sharded"
        np.testing.assert_array_equal(
            np.asarray(leaf), np.asarray(getattr(f, name)))

    # The 6-species GHG vector must NOT be face-sharded (name gate, not
    # shape gate) — nor the scalars.
    assert not _is_face_sharded(out.ghg_vmr, dc.mesh)
    np.testing.assert_array_equal(np.asarray(out.ghg_vmr),
                                  np.asarray(f.ghg_vmr))
    assert out.day_of_year is f.day_of_year
    assert out.s_0 is f.s_0
    # None optional fields stay None.
    assert out.sfc_shflx_override is None
    assert out.sfc_lhflx_override is None


def test_flat_column_major_o3_face_sharded():
    """PRODUCTION shape: _precompute_external_forcing flattens the 3-D
    radiation forcing to (ncol=6*n*n, nlev).  These must face-shard on
    dim 0 (one face's columns per mesh slot) — the leading-6 rule alone
    misses them (codex 2026-07-01)."""
    dc = create_device_mesh(n_devices=6)
    n, nlev = 4, 3
    ncol = 6 * n * n
    f = _forcing()._replace(
        o3_vmr=jnp.full((ncol, nlev), 1e-6),
        aerosol_od=jnp.zeros((ncol, nlev)),
        aerosol_lw_od=jnp.zeros((ncol, nlev)),
    )
    out = shard_forcing(f, dc)
    for name in ("o3_vmr", "aerosol_od", "aerosol_lw_od"):
        assert _is_face_sharded(getattr(out, name), dc.mesh), name
        np.testing.assert_array_equal(np.asarray(getattr(out, name)),
                                      np.asarray(getattr(f, name)))


def test_placeholder_o3_shape_gate():
    """A (0,) o3 placeholder (gray-radiation run) passes through unsharded
    (shard_pytree's leading-dim gate) without error."""
    dc = create_device_mesh(n_devices=6)
    f = _forcing()._replace(o3_vmr=jnp.zeros((0,)))
    out = shard_forcing(f, dc)
    assert out.o3_vmr.shape == (0,)
    assert not _is_face_sharded(out.o3_vmr, dc.mesh)


def test_field_list_matches_namedtuple():
    """Every name in the grid-shaped list is a real SegmentForcing field
    (a rename must update both)."""
    from legoesm.driver.compiled_segments import SegmentForcing
    for name in GRID_SHAPED_FORCING_FIELDS:
        assert name in SegmentForcing._fields
