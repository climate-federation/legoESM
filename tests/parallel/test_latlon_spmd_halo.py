"""Lat-lon band SPMD halo (multi-GPU, no mpi4jax) — bit-identity vs serial.

The foundation for the ocean lat-lon multi-GPU SPMD step: lat-band ppermute +
local lon-wrap + pole fold at the end bands.  Each band's padded block must
equal the serial ``pad_halo_latlon_local`` (scalar) / ``..._vector_local``
(vector, v sign-flip at pole) window for that band.  Runs on host CPU devices
(``XLA_FLAGS=--xla_force_host_platform_device_count=4``); the production target
is 2 GPUs but the shard_map/ppermute logic is device-agnostic.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.halo_latlon import (
    pad_halo_latlon_local, pad_halo_latlon_vector_local,
    pad_halo_latlon_3d_local, pad_halo_latlon_vector_3d_local,
)
from legoesm.parallel.latlon_spmd import pad_halo_latlon_band_spmd

N_DEV = 4
N_LAT = 16
NL = N_LAT // N_DEV
N_LON = 8


def _mesh():
    if len(jax.devices()) < N_DEV:
        pytest.skip(f"needs --xla_force_host_platform_device_count={N_DEV}")
    from jax.sharding import Mesh
    dev = np.array(jax.devices()[:N_DEV])
    return Mesh(dev, axis_names=("lat",))


@pytest.mark.parametrize("negate", [False, True])
@pytest.mark.parametrize("nlev", [None, 5])
def test_latlon_band_pad_matches_serial(negate, nlev):
    mesh = _mesh()
    from jax.sharding import NamedSharding, PartitionSpec as P

    rng = np.random.default_rng(91 + int(negate) + (0 if nlev is None else 100))
    shp = (N_LAT, N_LON) if nlev is None else (N_LAT, N_LON, nlev)
    field = jnp.asarray(rng.standard_normal(shp))

    # Serial reference: the 3D-native helpers when nlev is present (the generic
    # 2D helpers now also pad lon on axis 1; both are pinned equal in
    # tests/unit/test_operators_latlon.py).
    if nlev is None:
        serial_fn = pad_halo_latlon_vector_local if negate else pad_halo_latlon_local
    else:
        serial_fn = (pad_halo_latlon_vector_3d_local if negate
                     else pad_halo_latlon_3d_local)
    ref = np.asarray(serial_fn(field, halo=1))            # (N_LAT+2, N_LON+2[,lev])

    isp = P("lat", *((None,) * (field.ndim - 1)))
    field_sh = jax.device_put(field, NamedSharding(mesh, isp))
    out = np.asarray(pad_halo_latlon_band_spmd(mesh, halo=1, negate=negate)(field_sh))

    blk = NL + 2                                          # per-band padded lat
    assert out.shape[0] == N_DEV * blk
    worst = 0.0
    for b in range(N_DEV):
        t = out[b * blk:(b + 1) * blk]
        g = ref[b * NL: b * NL + NL + 2]
        worst = max(worst, float(np.max(np.abs(t - g))))
    assert worst < 1e-12, (
        f"latlon band SPMD halo vs serial (negate={negate}, nlev={nlev}): "
        f"{worst:.3e}")


def test_pad_halo_latlon_backend_dispatch_matches_serial():
    """The BACKEND routing: with the spmd backend armed, pad_halo_latlon called
    INSIDE an outer shard_map routes through the band body and matches serial —
    so the existing ocean/atm lat-lon step is backend-oblivious."""
    mesh = _mesh()
    from functools import partial
    from jax.sharding import NamedSharding, PartitionSpec as P
    try:
        from jax import shard_map
    except ImportError:  # pragma: no cover
        from jax.experimental.shard_map import shard_map
    from legoesm.grids.halo_latlon import pad_halo_latlon
    from legoesm.parallel.latlon_spmd import (
        activate_latlon_spmd_halo, deactivate_latlon_spmd_halo,
    )

    rng = np.random.default_rng(77)
    field = jnp.asarray(rng.standard_normal((N_LAT, N_LON)))
    ref = np.asarray(pad_halo_latlon(field, halo=1))      # local backend

    isp = P("lat", None)
    field_sh = jax.device_put(field, NamedSharding(mesh, isp))
    activate_latlon_spmd_halo(mesh)
    try:
        @partial(shard_map, mesh=mesh, in_specs=isp, out_specs=isp,
                 check_vma=False)
        def _ex(tile):
            return pad_halo_latlon(tile, halo=1)   # routes to the band body
        out = np.asarray(_ex(field_sh))
    finally:
        deactivate_latlon_spmd_halo()

    blk = NL + 2
    worst = max(
        float(np.max(np.abs(out[b * blk:(b + 1) * blk]
                            - ref[b * NL: b * NL + NL + 2])))
        for b in range(N_DEV))
    assert worst < 1e-12, f"pad_halo_latlon spmd-backend dispatch vs serial: {worst:.3e}"


# ---------------------------------------------------------------------------
# WALL pad (pad_with_pole_bc_lat) + pole-zero (zero_polar_lat_ends) SPMD bodies
# ---------------------------------------------------------------------------
# These two helpers carry the ocean step's wall-BC (lat-only constant pad at the
# physical poles, neighbour-band ppermute at interior cuts) and the post-stencil
# polar wall (zero only the GLOBAL pole rows).  They had NO spmd branch — the
# ocean lat-band SPMD step needs them so the band-cut v-rows are the neighbour's
# true edge row, not a constant wall.  Bit-identity vs the serial local helpers,
# per band, is the same proven methodology as the pad tests above.


@pytest.mark.parametrize("south_value,north_value", [(0.0, 0.0), (3.0, -2.0)])
@pytest.mark.parametrize("nlev", [None, 5])
def test_wall_pad_band_matches_serial(south_value, north_value, nlev):
    """The lat-ONLY WALL band body == the serial pad_with_pole_bc_lat per band:
    interior cuts read the neighbour band's edge row (ppermute), pole end bands
    get the south/north constant.  lon is NOT padded (wall pad is lat-only)."""
    mesh = _mesh()
    from jax.sharding import NamedSharding, PartitionSpec as P
    from legoesm.grids.halo_latlon import pad_with_pole_bc_lat
    from legoesm.parallel.latlon_spmd import make_latlon_band_wall_pad_body

    rng = np.random.default_rng(
        13 + int(south_value) + (0 if nlev is None else 50))
    shp = (N_LAT, N_LON) if nlev is None else (N_LAT, N_LON, nlev)
    field = jnp.asarray(rng.standard_normal(shp))
    # Serial reference: local backend constant pad along lat ONLY (n_lon kept).
    ref = np.asarray(pad_with_pole_bc_lat(
        field, halo=1, south_value=south_value, north_value=north_value))

    isp = P("lat", *((None,) * (field.ndim - 1)))
    field_sh = jax.device_put(field, NamedSharding(mesh, isp))
    body = make_latlon_band_wall_pad_body(
        mesh, halo=1, south_value=south_value, north_value=north_value)
    from functools import partial
    try:
        from jax import shard_map
    except ImportError:  # pragma: no cover
        from jax.experimental.shard_map import shard_map

    @partial(shard_map, mesh=mesh, in_specs=isp, out_specs=isp, check_vma=False)
    def _ex(tile):
        return body(tile)

    out = np.asarray(_ex(field_sh))
    blk = NL + 2                                          # per-band padded lat
    assert out.shape[0] == N_DEV * blk
    assert out.shape[1] == N_LON                          # lon UNCHANGED
    worst = 0.0
    for b in range(N_DEV):
        t = out[b * blk:(b + 1) * blk]
        g = ref[b * NL: b * NL + NL + 2]
        worst = max(worst, float(np.max(np.abs(t - g))))
    assert worst < 1e-12, (
        f"wall pad band vs serial (south={south_value}, north={north_value}, "
        f"nlev={nlev}): {worst:.3e}")


def test_wall_pad_backend_dispatch_matches_serial():
    """pad_with_pole_bc_lat called INSIDE a shard_map with the spmd backend armed
    routes to the wall band body and matches serial — so the ocean step's wall-BC
    pads are backend-oblivious under SPMD (the failure that walled interior band
    cuts to zero before the spmd branch existed)."""
    mesh = _mesh()
    from functools import partial
    from jax.sharding import NamedSharding, PartitionSpec as P
    try:
        from jax import shard_map
    except ImportError:  # pragma: no cover
        from jax.experimental.shard_map import shard_map
    from legoesm.grids.halo_latlon import pad_with_pole_bc_lat
    from legoesm.parallel.latlon_spmd import (
        activate_latlon_spmd_halo, deactivate_latlon_spmd_halo,
    )

    rng = np.random.default_rng(404)
    field = jnp.asarray(rng.standard_normal((N_LAT, N_LON)))
    ref = np.asarray(pad_with_pole_bc_lat(field, halo=1))   # local backend

    isp = P("lat", None)
    field_sh = jax.device_put(field, NamedSharding(mesh, isp))
    activate_latlon_spmd_halo(mesh)
    try:
        @partial(shard_map, mesh=mesh, in_specs=isp, out_specs=isp,
                 check_vma=False)
        def _ex(tile):
            return pad_with_pole_bc_lat(tile, halo=1)   # routes to wall body
        out = np.asarray(_ex(field_sh))
    finally:
        deactivate_latlon_spmd_halo()

    blk = NL + 2
    worst = max(
        float(np.max(np.abs(out[b * blk:(b + 1) * blk]
                            - ref[b * NL: b * NL + NL + 2])))
        for b in range(N_DEV))
    assert worst < 1e-12, (
        f"pad_with_pole_bc_lat spmd-backend dispatch vs serial: {worst:.3e}")


def test_zero_polar_lat_ends_backend_dispatch_matches_serial():
    """zero_polar_lat_ends under the spmd backend zeros ONLY the global pole rows
    (south band index 0, north band index -1) and leaves interior band cuts
    intact — matching a serial zero of just the two global ends.  The naive local
    branch would zero EVERY band's ends (clobbering the cut v-rows)."""
    mesh = _mesh()
    from functools import partial
    from jax.sharding import NamedSharding, PartitionSpec as P
    try:
        from jax import shard_map
    except ImportError:  # pragma: no cover
        from jax.experimental.shard_map import shard_map
    from legoesm.grids.halo_latlon import zero_polar_lat_ends
    from legoesm.parallel.latlon_spmd import (
        activate_latlon_spmd_halo, deactivate_latlon_spmd_halo,
    )

    # A v-face-style field (n_lat+1 rows would be the real shape; here the
    # post-stencil result is the band-local nl rows, so use N_LAT rows split
    # evenly).  Reference = zero ONLY the two global ends.
    rng = np.random.default_rng(505)
    field = jnp.asarray(rng.standard_normal((N_LAT, N_LON)))
    ref = np.asarray(field).copy()
    ref[0] = 0.0
    ref[-1] = 0.0

    isp = P("lat", None)
    field_sh = jax.device_put(field, NamedSharding(mesh, isp))
    activate_latlon_spmd_halo(mesh)
    try:
        @partial(shard_map, mesh=mesh, in_specs=isp, out_specs=isp,
                 check_vma=False)
        def _ex(tile):
            return zero_polar_lat_ends(tile)
        out = np.asarray(_ex(field_sh))
    finally:
        deactivate_latlon_spmd_halo()

    worst = float(np.max(np.abs(out - ref)))
    assert worst < 1e-12, (
        f"zero_polar_lat_ends spmd-backend dispatch vs serial: {worst:.3e}")
    # Non-vacuity: interior band-cut rows are NOT zeroed (a local-branch bug
    # would have zeroed every band's ends — e.g. row NL = band-1 start).
    assert float(np.max(np.abs(out[NL]))) > 1e-6, (
        "interior cut row was wrongly zeroed (local-branch leak)")


# ---------------------------------------------------------------------------
# apply_pole_end_masks — the shared SPMD pole-wall zeroing core (used by
# zero_polar_lat_ends_band_spmd AND the atm PE v-tendency pole clamp).  Masks
# are ARGUMENTS, so this is a device-free unit test: the (south, north) bool
# pair fully determines which band-end rows are zeroed.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("south,north", [
    (True, False),    # south-pole band: zero row 0 only
    (False, True),    # north-pole band: zero row -1 only
    (False, False),   # INTERIOR cut band: zero NOTHING (keeps cut gradient)
    (True, True),     # single-band / serial: both ends
])
def test_apply_pole_end_masks(south, north):
    from legoesm.parallel.latlon_spmd import apply_pole_end_masks
    rng = np.random.default_rng(7)
    field = jnp.asarray(rng.standard_normal((6, 4, 3)))
    out = np.asarray(apply_pole_end_masks(field, (south, north), offset=0))
    ref = np.array(field)
    if south:
        ref[0] = 0.0
    if north:
        ref[-1] = 0.0
    np.testing.assert_array_equal(out, ref)
    # Non-vacuity: an INTERIOR row (row 2) is NEVER touched on any mask combo.
    np.testing.assert_array_equal(out[2], np.asarray(field)[2])


def test_apply_pole_end_masks_offset():
    """offset>0 (MPI-style halo pad) zeros the OFFSET-th and (n-1-offset)-th
    rows, leaving the halo rows themselves untouched."""
    from legoesm.parallel.latlon_spmd import apply_pole_end_masks
    rng = np.random.default_rng(11)
    field = jnp.asarray(rng.standard_normal((7, 4)))
    out = np.asarray(apply_pole_end_masks(field, (True, True), offset=1))
    ref = np.array(field)
    ref[1] = 0.0        # south pole one halo row in
    ref[-2] = 0.0       # north pole one halo row in
    np.testing.assert_array_equal(out, ref)
    # halo rows (0 and -1) are untouched
    np.testing.assert_array_equal(out[0], np.asarray(field)[0])
    np.testing.assert_array_equal(out[-1], np.asarray(field)[-1])
