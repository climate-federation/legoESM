"""Single-controller SPMD halo for the lat-lon grid (multi-GPU, no mpi4jax).

The ocean lat-lon C-grid (and atm lat-lon) shard cleanly by LATITUDE BAND: each
device owns a contiguous lat band and the FULL longitude circle (lon is periodic
and kept local — the audit's "every rank owns all longitudes").  The halo is
therefore 1-D over the ``"lat"`` mesh axis:

  * longitude: periodic wrap — LOCAL ``jnp.pad(mode="wrap")`` (no comm), exactly
    like the serial :func:`legoesm.grids.halo_latlon.pad_halo_latlon_local`.
  * latitude interior band cuts: ``jax.lax.ppermute`` of the ``halo`` edge rows
    between adjacent bands (north neighbour's bottom rows / south neighbour's
    top rows).
  * poles: the end bands (axis_index 0 = south, N-1 = north) have no neighbour
    there, so they fold their OWN pole rows (mirror in lat + 180 deg in lon),
    selected by a ``jnp.where`` on the band index — bit-identical to the serial
    pole fold.

This is the lat-lon analogue of the cubed-sphere
``cubesphere_exchange.make_tiled_pad_body`` (which the cube SPMD uses), and the
foundation for the ocean lat-lon multi-GPU SPMD step (pure-jax ppermute over the
RTX8000 PCIe pair — no mpi4jax dependency).  Validated by BIT-IDENTITY vs the
serial local pad (``tests/parallel/test_latlon_spmd_halo.py``), the proven
methodology.  ``check_vma=False`` follows the cube SPMD halo bodies.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

try:  # JAX >= 0.8 top-level export
    from jax import shard_map
except ImportError:  # pragma: no cover
    from jax.experimental.shard_map import shard_map

from jax.sharding import PartitionSpec as P


def _latlon_band_perms(n_dev: int):
    """Static (src, dst) permutation pairs over the 1-D ``lat`` band axis.

    ``perm_north``: each band ``b`` receives band ``b+1``'s bottom rows as its
    NORTH ghost -> source ``b+1`` sends to ``b`` (pairs ``(s, s-1)``); the top
    band (``N-1``) is not a destination -> its north_recv is zeros, replaced by
    the north pole fold.  ``perm_south``: band ``b`` receives band ``b-1``'s top
    rows as its SOUTH ghost -> ``(s, s+1)``; the bottom band (0) gets zeros ->
    south pole fold.
    """
    perm_north = tuple((s, s - 1) for s in range(1, n_dev))   # send up->down
    perm_south = tuple((s, s + 1) for s in range(0, n_dev - 1))  # down->up
    return perm_north, perm_south


def _pole_fold(rows, negate: bool):
    """Serial pole fold of ``rows`` (lat-mirror + 180 deg lon roll [+ sign]).

    ``rows`` is ``(halo, n_lon_padded[, nlev])`` already lon-wrapped; matches
    :func:`legoesm.grids.halo_latlon.fold_pole_rows` (``half = n_lon_pad // 2``,
    ``roll(rows[::-1], half, axis=lon)``)."""
    half = rows.shape[1] // 2
    sign = -1.0 if negate else 1.0
    return sign * jnp.roll(rows[::-1], half, axis=1)


def activate_latlon_spmd_halo(mesh) -> None:
    """Arm the lat-lon band SPMD halo backend on a 1-D ``"lat"`` mesh.

    Sets the halo backend to ``"spmd"`` and stores ``mesh`` so the per-grid
    ``pad_halo_latlon*`` dispatch routes through :func:`make_latlon_band_pad_body`
    when called INSIDE a shard_map over the same ``"lat"`` axis (the ocean/atm
    lat-lon SPMD step).  Mirrors the cube
    ``cubesphere_exchange.activate_spmd_halo_backend``.
    """
    from legoesm.grids.halo import set_halo_backend, set_spmd_mesh
    if tuple(mesh.devices.shape) != (mesh.devices.size,):
        raise ValueError(
            f"activate_latlon_spmd_halo: needs a 1-D lat-band mesh; got "
            f"shape {tuple(mesh.devices.shape)}")
    # The pad_halo_latlon dispatch routes on ``"lat" in mesh.axis_names``; a
    # mesh named otherwise would activate but SILENTLY fall back to local
    # padding inside the shard_map (wrong interior band halos) — fail loud
    # (codex LOW).
    if "lat" not in tuple(mesh.axis_names):
        raise ValueError(
            f"activate_latlon_spmd_halo: mesh axis must be named 'lat' (the "
            f"pad_halo_latlon SPMD dispatch keys on it); got "
            f"{tuple(mesh.axis_names)}")
    set_spmd_mesh(mesh)
    set_halo_backend("spmd")


def deactivate_latlon_spmd_halo() -> None:
    """Clear the SPMD halo backend (-> ``"local"``)."""
    from legoesm.grids.halo import set_halo_backend, set_spmd_mesh
    set_spmd_mesh(None)
    set_halo_backend("local")


def make_latlon_band_pad_body(mesh, halo: int = 1, negate: bool = False):
    """Unwrapped lat-lon band halo body for use INSIDE a shard_map over the
    ``"lat"`` axis (the ocean/atm lat-lon SPMD step).

    Returns ``body(tile)`` where ``tile`` is one band's local
    ``(nl_lat, n_lon[, nlev])`` block and the result is the padded
    ``(nl_lat+2h, n_lon+2h[, nlev])`` block — lon-wrap (local) + lat-band
    ppermute (interior) + pole fold (ends).  ``negate=True`` folds with a sign
    flip for meridional-vector components (v).  Bit-identical to the serial
    ``pad_halo_latlon_local`` (scalar) / ``..._vector_local`` (vector) per band.
    """
    n_dev = mesh.devices.size
    if tuple(mesh.devices.shape) != (n_dev,):
        raise ValueError(
            f"make_latlon_band_pad_body: needs a 1-D (n_lat-band) mesh; got "
            f"shape {tuple(mesh.devices.shape)}")
    axis = mesh.axis_names[0]
    perm_north, perm_south = _latlon_band_perms(n_dev)

    def body(tile):
        # 1. longitude periodic wrap (LOCAL — full lon circle per band).
        pad_lon = ((0, 0),) + ((halo, halo),) + ((0, 0),) * (tile.ndim - 2)
        data_lon = jnp.pad(tile, pad_lon, mode="wrap")  # (nl, n_lon+2h[, lev])

        # 2. latitude band ppermute of the edge rows (axis 0 = south->north,
        # so row 0 is the SOUTH edge, row -1 the NORTH edge).
        south_edge = data_lon[:halo]   # my south rows -> band below (b-1)'s N ghost
        north_edge = data_lon[-halo:]  # my north rows -> band above (b+1)'s S ghost
        north_recv = jax.lax.ppermute(south_edge, axis, perm_north)  # b's N ghost = b+1's south edge
        south_recv = jax.lax.ppermute(north_edge, axis, perm_south)  # b's S ghost = b-1's north edge

        # 3. pole fold at the end bands (ppermute non-targets receive zeros).
        b = jax.lax.axis_index(axis)
        south_ghost = jnp.where(b == 0,
                                _pole_fold(data_lon[:halo], negate), south_recv)
        north_ghost = jnp.where(b == n_dev - 1,
                                _pole_fold(data_lon[-halo:], negate), north_recv)
        return jnp.concatenate([south_ghost, data_lon, north_ghost], axis=0)

    return body


def pad_halo_latlon_band_spmd(mesh, halo: int = 1, negate: bool = False):
    """Wrapped shard_map lat-lon band halo exchange (parity-test entry).

    ``fn(field)`` where ``field`` is the GLOBAL ``(n_lat, n_lon[, nlev])`` field
    lat-sharded ``P("lat", None[, None])``; returns the padded global field with
    band-local halos, gathered ``(n_dev*(nl+2h)... )`` — band ``b``'s block ==
    the serial pad's ``[b*nl : b*nl+nl+2h]`` window (the tiled methodology)."""
    body = make_latlon_band_pad_body(mesh, halo=halo, negate=negate)
    axis = mesh.axis_names[0]

    def fn(field):
        nd = field.ndim
        isp = P(axis, *((None,) * (nd - 1)))
        osp = P(axis, *((None,) * (nd - 1)))

        @partial(shard_map, mesh=mesh, in_specs=isp, out_specs=osp,
                 check_vma=False)
        def _ex(x):
            return body(x)

        return _ex(field)

    return fn
