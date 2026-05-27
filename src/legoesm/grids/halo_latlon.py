"""Halo exchange for the latitude-longitude grid.

- Longitude: periodic wrap (every rank holds full lon — no decomposition)
- Latitude: pole-folding at boundary ranks; inter-rank sendrecv at
  interior partition cuts (MPI backend only)

Backend dispatch
----------------
``pad_halo_latlon`` and the three sibling helpers branch on the
shared ``_halo_backend`` global from :mod:`legoesm.grids.halo`.  This
mirrors the cubed-sphere precedent:

* ``"local"`` — single-rank / single-process: lon-wrap + pole-fold
  at both lat ends (the historical behaviour, kept unchanged).
* ``"mpi"`` — multi-rank lat-lon band: lon-wrap + pole-fold at
  pole-touching ranks, MPI sendrecv at interior partition cuts.
  The implementation lives in
  :mod:`legoesm.parallel.latlon_mpi`; this module dispatches
  there when the active topology is a ``LatLonBandLayout`` and
  delegates back here for the local fallback.

Callers (operators) consume ``pad_halo_latlon`` without knowing the
backend — the dispatch is global state, same as cubed-sphere
``pad_halo``.  This is the architectural reuse the user asked for:
no separate operator entry points per backend.
"""

from __future__ import annotations

import jax.numpy as jnp


def _fold_pole_rows(
    data: jnp.ndarray,
    halo: int,
    negate: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute pole-folded halo rows for south and north poles.

    When crossing a pole, a point at latitude (90+delta) maps to
    latitude (90-delta) at longitude+180 deg.  For vector components
    the sign also reverses.

    Parameters
    ----------
    data : shape (n_lat, n_lon_padded)
        Field (already padded in longitude).
    halo : int
    negate : bool
        If True, negate the folded values (for vector components).

    Returns
    -------
    south, north : halo rows ready for concatenation.
    """
    half = data.shape[1] // 2
    sign = -1.0 if negate else 1.0

    # South pole: ghost rows mirror the first `halo` rows, reversed in
    # latitude order and shifted by 180 deg in longitude.
    south = sign * jnp.roll(data[:halo][::-1], half, axis=1)

    # North pole: mirror the last `halo` rows.
    north = sign * jnp.roll(data[-halo:][::-1], half, axis=1)

    return south, north


def _pad_halo_latlon_local(data: jnp.ndarray, halo: int = 1) -> jnp.ndarray:
    """Local (single-rank) scalar halo: lon-wrap + pole-fold on both lat ends.

    This is the historical implementation of ``pad_halo_latlon``,
    extracted so the public ``pad_halo_latlon`` can dispatch on
    ``_halo_backend``.  Serial bit-for-bit identical to the
    pre-refactor function.
    """
    # Longitude: periodic wrap.  ``jnp.pad(..., mode='wrap')`` lowers
    # to a single XLA Pad op; the previous ``concatenate([data[:, -halo:],
    # data, data[:, :halo]])`` materialised three buffers + a concat
    # HLO per call.
    pad_axes = ((0, 0),) * (data.ndim - 1)
    data_lon = jnp.pad(data, (*pad_axes, (halo, halo)), mode="wrap")

    # Latitude: pole-folding (scalar — no sign change).  Pole rows are
    # NOT periodic so we still concat the folded rows.
    south, north = _fold_pole_rows(data_lon, halo, negate=False)
    padded = jnp.concatenate([south, data_lon, north], axis=0)
    return padded


def pad_halo_latlon(data: jnp.ndarray, halo: int = 1) -> jnp.ndarray:
    """Pad a scalar field with halo cells using pole-folding.

    Backend-dispatched: ``"local"`` runs the serial pole-fold;
    ``"mpi"`` (set via
    :func:`legoesm.grids.halo.set_halo_backend` with a
    :class:`~legoesm.parallel.latlon_mpi.LatLonBandLayout` topology)
    routes through
    :func:`legoesm.parallel.latlon_mpi._pad_halo_latlon_mpi`, which
    does lon-wrap + pole-fold at boundary ranks + MPI sendrecv at
    interior partition cuts.  Callers stay backend-oblivious; same
    pattern as cubed-sphere ``pad_halo``.

    Parameters
    ----------
    data : jax.Array
        Scalar field, shape ``(n_lat_local, n_lon)`` or
        ``(n_lat_local, n_lon, nlev)``.  Under MPI, ``n_lat_local``
        is the rank's owned band; under local, it is the full
        global ``n_lat``.
    halo : int
        Number of halo cells on each side (default 1).

    Returns
    -------
    jax.Array : Padded field, shape ``(n_lat_local + 2*halo, n_lon + 2*halo[, nlev])``.
    """
    # Local import to avoid a module-import cycle: halo.py is
    # imported by many low-level modules, and the dispatch only
    # needs ``halo`` when the backend is non-local.
    from legoesm.grids.halo import get_halo_backend, get_mpi_topology
    if get_halo_backend() == "mpi":
        topology = get_mpi_topology()
        # Lat-lon topology = LatLonBandLayout.  Anything else means
        # an MPI run was activated for a different grid type (e.g.
        # cubed-sphere) and a lat-lon op was called by mistake; fall
        # back to the local serial path rather than crashing in the
        # MPI dispatch with an opaque error.
        from legoesm.parallel.latlon_mpi import (
            LatLonBandLayout, _pad_halo_latlon_mpi,
        )
        if isinstance(topology, LatLonBandLayout):
            return _pad_halo_latlon_mpi(
                data, topology, halo=halo, is_vector_v=False,
            )
    return _pad_halo_latlon_local(data, halo)


def _pad_halo_latlon_vector_local(
    data: jnp.ndarray, halo: int = 1,
) -> jnp.ndarray:
    """Local vector halo: lon-wrap + pole-fold with sign reversal."""
    # Longitude: periodic wrap (single Pad HLO via mode="wrap").
    pad_axes = ((0, 0),) * (data.ndim - 1)
    data_lon = jnp.pad(data, (*pad_axes, (halo, halo)), mode="wrap")

    # Latitude: pole-folding (vector — negate)
    south, north = _fold_pole_rows(data_lon, halo, negate=True)
    padded = jnp.concatenate([south, data_lon, north], axis=0)
    return padded


def pad_halo_latlon_vector(data: jnp.ndarray, halo: int = 1) -> jnp.ndarray:
    """Pad a single vector component with pole-folding and sign reversal.

    Backend-dispatched (see :func:`pad_halo_latlon` for details).

    Use this for any quantity that changes sign when crossing a
    pole — meridional velocity ``v``, meridional vector flux, etc.

    Parameters
    ----------
    data : jax.Array
        Vector component field, shape ``(n_lat_local, n_lon)`` or
        ``(n_lat_local, n_lon, nlev)``.
    halo : int
        Number of halo cells on each side (default 1).

    Returns
    -------
    jax.Array : Padded field, shape ``(n_lat_local + 2*halo, n_lon + 2*halo[, nlev])``.
    """
    from legoesm.grids.halo import get_halo_backend, get_mpi_topology
    if get_halo_backend() == "mpi":
        topology = get_mpi_topology()
        from legoesm.parallel.latlon_mpi import (
            LatLonBandLayout, _pad_halo_latlon_mpi,
        )
        if isinstance(topology, LatLonBandLayout):
            return _pad_halo_latlon_mpi(
                data, topology, halo=halo, is_vector_v=True,
            )
    return _pad_halo_latlon_vector_local(data, halo)


def pad_halo_vector_latlon(
    u: jnp.ndarray,
    v: jnp.ndarray,
    halo: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pad vector components with pole-folding and sign reversal.

    Parameters
    ----------
    u, v : jax.Array
        Vector components, shape (n_lat, n_lon).
    halo : int
        Number of halo cells on each side (default 1).

    Returns
    -------
    (u_padded, v_padded) : each shape (n_lat+2*halo, n_lon+2*halo).
    """
    return pad_halo_latlon_vector(u, halo), pad_halo_latlon_vector(v, halo)


# ==============================================================================
# Native 3D halo padding (batch across levels, no vmap)
# ==============================================================================


def _fold_pole_rows_3d(
    data: jnp.ndarray,
    halo: int,
    negate: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pole-fold for 3D arrays, shape (n_lat, n_lon_padded, nlev).

    Same logic as _fold_pole_rows but keeps the level axis intact.
    """
    half = data.shape[1] // 2
    sign = -1.0 if negate else 1.0

    south = sign * jnp.roll(data[:halo][::-1], half, axis=1)
    north = sign * jnp.roll(data[-halo:][::-1], half, axis=1)
    return south, north


def _pad_halo_latlon_3d_local(
    data: jnp.ndarray, halo: int = 1,
) -> jnp.ndarray:
    """Local 3-D scalar halo: lon-wrap + pole-fold."""
    # Longitude: periodic wrap via single Pad HLO (axis 1).
    data_lon = jnp.pad(data, ((0, 0), (halo, halo), (0, 0)), mode="wrap")
    # Latitude: pole-folding (scalar — no sign change)
    south, north = _fold_pole_rows_3d(data_lon, halo, negate=False)
    return jnp.concatenate([south, data_lon, north], axis=0)


def pad_halo_latlon_3d(data: jnp.ndarray, halo: int = 1) -> jnp.ndarray:
    """Pad a scalar 3D field with halo cells.  Backend-dispatched."""
    from legoesm.grids.halo import get_halo_backend, get_mpi_topology
    if get_halo_backend() == "mpi":
        topology = get_mpi_topology()
        from legoesm.parallel.latlon_mpi import (
            LatLonBandLayout, _pad_halo_latlon_mpi,
        )
        if isinstance(topology, LatLonBandLayout):
            return _pad_halo_latlon_mpi(
                data, topology, halo=halo, is_vector_v=False,
            )
    return _pad_halo_latlon_3d_local(data, halo)


def _pad_halo_latlon_vector_3d_local(
    data: jnp.ndarray, halo: int = 1,
) -> jnp.ndarray:
    """Local 3-D vector halo: lon-wrap + pole-fold with sign reversal."""
    data_lon = jnp.pad(data, ((0, 0), (halo, halo), (0, 0)), mode="wrap")
    south, north = _fold_pole_rows_3d(data_lon, halo, negate=True)
    return jnp.concatenate([south, data_lon, north], axis=0)


def pad_halo_latlon_vector_3d(data: jnp.ndarray, halo: int = 1) -> jnp.ndarray:
    """Pad a 3D vector component with pole-folding + sign reversal.

    Backend-dispatched."""
    from legoesm.grids.halo import get_halo_backend, get_mpi_topology
    if get_halo_backend() == "mpi":
        topology = get_mpi_topology()
        from legoesm.parallel.latlon_mpi import (
            LatLonBandLayout, _pad_halo_latlon_mpi,
        )
        if isinstance(topology, LatLonBandLayout):
            return _pad_halo_latlon_mpi(
                data, topology, halo=halo, is_vector_v=True,
            )
    return _pad_halo_latlon_vector_3d_local(data, halo)


def pad_halo_vector_latlon_3d(
    u: jnp.ndarray,
    v: jnp.ndarray,
    halo: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pad 3D vector components with pole-folding and sign reversal.

    Parameters
    ----------
    u, v : jax.Array
        Vector components, shape (n_lat, n_lon, nlev).
    halo : int
        Number of halo cells on each side (default 1).

    Returns
    -------
    (u_padded, v_padded) : each shape (n_lat+2*halo, n_lon+2*halo, nlev).
    """
    return pad_halo_latlon_vector_3d(u, halo), pad_halo_latlon_vector_3d(v, halo)


# ==============================================================================
# Wall-BC pad with backend dispatch
# ==============================================================================
#
# This is the lat-lon analog of cubed-sphere ``pad_halo``'s wall-BC
# behaviour.  Operators such as ``curl_vertex_cgrid``,
# ``gradient_y_cgrid`` and ``divergence_cgrid`` extend their fields
# along the lat axis with a constant value at the poles (``-1``/``+1``
# for ``sin_lat``, ``0`` for u / dx / df_interior).  Under serial
# execution the constant pad IS the correct boundary condition.
# Under latitude-band MPI, the same convention applies only at the
# pole-touching ranks; interior partition cuts must instead receive
# the neighbour rank's value via MPI sendrecv.
#
# We expose one helper, ``pad_with_pole_bc_lat``, and route every
# pole-BC-style pad through it (``pad_ns_zero`` and friends in
# ``ocean.dynamics.latlon_cgrid_operators`` are refactored in a
# follow-up commit to delegate here).


def pad_with_pole_bc_lat(
    interior: jnp.ndarray,
    halo: int = 1,
    south_value: float = 0.0,
    north_value: float = 0.0,
    *,
    is_vector_v: bool = False,
) -> jnp.ndarray:
    """Pad along lat axis with pole-BC constants; backend-dispatched.

    Local backend
    -------------
    ``jnp.pad(interior, ((halo, halo), (0, 0), ...))`` with
    ``constant_values=(south_value, north_value)`` — identical to
    the inline ``jnp.pad`` calls the operators currently make.

    MPI backend
    -----------
    Pole-touching south rank → pad with ``south_value``.
    Pole-touching north rank → pad with ``north_value``.
    Interior partition cuts → MPI sendrecv with the neighbour rank,
    reusing the AD-safe
    :func:`legoesm.parallel.latlon_mpi.exchange_halo_latlon` path.
    The ``is_vector_v`` flag is forwarded to the exchange so the
    pole fold applies its sign flip when called at an interior rank
    next to (but not owning) a pole — typically dormant for the
    typical band layouts but kept for safety.

    Parameters
    ----------
    interior : jax.Array
        Field to pad along axis 0.  Shape ``(n_lat_interior, ...)``;
        any trailing axes are passed through unchanged.
    halo : int
        Number of rows to add on each side (typically 1 for
        compact-stencil operators, sometimes 2 for biharmonic /
        PPM).
    south_value, north_value : float
        Constant pad values at pole-touching ranks (or both ends
        under the local backend).
    is_vector_v : bool
        Forwarded to the MPI exchange's pole-fold sign-flip
        convention.  For scalar wall BC (sin, dx, df, etc.) this
        stays False.

    Returns
    -------
    jax.Array : shape ``(n_lat_interior + 2*halo, ...)``.
    """
    if halo <= 0:
        return interior
    # Local fallback: same as the operators' previous inline pad.
    # We construct the per-axis pad-tuple manually so the function
    # works for arbitrary trailing dimensions (1D sin_lat, 2D u-face,
    # 3D u-face-with-levels, etc.).
    pad_widths = ((halo, halo),) + ((0, 0),) * (interior.ndim - 1)

    from legoesm.grids.halo import get_halo_backend, get_mpi_topology
    if get_halo_backend() != "mpi":
        # Symmetric constants → single Pad HLO via ``constant_values``
        # tuple (matches the operators' historical formulation).
        return jnp.pad(
            interior, pad_widths,
            constant_values=((south_value, north_value),)
            + ((0, 0),) * (interior.ndim - 1),
        )

    topology = get_mpi_topology()
    # Lat-lon topology is the only kind that maps onto this helper;
    # if the active backend is MPI but for a different grid, fall
    # back to the local serial pad.
    from legoesm.parallel.latlon_mpi import (
        LatLonBandLayout,
        _pad_with_pole_bc_lat_mpi,
    )
    if not isinstance(topology, LatLonBandLayout):
        return jnp.pad(
            interior, pad_widths,
            constant_values=((south_value, north_value),)
            + ((0, 0),) * (interior.ndim - 1),
        )
    return _pad_with_pole_bc_lat_mpi(
        interior, topology,
        halo=halo,
        south_value=south_value,
        north_value=north_value,
        is_vector_v=is_vector_v,
    )
