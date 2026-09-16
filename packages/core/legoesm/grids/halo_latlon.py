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

# --- Meridionally-periodic (y-re-entrant channel) mode -----------------------
# Default OFF → the historical meridionally-CLOSED (walled N/S v-faces) BC,
# bit-identical.  When ON (single-rank/local backend only for now), the lat-axis
# boundary helpers WRAP instead of walling: ``pad_with_pole_bc_lat`` pads with the
# periodic (rolled) rows and ``zero_polar_lat_ends`` is a no-op (the wrapped halo
# already carries the correct boundary v-faces).  This matches an Oceananigans
# ``Flat``/``Periodic`` meridional topology — required to reproduce idealized 2-D
# x–z cases (internal_tide #576) whose barotropic Rossby radius spans the basin,
# so a closed basin geostrophically adjusts instead of oscillating freely.
# Mirrors the ``_halo_backend`` global-state pattern (accessor, never import the
# global directly).
_MERIDIONALLY_PERIODIC = False


def set_meridionally_periodic(enabled: bool) -> None:
    """Enable/disable the meridionally-periodic (y-re-entrant) boundary mode.

    Local (single-rank) backend only; the MPI/SPMD lat-band paths still wall
    (a periodic-y band exchange is a follow-up).  Default OFF = bit-identical
    closed-basin BC.
    """
    global _MERIDIONALLY_PERIODIC
    _MERIDIONALLY_PERIODIC = bool(enabled)


def get_meridionally_periodic() -> bool:
    """Return whether the meridionally-periodic boundary mode is active."""
    return _MERIDIONALLY_PERIODIC


# --- Meridionally-FLAT (Oceananigans `Flat`-y topology) mode ------------------
# Default OFF.  When ON, every meridional DIFFERENCE operator returns 0 — exactly
# matching Oceananigans' `Flat` topology (`δyᵃᶜᵃ(grid::Flat, v) = zero`): v stays
# prognostic (so the Coriolis f×u → v rotation works) but no ∂/∂y ever exists, so
# (a) no meridional pressure gradient → no closed-basin geostrophic adjustment, and
# (b) no 2Δy mode can ever develop. This is the faithful 2-D x–z analog the oracle
# uses (`topology=(Periodic, Flat, Bounded)`), strictly cleaner than the periodic-y
# thin channel (which ADDS meridional d.o.f. the oracle does not have). Gates
# `gradient_y_cgrid` (→0) and `divergence_cgrid`'s meridional flux (→0); pairs with
# the wrap v_mask so v is unwalled. Default OFF = bit-identical.
_MERIDIONALLY_FLAT = False


def set_meridionally_flat(enabled: bool) -> None:
    """Enable/disable the meridionally-flat (Oceananigans `Flat`-y) mode."""
    global _MERIDIONALLY_FLAT
    _MERIDIONALLY_FLAT = bool(enabled)


def get_meridionally_flat() -> bool:
    """Return whether the meridionally-flat (`Flat`-y) mode is active."""
    return _MERIDIONALLY_FLAT


def fold_pole_rows(
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


def pad_halo_latlon_local(data: jnp.ndarray, halo: int = 1) -> jnp.ndarray:
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
    south, north = fold_pole_rows(data_lon, halo, negate=False)
    padded = jnp.concatenate([south, data_lon, north], axis=0)
    return padded


def _try_spmd_latlon_pad(data: jnp.ndarray, halo: int, negate: bool):
    """Route a lat-lon halo through the band SPMD ppermute body if the ``spmd``
    backend + a ``"lat"`` mesh are active; else return ``None`` (caller falls
    through to mpi/local).  Used by every ``pad_halo_latlon*`` dispatcher so the
    ocean/atm lat-lon step is backend-oblivious — same pattern as the cube.
    Handles 2-D and 3-D (the body's lon-pad is ndim-agnostic).

    A 2-D ``("lat", "lon")`` mesh (M3a native 2-D tiling) routes to
    :func:`legoesm.parallel.latlon_spmd.make_latlon_2d_pad_body` — lat
    ppermute + periodic lon ring ppermute + the exact serial 180-deg pole
    fold (all_gather'd over the lon ring at the pole tiles).  A degenerate
    ``p_lon == 1`` lon axis takes the body's static local-wrap branch,
    bit-identical to the 1-D band body."""
    from legoesm.grids.halo import get_halo_backend, get_spmd_mesh
    if get_halo_backend() != "spmd":
        return None
    mesh = get_spmd_mesh()
    if mesh is None or "lat" not in getattr(mesh, "axis_names", ()):
        return None
    if "lon" in tuple(mesh.axis_names):
        from legoesm.parallel.latlon_spmd import make_latlon_2d_pad_body
        return make_latlon_2d_pad_body(mesh, halo=halo, negate=negate)(data)
    from legoesm.parallel.latlon_spmd import make_latlon_band_pad_body
    return make_latlon_band_pad_body(mesh, halo=halo, negate=negate)(data)


def _spmd_lat_mesh():
    """Return the armed lat-band SPMD mesh, or ``None`` if the ``spmd`` backend
    is not active (caller falls through to the mpi/local branch).  Shared by the
    wall-pad / pole-zero SPMD routing so they key on the backend the same way as
    :func:`_try_spmd_latlon_pad`."""
    from legoesm.grids.halo import get_halo_backend, get_spmd_mesh
    if get_halo_backend() != "spmd":
        return None
    mesh = get_spmd_mesh()
    if mesh is None or "lat" not in getattr(mesh, "axis_names", ()):
        return None
    return mesh


def _dispatch_latlon_2d_fold(data, topology, halo, *, is_vector_v):
    """Fold-family ``pad_halo_latlon*`` dispatch for a ``LatLon2DLayout``.

    ``proc_lon == 1`` is a pure latitude band (every rank owns the full lon
    circle), so the 180-deg pole fold is LOCAL — reuse the validated band
    fold (:func:`legoesm.parallel.latlon_mpi.pad_halo_latlon_mpi` on the
    equivalent :class:`~legoesm.parallel.latlon_mpi.LatLonBandLayout`),
    which is bit-identical to the serial fold and correct for EVERY scalar
    caller (incl. the A-grid operators that read the pole ghost directly).
    This is why the global ``pad_halo_latlon`` 2-D dispatch is SAFE — it
    does not silently swap pole-fold for a wall-zero ghost at proc_lon==1.

    ``proc_lon > 1`` splits longitude, so the fold's 180-deg shift needs the
    lat-pencil transpose (not yet wired) — fall back to the labeled
    wall-pole benchmark pad (``pad_halo_latlon_2d(pole_bc="wall")``).
    :func:`legoesm.parallel.latlon_mpi.make_latlon_2d_mpi_step` now WIRES
    proc_lon>1 for regular / wall-pole grids (so an atmospheric wall-pole
    step DOES reach here), but still refuses the TRIPOLAR fold — the 180-deg
    fold path below is the wall-pole case only.
    Handles 2-D and 3-D (the level axis rides through both backends).
    """
    from legoesm.parallel.latlon_mpi import (
        make_latlon_band_layout,
        pad_halo_latlon_2d,
        pad_halo_latlon_mpi,
    )
    if topology.proc_lon == 1:
        band = make_latlon_band_layout(
            topology.proc_row, topology.proc_lat,
            topology.n_lat_global, topology.n_lon_global,
            fold=topology.fold,
        )
        return pad_halo_latlon_mpi(
            data, band, halo=halo, is_vector_v=is_vector_v)
    return pad_halo_latlon_2d(data, topology, halo=halo, pole_bc="wall")


def pad_halo_latlon(data: jnp.ndarray, halo: int = 1) -> jnp.ndarray:
    """Pad a scalar field with halo cells using pole-folding.

    Backend-dispatched: ``"local"`` runs the serial pole-fold;
    ``"mpi"`` (set via
    :func:`legoesm.grids.halo.set_halo_backend` with a
    :class:`~legoesm.parallel.latlon_mpi.LatLonBandLayout` topology)
    routes through
    :func:`legoesm.parallel.latlon_mpi.pad_halo_latlon_mpi`, which
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
            LatLon2DLayout, LatLonBandLayout, pad_halo_latlon_mpi,
        )
        if isinstance(topology, LatLonBandLayout):
            return pad_halo_latlon_mpi(
                data, topology, halo=halo, is_vector_v=False,
            )
        if isinstance(topology, LatLon2DLayout):
            return _dispatch_latlon_2d_fold(
                data, topology, halo, is_vector_v=False,
            )
    _spmd = _try_spmd_latlon_pad(data, halo, negate=False)
    if _spmd is not None:
        return _spmd
    return pad_halo_latlon_local(data, halo)


def pad_halo_latlon_vector_local(
    data: jnp.ndarray, halo: int = 1,
) -> jnp.ndarray:
    """Local vector halo: lon-wrap + pole-fold with sign reversal."""
    # Longitude: periodic wrap (single Pad HLO via mode="wrap").
    pad_axes = ((0, 0),) * (data.ndim - 1)
    data_lon = jnp.pad(data, (*pad_axes, (halo, halo)), mode="wrap")

    # Latitude: pole-folding (vector — negate)
    south, north = fold_pole_rows(data_lon, halo, negate=True)
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
            LatLon2DLayout, LatLonBandLayout, pad_halo_latlon_mpi,
        )
        if isinstance(topology, LatLonBandLayout):
            return pad_halo_latlon_mpi(
                data, topology, halo=halo, is_vector_v=True,
            )
        if isinstance(topology, LatLon2DLayout):
            return _dispatch_latlon_2d_fold(
                data, topology, halo, is_vector_v=True,
            )
    _spmd = _try_spmd_latlon_pad(data, halo, negate=True)
    if _spmd is not None:
        return _spmd
    return pad_halo_latlon_vector_local(data, halo)


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


def fold_pole_rows_3d(
    data: jnp.ndarray,
    halo: int,
    negate: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pole-fold for 3D arrays, shape (n_lat, n_lon_padded, nlev).

    Same logic as fold_pole_rows but keeps the level axis intact.
    """
    half = data.shape[1] // 2
    sign = -1.0 if negate else 1.0

    south = sign * jnp.roll(data[:halo][::-1], half, axis=1)
    north = sign * jnp.roll(data[-halo:][::-1], half, axis=1)
    return south, north


def pad_halo_latlon_3d_local(
    data: jnp.ndarray, halo: int = 1,
) -> jnp.ndarray:
    """Local 3-D scalar halo: lon-wrap + pole-fold."""
    # Longitude: periodic wrap via single Pad HLO (axis 1).
    data_lon = jnp.pad(data, ((0, 0), (halo, halo), (0, 0)), mode="wrap")
    # Latitude: pole-folding (scalar — no sign change)
    south, north = fold_pole_rows_3d(data_lon, halo, negate=False)
    return jnp.concatenate([south, data_lon, north], axis=0)


def pad_halo_latlon_3d(data: jnp.ndarray, halo: int = 1) -> jnp.ndarray:
    """Pad a scalar 3D field with halo cells.  Backend-dispatched."""
    from legoesm.grids.halo import get_halo_backend, get_mpi_topology
    if get_halo_backend() == "mpi":
        topology = get_mpi_topology()
        from legoesm.parallel.latlon_mpi import (
            LatLon2DLayout, LatLonBandLayout, pad_halo_latlon_mpi,
        )
        if isinstance(topology, LatLonBandLayout):
            return pad_halo_latlon_mpi(
                data, topology, halo=halo, is_vector_v=False,
            )
        if isinstance(topology, LatLon2DLayout):
            return _dispatch_latlon_2d_fold(
                data, topology, halo, is_vector_v=False,
            )
    _spmd = _try_spmd_latlon_pad(data, halo, negate=False)
    if _spmd is not None:
        return _spmd
    return pad_halo_latlon_3d_local(data, halo)


def pad_halo_latlon_vector_3d_local(
    data: jnp.ndarray, halo: int = 1,
) -> jnp.ndarray:
    """Local 3-D vector halo: lon-wrap + pole-fold with sign reversal."""
    data_lon = jnp.pad(data, ((0, 0), (halo, halo), (0, 0)), mode="wrap")
    south, north = fold_pole_rows_3d(data_lon, halo, negate=True)
    return jnp.concatenate([south, data_lon, north], axis=0)


def pad_halo_latlon_vector_3d(data: jnp.ndarray, halo: int = 1) -> jnp.ndarray:
    """Pad a 3D vector component with pole-folding + sign reversal.

    Backend-dispatched."""
    from legoesm.grids.halo import get_halo_backend, get_mpi_topology
    if get_halo_backend() == "mpi":
        topology = get_mpi_topology()
        from legoesm.parallel.latlon_mpi import (
            LatLon2DLayout, LatLonBandLayout, pad_halo_latlon_mpi,
        )
        if isinstance(topology, LatLonBandLayout):
            return pad_halo_latlon_mpi(
                data, topology, halo=halo, is_vector_v=True,
            )
        if isinstance(topology, LatLon2DLayout):
            return _dispatch_latlon_2d_fold(
                data, topology, halo, is_vector_v=True,
            )
    _spmd = _try_spmd_latlon_pad(data, halo, negate=True)
    if _spmd is not None:
        return _spmd
    return pad_halo_latlon_vector_3d_local(data, halo)


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


def zero_polar_lat_ends(field: jnp.ndarray) -> jnp.ndarray:
    """Zero a field at its lat-boundary rows (wall BC), backend-aware.

    Used by operators that pre-pad a field via :func:`pad_halo_latlon`
    and then compute a compact stencil — the resulting gradient
    contains a (generally non-zero) value at the polar v-faces from
    the pole-fold ghost, which must be overridden with zero to match
    the dycore's wall-BC convention.

    Local backend
        Always zeros indices ``0`` and ``-1`` of axis 0 (the historical
        "both ends are poles" assumption).

    MPI backend (LatLonBandLayout topology)
        Zeros index 0 only if this rank touches the south pole
        (``layout.south_rank is None``); zeros index ``-1`` only if
        this rank touches the north pole.  Interior partition cuts
        are left intact so the cross-partition gradient computed
        from neighbour-sendrecv'd halo values is preserved.

    Parameters
    ----------
    field : jax.Array
        Shape ``(n_lat_v, ...)``.  Axis 0 is the lat (v-face) axis.

    Returns
    -------
    jax.Array : same shape as ``field``.
    """
    # Meridionally-periodic (y-re-entrant) mode: the lat-axis WRAPS, so the
    # boundary v-faces are genuine periodic interfaces, not walls — do NOT zero
    # them.  Local backend only (the MPI/SPMD band paths below still wall).
    if _MERIDIONALLY_PERIODIC:
        from legoesm.grids.halo import get_halo_backend
        if get_halo_backend() != "mpi" and _spmd_lat_mesh() is None:
            return field
    # SPMD lat-band backend (single-controller multi-GPU): zero index 0 only on
    # the south band, index -1 only on the north band; interior band cuts keep
    # their cross-band gradient (the analogue of the MPI pole-touch test).
    _spmd_mesh = _spmd_lat_mesh()
    if _spmd_mesh is not None:
        from legoesm.parallel.latlon_spmd import zero_polar_lat_ends_band_spmd
        return zero_polar_lat_ends_band_spmd(field, _spmd_mesh)
    from legoesm.grids.halo import get_halo_backend, get_mpi_topology
    if get_halo_backend() == "mpi":
        topology = get_mpi_topology()
        from legoesm.parallel.latlon_mpi import (
            LatLon2DLayout, LatLonBandLayout,
        )
        # Band AND 2-D pencil share the pole-touch test: a rank zeros its
        # south end only if it OWNS the south pole (``south_rank is None``)
        # and its north end only if it owns the north pole.  In the 2-D
        # pencil this is the proc_row-0 / proc_row-last test (the riskiest
        # 2-D bug — an interior proc row must NOT wall its lat cut, or the
        # cross-cut gradient sendrecv'd from the neighbour is destroyed).
        if isinstance(topology, (LatLonBandLayout, LatLon2DLayout)):
            out = field
            if topology.south_rank is None:
                out = out.at[0].set(jnp.zeros_like(out[0]))
            if topology.north_rank is None:
                out = out.at[-1].set(jnp.zeros_like(out[-1]))
            return out
    # Local fallback (and non-LatLonBandLayout MPI): always zero both
    # ends — preserves the single-rank wall-BC convention.
    out = field.at[0].set(jnp.zeros_like(field[0]))
    out = out.at[-1].set(jnp.zeros_like(out[-1]))
    return out


def pad_with_pole_bc_lat(
    interior: jnp.ndarray,
    halo: int = 1,
    south_value: float = 0.0,
    north_value: float = 0.0,
    *,
    is_vector_v: bool = False,
    is_vector_u: bool = False,
    north_fold: bool = False,
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
    is_vector_u : bool
        Forwarded to the MPI exchange for tripolar grids (issue #353):
        a u-face field flips sign across the north fold seam
        (``vector_sign_u``).  No effect on the local backend or on
        regular lat-lon (north is a wall, not a fold).
    north_fold : bool
        Opt-in tripolar north fold (issue #353).  Default ``False`` keeps
        the historical WALL semantics at the north boundary even on an
        active tripolar layout, so existing wall-BC callers are unchanged.
        Pass ``True`` only for a quantity whose north boundary is the
        ORCA fold seam (then ``is_vector_u`` / ``is_vector_v`` choose the
        sign convention).  No effect on the local backend.

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

    # Meridionally-periodic (y-re-entrant) mode: WRAP-pad the lat axis instead of
    # padding with the south/north wall constants.  ``mode="wrap"`` fills the
    # south halo with ``interior[-halo:]`` and the north halo with
    # ``interior[:halo]`` — the periodic-channel BC.  Local backend only (the
    # MPI/SPMD band paths below keep the wall constant; periodic-y band exchange
    # is a follow-up).  No vector sign flip: a y-periodic f-plane channel has no
    # pole fold.
    if _MERIDIONALLY_PERIODIC:
        from legoesm.grids.halo import get_halo_backend
        if get_halo_backend() != "mpi" and _spmd_lat_mesh() is None:
            return jnp.pad(interior, pad_widths, mode="wrap")

    # SPMD lat-band backend (single-controller multi-GPU): interior band cuts
    # must read the NEIGHBOUR band's edge row (ppermute), not a constant wall;
    # only the PHYSICAL pole end bands get the south/north wall constant.  The
    # local jnp.pad below would wall EVERY band's boundary (wrong cut rows).
    _spmd_mesh = _spmd_lat_mesh()
    if _spmd_mesh is not None:
        if north_fold or is_vector_u or is_vector_v:
            # The wall band body handles the regular-grid wall BC only; the
            # tripolar fold seam (sign-flipped permutation across the north
            # boundary) is a follow-up.  Fail loud rather than silently wall it.
            raise NotImplementedError(
                "pad_with_pole_bc_lat SPMD: tripolar north fold / vector-sign "
                "flags are a follow-up (the regular-grid wall BC is wired). "
                "north_fold=%r is_vector_u=%r is_vector_v=%r"
                % (north_fold, is_vector_u, is_vector_v))
        from legoesm.parallel.latlon_spmd import make_latlon_band_wall_pad_body
        return make_latlon_band_wall_pad_body(
            _spmd_mesh, halo=halo,
            south_value=south_value, north_value=north_value)(interior)

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
        LatLon2DLayout,
        LatLonBandLayout,
        pad_with_pole_bc_lat_2d,
        pad_with_pole_bc_lat_mpi,
    )
    if isinstance(topology, LatLon2DLayout):
        # 2-D pencil: lat-axis-ONLY wall pad (interior lat cut sendrecv +
        # pole wall constant).  Longitude is left untouched — the band path
        # never split lon, so its wall-BC pad is lat-only; the 2-D path
        # keeps that contract and the operator adds lon ghosts through its
        # own dispatched lon halo.  The tripolar north fold / vector-u sign
        # seam needs the lat-pencil transpose (the wall-pole 2-D benchmark
        # excludes it) — fail loud rather than silently wall a fold seam.
        if north_fold or is_vector_u:
            raise NotImplementedError(
                "pad_with_pole_bc_lat 2-D pencil: tripolar north_fold / "
                "is_vector_u need the lat-pencil transpose (wall-pole 2-D "
                f"only). north_fold={north_fold!r} is_vector_u={is_vector_u!r}"
            )
        return pad_with_pole_bc_lat_2d(
            interior, topology, halo=halo,
            south_value=south_value, north_value=north_value,
        )
    if not isinstance(topology, LatLonBandLayout):
        return jnp.pad(
            interior, pad_widths,
            constant_values=((south_value, north_value),)
            + ((0, 0),) * (interior.ndim - 1),
        )
    return pad_with_pole_bc_lat_mpi(
        interior, topology,
        halo=halo,
        south_value=south_value,
        north_value=north_value,
        is_vector_v=is_vector_v,
        is_vector_u=is_vector_u,
        north_fold=north_fold,
    )


def pad_with_pole_bc_lat_multi(
    fields,
    halo: int = 1,
    south_values=None,
    north_values=None,
) -> tuple:
    """Batched :func:`pad_with_pole_bc_lat` for independent wall-BC scalars.

    Pads every field in ``fields`` along the lat axis with constant
    boundary values, backend-dispatched.  Value-identical to calling
    :func:`pad_with_pole_bc_lat` once per field — but under the MPI
    lat-lon band backend the interior partition cuts are exchanged in
    ONE fused sendrecv pair per cut per dtype group instead of one pair
    per field.  mpi4jax sendrecvs are token-serialized (no overlap), so
    each fused cluster of N pads saves ``(N-1) x 2`` sendrecv latencies
    per step — the measured rank-growing term of the ocean baroclinic
    phase (scaling campaign audit lever O4).

    Scalar wall-BC fields only: no ``is_vector_*`` / ``north_fold``
    support (those callers keep the single-field path; their boundary
    handling is field-specific, while the interior-cut exchange this
    helper fuses is flag-independent).

    Set ``LEGOESM_LATLON_FUSED_HALO=0`` to force the per-field
    single-exchange fallback (A/B lever; trace-time Python switch, same
    pattern as the other feature gates).

    Parameters
    ----------
    fields : sequence of jax.Array
        Fields to pad along axis 0.  Must share ``n_lat`` (axis 0);
        trailing shapes / dtypes may differ.
    halo : int
    south_values, north_values : sequence of float, optional
        Per-field boundary constants (default all-zero, i.e.
        ``pad_ns_zero`` semantics).

    Returns
    -------
    tuple of jax.Array, in input order.
    """
    fields = tuple(fields)
    n = len(fields)
    if n == 0:
        return ()
    if south_values is None:
        south_values = (0.0,) * n
    if north_values is None:
        north_values = (0.0,) * n
    south_values = tuple(south_values)
    north_values = tuple(north_values)
    if len(south_values) != n or len(north_values) != n:
        raise ValueError(
            "pad_with_pole_bc_lat_multi: south_values/north_values must "
            f"match len(fields)={n}; got {len(south_values)}/"
            f"{len(north_values)}."
        )

    import os

    from legoesm.grids.halo import get_halo_backend, get_mpi_topology

    # SPMD leg of the message-aggregation lever (audit item 7): ONE
    # ppermute pair per direction per dtype group instead of one per
    # field.  Value-identical to the per-field pads (the exchange is a
    # bit-copy), so this is a pure collective-COUNT reduction: census
    # 41 -> 29 collective-permutes/step at IDENTICAL bytes.
    #
    # DEFAULT ON since the audit item's contract -- "flip per deck only
    # with a measured GPU A/B receipt" -- is satisfied on both lanes:
    #   atm  LL2048, jobs 26681636/26681858:  -5.4 % @64, -4.5 % @128
    #        (fused alone; A/A2 off-arms bracket at 6.590/6.650 @64)
    #   ocean LL2304@128, job 26692291:       -4.9 % (A/A2 drift 0.06 %)
    # Both are the fused-alone arm, i.e. exactly this flag; the larger
    # atm -20.3 %/-15.0 % numbers stack XLA overlap, which is a separate
    # job-level env knob (LEGOESM_XLA_OVERLAP, _env.sh) and is NOT
    # implied here -- it showed no ocean benefit.
    #
    # Set LEGOESM_LATLON_SPMD_FUSED_HALO=0 to restore the per-field
    # ppermutes (byte-identical, just more collectives).
    _spmd_fused = os.environ.get(
        "LEGOESM_LATLON_SPMD_FUSED_HALO", "1") != "0"
    if _spmd_fused:
        mesh = _spmd_lat_mesh()
        if mesh is not None:
            from legoesm.parallel.latlon_spmd import (
                make_latlon_band_wall_multi_pad_body,
            )
            body = make_latlon_band_wall_multi_pad_body(
                mesh, halo=halo,
                south_values=south_values, north_values=north_values,
                n_fields=n)
            return body(*fields)

    fused = os.environ.get("LEGOESM_LATLON_FUSED_HALO", "1") != "0"
    if get_halo_backend() == "mpi" and fused:
        from legoesm.parallel.latlon_mpi import (
            LatLonBandLayout,
            pad_with_pole_bc_lat_multi_mpi,
        )
        topology = get_mpi_topology()
        if isinstance(topology, LatLonBandLayout):
            return pad_with_pole_bc_lat_multi_mpi(
                fields, topology, halo=halo,
                south_values=south_values, north_values=north_values,
            )
    # Local backend / non-latlon topology / 2-D pencil / fused-off:
    # per-field pads (bit-identical semantics; under band MPI this is the
    # legacy one-sendrecv-pair-per-field schedule).  The 2-D pencil routes
    # HERE on purpose — the fused path above is keyed to LatLonBandLayout,
    # so each field re-enters ``pad_with_pole_bc_lat`` and takes its
    # lat-only ``pad_with_pole_bc_lat_2d`` branch (fusing the 2-D lat
    # sendrecv is a later perf increment, not a correctness gap).
    return tuple(
        pad_with_pole_bc_lat(
            f, halo=halo,
            south_value=south_values[i], north_value=north_values[i],
        )
        for i, f in enumerate(fields)
    )


# ============================================================================
# Wide-halo band widening (opt-in wide-halo split-explicit barotropic)
# ============================================================================

def band_pole_flags():
    """(south_is_pole, north_is_pole) for the ACTIVE lat-band backend.

    Booleans may be Python bools (local / MPI band — static at trace time)
    or traced scalars (SPMD: derived from ``lax.axis_index`` inside the
    shard_map body, where a rank-static branch is impossible because the
    body is one uniform program).  Callers must therefore consume them with
    ``jnp.where``-style selects, never Python ``if``.

    Local backend: a single band owns both physical poles → (True, True).
    MPI band layout: pole ownership is ``south_rank/north_rank is None``.
    2-D pencil: pole ownership of the proc-row (lat) axis.
    """
    mesh = _spmd_lat_mesh()
    if mesh is not None:
        import jax

        idx = jax.lax.axis_index("lat")
        n_bands = mesh.shape["lat"]
        return idx == 0, idx == n_bands - 1
    from legoesm.grids.halo import get_halo_backend, get_mpi_topology
    if get_halo_backend() == "mpi":
        from legoesm.parallel.latlon_mpi import (
            LatLon2DLayout,
            LatLonBandLayout,
        )
        topology = get_mpi_topology()
        if isinstance(topology, (LatLonBandLayout, LatLon2DLayout)):
            return topology.south_rank is None, topology.north_rank is None
    return True, True


def _clamp_pole_pad_rows(x_ext: jnp.ndarray, halo: int,
                         south_is_pole, north_is_pole) -> jnp.ndarray:
    """Overwrite beyond-pole pad rows with the nearest PHYSICAL row.

    The wide exchange fills pole-side pad rows with the wall constant (0).
    Metric arrays (area, dx, dy) must stay FINITE and non-zero there — the
    interior operators divide by them, and ``0/0 → NaN`` would leak through
    a masked row into the owned region via ``min``/flux stencils (NaN is not
    absorbed by ``* mask``).  Values are irrelevant (the rows are permanently
    land-masked); finiteness is the contract.

    Uniform ``jnp.where`` select so the SPMD (traced pole flags) and the
    static local/MPI cases share one code path.
    """
    n_ext = x_ext.shape[0]
    idx = jnp.arange(n_ext)
    shape_tail = (1,) * (x_ext.ndim - 1)
    south_sel = ((idx < halo).reshape((n_ext,) + shape_tail)
                 & jnp.asarray(south_is_pole))
    north_sel = ((idx >= n_ext - halo).reshape((n_ext,) + shape_tail)
                 & jnp.asarray(north_is_pole))
    south_row = x_ext[halo][None]
    north_row = x_ext[n_ext - halo - 1][None]
    out = jnp.where(south_sel, south_row, x_ext)
    return jnp.where(north_sel, north_row, out)


def widen_band_cell_fields(fields, halo: int, *, clamp_poles: bool = False):
    """Widen cell-row (leading dim ``n_lat``) fields by ``halo`` rows/side.

    ONE fused exchange under the MPI band backend
    (:func:`pad_with_pole_bc_lat_multi`); wall-zero at physical poles,
    optionally clamped to the nearest physical row (metric arrays — see
    :func:`_clamp_pole_pad_rows`).  Applies to T-point AND u-point fields
    (both carry one row per cell).
    """
    padded = pad_with_pole_bc_lat_multi(fields, halo=halo)
    if not clamp_poles:
        return padded
    south_is_pole, north_is_pole = band_pole_flags()
    return tuple(
        _clamp_pole_pad_rows(p, halo, south_is_pole, north_is_pole)
        for p in padded
    )


def widen_band_vface_fields(fields, halo: int, *, clamp_poles: bool = False):
    """Widen v-face (leading dim ``n_lat+1``) fields by ``halo`` rows/side.

    A band's v array carries one duplicated boundary face, so faces cannot
    be exchanged like cell rows.  Trick: drop the top face — ``v[:-1]`` is
    exactly one "south face" per cell row — exchange THAT as a cell field
    with ``halo+1`` rows, and re-slice: ``padded[1:]`` are the south faces
    of the ``n+2*halo`` extended cells plus the top face of the northmost
    extended cell, i.e. the ``(n + 2*halo) + 1`` faces of the extended
    band.  Value-identical to the serial extended domain; at physical poles
    the wall constant fills beyond-pole faces (repaired to the nearest
    physical face for metric arrays via ``clamp_poles``).

    The band's OWN TOP face (a value the ``[:-1]`` drop discarded) is
    restored verbatim at extended index ``halo + n``: on an interior band
    the exchange already delivered the neighbour's bit-equal copy of that
    shared face, but on the NORTH-POLE band the exchange fills it with the
    wall constant — zeroing a real metric row (``dy_v``/``area_q``/``f_v``
    at the pole face), which poisons every division at that face.  The
    restore is unconditional (bit-neutral on interior bands), so it stays
    uniform under SPMD.
    """
    interiors = tuple(f[:-1] for f in fields)
    padded = pad_with_pole_bc_lat_multi(interiors, halo=halo + 1)
    out = tuple(
        p[1:].at[halo + f.shape[0] - 1].set(f[-1])
        for p, f in zip(padded, fields)
    )
    if not clamp_poles:
        return out
    south_is_pole, north_is_pole = band_pole_flags()
    return tuple(
        _clamp_pole_pad_rows(p, halo, south_is_pole, north_is_pole)
        for p in out
    )


def widen_cgrid_geometry_band(geom, halo: int):
    """Extended-band twin of ``slice_cgrid_geometry_to_band``: widen a
    band-local :class:`~legoesm.grids.latlon.LatLonCGridGeometry` by
    ``halo`` ghost rows per side via the ACTIVE halo backend.

    Field rules mirror the slicer (stagger-aware):

    * T-/u-point metrics + 1-D lat arrays → cell-row widening;
    * v-/q-point metrics → v-face widening;
    * ``lon``/``dlon``/``dlat``/``radius``/``total_area`` unchanged
      (``total_area`` stays the GLOBAL denominator);
    * ``n_lat`` → ``n_lat + 2*halo`` (static Python int);
    * ``fold`` must be inactive/non-local — the wide-halo path refuses
      tripolar folds (caller-validated; this helper asserts).

    ALL metric rows are pole-clamped (finite beyond-pole values): the
    extended rows are permanently land-masked, so their values never enter
    the owned region, but a zero metric would create ``0/0 = NaN`` that
    masks do NOT absorb.

    Traced (per-step) op: two fused exchanges (cell group + v-face group).
    """
    # Refuse on an ACTIVE fold (``is_active`` is rank-CONSISTENT on sliced
    # bands — unlike ``fold_is_local``, which is True only on the north band
    # and would raise on one rank while the others proceed into a fused
    # collective: a deadlock, not an error).
    fold = getattr(geom, "fold", None)
    if fold is not None and getattr(fold, "is_active", False):
        raise NotImplementedError(
            "widen_cgrid_geometry_band: tripolar north fold is not "
            "supported by the wide-halo barotropic path (the fold row "
            "needs a permuted, sign-flipped wide exchange — follow-up)."
        )
    cell_names = ("lat_T", "lon_T", "dx_T", "dy_T", "area_T", "dx_u",
                  "dy_u", "f_T", "f_u", "cos_alpha_u", "sin_alpha_u",
                  "cos_lat", "sin_lat", "lat")
    # The partial-periodic seam wall is a per-lat-row (n_lat,) profile —
    # widen it as a cell-row field so it stays aligned with n_lat under the
    # wide-halo barotropic path.  Ghost rows are permanently land-masked, so
    # the pole-clamped ghost value never enters the owned region.  Omitted
    # when unset (None), keeping non-seam grids byte-identical.
    if getattr(geom, "seam_wall_rows", None) is not None:
        cell_names = cell_names + ("seam_wall_rows",)
    vface_names = ("dx_v", "dy_v", "area_q", "f_v", "cos_alpha_v",
                   "sin_alpha_v", "cos_lat_v")
    # lat_v rides the same v-face axis; omitted when unset (None) so grids
    # without a face array stay byte-identical.
    if getattr(geom, "lat_v", None) is not None:
        vface_names = vface_names + ("lat_v",)
    cell_wide = widen_band_cell_fields(
        tuple(getattr(geom, n) for n in cell_names), halo, clamp_poles=True)
    vface_wide = widen_band_vface_fields(
        tuple(getattr(geom, n) for n in vface_names), halo, clamp_poles=True)
    updates = dict(zip(cell_names, cell_wide))
    updates.update(zip(vface_names, vface_wide))
    updates["n_lat"] = int(geom.n_lat) + 2 * halo
    return geom._replace(**updates)
