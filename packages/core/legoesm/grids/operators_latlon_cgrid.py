"""Shared lat-lon C-grid differential operators.

The finite-difference C-grid stencils (gradient / divergence / curl / Laplacian /
vector-Laplacian / cell<->face interpolation) used by BOTH the ocean lat-lon
C-grid dynamics AND the atmosphere lat-lon C-grid dycores.  They live in
``grids`` (the shared substrate) so the two components share one implementation
without an atmosphere->ocean import — they were historically defined in
``ocean.dynamics.latlon_cgrid_operators`` and moved here verbatim.

Pure grid operators: they depend only on a ``LatLonGrid`` (its dx/dy/masks) and
jax; the ocean-specific physics (Smagorinsky/Leith viscosity, SMC03 pressure
gradient, Coriolis, strain/stress) stays in ``ocean.dynamics.latlon_cgrid_operators``,
which imports these operators from here.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants  # noqa: F401 (kept for parity with the source module)
from legoesm.grids.latlon import LatLonGrid  # noqa: F401 — type compat


def pad_ns_zero(interior: jnp.ndarray) -> jnp.ndarray:
    """Zero-pad south and north rows (wall BC), backend-dispatched.

    Local backend
        Pads ``interior`` with zeros at both lat ends (the historical
        single-rank wall BC).  Bit-identical to the previous
        implementation.

    MPI backend (lat-lon band layout)
        Pole-touching ranks pad their pole side with zero (wall BC at
        the actual pole); interior partition cuts MPI-sendrecv with
        the neighbour rank so the gradient / divergence stencil sees
        continuous data across the cut.  Routes through
        :func:`legoesm.grids.halo_latlon.pad_with_pole_bc_lat` which
        in turn delegates to
        :func:`legoesm.parallel.latlon_mpi.pad_with_pole_bc_lat_mpi`.

    Parameters
    ----------
    interior : (..., n_interior, n_lon, ...) — missing the first and last
        rows along the latitude axis (axis 0 for 2D/3D fields).

    Returns
    -------
    padded : (..., n_interior+2, n_lon, ...)
    """
    # Deferred import to avoid cycles — ocean.dynamics is imported by
    # many grid-related modules.
    from legoesm.grids.halo_latlon import pad_with_pole_bc_lat
    return pad_with_pole_bc_lat(
        interior, halo=1, south_value=0.0, north_value=0.0,
    )


def pad_ns_zero_multi(*fields: jnp.ndarray) -> tuple:
    """Batched :func:`pad_ns_zero` for independent same-``n_lat`` fields.

    Value-identical to ``tuple(pad_ns_zero(f) for f in fields)``; under
    the MPI lat-lon band backend the interior partition cuts are
    exchanged in ONE fused sendrecv pair per cut per dtype group instead
    of one pair per field (see
    :func:`legoesm.grids.halo_latlon.pad_with_pole_bc_lat_multi`).
    Use for clusters of pads at the same dataflow level (no data
    dependency between the fields).
    """
    from legoesm.grids.halo_latlon import pad_with_pole_bc_lat_multi
    return pad_with_pole_bc_lat_multi(fields, halo=1)


def _vface_cos_lat_core(grid) -> jnp.ndarray:
    """Raw ``cos(lat_v)`` at the ``n_lat+1`` v-face midpoints (#515 consolidation).

    The single source for the regular-branch v-face zonal-metric cosine shared by
    ``divergence_cgrid`` and ``gradient_curl_to_v``: pad ``grid.lat`` with the
    halo-aware pole/cut BC (``pad_with_pole_bc_lat`` — at an interior MPI band cut
    the ghost row is the neighbour rank's true edge latitude via the AD-safe
    sendrecv), take the v-face midpoint latitude, return its cosine.  Callers apply
    their OWN pole step (``zero_polar_lat_ends`` vs a ``1e-30`` floor) and the
    ``R*dlon`` scaling.  Kept on the STORED ``grid.lat`` dtype (NOT
    ``result_type``-cast — unlike the ocean ``vface_zonal_cos_lat``) so the core
    path stays byte-identical to the former inline recompute.
    """
    from legoesm.grids.halo_latlon import pad_with_pole_bc_lat
    lat_pad = pad_with_pole_bc_lat(
        grid.lat, halo=1, south_value=0.0, north_value=0.0,
    )
    lat_v = 0.5 * (lat_pad[:-1] + lat_pad[1:])  # (n_lat+1,)
    return jnp.cos(lat_v)


def is_tripolar(grid) -> bool:
    """Return True if ``grid`` carries an active tripolar fold descriptor.

    Replaces the legacy ``hasattr(grid, "<metric>") and grid.dlat == 0.0``
    sentinel dispatch.  Tripolar geometries always carry a ``fold`` field
    with ``is_active=True``; regular/Mercator ``LatLonCGridGeometry`` have
    ``is_active=False``; the old ``LatLonGrid`` lacks the field entirely.

    Under MPI latitude-band decomposition, ``is_tripolar`` returns True on
    ALL ranks of a tripolar run (the fold is active on every rank, even
    though the fold seam is physically present only on the northernmost
    rank).  Use :func:`fold_is_local` to test whether the fold seam is
    locally present for fold-specific computations.
    """
    fold = getattr(grid, "fold", None)
    return fold is not None and bool(fold.is_active)


def fold_is_local(grid) -> bool:
    """Return True if the tripolar fold seam is physically local to this rank.

    Under MPI decomposition, ``is_tripolar(grid)`` is True on all ranks of
    a tripolar run, but the fold seam (``fold_j >= 0``) exists only on the
    northernmost rank.  Non-northernmost ranks carry a sentinel fold with
    ``fold_j == -1`` and ``cap_j == -1`` to indicate "fold exists but is
    not local."  This function checks that the fold is both active AND
    locally present, which is the correct guard for fold-specific
    computations (fold-partner lookups, fold-permutation padding, etc.).
    """
    fold = getattr(grid, "fold", None)
    return fold is not None and bool(fold.is_active) and fold.fold_j >= 0


def _has_2d_vface_metric(grid) -> bool:
    """True if ``grid`` carries an explicit 2D stored v-face zonal metric.

    Rich ``LatLonCGridGeometry`` (tripolar, spherical/Mercator, beta-plane)
    builds ``dx_v`` of shape ``(n_lat+1, n_lon)`` at construction; the lean
    ``LatLonGrid`` lacks the field entirely.
    """
    dxv = getattr(grid, "dx_v", None)
    return dxv is not None and getattr(dxv, "ndim", 0) == 2


def reads_stored_vface_metric(grid) -> bool:
    """True if the v-face zonal length metric must be READ from the grid's
    stored ``dx_v`` rather than recomputed as ``R*cos(grid.lat_v)*dlon``.

    Generalizes :func:`is_tripolar` (#514/#515): the recompute reconstructs the
    metric from ``cos(grid.lat)``, which is WRONG on a Cartesian **beta-plane**
    (whose stored metric is the uniform ``dx_m`` but whose pseudo-lat is a
    nonzero ``y_c/radius``) and merely redundant on a spherical rich geometry
    (whose stored ``dx_v`` is bit-identical to the recompute in the core).

    Reads stored when the grid carries an explicit 2D ``dx_v`` (any rich
    geometry) EXCEPT under a meridionally-periodic (y-reentrant) topology,
    where the pole v-faces must be wrap-padded (recomputed via
    ``pad_with_pole_bc_lat``), not the stored closed-domain pole-zeros.
    Tripolar always reads stored (its fold metric is never recomputable).

    Resolves statically: ``grid`` is a trace-time constant and
    ``get_meridionally_periodic()`` is a concrete module bool, so the calling
    ``if`` is a compile-time branch (the sanctioned feature-gating pattern),
    never traced control flow.
    """
    if is_tripolar(grid):
        return True
    from legoesm.grids.halo_latlon import get_meridionally_periodic

    return _has_2d_vface_metric(grid) and not get_meridionally_periodic()


def pad_ns_scalar(interior: jnp.ndarray, grid) -> jnp.ndarray:
    """Pad south/north rows for a v-face or vertex scalar quantity.

    On regular lat-lon: zero-pad (wall BC).
    On tripolar (fold.is_active): south = zero, north = fold-reflected.

    Under MPI, all ranks call ``pad_ns_zero`` first (ensuring consistent
    MPI sendrecv call counts), then the northernmost rank replaces the
    north ghost row with fold-permuted data via :func:`fold_is_local`.

    Parameters
    ----------
    interior : (n_lat-1, n_lon, ...) — interior v-face or vertex rows.
    grid : LatLonGrid or LatLonCGridGeometry.

    Returns
    -------
    padded : (n_lat+1, n_lon, ...)
    """
    padded = pad_ns_zero(interior)
    fold = getattr(grid, "fold", None)
    nmask = north_fold_mask(grid)
    if fold_is_local(grid) or nmask is not None:
        # Fold: last interior row, i-reversed via perm_T (scalar sign +1).
        # fold_row handles the n_lon+1 wrap column (vertex / u-face fields).
        north = fold_row(interior[-1:], fold.perm_T, 1.0, fold.perm_T.shape[0])
        padded = apply_north_fold(padded, north, grid, north_mask=nmask)
    return padded


def fold_row(last_row, perm, sign, n_lon):
    """Apply fold permutation with optional sign flip to one row."""
    n_cols = last_row.shape[1]
    if n_cols == n_lon:
        return sign * last_row[:, perm]
    else:
        core = sign * last_row[:, :n_lon][:, perm]
        return jnp.concatenate([core, core[:, 0:1]], axis=1)


def north_fold_mask(grid):
    """Traced north-band scalar bool for the tripolar fold under SPMD, else None.

    Companion to :func:`fold_is_local`.  ``fold_is_local`` is the STATIC
    (serial / MPI) "this rank owns the fold seam" test used in a Python ``if``;
    under the lat-band SPMD backend ONE ``shard_map`` trace runs on every band,
    so ``fold_is_local`` is uniformly False (the slicer sets ``fold_j=-1`` on
    every band) and the seam must instead be selected DATA-dependently on the
    north band (``axis_index("lat") == N-1``).  This returns that traced mask
    when the SPMD backend is armed AND ``grid.fold`` is active, else ``None`` so
    the caller keeps its existing ``if fold_is_local`` path unchanged
    (serial / MPI / regular grid byte-for-byte identical).

    The band geometry preserves ``grid.fold.perm_T``/``perm_v``/signs (the
    slicer only zeroes ``fold_j``/``cap_j``), so the caller computes the fold
    row from ``grid.fold`` exactly as in the serial path."""
    fold = getattr(grid, "fold", None)
    if fold is None or not bool(getattr(fold, "is_active", False)):
        return None
    # Function-scope import: core/grids must not import parallel/ at module top.
    from legoesm.parallel.latlon_spmd import spmd_pole_end_masks
    pm = spmd_pole_end_masks()
    return None if pm is None else pm[1]


def apply_north_fold(padded, north_row, grid, *, north_mask=None):
    """Write the tripolar fold-partner row into ``padded``'s north (last) lat-row.

    Serial / MPI northernmost rank (:func:`fold_is_local`): overwrite the row
    (the historical path).  Lat-band SPMD: select it on the north band only
    (``north_mask`` from :func:`north_fold_mask`) via ``jnp.where`` — the single
    shard_map trace runs on every band, so a static overwrite would fold every
    band's interior cut.  Non-seam (regular grid / interior MPI rank / no mask):
    ``padded`` unchanged.

    padded    : (n_lat+1, n_cols, ...) wall-padded field (:func:`pad_ns_zero`).
    north_row : (1, n_cols, ...) fold-partner row (:func:`fold_row`)."""
    if fold_is_local(grid):
        return jnp.concatenate([padded[:-1], north_row], axis=0)
    if north_mask is None:
        north_mask = north_fold_mask(grid)
    if north_mask is not None:
        return padded.at[-1].set(jnp.where(north_mask, north_row[0], padded[-1]))
    return padded


def pad_ns_vector_v(interior: jnp.ndarray, grid) -> jnp.ndarray:
    """Pad south/north for a v-component at v-face latitudes.

    On regular lat-lon: zero-pad.
    On tripolar: south = zero, north = fold-reflected with sign flip.

    Under MPI, all ranks call ``pad_ns_zero`` first (consistent MPI call
    counts), then the northernmost rank applies the fold correction.
    """
    padded = pad_ns_zero(interior)
    fold = getattr(grid, "fold", None)
    nmask = north_fold_mask(grid)
    if fold_is_local(grid) or nmask is not None:
        north = fold_row(interior[-1:], fold.perm_v, fold.vector_sign_v,
                         fold.perm_v.shape[0])
        padded = apply_north_fold(padded, north, grid, north_mask=nmask)
    return padded


def pad_lon_cgrid(f: jnp.ndarray, halo: int = 1) -> jnp.ndarray:
    """Periodic LONGITUDE halo for a C-grid field, backend-dispatched.

    Adds ``halo`` ghost columns on each lon side so a compact zonal stencil
    spans longitude partition cuts.  Backend dispatch:

    * local / band / SPMD-lat / non-2-D MPI — every rank owns the full
      longitude circle, so the wrap is LOCAL: ``jnp.pad(mode="wrap")``.
    * 2-D pencil (``LatLon2DLayout``) — longitude is split, so the wrap
      becomes an MPI ring exchange with the W/E neighbour
      (:func:`legoesm.parallel.latlon_mpi.exchange_halo_lon`).

    BIT-IDENTICAL at ``proc_lon == 1`` (``exchange_halo_lon``'s single-member
    ring is the same local wrap), so the cell→face / vertex operators that
    pad-then-stencil through this helper stay byte-for-byte unchanged on the
    serial / band / SPMD paths and only gain the true neighbour columns under
    a genuine longitude split.  AD-safe (the exchange uses the shared
    sendrecv VJP).  ``f`` may be 2-D ``(n_lat, n_lon[_local], ...)`` or 3-D;
    the lon axis is axis 1.
    """
    if halo <= 0:
        return f
    from legoesm.grids.halo import get_halo_backend, get_mpi_topology
    if get_halo_backend() == "mpi":
        topology = get_mpi_topology()
        from legoesm.parallel.latlon_mpi import (
            LatLon2DLayout, exchange_halo_lon,
        )
        if isinstance(topology, LatLon2DLayout):
            return exchange_halo_lon(
                f, topology.west_rank, topology.east_rank,
                topology.rank, halo=halo,
            )
    # Local periodic wrap (lon = axis 1); single Pad HLO.
    pad = [(0, 0)] * f.ndim
    pad[1] = (halo, halo)
    return jnp.pad(f, tuple(pad), mode="wrap")


def interp_cell_to_uface(f: jnp.ndarray) -> jnp.ndarray:
    """Interpolate a cell-center field to u-face (lon interface) positions.

    Simple average of the two cells sharing each lon face.
    Periodic in longitude: face n_lon wraps to face 0.

    Parameters
    ----------
    f : (n_lat, n_lon, ...) at cell centers.

    Returns
    -------
    f_u : (n_lat, n_lon+1, ...) at u-faces.
    """
    # Pad-then-average: a lon halo (local wrap, or the 2-D ring exchange) +
    # the 2-pt face average over the padded cells.  Bit-identical to the
    # former ``0.5*(roll(f,1)+f)`` + wrap-column concat at proc_lon==1, and
    # spans lon partition cuts under a 2-D split.  Output face j = mean of
    # the two cells sharing it; n_lon+1 faces (the last the periodic closure).
    f_pad = pad_lon_cgrid(f, halo=1)
    return 0.5 * (f_pad[:, :-1] + f_pad[:, 1:])


def interp_cell_to_vface(f: jnp.ndarray, grid=None) -> jnp.ndarray:
    """Interpolate a cell-center field to v-face (lat interface) positions.

    Interior faces: average of adjacent cells.

    Boundary faces depend on ``grid``:

    - ``grid is None`` (legacy default): south/north pole faces copy the
      adjacent cell value.  The pole value is numerically inert since
      ``v = 0`` at the wall.  Bit-exact backwards-compat.
    - ``grid`` provided: use :func:`pad_ns_scalar` — south = 0 and north
      = 0 (wall) on regular lat-lon, or north = fold-reflected on a
      tripolar grid where the north boundary is an active fold rather
      than a wall.  This is the form needed when the v-face value at the
      north fold is physically meaningful (e.g. interpolating a vertical
      velocity for momentum advection on eORCA1).

    Parameters
    ----------
    f : (n_lat, n_lon, ...) at cell centers.
    grid : optional LatLonGrid or LatLonCGridGeometry.

    Returns
    -------
    f_v : (n_lat+1, n_lon, ...) at v-faces.
    """
    f_v_interior = 0.5 * (f[:-1] + f[1:])  # (n_lat-1, ...)
    if grid is not None:
        return pad_ns_scalar(f_v_interior, grid)
    return jnp.concatenate([f[0:1], f_v_interior, f[-1:]], axis=0)


def get_band_mpi_cut_layout():
    """Return the armed band layout iff it has an interior partition cut.

    Static dispatch predicate shared by the band-aware operator
    variants (:func:`interp_cell_to_vface_halo`, the
    ``absolute_vorticity_coriolis`` band branch): returns the active
    :class:`~legoesm.parallel.latlon_mpi.LatLonBandLayout` when the
    MPI halo backend is armed with one AND at least one of the rank's
    lat ends is an interior cut (``south_rank``/``north_rank`` not
    ``None``); returns ``None`` for the local backend, a non-band MPI
    topology (e.g. cubed-sphere ``CommTopology``), or a single-rank
    band (both ends are true poles — the serial conventions are
    already exact there).

    Evaluated at trace time on Python globals — callers branch on the
    result with plain ``if`` (static, never traced).
    """
    from legoesm.grids.halo import get_halo_backend, get_mpi_topology
    if get_halo_backend() != "mpi":
        return None
    topology = get_mpi_topology()
    from legoesm.parallel.latlon_mpi import (
        LatLon2DLayout, LatLonBandLayout,
    )
    # Band AND 2-D pencil expose the same pole-terminated lat-LINE
    # semantics (``south_rank``/``north_rank is None`` at the physical
    # poles), so the pole-touch test is identical.  Recognising the 2-D
    # layout here is what makes ``lat_ends_are_poles`` /
    # ``interp_cell_to_vface_halo`` correct on a ``proc_lat>1`` pencil rank
    # — without it an interior lat-cut rank would clamp its band edge to a
    # physical pole (e.g. curl_vertex's sin clamp), corrupting metrics.
    if isinstance(topology, (LatLonBandLayout, LatLon2DLayout)) and (
        topology.south_rank is not None
        or topology.north_rank is not None
    ):
        return topology
    return None


def lat_ends_are_poles() -> tuple[bool, bool]:
    """``(south_is_pole, north_is_pole)`` for the active halo backend.

    Static Python control flow at trace time — mirrors the dispatch of
    :func:`legoesm.grids.halo_latlon.zero_polar_lat_ends`: under the
    local backend both latitude ends of a rank's arrays are physical
    poles; under latitude-band MPI only the pole-touching ranks' outer
    ends are (``south_rank is None`` / ``north_rank is None``), while
    interior partition cuts are real interior rows.  Shared by
    ``tvd_to_v_points`` (ocean) and any operator that must restore a
    serial pole convention at physical ends only — ONE implementation,
    delegating to :func:`get_band_mpi_cut_layout`.
    """
    band = get_band_mpi_cut_layout()
    if band is None:
        return True, True
    return band.south_rank is None, band.north_rank is None


def interp_cell_to_vface_halo(
    f: jnp.ndarray, *, f_pad: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Cell→v-face interpolation with MPI-band-correct interior cut faces.

    Backend-dispatched twin of :func:`interp_cell_to_vface` (legacy
    ``grid=None`` convention) for fields that live on a latitude band
    under MPI.  The legacy function copies the adjacent cell row onto
    the two end faces of the array — correct at a true pole (the value
    is numerically inert behind the ``v = 0`` wall), but WRONG at an
    interior partition cut, where the end face is a real interior
    v-face whose serial value is the average ``0.5 * (f[j-1] + f[j])``
    spanning the cut.

    Dispatch (static Python control flow at trace time — never traced):

    * **Local backend, non-band MPI topology, or single-rank band**
      (``south_rank is None and north_rank is None``) — delegates to
      :func:`interp_cell_to_vface` unchanged.  Bit-identical serial
      behaviour, including the pole edge-copy convention
      ``f_v[0] = f[0]``, ``f_v[-1] = f[-1]``.
    * **Band MPI with at least one interior cut** — pads ``f`` by one
      latitude row through the backend-dispatched
      :func:`legoesm.grids.halo_latlon.pad_with_pole_bc_lat` (interior
      cuts receive the neighbour rank's row via the AD-safe
      ``_sendrecv_vjp`` sendrecv; a pole-touching end receives a
      constant ghost that never reaches the output, see below),
      computes every face with the interior average on the padded
      array, then restores the legacy edge copy at any pole-touching
      end so pole-touching ranks reproduce serial bit-for-bit.

    The u-face twin needs no halo variant: ``interp_cell_to_uface``
    averages along longitude only, and every band rank owns the full
    periodic lon axis, so its output is row-by-row identical to serial
    under latitude-band decomposition.

    Parameters
    ----------
    f : (n_lat_local, n_lon, ...) at cell centers (ndim 2 or 3).

    Returns
    -------
    f_v : (n_lat_local + 1, n_lon, ...) at v-faces.
    """
    band = get_band_mpi_cut_layout()
    # Single-program SPMD twin (the lat-band shard_map backend, for which
    # ``get_band_mpi_cut_layout()`` is None): the cross-cut interior faces need
    # the same neighbour-row average, but the pole edge-copy must be restored
    # only at the PHYSICAL pole bands, selected DATA-dependently per band.
    # ``None`` for serial/MPI/cube (those keep their static branch — additive).
    from legoesm.parallel.latlon_spmd import spmd_pole_end_masks
    spmd_pm = spmd_pole_end_masks()
    if band is not None or spmd_pm is not None:
        from legoesm.grids.halo_latlon import pad_with_pole_bc_lat
        # Interior cuts: neighbour row via the backend-aware pad (AD-safe MPI
        # sendrecv / SPMD lat-band ppermute).  A pole-touching end gets a
        # constant-0 ghost row here that is immediately overridden by the
        # legacy edge copy below, so the constant never reaches the output.
        # ``f_pad`` may be supplied pre-padded by a caller that fused this
        # exchange with others (must be exactly this pad call's output).
        if f_pad is None:
            f_pad = pad_with_pole_bc_lat(
                f, halo=1, south_value=0.0, north_value=0.0,
            )
        f_v = 0.5 * (f_pad[:-1] + f_pad[1:])  # (n_lat_local+1, ...)
        if spmd_pm is not None:
            # SPMD: restore the legacy pole edge-copy at the south / north pole
            # bands only (interior cuts keep the cross-cut average). south then
            # north, sequenced so a single band reproduces serial at both ends.
            south_m, north_m = spmd_pm
            f_v = jnp.where(
                south_m, jnp.concatenate([f[0:1], f_v[1:]], axis=0), f_v)
            f_v = jnp.where(
                north_m, jnp.concatenate([f_v[:-1], f[-1:]], axis=0), f_v)
        else:
            # MPI: static per-rank pole answer (None ⟺ this rank owns the pole).
            if band.south_rank is None:
                f_v = jnp.concatenate([f[0:1], f_v[1:]], axis=0)
            if band.north_rank is None:
                f_v = jnp.concatenate([f_v[:-1], f[-1:]], axis=0)
        return f_v
    return interp_cell_to_vface(f)


def interp_u_to_vface_4pt(u: jnp.ndarray, grid) -> jnp.ndarray:
    """Sadourny 4-point average of a u-point field to v-points.

    ``u_at_v[i, j] = 0.25 * (u[i-1, j] + u[i-1, j+1] + u[i, j] + u[i, j+1])``
    — the Coriolis-term staggering average used by the Matsuno
    forward-backward split (baroclinic and barotropic).

    Cell-pad-first (PR357 Bug-2 pattern): ``u`` is padded by one
    latitude row through the backend-dispatched
    :func:`legoesm.grids.halo_latlon.pad_with_pole_bc_lat` (local
    ``jnp.pad`` / AD-safe MPI sendrecv at interior band cuts), so ALL
    ``n_lat+1`` local v-faces — including partition-cut faces — average
    the true u rows.  The historical pattern (interior faces then
    ``pad_ns_vector_u`` refill) delivered the neighbour's ADJACENT face
    value at a cut — one face row off.  Serial bit-identity: interior
    faces average exactly the previous operands in the previous order;
    physical pole faces are zeroed (the old ``pad_ns_zero`` ends) and
    the tripolar fold row is ``vector_sign_u * perm_T`` of the last
    interior face row, exactly as ``pad_ns_vector_u`` built it.

    Parameters
    ----------
    u : (n_lat, n_lon+1, ...) at u-points (ndim 2 or 3).
    grid : LatLonGrid or LatLonCGridGeometry (fold-aware).

    Returns
    -------
    u_at_v : (n_lat+1, n_lon, ...) at v-points.
    """
    from legoesm.grids.halo_latlon import (
        pad_with_pole_bc_lat,
        zero_polar_lat_ends,
    )
    u_pad = pad_with_pole_bc_lat(
        u, halo=1, south_value=0.0, north_value=0.0,
    )  # (n_lat+2, n_lon+1, ...)
    u_at_v = 0.25 * (
        u_pad[:-1, :-1] + u_pad[:-1, 1:]
        + u_pad[1:, :-1] + u_pad[1:, 1:]
    )  # (n_lat+1, n_lon, ...)
    u_at_v = zero_polar_lat_ends(u_at_v)
    nmask = north_fold_mask(grid)
    if fold_is_local(grid) or nmask is not None:
        fold = grid.fold
        north = fold_row(
            u_at_v[-2:-1], fold.perm_T, fold.vector_sign_u,
            fold.perm_T.shape[0],
        )
        u_at_v = apply_north_fold(u_at_v, north, grid, north_mask=nmask)
    return u_at_v


def cell_to_cgrid_winds(
    u_cell: jnp.ndarray,
    v_cell: jnp.ndarray,
    grid=None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert cell-centered winds to C-grid face-staggered winds.

    Works for both 2D (n_lat, n_lon) and 3D (n_lat, n_lon, nlev).
    v = 0 at pole walls on regular lat-lon; fold-reflected on tripolar.

    Parameters
    ----------
    u_cell, v_cell : cell-centered wind components.
    grid : optional LatLonGrid or LatLonCGridGeometry.

    Returns
    -------
    u_face : (..., n_lon+1, ...) at lon interfaces.
    v_face : (n_lat+1, ...) at lat interfaces.
    """
    u_face = interp_cell_to_uface(u_cell)
    v_interior = 0.5 * (v_cell[:-1] + v_cell[1:])
    fold = getattr(grid, "fold", None) if grid is not None else None
    # Route to pad_ns_vector_v on ANY active tripole (not just the fold-local
    # serial rank): pad_ns_vector_v itself folds only on the fold-local rank /
    # the SPMD north band, and no-ops to pad_ns_zero elsewhere — so dropping the
    # fold_j>=0 check is serial/MPI byte-identical AND wires the SPMD north fold.
    if fold is not None and fold.is_active:
        v_face = pad_ns_vector_v(v_interior, grid)
    else:
        v_face = pad_ns_zero(v_interior)
    return u_face, v_face


def gradient_x_cgrid(
    f: jnp.ndarray,
    grid: LatLonGrid,
) -> jnp.ndarray:
    """Zonal gradient df/dx at u-points (lon interfaces).

    Compact stencil: (f[i, j+1] - f[i, j]) / dx_u

    Uses periodic wrapping in longitude.

    Parameters
    ----------
    f : array, shape (n_lat, n_lon) or (n_lat, n_lon, nlev)
        Scalar field at cell centers.
    grid : LatLonGrid

    Returns
    -------
    df_dx : array, shape (n_lat, n_lon+1, ...) at u-points.
    """
    # Face j sits between cell (j-1) mod n_lon (west) and cell j (east),
    # matching the divergence convention (cell j: west=face j, east=face j+1).
    # Gradient at face j: (f[j] - f[(j-1) mod n_lon]) / dx
    # Pad-then-diff: a lon halo (local wrap, or the 2-D ring exchange) then
    # the compact zonal difference over the padded cells.  df_full[j] =
    # f[j] - f[j-1] at u-face j; n_lon+1 faces (the last the periodic
    # closure).  ndim-agnostic (lon = axis 1).  Bit-identical to the former
    # ``roll(f,1)`` + wrap-column concat at proc_lon==1, and spans lon
    # partition cuts under a 2-D split.
    f_pad = pad_lon_cgrid(f, halo=1)
    df_full = f_pad[:, 1:] - f_pad[:, :-1]

    # dx at u-point.  On a regular lat-lon grid (dlat > 0), use the
    # legacy 1D path for bit-exact backward compat.  On a tripolar
    # grid (dlat == 0 sentinel), use the pre-computed 2D metric.
    if is_tripolar(grid):
        dx_u = grid.dx_u  # (n_lat, n_lon+1)
        if f.ndim == 2:
            return df_full / dx_u
        else:
            return df_full / dx_u[:, :, jnp.newaxis]
    else:
        dx_u = grid.radius * grid.dlon * grid.cos_lat
        if f.ndim == 2:
            return df_full / dx_u[:, jnp.newaxis]
        else:
            return df_full / dx_u[:, jnp.newaxis, jnp.newaxis]


def gradient_y_cgrid(
    f: jnp.ndarray,
    grid: LatLonGrid,
) -> jnp.ndarray:
    """Meridional gradient df/dy at v-points (lat interfaces).

    Compact stencil: (f[i+1, j] - f[i, j]) / dy_v

    Wall BC: v=0 at poles, so gradient at pole boundaries is zero.

    Parameters
    ----------
    f : array, shape (n_lat, n_lon) or (n_lat, n_lon, nlev)
        Scalar field at cell centers.
    grid : LatLonGrid

    Returns
    -------
    df_dy : array, shape (n_lat+1, n_lon, ...) at v-points.
    """
    # Unified pre-pad-then-compact-stencil for ALL grids (regular,
    # Mercator, tripolar).  The old tripolar path used a rank-local
    # compact-stencil-then-pad that zeroed the south v-face and read only
    # local ``f``, so under MPI it dropped the neighbour band's row at a
    # partition cut (#356 follow-up, PR358).  Pre-padding ``f`` via the
    # backend-dispatched halo (local pole-fold OR MPI sendrecv + boundary
    # pole-fold) makes the compact stencil span partition cuts on every
    # grid; ``zero_polar_lat_ends`` restores the wall BC at the physical
    # poles only, and the north fold-face gradient is replaced with the
    # fold-partner gradient on the rank that owns the seam.
    #
    # Serial bit-exactness: the local backend's pole-fold gives the same
    # interior gradients, ``zero_polar_lat_ends`` zeros the two physical
    # ends (== the historical south=0 / pole pad), and the fold overwrite
    # reproduces the old tripolar north row.
    from legoesm.grids.halo_latlon import (
        get_meridionally_flat,
        pad_halo_latlon,
        pad_halo_latlon_3d,
        zero_polar_lat_ends,
    )
    # Oceananigans `Flat`-y topology: δy ≡ 0 (no meridional gradient ever).
    if get_meridionally_flat():
        n_lat = f.shape[0]
        out_shape = (n_lat + 1,) + f.shape[1:]
        return jnp.zeros(out_shape, dtype=f.dtype)
    if f.ndim == 2:
        f_padded = pad_halo_latlon(f, halo=1)
        # Strip the lon halo — gradient_y only needs the lat halo.
        f_padded = f_padded[:, 1:-1]
    elif f.ndim == 3:
        f_padded = pad_halo_latlon_3d(f, halo=1)
        f_padded = f_padded[:, 1:-1, :]
    else:
        raise ValueError(
            f"gradient_y_cgrid: f.ndim must be 2 or 3, got {f.ndim}"
        )

    # Compact stencil on padded f — gradient at ALL v-faces of the
    # rank-local band, including the partition cuts.
    f_diff = f_padded[1:] - f_padded[:-1]  # (n_lat_v, n_lon[, nlev])

    if is_tripolar(grid):
        # Full 2D dy_v metric (varies in lon on the bipolar cap).  Floor the
        # denominator: a tripole built with ``min_dx_m=0.0`` has an exact
        # zero south pole row, and dividing every f_diff row before
        # zero_polar_lat_ends would compute nonzero/0 = inf/NaN there (poisons
        # jax_debug_nans and AD even though the row is overwritten to zero).
        dy_v = jnp.maximum(grid.dy_v, 1.0e-30)  # (n_lat+1, n_lon)
        bcast = (slice(None), slice(None)) + (jnp.newaxis,) * (f_diff.ndim - 2)
        df_dy = f_diff / dy_v[bcast]
    else:
        # Regular / Mercator: 1D dy over latitude.  Uniform-dlat is a
        # constant (= R * dlat); ``mode='edge'`` extends Mercator's
        # variable dlat by one row (unchanged behaviour).
        dy_h = grid.dy * 0.5                           # (n_lat,)
        dy_h_padded = jnp.pad(dy_h, (1, 1), mode="edge")
        dy_v = 0.5 * (dy_h_padded[1:] + dy_h_padded[:-1])  # (n_lat+1,)
        bcast = (slice(None),) + (jnp.newaxis,) * (f_diff.ndim - 1)
        df_dy = f_diff / dy_v[bcast]

    # Wall BC at the physical pole v-faces (backend-aware: interior
    # partition cuts are left intact).
    df_dy = zero_polar_lat_ends(df_dy)

    # Tripolar north fold seam: replace the north polar v-face gradient
    # with the fold-partner gradient on the rank/band that owns the seam.
    nmask = north_fold_mask(grid)
    if fold_is_local(grid) or nmask is not None:
        fold = grid.fold
        f_partner = f[-1:, fold.perm_T]
        dy_fold = grid.dy_v[-1:]
        if f.ndim == 3:
            dy_fold = dy_fold[:, :, jnp.newaxis]
        dy_fold_safe = jnp.maximum(dy_fold, 1.0e-30)
        df_fold = (f_partner - f[-1:]) / dy_fold_safe
        df_dy = apply_north_fold(df_dy, df_fold, grid, north_mask=nmask)

    return df_dy


def upwind_cell_to_uface(
    f: jnp.ndarray,
    flux_u: jnp.ndarray,
) -> jnp.ndarray:
    """First-order (donor-cell) upwind reconstruction of a cell-center field
    at u-faces, selected by the face-normal flux/velocity sign.

    PROMOTED from ``ocean.dynamics.ocean_pe_latlon_cgrid.upwind_to_u_points``
    (verbatim numerics) so non-ocean components (sea-ice C-grid transport) can
    use it without an ice->ocean layering break; the ocean re-imports this as
    its canonical implementation.

    Parameters
    ----------
    f : array, shape (n_lat, n_lon, ...) at cell centers.
    flux_u : array, shape (n_lat, n_lon+1, ...) at u-faces.
        Sign convention: positive = flow in +j (eastward) direction.

    Returns
    -------
    f_u : array, shape (n_lat, n_lon+1, ...) at u-faces.
    """
    # Face j is between cell (j-1) mod n_lon and cell j.
    # Positive flux => flow from cell j-1 to cell j => upwind is cell j-1.
    # Negative flux => flow from cell j to cell j-1 => upwind is cell j.
    f_left = jnp.roll(f, 1, axis=1)   # f_left[:, j] = f[:, j-1]
    f_right = f                        # f_right[:, j] = f[:, j]

    # Build upwind at interior faces (n_lat, n_lon)
    f_upwind = jnp.where(flux_u[:, :-1] > 0, f_left, f_right)

    # Wrap: face n_lon is the same as face 0 (periodic in longitude)
    if f.ndim >= 3:
        return jnp.concatenate([f_upwind, f_upwind[:, 0:1, :]], axis=1)
    else:
        return jnp.concatenate([f_upwind, f_upwind[:, 0:1]], axis=1)


def upwind_cell_to_vface(
    f: jnp.ndarray,
    flux_v: jnp.ndarray,
    grid=None,
) -> jnp.ndarray:
    """First-order (donor-cell) upwind reconstruction of a cell-center field
    at v-faces, wall-zeroed at physical poles and fold-aware on the tripolar
    north seam.

    PROMOTED from ``ocean.dynamics.ocean_pe_latlon_cgrid.upwind_to_v_points``
    (verbatim numerics — cell-pad-first through the backend-dispatched
    ``pad_with_pole_bc_lat``; pole faces zeroed; tripolar fold row = the
    fold-permuted last interior face row, exactly as ``pad_ns_scalar`` built
    it) so non-ocean components can use it; the ocean re-imports this.

    Parameters
    ----------
    f : array, shape (n_lat, n_lon, ...) at cell centers.
    flux_v : array, shape (n_lat+1, n_lon, ...) at v-faces.
        Sign convention: positive = flow in +i (northward) direction.
    grid : optional LatLonGrid or LatLonCGridGeometry (fold handling).

    Returns
    -------
    f_v : array, shape (n_lat+1, n_lon, ...) at v-faces.
    """
    from legoesm.grids.halo_latlon import (
        pad_with_pole_bc_lat,
        zero_polar_lat_ends,
    )
    # Face i sits between cell i-1 and cell i (m_pad rows i and i+1).
    # Positive flux => flow from cell i-1 to cell i => upwind is cell i-1.
    # Negative flux => flow from cell i to cell i-1 => upwind is cell i.
    f_pad = pad_with_pole_bc_lat(
        f, halo=1, south_value=0.0, north_value=0.0,
    )
    f_south = f_pad[:-1]   # cell i-1 for face i
    f_north = f_pad[1:]    # cell i   for face i
    f_v = jnp.where(flux_v > 0, f_south, f_north)

    # Wall BC at the physical pole faces only (backend-aware), then the
    # tripolar north fold row exactly as pad_ns_scalar produced it.
    f_v = zero_polar_lat_ends(f_v)
    nmask = north_fold_mask(grid)
    if fold_is_local(grid) or nmask is not None:
        north = f_v[-2:-1][:, grid.fold.perm_T]
        f_v = apply_north_fold(f_v, north, grid, north_mask=nmask)
    return f_v


def divergence_cgrid(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: LatLonGrid,
    *,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Conservative FV divergence on the C-grid.

    div = (1/A) * [u_e*dy_e - u_w*dy_w + v_n*dx_n - v_s*dx_s]

    where u_e, u_w are zonal velocity at east/west faces and
    v_n, v_s are meridional velocity at north/south faces.

    Parameters
    ----------
    u : array, shape (n_lat, n_lon+1) or (n_lat, n_lon+1, nlev)
        Zonal velocity at lon interfaces.
    v : array, shape (n_lat+1, n_lon) or (n_lat+1, n_lon, nlev)
        Meridional velocity at lat interfaces.
    grid : LatLonGrid
    u_mask, v_mask : array, optional
        Face masks. Applied before flux computation.

    Returns
    -------
    div : array, shape (n_lat, n_lon) or (n_lat, n_lon, nlev)
    """
    u_eff = u
    v_eff = v
    if u_mask is not None:
        um = u_mask[..., jnp.newaxis] if u.ndim == 3 and u_mask.ndim == 2 else u_mask
        u_eff = u * um
    if v_mask is not None:
        vm = v_mask[..., jnp.newaxis] if v.ndim == 3 and v_mask.ndim == 2 else v_mask
        v_eff = v * vm

    # --- Zonal face length (meridional extent of u-face) ---
    if is_tripolar(grid):
        # Tripolar: use full 2D dy_u.  On a regular lat-lon grid dy_u
        # is constant in longitude, but on the bipolar cap it varies
        # significantly — column-0 extraction is NOT valid.
        face_dy = grid.dy_u  # (n_lat, n_lon+1) — full 2D
    else:
        # Regular or Mercator: variable-dy safe (1D over latitude).
        face_dy = grid.dy * 0.5  # (n_lat,)

    # East face flux - west face flux
    _is_2d_dy = hasattr(face_dy, 'ndim') and face_dy.ndim == 2
    if u.ndim == 2:
        u_east = u_eff[:, 1:]
        u_west = u_eff[:, :-1]
        if _is_2d_dy:
            face_dy_e = face_dy[:, 1:]
            face_dy_w = face_dy[:, :-1]
        elif hasattr(face_dy, 'ndim') and face_dy.ndim == 1:
            face_dy = face_dy[:, jnp.newaxis]
    else:
        u_east = u_eff[:, 1:, :]
        u_west = u_eff[:, :-1, :]
        if _is_2d_dy:
            face_dy_e = face_dy[:, 1:, jnp.newaxis]
            face_dy_w = face_dy[:, :-1, jnp.newaxis]
        elif hasattr(face_dy, 'ndim') and face_dy.ndim == 1:
            face_dy = face_dy[:, jnp.newaxis, jnp.newaxis]

    if _is_2d_dy:
        # Per-face dy: each u-face has its own meridional extent.
        net_zonal = u_east * face_dy_e - u_west * face_dy_w
    else:
        net_zonal = (u_east - u_west) * face_dy  # preserve arithmetic order

    # --- Meridional face length (zonal extent of v-face) ---
    # On a lean lat-lon grid this is the 1D array R*cos(lat_v)*dlon; a rich
    # geometry that carries an explicit 2D stored metric (tripolar, beta-plane,
    # spherical) reads grid.dx_v directly (#514) — the recompute reconstructs
    # cos(grid.lat_v), which is WRONG on a Cartesian beta-plane.
    if reads_stored_vface_metric(grid):
        face_dx = grid.dx_v  # (n_lat+1, n_lon) — 2D stored metric
    else:
        # v-face latitudes at ALL n_lat+1 local faces, cell-pad-first:
        # pad the CELL-CENTRE latitudes by one row through the
        # backend-dispatched pad (at an interior MPI band cut the ghost
        # row is the neighbour rank's true edge cell latitude via the
        # AD-safe sendrecv), then take midpoints.  The previous code
        # padded the INTERIOR-FACE cos array instead, so a cut face
        # received the neighbour's ADJACENT face metric — one face row
        # off — breaking both np>=2 parity and cross-cut flux
        # telescoping.  Wall BC: cos at the physical polar v-faces is
        # exactly 0 (no flux through the pole), which also discards the
        # pole-side constant ghost; bit-identical to the historical
        # jnp.pad(cos_interior, (1, 1)) on the local backend.
        from legoesm.grids.halo_latlon import zero_polar_lat_ends
        cos_lat_v = zero_polar_lat_ends(_vface_cos_lat_core(grid))
        face_dx = grid.radius * cos_lat_v * grid.dlon  # (n_lat+1,)

    # North face flux - south face flux
    if v.ndim == 2:
        v_north = v_eff[1:]
        v_south = v_eff[:-1]
        fd = face_dx
        if fd.ndim == 1:
            net_merid = v_north * fd[1:, jnp.newaxis] - v_south * fd[:-1, jnp.newaxis]
        else:
            net_merid = v_north * fd[1:] - v_south * fd[:-1]
    else:
        v_north = v_eff[1:, :, :]
        v_south = v_eff[:-1, :, :]
        fd = face_dx
        if fd.ndim == 1:
            net_merid = (v_north * fd[1:, jnp.newaxis, jnp.newaxis]
                         - v_south * fd[:-1, jnp.newaxis, jnp.newaxis])
        else:
            net_merid = (v_north * fd[1:, :, jnp.newaxis]
                         - v_south * fd[:-1, :, jnp.newaxis])

    # Oceananigans `Flat`-y topology: the meridional flux divergence is 0
    # (δy(Ay·v) ≡ 0), so η responds only to the zonal transport divergence.
    from legoesm.grids.halo_latlon import get_meridionally_flat
    if get_meridionally_flat():
        net_merid = jnp.zeros_like(net_zonal)

    # Cell area
    area = grid.area  # (n_lat, n_lon)
    if u.ndim == 3:
        area = area[..., jnp.newaxis]

    div = (net_zonal + net_merid) / area
    return div


def laplacian_cgrid(
    f: jnp.ndarray,
    grid: LatLonGrid,
    *,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Scalar Laplacian on C-grid cell centers using compact stencils.

    Computes lap(f) = div(grad(f)) using gradient_x/y_cgrid and
    divergence_cgrid. This is a 5-point Laplacian, not the 2*dx
    centered-difference Laplacian of the A-grid.

    Parameters
    ----------
    f : array, shape (n_lat, n_lon) or (n_lat, n_lon, nlev)
    grid : LatLonGrid
    mask : array, optional

    Returns
    -------
    lap_f : array, same shape as f
    """
    is_3d = f.ndim == 3

    # ``gradient_x_cgrid`` / ``gradient_y_cgrid`` / ``divergence_cgrid``
    # all natively support 3D inputs (they internally branch on ndim
    # for the dx_u / dy_v / cos_lat broadcasting).  Call them directly
    # on 3D — the previous moveaxis + vmap + moveaxis round-trip was
    # redundant.
    grad_x = gradient_x_cgrid(f, grid)  # (n_lat, n_lon+1[, nlev])
    grad_y = gradient_y_cgrid(f, grid)  # (n_lat+1, n_lon[, nlev])

    if mask is not None:
        # Zero gradient at land-ocean boundaries.  u-face j is active iff both
        # adjacent cells (j-1, j) are wet — pad-then-product over the lon halo
        # (local wrap / 2-D ring) so the west cell at j=0 is the true neighbour
        # under a 2-D split.  Bit-identical to ``mask*roll(mask,1)`` + wrap
        # concat at proc_lon==1; n_lon+1 faces (last = periodic closure).
        mask_pad = pad_lon_cgrid(mask, halo=1)
        u_mask = mask_pad[:, 1:] * mask_pad[:, :-1]
        # Boundary: wall BC on regular lat-lon; fold on tripolar.
        v_mask_interior = mask[:-1] * mask[1:]
        v_mask = pad_ns_scalar(v_mask_interior, grid)
        if is_3d:
            grad_x = grad_x * u_mask[..., jnp.newaxis]
            grad_y = grad_y * v_mask[..., jnp.newaxis]
        else:
            grad_x = grad_x * u_mask
            grad_y = grad_y * v_mask

    lap = divergence_cgrid(grad_x, grad_y, grid)

    if mask is not None:
        if is_3d:
            lap = lap * mask[..., jnp.newaxis]
        else:
            lap = lap * mask

    return lap


def curl_vertex_cgrid(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: LatLonGrid,
    *,
    u_ext: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Relative vorticity at vertex (corner) points via circulation integral.

    Vertices sit at the intersection of u-face latitudes and v-face
    longitudes: shape (n_lat+1, n_lon+1), with periodic wrap in
    longitude (column n_lon == column 0).

    Parameters
    ----------
    u : (n_lat, n_lon+1) or (n_lat, n_lon+1, nlev)
    v : (n_lat+1, n_lon) or (n_lat+1, n_lon, nlev)
    grid : LatLonGrid

    Returns
    -------
    zeta : (n_lat+1, n_lon+1) or (n_lat+1, n_lon+1, nlev)
    """
    # Native 2D and 3D — broadcast 1D metric over the trailing level
    # axis when needed.  Eliminates the prior moveaxis + vmap +
    # moveaxis round-trip.
    is_3d = u.ndim == 3

    if is_tripolar(grid):
        # Tripolar: use full 2D per-cell metrics.  On the bipolar cap,
        # metrics vary significantly in BOTH lat and lon — column-0
        # extraction is not valid.
        A_vertex_full = grid.area_q                    # (n_lat+1, n_lon+1)
        dx_cell = grid.dx_T                            # (n_lat, n_lon)
        # dy at each v-face edge of the circulation loop: east and west
        # edges have different dy on the distorted cap.
        dy_v_2d = grid.dy_v                            # (n_lat+1, n_lon)
        _tripolar_curl = True
    else:
        R = grid.radius
        dlon = grid.dlon
        dlat = grid.dlat
        lat = grid.lat
        cos_lat = grid.cos_lat
        # Wall-BC pad of sin_lat: sin(south_pole)=-1, sin(north_pole)=+1.
        # Backend-aware so MPI interior ranks sendrecv from neighbour
        # rather than apply pole BC at the wrong location.
        # Pad the CONCRETE ``lat``, THEN take sin: ``jnp.sin(lat)``
        # inside jit stages to a Tracer, which kept this pad on the
        # traced per-step sendrecv path (census 8459326); the
        # concrete-lat pad constant-folds at trace time instead, and
        # ``jnp.sin`` of that constant folds at XLA compile time.
        # Interior/cut rows are bit-identical (same jnp.sin applied to
        # the same f64 inputs on every rank).  Pole rows are then
        # OVERWRITTEN with the literal ±1.0 the historical path padded
        # (codex 2026-06-11 MAJOR: relying on jnp.sin(±π/2) == ±1.0 is
        # backend-dependent; the literals make it exact by fiat).
        from legoesm.grids.halo_latlon import pad_with_pole_bc_lat
        import math
        lat_ext_q = pad_with_pole_bc_lat(
            lat, halo=1,
            south_value=-math.pi / 2.0, north_value=math.pi / 2.0,
        )
        sin_ext = jnp.sin(lat_ext_q)
        # Restore the EXACT pole sin (±1.0) at PHYSICAL poles.  Under the
        # single-program SPMD backend ``lat_ends_are_poles()`` is (True, True) on
        # every band, so the static ``if`` would clamp every band's INTERIOR cut
        # (the SPMD-blind bug) — select the clamp DATA-dependently per band via
        # ``axis_index`` there (function-scope import: core -> parallel).
        from legoesm.parallel.latlon_spmd import spmd_pole_end_masks
        _spmd_pm = spmd_pole_end_masks()
        sin_ext_s = jnp.concatenate(
            [jnp.full((1,), -1.0, dtype=sin_ext.dtype), sin_ext[1:]])
        sin_ext_n = jnp.concatenate(
            [sin_ext[:-1], jnp.full((1,), 1.0, dtype=sin_ext.dtype)])
        if _spmd_pm is not None:
            south_mask, north_mask = _spmd_pm
            sin_ext = jnp.where(south_mask, sin_ext_s, sin_ext)
            sin_ext = jnp.where(north_mask, sin_ext_n, sin_ext)
        else:
            south_is_pole_q, north_is_pole_q = lat_ends_are_poles()
            if south_is_pole_q:
                sin_ext = sin_ext_s
            if north_is_pole_q:
                sin_ext = sin_ext_n
        A_vertex_full = R**2 * dlon * jnp.abs(sin_ext[1:] - sin_ext[:-1])
        # Prefer the grid's STORED dual-cell area when present (the C-grid ocean
        # geometries carry ``area_q``).  On a spherical lat-lon grid the stored
        # ``area_q`` is computed by the SAME ``R²·dlon·|Δsin(lat)|`` formula
        # (create_latlon_cgrid_geometry: "Matches curl_vertex_cgrid") and agrees
        # with the recompute above to ~1e-7 relative — NOT bit-exact: ``area_q`` is
        # built in float64 while this recompute does ``sin(grid.lat)`` on the stored
        # float32 ``lat``, losing precision in ``Δsin`` near the equator.  Reading
        # ``area_q`` is thus a small ACCURACY IMPROVEMENT on spherical grids (any
        # curl/vorticity regression pinned tighter than ~1e-7 will shift).  On the
        # CARTESIAN beta-plane the recompute COLLAPSES
        # to ~0 in the interior — the pseudo-``lat`` is pinned to 0 for the
        # divergence/gradient metric consistency, so ``Δsin(lat)=0`` and the
        # ``safe_A`` pole guard then divides by 1.0, blowing the vorticity (and the
        # vector Laplacian/biharmonic) up ~1e8×.  The stored ``area_q = dx·dy`` is
        # correct there.  The bare ``LatLonGrid`` (atmos dycore, plane tests) has no
        # ``area_q`` -> recompute unchanged.
        if hasattr(grid, "area_q"):
            A_vertex_full = jnp.abs(grid.area_q[:, 0]).astype(A_vertex_full.dtype)  # noqa: N806
        dy_edge = R * dlat
        _tripolar_curl = False

    # v contribution: circulation from v-edges (east minus west).  Each
    # branch builds the full n_lon+1 vertex-column ``dv_circ_full`` HERE (no
    # shared wrap-column append later), so the regular path can pad-then-diff
    # through the lon halo and span 2-D longitude partition cuts.
    if _tripolar_curl:
        # Tripolar per-face dy.  Local roll + wrap-column append (the
        # tripolar cap's 2-D-lon-split fold is a separate follow-up; this is
        # correct at proc_lon==1 / band, as before).
        v_west = jnp.roll(v, 1, axis=1)
        dy_east = dy_v_2d
        dy_west = jnp.roll(dy_v_2d, 1, axis=1)
        if is_3d:
            dy_east = dy_east[:, :, jnp.newaxis]
            dy_west = dy_west[:, :, jnp.newaxis]
        dv_circ = v * dy_east - v_west * dy_west
        dv_circ_full = jnp.concatenate([dv_circ, dv_circ[:, 0:1]], axis=1)
    else:
        # Meridional edge length at vertex rows: variable-dy safe.
        dy_h = grid.dy * 0.5                              # (n_lat,) cell heights
        dy_edge_interior = 0.5 * (dy_h[1:] + dy_h[:-1])    # (n_lat-1,)
        dy_edge = jnp.pad(dy_edge_interior, (1, 1), mode='edge')  # (n_lat+1,)
        bcast_lat = (slice(None),) + (jnp.newaxis,) * (v.ndim - 1)
        # Pad-then-diff: a lon halo (local wrap / 2-D ring exchange) then the
        # compact vertex difference (v[j]-v[j-1]) over padded v => n_lon+1
        # vertex columns directly.  Bit-identical to ``roll(v,1)`` + wrap-
        # column concat at proc_lon==1; spans lon cuts under a 2-D split.
        v_pad = pad_lon_cgrid(v, halo=1)
        dv_circ_full = (v_pad[:, 1:] - v_pad[:, :-1]) * dy_edge[bcast_lat]

    # u contribution: u[i-1, j]*dx[i-1] - u[i, j]*dx[i].
    # Pad with zeros at poles along the lat axis (axis 0).  Wall BC
    # at the pole; under MPI on an interior rank the same call
    # sendrecv's the neighbour's u row instead (the pole pad fires
    # only at boundary ranks).
    # ``u_ext`` may be supplied pre-padded by the caller (the atm
    # tendency fuses this pad with its other entry-level lat pads and
    # the absolute-vorticity 4-pt average reuses the SAME padded u —
    # census probe 8460424 showed u padded twice per RK stage).  Must
    # be exactly pad_with_pole_bc_lat(u, halo=1, 0, 0).
    from legoesm.grids.halo_latlon import pad_with_pole_bc_lat
    if u_ext is None:
        u_ext = pad_with_pole_bc_lat(
            u, halo=1, south_value=0.0, north_value=0.0,
        )
    if _tripolar_curl:
        # dx_cell is 2D (n_lat, n_lon); pad lat axis, append wrap column.
        # Use the backend-aware pad so an interior MPI rank sendrecv's the
        # neighbour band's dx_T row at a partition cut, matching the u halo
        # above.  A plain jnp.pad would force the metric to zero at the cut
        # and corrupt the first/last local vertex circulation (#357 review).
        # Bit-identical to jnp.pad((1,1),(0,0)) on the local backend.
        dx_pad = pad_with_pole_bc_lat(
            dx_cell, halo=1, south_value=0.0, north_value=0.0,
        )  # (n_lat+2, n_lon)
        dx_pad = jnp.concatenate([dx_pad, dx_pad[:, 0:1]], axis=1)  # (n_lat+2, n_lon+1)
        dx_south = dx_pad[:-1]  # (n_lat+1, n_lon+1)
        dx_north = dx_pad[1:]   # (n_lat+1, n_lon+1)
        if is_3d:
            dx_south = dx_south[:, :, jnp.newaxis]
            dx_north = dx_north[:, :, jnp.newaxis]
        u_south = u_ext[:-1]
        u_north = u_ext[1:]
        du_circ = u_south * dx_south - u_north * dx_north
    else:
        # Wall-BC pad of dx_cell at poles (zero contribution beyond
        # the pole); under MPI interior ranks pad with neighbour's
        # dx via sendrecv instead.
        # Pad the CONCRETE ``cos_lat`` first (static-folds at trace
        # time — same recipe as the lat/sin pad above), then form
        # ``R * cos * dlon`` on the padded array: row-wise identical
        # arithmetic to padding the precomputed dx_cell, and the zero
        # pole constant maps to an exactly-zero dx row.
        cos_ext_q = pad_with_pole_bc_lat(
            cos_lat, halo=1, south_value=0.0, north_value=0.0,
        )
        dx_ext = R * cos_ext_q * dlon
        u_south = u_ext[:-1]
        u_north = u_ext[1:]
        dx_south = dx_ext[:-1]
        dx_north = dx_ext[1:]
        bcast = (slice(None),) + (jnp.newaxis,) * (u.ndim - 1)
        du_circ = (u_south * dx_south[bcast] - u_north * dx_north[bcast])

    # ``dv_circ_full`` (n_lon+1 vertex columns) is built per-branch in the v
    # contribution above (pad-then-diff for the regular path, roll + wrap
    # append for tripolar), so no shared wrap-column append is needed here.
    circ = du_circ + dv_circ_full

    # Vorticity at ALL local vertex rows.  ``circ`` already spans the
    # rank's partition cuts (u and the dx_T metric are halo-exchanged
    # above), so every interior cut row carries the true circulation and
    # must be divided through — not dropped and refilled from a neighbour
    # zeta halo, which would be off by one vertex row (#357 review).  Safe
    # division guards the real pole rows (``A_vertex == 0`` there), which
    # are then overwritten with the wall / fold BC below.
    safe_A = jnp.where(A_vertex_full > 0, A_vertex_full, 1.0)
    if safe_A.ndim == 1:
        # Regular / Mercator: A_vertex is 1D over latitude — broadcast over
        # the longitude (and level) axes.
        safe_A = safe_A[(slice(None),) + (jnp.newaxis,) * (circ.ndim - 1)]
    elif is_3d:
        # Tripolar: A_vertex is 2D (lat, lon) — add the level axis.
        safe_A = safe_A[:, :, jnp.newaxis]
    zeta = circ / safe_A

    # Wall BC at the real pole rows only (backend-aware: interior partition
    # cuts are left intact so the cross-cut vorticity survives).
    from legoesm.grids.halo_latlon import zero_polar_lat_ends
    zeta = zero_polar_lat_ends(zeta)

    # Tripolar north fold: the rank that owns the seam overwrites its north
    # pole row with the fold-permuted sub-polar vertex row (matches the
    # pad_ns_scalar fold convention; vertex fields carry an n_lon+1 wrap
    # column).
    nmask = north_fold_mask(grid)
    if fold_is_local(grid) or nmask is not None:
        fold = grid.fold
        # fold_row handles the n_lon+1 vertex wrap column (scalar sign +1).
        north = fold_row(zeta[-2:-1], fold.perm_T, 1.0, fold.perm_T.shape[0])
        zeta = apply_north_fold(zeta, north, grid, north_mask=nmask)

    return zeta


def gradient_curl_to_u(
    zeta: jnp.ndarray,
    grid: LatLonGrid,
) -> jnp.ndarray:
    """Meridional gradient of vertex scalar to u-face points.

    The tangential direction at a u-face is south-to-north.
    result(i, j) = (zeta[i+1, j] - zeta[i, j]) / (R * dlat)

    Parameters
    ----------
    zeta : (n_lat+1, n_lon+1) or (n_lat+1, n_lon+1, nlev)

    Returns
    -------
    grad : (n_lat, n_lon+1) or (n_lat, n_lon+1, nlev)
    """
    if is_tripolar(grid):
        # Tripolar: full 2D dy at u-face stagger.
        dy = grid.dy_u  # (n_lat, n_lon+1)
        if zeta.ndim == 3:
            return (zeta[1:] - zeta[:-1]) / dy[:, :, jnp.newaxis]
        return (zeta[1:] - zeta[:-1]) / dy
    else:
        # Regular or Mercator: variable-dy safe.
        dy = grid.dy * 0.5  # (n_lat,)
        diff = zeta[1:] - zeta[:-1]
        bcast = (slice(None),) + (jnp.newaxis,) * (diff.ndim - 1)
        return diff / dy[bcast]


def gradient_curl_to_v(
    zeta: jnp.ndarray,
    grid: LatLonGrid,
) -> jnp.ndarray:
    """Zonal gradient of vertex scalar to v-face points.

    The tangential direction at a v-face is west-to-east.
    result(i, j) = (zeta[i, j+1] - zeta[i, j]) / (R * cos(lat_v[i]) * dlon)

    Parameters
    ----------
    zeta : (n_lat+1, n_lon+1) or (n_lat+1, n_lon+1, nlev)

    Returns
    -------
    grad : (n_lat+1, n_lon) or (n_lat+1, n_lon, nlev)

    Notes
    -----
    The gradient is computed at ALL ``n_lat+1`` local v-face rows
    directly from ``zeta`` (a vertex field — its band-end rows are
    locally available, no halo needed); only the 1D regular/Mercator
    metric needs one backend-dispatched cell-latitude pad so the end
    faces carry the exact metric at an interior MPI band cut.  The
    previous implementation computed interior rows only and refilled
    the two end rows via ``pad_ns_scalar``, which at a cut delivered
    the neighbour's ADJACENT-face gradient — one face row off (np>=2
    parity bug in the default A_h vector-Laplacian path).  Pole / fold
    conventions are restored at physical ends only:
    ``zero_polar_lat_ends`` (wall) then the fold-permuted last interior
    face row on the rank that owns the tripolar seam — bit-identical to
    the old ``pad_ns_scalar`` output on the local backend.
    """
    if reads_stored_vface_metric(grid):
        # Full 2D dx_v at ALL n_lat+1 v-faces.  The band slice already
        # carries the exact global metric at partition-cut rows
        # ([s:e+1]).  Floor the denominator like gradient_y_cgrid
        # (#358 review): the pole/fold rows may carry a zero metric and
        # are overwritten below, but dividing by exact zero first would
        # poison jax_debug_nans and AD.
        dx_v = jnp.maximum(grid.dx_v, 1e-30)  # (n_lat+1, n_lon)
        bcast = (slice(None), slice(None)) + (
            (jnp.newaxis,) if zeta.ndim == 3 else ()
        )
    else:
        # Regular / Mercator: 1D metric, cell-pad-first (same recipe as
        # divergence_cgrid): at an interior MPI band cut the ghost row
        # is the neighbour's true edge cell latitude via the AD-safe
        # sendrecv, so the end faces divide by the exact serial metric.
        R = grid.radius
        dlon = grid.dlon
        cos_lat_v = _vface_cos_lat_core(grid)
        # Floor only guards the ghost-derived pole entries (overwritten
        # below); interior/cut faces are O(1e5 m) — bit-identical.
        dx_v = jnp.maximum(R * cos_lat_v * dlon, 1e-30)  # (n_lat+1,)
        bcast = (slice(None),) + (jnp.newaxis,) * (zeta.ndim - 1)

    # zeta[:, j+1] - zeta[:, j] for j=0..n_lon-1, at every local v-face
    # row including the band ends.
    dzeta = zeta[:, 1:] - zeta[:, :-1]  # (n_lat+1, n_lon [, nlev])
    grad = dzeta / dx_v[bcast]

    # Wall BC at the physical pole v-faces only (backend-aware:
    # interior partition cuts keep the computed gradient), then the
    # tripolar north fold row exactly as pad_ns_scalar produced it
    # (perm_T of the last interior face row; n_lon columns — no wrap
    # column on this stagger).
    from legoesm.grids.halo_latlon import zero_polar_lat_ends
    grad = zero_polar_lat_ends(grad)
    nmask = north_fold_mask(grid)
    if fold_is_local(grid) or nmask is not None:
        north = fold_row(grad[-2:-1], grid.fold.perm_T, 1.0,
                         grid.fold.perm_T.shape[0])
        grad = apply_north_fold(grad, north, grid, north_mask=nmask)
    return grad


def vector_laplacian_cgrid(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: LatLonGrid,
    *,
    mask: jnp.ndarray | None = None,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
    vertex_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Vector Laplacian on the C-grid: grad(div) - k x grad(curl).

    Operates directly on face velocities without cell-center detour.
    This is the rectangular-grid analog of TRiSK's vector_laplacian_del2.

    Parameters
    ----------
    u : (n_lat, n_lon+1) or (n_lat, n_lon+1, nlev)
    v : (n_lat+1, n_lon) or (n_lat+1, n_lon, nlev)
    grid : LatLonGrid
    mask : (n_lat, n_lon) cell-center land mask, optional
    u_mask : (n_lat, n_lon+1) u-face mask, optional
    v_mask : (n_lat+1, n_lon) v-face mask, optional

    Returns
    -------
    vlap_u : same shape as u
    vlap_v : same shape as v
    """
    # Native 2D and 3D — all underlying operators (``divergence_cgrid``,
    # ``gradient_*_cgrid``, ``curl_vertex_cgrid``, ``_gradient_curl_to_*``)
    # natively support 3D inputs.  Masks remain 2D and broadcast over
    # the trailing level axis when present.
    is_3d = u.ndim == 3

    def _bcast(m, like):
        # Broadcast 2D ``m`` over trailing axes of ``like``.
        return m[..., jnp.newaxis] if is_3d else m

    # Apply face masks before computing div and curl
    u_eff = u if u_mask is None else u * _bcast(u_mask, u)
    v_eff = v if v_mask is None else v * _bcast(v_mask, v)

    # 1. Divergence at cell centers
    div = divergence_cgrid(u_eff, v_eff, grid)
    if mask is not None:
        div = div * _bcast(mask, div)

    # 2. grad(div) at faces
    grad_div_u = gradient_x_cgrid(div, grid)  # (n_lat, n_lon+1[, nlev])
    grad_div_v = gradient_y_cgrid(div, grid)  # (n_lat+1, n_lon[, nlev])

    # 3. Curl at vertices
    zeta = curl_vertex_cgrid(u_eff, v_eff, grid)  # (n_lat+1, n_lon+1[, nlev])

    # Mask curl at land-adjacent vertices.  ``vertex_mask`` is the
    # model-init precomputed mask (a closure CONSTANT — the per-step
    # recompute paid an N-S exchange even though land_mask is constant
    # per run, because state.land_mask is a step-input tracer; halo
    # census 8474554).  Fallback recompute keeps standalone callers
    # working unchanged.
    if mask is not None:
        vmask = (vertex_mask if vertex_mask is not None
                 else compute_vertex_mask(mask, grid=grid))
        zeta = zeta * _bcast(vmask, zeta)

    # 4. Tangential gradient of curl at faces
    grad_curl_u = gradient_curl_to_u(zeta, grid)
    grad_curl_v = gradient_curl_to_v(zeta, grid)

    # 5. Vector Laplacian = grad(div) - curl(curl).  curl(curl F) =
    # k × ∇ζ = (-∂ζ/∂y, +∂ζ/∂x).  Signs verified via bump tests.
    vlap_u = grad_div_u - grad_curl_u
    vlap_v = grad_div_v + grad_curl_v

    # Apply face masks to output
    if u_mask is not None:
        vlap_u = vlap_u * _bcast(u_mask, vlap_u)
    if v_mask is not None:
        vlap_v = vlap_v * _bcast(v_mask, vlap_v)

    return vlap_u, vlap_v


def compute_vertex_mask(land_mask: jnp.ndarray, grid=None) -> jnp.ndarray:
    """Compute vertex mask: wet only if all four surrounding cells are wet.

    Parameters
    ----------
    land_mask : (n_lat, n_lon)
    grid : optional LatLonGrid or LatLonCGridGeometry.
        When provided and a tripolar fold is active, the north-boundary
        vertex mask is computed from the fold-partner cells rather than
        being zero (wall BC).

    Returns
    -------
    vertex_mask : (n_lat+1, n_lon+1)

    Notes
    -----
    Cell-pad-first: the cell mask is padded by one latitude row through
    the backend-dispatched pad (at an interior MPI band cut the ghost
    row is the neighbour rank's true edge cell row), so EVERY local
    vertex row — including the partition-cut rows — is the product of
    its four true surrounding cells.  The previous implementation
    computed interior vertex rows only and refilled the end rows via
    ``pad_ns_zero``/``pad_ns_scalar``, which at a cut delivered the
    neighbour's ADJACENT vertex row (one row off — wrong wherever land
    touches a cut).  Physical pole rows stay zero (wall BC) and the
    tripolar fold row is reproduced exactly as ``pad_ns_scalar`` built
    it (fold-permuted last interior row + wrap column).
    """
    from legoesm.grids.halo_latlon import (
        pad_with_pole_bc_lat,
        zero_polar_lat_ends,
    )
    if land_mask.ndim != 2:
        # Preserve the previous implicit 2D contract (it unpacked
        # ``m.shape`` into two names).
        raise ValueError(
            f"compute_vertex_mask: land_mask must be 2D (n_lat, n_lon), "
            f"got ndim={land_mask.ndim}"
        )
    # Vertex (i, j) is surrounded by cells (i-1, j-1), (i-1, j),
    # (i, j-1), (i, j); with the lat pad, rows i-1 / i of the global
    # mask are m_pad[i] / m_pad[i+1].
    m_pad = pad_with_pole_bc_lat(
        land_mask, halo=1, south_value=0.0, north_value=0.0,
    )  # (n_lat+2, n_lon[_local])
    # LON halo on the lat-padded mask (local wrap / 2-D ring), then the
    # 4-cell vertex product over padded cells.  ``east``/``west`` are the
    # cell (j) / cell (j-1) columns at each vertex; vertex (i,j) = product of
    # the four surrounding cells.  Builds the full n_lon+1 vertex columns
    # directly (no wrap-column append) — bit-identical to ``roll(m_pad,1)`` +
    # wrap concat at proc_lon==1; spans lon partition cuts under a 2-D split.
    m_pad_lon = pad_lon_cgrid(m_pad, halo=1)  # (n_lat+2, n_lon_local+2)
    east = m_pad_lon[:, 1:]                    # cell j   at each vertex column
    west = m_pad_lon[:, :-1]                   # cell j-1 at each vertex column
    full = east[:-1] * east[1:] * west[:-1] * west[1:]  # (n_lat+1, n_lon+1)

    # Wall BC at the physical pole vertex rows only (backend-aware).
    full = zero_polar_lat_ends(full)
    nmask = north_fold_mask(grid)
    if fold_is_local(grid) or nmask is not None:
        fold = grid.fold
        # fold_row handles the n_lon+1 vertex wrap column (scalar sign +1).
        north = fold_row(full[-2:-1], fold.perm_T, 1.0, fold.perm_T.shape[0])
        full = apply_north_fold(full, north, grid, north_mask=nmask)
    return full
