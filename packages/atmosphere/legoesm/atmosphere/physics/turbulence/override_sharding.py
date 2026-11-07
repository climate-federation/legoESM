"""Localize a GLOBAL per-column turbulence override to a rank's MPI tile.

The LES-informed correction loop deploys a GLOBAL ``(ncol,)`` per-column
``clubb_lite`` override (``ExperimentConfig.turbulence_override``).  Under MPI
lat-lon decomposition each rank owns a sub-tile of columns, so the physics runs on
rank-local ``(ncol_local, nlev)`` shapes and a global override no longer matches
``broadcast_column_param``.  :func:`localize_turbulence_override` slices the global
override down to the rank's columns at dycore-build time — the driver's single
source of truth (``turbulence_config_for``) calls it with the active MPI layout.

This is the DRIVER-path (lenient) counterpart to the deploy-context (strict)
slicers in :mod:`legoesm.training.deploy_correction`: it passes through anything
that does not need slicing (a non-clubb scheme, an all-scalar override, or a
per-column field already at the rank-local length) and slices only a GLOBAL
per-column override, so it is a safe no-op on every serial / already-local build.
Lives in ``atmosphere`` (not ``ml``) so the ``coupler`` driver can import it
without a reverse-federation dependency.
"""

from __future__ import annotations

from typing import Any

_CLUBB_FIELDS = ("C_K", "Pr_t", "C_eps")


def localize_turbulence_override(override: Any, layout: Any) -> Any:
    """Slice a GLOBAL per-column ``clubb_lite`` override to ``layout``'s rank tile.

    ``layout`` is the active MPI layout (``LatLonBandLayout`` — a latitude band, or
    ``LatLon2DLayout`` — a lat×lon pencil).  Returns ``override`` UNCHANGED when no
    slicing is needed (non-clubb scheme, all-scalar fields, or per-column fields not
    at the global column count — i.e. already rank-local).  Supports lat-lon
    (``LatLonBandLayout`` / ``LatLon2DLayout``), cubed-sphere (``DistributedLayout``
    face-only / tiled, via the model's ``scatter``), AND MPAS/Voronoi
    (``VoronoiPartitionLayout``, gathered at the rank's ``local_cells``).  An
    UNRECOGNIZED decomposition passes the override THROUGH unchanged (honoring an
    advanced user who pre-sliced per rank via
    :func:`legoesm.training.deploy_correction.slice_override_columns`); a GLOBAL
    override there fails LOUDLY downstream in ``broadcast_column_param`` (a length
    mismatch), never silently.
    """
    import jax.numpy as jnp

    if getattr(override, "scheme", None) != "clubb_lite":
        return override  # only clubb_lite carries per-column coefficient fields
    clubb = override.clubb_lite
    per_column = {}
    for field in _CLUBB_FIELDS:
        arr = jnp.asarray(getattr(clubb, field))
        if arr.ndim == 1:
            per_column[field] = arr
    if not per_column:
        return override  # all-scalar override broadcasts on every rank unchanged

    part = _layout_partition(layout)
    if part is None:
        # An unrecognized decomposition has no global shape to slice against. Pass
        # the override THROUGH so an advanced user who pre-sliced it per rank (the
        # `slice_override_columns` escape hatch) is honored; a GLOBAL override here
        # fails LOUDLY downstream in broadcast_column_param (a length mismatch),
        # never silently.
        return override
    global_shape, slice_to_local = part
    ncol = 1
    for dim in global_shape:
        ncol *= int(dim)
    lengths = {int(a.shape[0]) for a in per_column.values()}
    if lengths == {ncol}:
        sliced = {f: slice_to_local(a.reshape(global_shape)).reshape(-1)
                  for f, a in per_column.items()}
        # _replace, not a rebuild: every other field of the override (e.g.
        # liquid_partition, update_interval_steps) must survive localization,
        # or the factory's guards see defaults instead of what was selected.
        return override._replace(clubb_lite=clubb._replace(**sliced))
    if ncol in lengths:
        raise ValueError(
            f"per-column override fields have inconsistent column counts {lengths} "
            f"(some match the global ncol {ncol}, some do not); every corrected "
            "field must span the same global grid.")
    return override  # already rank-local (or a different grid) → leave as-is


def active_column_layout():
    """The active per-column MPI layout across all grid families, or ``None`` (serial).

    The three grid families register their layout DIFFERENTLY: lat-lon sets it as
    the halo topology (``get_mpi_topology``); cubed-sphere sets a halo
    ``CommTopology`` (the column ``DistributedLayout`` is resolved inside
    :func:`_layout_partition`); MPAS/Voronoi sets ONLY the active voronoi layout (no
    halo topology), so fall back to it when no halo topology is active.  This is the
    single accessor ``turbulence_config_for`` uses to find the rank's partition.

    Precedence is halo-topology-FIRST, which assumes ONE grid family per process —
    the production case (a run drives a single dycore).  A same-process run that
    switched grid families without resetting (a test) could leave a stale halo
    topology shadowing an active voronoi layout; distributed tests must reset it
    (``set_halo_backend('local')`` / ``reset_distributed_topology()``), exactly as a
    fresh MPAS process has no halo topology set.
    """
    from legoesm.grids.halo import get_mpi_topology

    layout = get_mpi_topology()
    if layout is not None:
        return layout  # lat-lon band/2-D, or a cubed CommTopology
    from legoesm.parallel.voronoi_mpi import get_active_voronoi_layout

    return get_active_voronoi_layout()  # MPAS Voronoi layout, or None (serial)


def _layout_partition(layout):
    """``(global_shape, slice_to_local)`` for a supported MPI layout, else ``None``.

    ``global_shape`` is the per-column field's native horizontal shape (``(n_lat,
    n_lon)`` lat-lon, ``(6, n, n)`` cubed-sphere, ``(nCells_global,)`` MPAS) and
    ``slice_to_local`` is the rank's deterministic block extractor — reusing the
    MODEL's own partition (the lat-lon band/2-D slice, :func:`legoesm.parallel.
    layout.scatter` for cubed-sphere, or a ``jnp.take`` over the MPAS partition's
    owned+halo ``local_cells``), so the local override lands on EXACTLY the columns
    the rank's physics consumes.  ``None`` ⇒ an unrecognized decomposition.
    """
    import jax.numpy as jnp
    from legoesm.parallel.comm import CommTopology
    from legoesm.parallel.latlon_mpi import LatLon2DLayout, LatLonBandLayout
    from legoesm.parallel.layout import (
        DistributedLayout,
        SingleRankLayout,
        scatter,
    )
    from legoesm.parallel.voronoi_mpi import VoronoiPartitionLayout

    if isinstance(layout, LatLon2DLayout):
        return ((layout.n_lat_global, layout.n_lon_global),
                lambda f: f[layout.lat_start:layout.lat_end,
                            layout.lon_start:layout.lon_end])
    if isinstance(layout, LatLonBandLayout):
        return ((layout.n_lat_global, layout.n_lon_global),
                lambda f: f[layout.lat_start:layout.lat_end, :])  # lon not split
    if isinstance(layout, CommTopology):
        # Cubed-sphere: get_mpi_topology() returns the halo CommTopology, but the
        # COLUMN partition (carrying global_n) is the active DistributedLayout.
        from legoesm.parallel.distributed import get_active_layout

        layout = get_active_layout()
        if layout is None:
            # A cubed halo topology is active but no column layout was registered
            # (initialize_distributed needs global_n). Fail LOUDLY + clearly here
            # rather than passing a global override through to a cryptic downstream
            # length error.
            raise RuntimeError(
                "cubed-sphere MPI is active but no DistributedLayout is registered "
                "(call initialize_distributed(global_n=...)); cannot localize the "
                "per-column turbulence override.")
    if isinstance(layout, (DistributedLayout, SingleRankLayout)):  # cubed-sphere
        n = int(layout.global_n)
        return ((6, n, n), lambda f: scatter(f, layout))
    if isinstance(layout, VoronoiPartitionLayout):  # MPAS / Voronoi (unstructured)
        part = layout.partition
        # The rank's physics runs on the LOCAL mesh (owned + halo cells); gather the
        # override at those cells' GLOBAL indices (halo cells take the owned value
        # they duplicate). 1-D field, so the "global_shape" is just (nCells_global,).
        local_cells = jnp.asarray(part.local_cells)
        return ((int(part.nCells_global),),
                lambda f: jnp.take(f, local_cells, axis=0))
    return None
