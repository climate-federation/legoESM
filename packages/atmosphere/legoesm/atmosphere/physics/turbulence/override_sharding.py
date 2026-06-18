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
    at the global column count — i.e. already rank-local).  Auto-localization is
    lat-lon-only: under a non-lat-lon decomposition (e.g. cubed-sphere) the override
    is passed THROUGH unchanged (honoring an advanced user who pre-sliced per rank
    via :func:`legoesm.training.deploy_correction.slice_override_columns`); a GLOBAL
    override there is unsupported and fails LOUDLY downstream in
    ``broadcast_column_param`` (a length mismatch), never silently.
    """
    import jax.numpy as jnp
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from legoesm.parallel.latlon_mpi import LatLon2DLayout, LatLonBandLayout

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

    if isinstance(layout, LatLon2DLayout):
        def _slice2d(f2d):
            return f2d[layout.lat_start:layout.lat_end,
                       layout.lon_start:layout.lon_end]
    elif isinstance(layout, LatLonBandLayout):
        def _slice2d(f2d):
            return f2d[layout.lat_start:layout.lat_end, :]  # lon not split
    else:
        # A non-lat-lon decomposition (e.g. cubed-sphere) has no (n_lat, n_lon)
        # global shape to slice against — auto-localization is lat-lon-only. Pass
        # the override THROUGH so an advanced user who pre-sliced it per rank (the
        # `slice_override_columns` escape hatch) is honored; a GLOBAL override here
        # is genuinely unsupported and fails LOUDLY downstream in
        # broadcast_column_param (a global-vs-local length mismatch), never silently.
        return override

    n_lat, n_lon = int(layout.n_lat_global), int(layout.n_lon_global)
    ncol = n_lat * n_lon
    lengths = {int(a.shape[0]) for a in per_column.values()}
    if lengths == {ncol}:
        sliced = {f: _slice2d(a.reshape(n_lat, n_lon)).reshape(-1)
                  for f, a in per_column.items()}
        return TurbulenceConfig(
            scheme="clubb_lite", clubb_lite=clubb._replace(**sliced))
    if ncol in lengths:
        raise ValueError(
            f"per-column override fields have inconsistent column counts {lengths} "
            f"(some match the global ncol {ncol}, some do not); every corrected "
            "field must span the same global grid.")
    return override  # already rank-local (or a different grid) → leave as-is
