"""Restart harness for the legoESM ocean dycores.

Saves the full ocean prognostic state to disk as a numpy ``.npz``
archive and loads it back into an empty state container of the
matching grid. Round-trip preserves every prognostic field to
bit-exact precision (``np.array_equal``) so a model started from a
restart steps to bit-identical results vs. an uninterrupted run.

Currently supports the three production grids:

* lat-lon C-grid (``LatLonCGridOceanState``)
* MPAS Voronoi (``MPASOceanState`` -- accessed via duck-typing on
  the ``state.u.dims`` shape)
* cubed-sphere (``OceanState``)

The serialised payload is a dict-of-numpy-arrays keyed by field
name; auxiliary metadata (model time in seconds, step index, source
sha) is stored under reserved underscore keys to keep field names
clean.

Usage::

    from legoesm.ocean.restart import save_restart, load_restart

    save_restart(state, "results/ocean/restart_t0.npz",
                 time_s=86400.0 * 30, step=10_000)
    state_loaded = load_restart("results/ocean/restart_t0.npz", state)
    # state_loaded is bit-identical to state.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from legoesm.core.field import Field


_RESERVED_KEYS: tuple[str, ...] = ("_time_s", "_step", "_sha")

# State slots this archive format deliberately neither writes nor restores:
# pure DIAGNOSTICS the next step rewrites unconditionally from the prognostic
# state.  Currently the #1442 ``store_mass_flux`` capture.
#
# Two reasons, and the second is a correctness one (codex round-7 YELLOW 2):
#  * a fresh template leaves these slots ``None``, and the loader's
#    ``ref_field is None`` branch then rebuilds them as bare
#    ``Field(data, name)`` -- WITHOUT dims/units.  Field metadata is pytree
#    AUX data, so such a state has a different treedef from what the step
#    writes and would abort the next ``lax.scan``;
#  * they are large (three face/interface-shaped arrays) and carry nothing a
#    restart needs.
# Not writing them makes save and load agree by construction.  Mirrors
# ``run_omip._RESTART_DIAGNOSTIC_SLOTS`` for the other npz lane.
DIAGNOSTIC_SLOTS: tuple[str, ...] = (
    "mass_flux_u", "mass_flux_v", "mass_flux_w",
    "salt_flux_u_int", "salt_flux_v_int")


def _iter_state_fields(state) -> list[str]:
    """Return the prognostic field names of the state NamedTuple."""
    # NamedTuple subclasses expose ``_fields``; fall back to dir() filtered.
    fields = getattr(state, "_fields", None)
    if fields is not None:
        return list(fields)
    return [k for k in dir(state) if isinstance(getattr(state, k), Field)]


def save_restart(state, path: str | Path, *,
                 time_s: float | None = None,
                 step: int | None = None,
                 sha: str | None = None) -> Path:
    """Serialise every ``Field`` of ``state`` to a ``.npz`` archive.

    Returns the resolved ``Path`` of the written file.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for name in _iter_state_fields(state):
        if name in DIAGNOSTIC_SLOTS:
            continue
        attr = getattr(state, name)
        if attr is None:
            continue
        if isinstance(attr, Field):
            payload[name] = np.asarray(attr.data)
    if time_s is not None:
        payload["_time_s"] = np.asarray(float(time_s))
    if step is not None:
        payload["_step"] = np.asarray(int(step))
    if sha is not None:
        payload["_sha"] = np.asarray(str(sha))
    np.savez(out_path, **payload)
    return out_path


def load_restart(path: str | Path, template_state) -> tuple:
    """Read a ``.npz`` archive and return a state with the loaded data.

    ``template_state`` provides the ``Field`` ``dims`` / ``units`` /
    ``name`` metadata; the returned state has the same NamedTuple type.
    """
    in_path = Path(path)
    with np.load(in_path, allow_pickle=False) as f:
        loaded = {k: f[k] for k in f.files}

    replace_kw: dict[str, Field | None] = {}
    for name in _iter_state_fields(template_state):
        if name in DIAGNOSTIC_SLOTS:
            # CLEARED to None, not merely skipped (codex round-8 RED 3).
            # Skipping leaves a POPULATED template's slot in place, so a
            # restart would carry a STALE diagnostic next to freshly loaded
            # prognostics -- and worse, this loader rebuilds the loaded Fields
            # WITHOUT their staggering, so a retained diagnostic (which kept
            # its donor's staggering) and a reloaded ``u``/``v`` would no
            # longer agree, giving the state a treedef the next step's output
            # does not match.  ``None`` is the unambiguous state: the seeding
            # helper rebuilds all three canonically before any scan.
            replace_kw[name] = None
            continue
        if name not in loaded:
            continue
        ref_field = getattr(template_state, name)
        if not isinstance(ref_field, Field):
            if ref_field is None:
                # Optional carry slot (e.g. prognostic tke) that the writer
                # saved but the template left unseeded: reconstruct a Field
                # with generic metadata rather than silently dropping the
                # carry (codex MED 2026-07-27 — a dropped carry re-spins the
                # turbulence from the background seed on restart).
                replace_kw[name] = Field(
                    data=jnp.asarray(loaded[name]), name=name)
            continue
        replace_kw[name] = Field(
            data=jnp.asarray(loaded[name]),
            name=ref_field.name,
            dims=ref_field.dims,
            units=ref_field.units,
        )
    return template_state._replace(**replace_kw)


def grid_lat2d_lon2d_deg(grid, grid_type: str) -> tuple[np.ndarray, np.ndarray]:
    """Tracer-point latitude/longitude in DEGREES for any production grid.

    The grids store coordinates in RADIANS (lat-lon ``grid.lat_T``/``lon_T``,
    cubed-sphere ``grid.lat``/``lon``, MPAS ``grid.latCell``/``lonCell``); the
    offline MLD / OMIP-NEMO scorers consume DEGREES.  This is the single
    radian->degree extraction used by the snapshot writer so every grid emits
    the same ``lat_T``/``lon_T`` (degrees) convention.

    Raises ``ValueError`` on an unknown ``grid_type`` (dispatch hardening: a
    silent fallthrough would emit a wrong-shaped / mis-unit coordinate).
    """
    if grid_type == "cubed_sphere":
        # Per-cell lat/lon (already 2-D-per-face, e.g. (6, n, n)).
        return (np.rad2deg(np.asarray(grid.lat)),
                np.rad2deg(np.asarray(grid.lon)))
    if grid_type == "mpas":
        # Voronoi cell centres: 1-D (nCells,); the scorer flattens any source.
        return (np.rad2deg(np.asarray(grid.latCell)),
                np.rad2deg(np.asarray(grid.lonCell)))
    if grid_type == "fesom":
        # FESOM2 unstructured triangular mesh: one lat/lon per NODE, 1-D
        # (nod2D,) -- the same convention as the MPAS branch above (one value
        # per cell). FesomOceanGrid.lat/.lon are geographic radians.
        return (np.rad2deg(np.asarray(grid.lat)),
                np.rad2deg(np.asarray(grid.lon)))
    if grid_type == "tripole":
        # Curvilinear tracer points are genuinely 2-D.
        return (np.rad2deg(np.asarray(grid.lat_T)),
                np.rad2deg(np.asarray(grid.lon_T)))
    if grid_type == "latlon":
        # Regular grid: 1-D radian axes -> 2-D degree meshgrid (n_lat, n_lon).
        lon2d, lat2d = np.meshgrid(np.rad2deg(np.asarray(grid.lon)),
                                   np.rad2deg(np.asarray(grid.lat)))
        return lat2d, lon2d
    raise ValueError(
        f"grid_lat2d_lon2d_deg: unknown grid_type {grid_type!r} "
        "(expected one of: latlon, tripole, cubed_sphere, mpas, fesom)")


def save_mld_snapshot(state, path: str | Path, *,
                      z_coord,
                      lat2d: np.ndarray,
                      lon2d: np.ndarray,
                      time_s: float | None = None,
                      step: int | None = None) -> Path:
    """Write a compact ocean snapshot for the offline mixed-layer-depth scorers.

    Unlike :func:`save_restart` (every prognostic field, for a bit-exact
    restart), this writes ONLY what the de Boyer Montegut / Treguier (2023)
    MLD diagnostic and the OMIP-NEMO comparison consume: ``T``, ``S``,
    ``land_mask`` and ``H_bathy`` from the state, the horizontal grid
    (``lat_T``, ``lon_T`` in DEGREES), and the reference level-centre depths
    (``z_center_ref``, positive-down) the scorers use to build the per-level
    wet mask ``z_center_ref < H_bathy``.  ``u``/``v``/``eta`` are included
    when present (``v`` is absent on MPAS; ``eta`` lets a run restart from the
    snapshot without a barotropic-adjustment shock).

    Shared writer for the snapshot contract read by
    ``scripts/validate/compare_mld_dbm.py`` and ``compare_omip_nemo.py``; the
    OMIP drivers (``run_omip`` / ``run_omip_core2``) call this so all four
    grids emit a single, identical snapshot format.  ``H_bathy`` and
    ``z_center_ref`` are REQUIRED by the consumers, so they are required here
    too (a clear write-time error beats a cryptic ``SystemExit`` at read).

    Parameters
    ----------
    state : ocean state NamedTuple
        Must expose ``T``/``S``/``land_mask``/``H_bathy`` Fields.
    path : str or Path
        Output ``.npz`` path.
    z_coord : ocean vertical coordinate
        Must provide ``z_half_ref`` (nlev+1, <=0) -> reference centre depths.
    lat2d, lon2d : array
        Tracer-point latitudes/longitudes in DEGREES (use
        :func:`grid_lat2d_lon2d_deg`).
    time_s, step : optional provenance scalars.

    Returns
    -------
    Path
        The resolved path of the written archive.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kw: dict[str, np.ndarray] = {
        "T": np.asarray(state.T.data),
        "S": np.asarray(state.S.data),
        "land_mask": np.asarray(state.land_mask.data),
        "lat_T": np.asarray(lat2d),
        "lon_T": np.asarray(lon2d),
    }
    # u/v/eta when present (MPAS lacks a separate v; eta enables shock-free
    # --restart-from). The MLD scorer reads only T/S/land_mask/geometry.
    for opt in ("u", "v", "eta"):
        fld = getattr(state, opt, None)
        if fld is not None and hasattr(fld, "data"):
            save_kw[opt] = np.asarray(fld.data)
    # H_bathy + z_center_ref are part of the consumer contract -> required.
    H_bathy = getattr(state, "H_bathy", None)
    if H_bathy is None or not hasattr(H_bathy, "data"):
        raise ValueError(
            "save_mld_snapshot: state has no H_bathy Field, but the MLD "
            "scorers require it (per-level wet mask z_center_ref < H_bathy).")
    save_kw["H_bathy"] = np.asarray(H_bathy.data)
    zh_ref = getattr(z_coord, "z_half_ref", None)
    if zh_ref is None:
        raise ValueError(
            "save_mld_snapshot: z_coord has no z_half_ref, required to build "
            "z_center_ref for the MLD scorers.")
    zh = np.asarray(zh_ref)                                      # (nlev+1,), <=0
    z_center_ref = np.abs(0.5 * (zh[:-1] + zh[1:]))             # (nlev,) +down
    # mixed_layer_depth assumes strictly-increasing positive-down centres; a
    # malformed coordinate that violates this would silently corrupt the MLD.
    if not (z_center_ref.ndim == 1 and z_center_ref.shape[0] >= 1
            and np.all(np.diff(z_center_ref) > 0.0)):
        raise ValueError(
            "save_mld_snapshot: z_center_ref is not strictly increasing "
            f"positive-down (got {z_center_ref}); check z_coord.z_half_ref.")
    save_kw["z_center_ref"] = z_center_ref
    if time_s is not None:
        save_kw["_time_s"] = np.asarray(float(time_s))
    if step is not None:
        save_kw["_step"] = np.asarray(int(step))
    # Atomic write: a killed process (or two runs sharing an output dir) must
    # not leave a partial/corrupt npz that a later scorer silently mis-reads.
    # A UNIQUE temp file (mkstemp) — not a fixed ".tmp.npz" name — so two
    # writers targeting the same out_path can't truncate each other's temp.
    fd, tmp_name = tempfile.mkstemp(
        dir=str(out_path.parent), prefix=out_path.name + ".", suffix=".tmp.npz",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez_compressed(fh, **save_kw)
        tmp_path.replace(out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return out_path


def restart_metadata(path: str | Path) -> dict[str, Any]:
    """Return ``{time_s, step, sha}`` (any present) from the archive."""
    in_path = Path(path)
    out: dict[str, Any] = {}
    with np.load(in_path, allow_pickle=False) as f:
        for key in _RESERVED_KEYS:
            if key in f.files:
                clean = key.lstrip("_")
                val = f[key]
                # Decode scalars + bytestrings.
                if val.dtype.kind in ("U", "S"):
                    out[clean] = str(val)
                else:
                    out[clean] = val.item()
    return out


__all__ = [
    "save_restart", "load_restart", "restart_metadata", "save_mld_snapshot",
    "grid_lat2d_lon2d_deg",
]
