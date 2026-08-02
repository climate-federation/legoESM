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

For RESUMING a driver mid-integration use the run-restart pair instead
(:func:`save_run_restart` / :func:`load_run_restart`): it persists every
PROGNOSTIC and STATIC state slot — including the non-``Field`` carries
``save_restart`` drops silently (rigid-lid ``psi``/``dpsi``/``dpsin``, the
``bt_hist`` tuple) — plus the unmangled sea-ice state and the absolute step
counter that every OMIP forcing index is a pure function of.  Slots labelled
DIAGNOSTIC in :data:`_SLOT_POLICY` are deliberately excluded, and a slot in
neither bucket raises::

    save_run_restart("results/omip/restart.npz", state, step=n, time_days=d,
                     grid_type="tripole", ice_state=ice)
    state, ice, meta = load_run_restart("results/omip/restart.npz", template,
                                        ice_template=ice0,
                                        grid_type="tripole")
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.field import Field


_RESERVED_KEYS: tuple[str, ...] = ("_time_s", "_step", "_sha")

# ---------------------------------------------------------------- run restart
# Reserved (underscore) keys of the RUN restart archive written by
# ``save_run_restart``.  Kept distinct from ``_RESERVED_KEYS`` (the older
# ``save_restart`` provenance triple) so the two formats never alias.
_RUN_RESERVED_KEYS: tuple[str, ...] = (
    "_format", "_step", "_time_days", "_grid_type", "_sha",
    "_slot_kinds", "_ice_slot_kinds", "_excluded",
    # Continuation FINGERPRINT + exact per-slot inventory (codex r1 HIGH):
    # step alone does not pin the forcing — ``_idx_t(step, dt, n_rec)`` also
    # depends on dt and the record count — and a manifest that merely omits a
    # slot would silently leave the fresh template's cold-start value in place.
    "_dt_seconds", "_n_forcing_records", "_x64",
    "_state_class", "_ice_class", "_inventory", "_ice_inventory",
)
# Bumped when the on-disk layout changes incompatibly.
_RUN_RESTART_FORMAT: int = 2
# Prefix under which the sea-ice state's slots are stored (mirrors the
# ``ice_<field>`` convention run_omip already writes).
_ICE_PREFIX: str = "ice_"
# Separator for the per-element keys of a tuple-valued carry slot.  Not a valid
# Python identifier character pair, so it can never collide with a field name.
_TUPLE_SEP: str = "::"

# Slot value KINDS the serialiser understands.  A state slot whose value is
# none of these raises at SAVE time — a silently dropped carry is exactly the
# defect this format exists to prevent (an uncheckpointed carry re-spins from
# its cold-start seed and the resumed run is not a continuation).
_KIND_FIELD = "field"
_KIND_ARRAY = "array"
_KIND_TUPLE = "tuple"

# --- persistence policy: an EXPLICIT, CLOSED partition of every state slot ---
#
# Every slot of every supported state class is labelled here.  ``save_run_restart``
# RAISES on a slot it does not recognise, and ``validate_restart_policy`` runs the
# same check at driver SETUP so an unclassified new field aborts in seconds
# rather than at the first mid-run checkpoint.  The point is that a new state
# field cannot fall into "neither bucket" and be silently dropped — the class of
# bug that broke bit-exact restart in this repo before.
#
# PROGNOSTIC: integrator state or a carry the next step reads.  MUST persist.
# STATIC:     geometry / reference profiles the resuming run rebuilds itself.
#             Persisted AND verified against the fresh template on load — a
#             resume whose mesh/bathymetry/--min-levels drifted is a confound,
#             not a continuation.
# DIAGNOSTIC: recomputed from the prognostic state every step.  DELIBERATELY
#             EXCLUDED — persisting one and restoring it resurrects a stale
#             value, and on the scan path it can flip an optional slot from
#             None to Field and break the carry treedef.
# Inventory labels recorded per slot so the loader can demand an EXACT layout
# rather than accepting any subset of the manifest.
_INV_PERSISTED = "persisted"     # written to the archive, MUST be restored
_INV_DIAGNOSTIC = "diagnostic"   # policy-excluded, recomputed by the next step
_INV_ABSENT = "absent"           # slot was None in the writing run (gate off)

_SLOT_PROGNOSTIC = "prognostic"
_SLOT_STATIC = "static"
_SLOT_DIAGNOSTIC = "diagnostic"

_SLOT_POLICY: dict[str, str] = {
    # --- ocean prognostics -------------------------------------------------
    "u": _SLOT_PROGNOSTIC, "v": _SLOT_PROGNOSTIC,
    "T": _SLOT_PROGNOSTIC, "S": _SLOT_PROGNOSTIC, "eta": _SLOT_PROGNOSTIC,
    # SOM (Prather) advection moments — carried by the scheme.
    "T_som": _SLOT_PROGNOSTIC, "S_som": _SLOT_PROGNOSTIC,
    # AB2 tracer-advection history.
    "T_flux_div_prev": _SLOT_PROGNOSTIC, "S_flux_div_prev": _SLOT_PROGNOSTIC,
    # Prognostic eddy / turbulent kinetic energy + their histories.
    "eke": _SLOT_PROGNOSTIC, "tke": _SLOT_PROGNOSTIC,
    "dtke": _SLOT_PROGNOSTIC, "eke_diss": _SLOT_PROGNOSTIC,
    # AB2 outer-integrator increments.
    "T_incr_prev": _SLOT_PROGNOSTIC, "S_incr_prev": _SLOT_PROGNOSTIC,
    "u_incr_prev": _SLOT_PROGNOSTIC, "v_incr_prev": _SLOT_PROGNOSTIC,
    # NEMO modified-leapfrog "before" state (Nbb).
    "u_before": _SLOT_PROGNOSTIC, "v_before": _SLOT_PROGNOSTIC,
    "T_before": _SLOT_PROGNOSTIC, "S_before": _SLOT_PROGNOSTIC,
    "eta_before": _SLOT_PROGNOSTIC,
    # Barotropic slow-forcing AB2 history.
    "F_slow_u_prev": _SLOT_PROGNOSTIC, "F_slow_v_prev": _SLOT_PROGNOSTIC,
    # Rigid-lid streamfunction quintet (raw arrays, not Fields).
    "psi": _SLOT_PROGNOSTIC, "dpsi": _SLOT_PROGNOSTIC,
    "dpsi_prev": _SLOT_PROGNOSTIC, "dpsin": _SLOT_PROGNOSTIC,
    "dpsin_prev": _SLOT_PROGNOSTIC,
    # NEMO AB3/AM4 cross-window barotropic history (tuple of arrays).
    "bt_hist": _SLOT_PROGNOSTIC,
    # Centred barotropic forcing history.
    "tau_x_prev": _SLOT_PROGNOSTIC, "tau_y_prev": _SLOT_PROGNOSTIC,
    "freshwater_eta_prev": _SLOT_PROGNOSTIC,
    # --- sea ice (SeaIceState + the 12-field DynamicSeaIceState) -----------
    "h_ice": _SLOT_PROGNOSTIC, "T_ice": _SLOT_PROGNOSTIC,
    "concentration": _SLOT_PROGNOSTIC, "u_ice": _SLOT_PROGNOSTIC,
    "v_ice": _SLOT_PROGNOSTIC, "sigma_11": _SLOT_PROGNOSTIC,
    "sigma_22": _SLOT_PROGNOSTIC, "sigma_12": _SLOT_PROGNOSTIC,
    "h_snow": _SLOT_PROGNOSTIC, "S_ice": _SLOT_PROGNOSTIC,
    "pond_area": _SLOT_PROGNOSTIC, "pond_depth": _SLOT_PROGNOSTIC,
    # --- static geometry / reference profiles ------------------------------
    "H_bathy": _SLOT_STATIC, "land_mask": _SLOT_STATIC,
    "u_mask": _SLOT_STATIC, "v_mask": _SLOT_STATIC,
    # MPAS reference density profile: READ by the pressure-gradient path
    # (ocean_pe_mpas rho_ref_z_static), fixed for the run.
    "rho_ref_z": _SLOT_STATIC,
    # --- diagnostics: recomputed every step, never persisted ---------------
    # ``w``: VERIFIED in code, not just from the docstring — the lat-lon C-grid
    # step's ONLY use of ``state.w`` is ``state.w.replace(data=w_full)`` (i.e.
    # the slot supplies Field METADATA, never a value the step reads), and it is
    # rewritten unconditionally every step from the flux divergence.  MPAS
    # declares the same ("Diagnostic field computed from flux divergence").
    "w": _SLOT_DIAGNOSTIC,
    # ``mass_flux_u``/``mass_flux_v`` (#1442, opt-in ``store_mass_flux``): the
    # h*u the step used for tracer advection, captured for the transport
    # diagnostic and recomputed from the prognostic state on the next step.
    # Pre-registered here so that change lands without tripping this gate, and
    # so it cannot repeat the run_omip asymmetry (written on save, silently
    # dropped on load because the fresh template slot is None).
    "mass_flux_u": _SLOT_DIAGNOSTIC, "mass_flux_v": _SLOT_DIAGNOSTIC,
}

# Derived from the policy so the two can never drift apart.
_STATIC_GEOMETRY_SLOTS: tuple[str, ...] = tuple(
    name for name, kind in _SLOT_POLICY.items() if kind == _SLOT_STATIC)


def _iter_state_fields(state) -> list[str]:
    """Return the prognostic field names of the state NamedTuple."""
    # NamedTuple subclasses expose ``_fields``; fall back to dir() filtered.
    fields = getattr(state, "_fields", None)
    if fields is not None:
        return list(fields)
    return [k for k in dir(state) if isinstance(getattr(state, k), Field)]


def _atomic_savez(out_path: Path, payload: dict) -> Path:
    """Write ``payload`` to ``out_path`` atomically (temp file + rename).

    A killed process (or two runs sharing an output directory) must never
    leave a partial/corrupt ``.npz`` that a later reader silently mis-reads.
    A UNIQUE temp name (``mkstemp``) — not a fixed ``.tmp.npz`` — so two
    writers targeting the same ``out_path`` cannot truncate each other's temp.
    Shared by :func:`save_mld_snapshot` and :func:`save_run_restart`, both of
    which wrote compressed archives before this was factored out.
    (:func:`save_restart` is deliberately NOT routed through here: it is the
    older format with existing consumers, and this change set does not alter
    its behaviour.)

    NOTE on the ``.npz`` suffix: ``np.savez*`` APPENDS ``.npz`` to a *path*
    that lacks it, but not to an open *file handle*.  Writing through a handle
    (as here) means the file lands at exactly ``out_path``, so the returned
    Path is always the file that was actually written — callers must pass the
    full name including ``.npz``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(out_path.parent), prefix=out_path.name + ".", suffix=".tmp.npz",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez_compressed(fh, **payload)
        tmp_path.replace(out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return out_path


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

    replace_kw: dict[str, Field] = {}
    for name in _iter_state_fields(template_state):
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
        "(expected one of: latlon, tripole, cubed_sphere, mpas)")


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
    # Atomic write (shared helper): a killed process must not leave a partial
    # npz that a later scorer silently mis-reads.  Compressed, as before.
    return _atomic_savez(out_path, save_kw)


# ============================================================================
# RUN restart: the full integrator carry + the sea-ice state + the step counter
# ============================================================================
#
# WHY A SECOND PAIR (not an extension of save_restart / save_mld_snapshot):
#
# * ``save_mld_snapshot`` is a DIAGNOSTIC artifact with several downstream
#   readers (``compare_mld_dbm.py``, ``compare_omip_nemo.py``); its key set is
#   a consumer contract, and it deliberately mangles the ice fields for
#   plotting (land-masked, category-aggregated).  A restart needs the raw,
#   unmangled prognostic values.
# * ``save_restart`` serialises only ``Field``-valued slots.  Several REAL
#   carries of ``LatLonCGridOceanState`` are NOT Fields — the rigid-lid
#   streamfunction quintet (``psi``/``dpsi``/``dpsi_prev``/``dpsin``/
#   ``dpsin_prev``, raw arrays) and the NEMO AB3/AM4 barotropic history
#   ``bt_hist`` (a 6-tuple of arrays) — so it drops them SILENTLY.
#
# The contract here is the inverse, and it is CLOSED rather than best-effort:
# every non-``None`` slot is looked up in :data:`_SLOT_POLICY`, the PROGNOSTIC
# and STATIC ones are serialised (with their value kind and Field aux-data
# recorded per slot), the DIAGNOSTIC ones are deliberately excluded and listed
# in the archive's ``_excluded`` record, and a slot in NEITHER bucket — or one
# whose container type the serialiser does not understand — is a hard error at
# save time.  So a field added to the state in future cannot be silently
# dropped (resuming from its cold-start seed) nor silently resurrected (a stale
# diagnostic restored over a fresh one).  The step counter is recorded because
# every forcing index in the OMIP drivers is a pure function of it (``_idx_t``,
# ``_runoff_month_idx``), so persisting it reproduces the forcing exactly.


def _field_meta(f: Field) -> list:
    """Field aux-data as a JSON-able list.

    ``Field.tree_flatten`` puts ALL FIVE metadata items in ``aux_data``, i.e.
    they are part of the pytree TREEDEF.  Rebuilding a restored carry with
    generic metadata therefore changes the treedef and breaks the very things
    the carry exists for — ``lax.scan``'s equal-treedef contract (the leapfrog
    carry deliberately stores ``u_before`` under the name ``'u'`` for exactly
    this reason) and any ``tree_map`` against a step output.  So the metadata
    is persisted and restored verbatim rather than re-derived from a template
    that, for an optional slot, is ``None`` anyway.
    """
    return [f.name, list(f.dims), f.units, f.long_name, f.staggering]


def _field_from_meta(data, meta: list | None, fallback_name: str) -> Field:
    if meta is None:            # archive predates metadata for this slot
        return Field(data=data, name=fallback_name)
    name, dims, units, long_name, staggering = meta
    return Field(data=data, name=name, dims=tuple(dims), units=units,
                 long_name=long_name, staggering=staggering)


def _encode_slot(name: str, value, payload: dict) -> dict:
    """Serialise one state slot into ``payload``; return its manifest entry.

    Raises ``TypeError`` for a value the format does not understand — see the
    module note above: a silently dropped carry is the defect being prevented.
    """
    if isinstance(value, Field):
        payload[name] = np.asarray(value.data)
        return {"kind": _KIND_FIELD, "meta": _field_meta(value)}
    # EXACT tuple, not any sequence: a list or a NamedTuple would be restored
    # as a plain tuple, silently changing the pytree treedef (codex r1 LOW).
    if type(value) is tuple:
        metas: list = []
        for i, elem in enumerate(value):
            key = f"{name}{_TUPLE_SEP}{i}"
            if isinstance(elem, Field):
                payload[key] = np.asarray(elem.data)
                metas.append(_field_meta(elem))
            elif hasattr(elem, "shape") and hasattr(elem, "dtype"):
                payload[key] = np.asarray(elem)
                metas.append(None)
            else:
                raise TypeError(
                    f"save_run_restart: carry slot {name!r} element {i} has "
                    f"unsupported type {type(elem).__name__}; the restart "
                    "format serialises Fields, arrays and tuples of them.  "
                    "Extend _encode_slot/_decode_slot rather than letting the "
                    "carry be dropped.")
        return {"kind": _KIND_TUPLE, "n": len(value), "elems": metas}
    # Arrays only — NOT bare Python scalars, which would come back as a JAX
    # 0-D array and change the leaf type (codex r1 LOW).
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        payload[name] = np.asarray(value)
        return {"kind": _KIND_ARRAY}
    raise TypeError(
        f"save_run_restart: carry slot {name!r} has unsupported type "
        f"{type(value).__name__}; the restart format serialises Fields, "
        "arrays and tuples of them.  Extend _encode_slot/_decode_slot rather "
        "than letting the carry be dropped (an uncheckpointed carry re-spins "
        "from its cold-start seed and the resumed run is not a continuation).")


def _decode_slot(name: str, entry: dict, loaded: dict, path: Path):
    """Rebuild one state slot from ``loaded`` using its manifest ``entry``.

    The manifest is authoritative for BOTH the container kind and the ``Field``
    metadata: the template's optional carry slots are ``None`` and so carry no
    metadata to copy, and the writer's metadata is what the step that produced
    the carry actually used.
    """
    def _need(key):
        if key not in loaded:
            raise ValueError(
                f"load_run_restart: {path} records carry slot {name!r} in its "
                f"manifest but the array {key!r} is missing — the archive is "
                "truncated or was written by an incompatible writer.  Refusing "
                "to resume with a silently cold-started carry.")
        return loaded[key]

    def _as_jnp(arr):
        # The SAVED dtype is authoritative (it is the precision the writing run
        # integrated at).  Refuse a silent demotion: an f64 archive read back
        # as f32 because JAX_ENABLE_X64 is unset is not a continuation, and it
        # would also flip the carry pytree's dtype mid-integration.
        out = jnp.asarray(arr)
        if out.dtype != arr.dtype:
            raise ValueError(
                f"load_run_restart: carry slot {name!r} was saved as "
                f"{arr.dtype} but JAX reads it back as {out.dtype}.  Set "
                "JAX_ENABLE_X64 to match the run that wrote the restart; "
                "resuming at a demoted precision is not a continuation.")
        return out

    kind = entry.get("kind")
    if kind == _KIND_FIELD:
        return _field_from_meta(_as_jnp(_need(name)), entry.get("meta"), name)
    if kind == _KIND_ARRAY:
        # Raw-array carries (the rigid-lid psi/dpsi/dpsin quintet) have no
        # Field metadata by construction.
        return _as_jnp(_need(name))
    if kind == _KIND_TUPLE:
        n = int(entry["n"])
        metas = entry.get("elems") or [None] * n
        out = []
        for i in range(n):
            data = _as_jnp(_need(f"{name}{_TUPLE_SEP}{i}"))
            out.append(_field_from_meta(data, metas[i], f"{name}_{i}")
                       if metas[i] is not None else data)
        return tuple(out)
    raise ValueError(
        f"load_run_restart: {path} declares unknown kind {kind!r} for carry "
        f"slot {name!r} (expected one of {_KIND_FIELD!r}, {_KIND_ARRAY!r}, "
        f"{_KIND_TUPLE!r}).")


def classify_restart_slots(state) -> tuple[list[str], list[str]]:
    """Partition ``state``'s NON-``None`` slots into (persist, excluded).

    ``persist`` = the prognostic + static slots; ``excluded`` = the slots
    labelled DIAGNOSTIC in :data:`_SLOT_POLICY` (recomputed every step).

    Raises ``KeyError`` if ANY declared field of the state class — populated or
    not — is missing from the policy.  Checking every field rather than only
    the live ones matters because a newly added carry defaults to ``None``: a
    value-only check would pass at setup and only blow up months later, at the
    first checkpoint of the first run that turns the new gate on (codex r1).
    """
    unclassified = sorted(n for n in _iter_state_fields(state)
                          if n not in _SLOT_POLICY)
    if unclassified:
        raise KeyError(
            f"restart persistence policy has no entry for "
            f"{type(state).__name__} slot(s) {unclassified}.  Classify each "
            f"one in legoesm.ocean.restart._SLOT_POLICY as "
            f"{_SLOT_PROGNOSTIC!r} (integrator state / carry the next step "
            f"reads), {_SLOT_STATIC!r} (geometry or a fixed reference profile) "
            f"or {_SLOT_DIAGNOSTIC!r} (recomputed every step).  Leaving a slot "
            "unclassified is refused because a dropped carry resumes from its "
            "cold-start seed and the run is silently not a continuation.")
    persist, excluded = [], []
    for name in _iter_state_fields(state):
        if getattr(state, name) is None:
            continue
        (excluded if _SLOT_POLICY[name] == _SLOT_DIAGNOSTIC
         else persist).append(name)
    return persist, excluded


def _state_inventory(state) -> dict[str, str]:
    """EXACT per-slot inventory of ``state``: {slot: persisted|diagnostic|absent}.

    Recorded in the archive so the loader can demand an exact layout instead of
    accepting any subset of the manifest.  Without it, deleting a slot from the
    manifest silently leaves the fresh template's cold-start value in place, and
    a 3-field ``SeaIceState`` archive would load into a 12-field
    ``DynamicSeaIceState`` template with nine fields cold (codex r1 HIGH).
    """
    persist, excluded = classify_restart_slots(state)   # also validates policy
    inv = {}
    for name in _iter_state_fields(state):
        if name in excluded:
            inv[name] = _INV_DIAGNOSTIC
        elif name in persist:
            inv[name] = _INV_PERSISTED
        else:
            inv[name] = _INV_ABSENT
    return inv


def validate_restart_policy(*states) -> None:
    """Fail FAST if any state class declares a slot the policy misses.

    Call once at driver setup when ``--restart-save``/``--restart-from`` is in
    play: an unclassified field then aborts within seconds instead of at the
    first mid-run checkpoint, hours into an integration.
    """
    for state in states:
        if state is not None:
            classify_restart_slots(state)


def _encode_state(state, payload: dict, prefix: str = "") -> dict[str, dict]:
    """Serialise the PERSIST slots of ``state``; return {slot: entry}.

    Diagnostic slots are excluded by policy (see :data:`_SLOT_POLICY`); an
    unclassified slot raises rather than being dropped.
    """
    persist, _excluded = classify_restart_slots(state)
    kinds: dict[str, dict] = {}
    for name in persist:
        kinds[name] = _encode_slot(prefix + name, getattr(state, name), payload)
    return kinds


def save_run_restart(path: str | Path, state, *,
                     step: int,
                     time_days: float,
                     grid_type: str,
                     dt_seconds: float | None = None,
                     n_forcing_records: int | None = None,
                     ice_state=None,
                     sha: str | None = None) -> Path:
    """Write a RESUMABLE checkpoint: full ocean carry + sea ice + step counter.

    Parameters
    ----------
    path : str or Path
        Output ``.npz`` path (written atomically).
    state : ocean state NamedTuple
        Every non-``None`` PROGNOSTIC and STATIC slot is serialised —
        prognostics AND integrator carries (AB2 / leapfrog / centred-forcing /
        EKE / TKE / rigid-lid / ``bt_hist``).  DIAGNOSTIC slots are excluded by
        policy; an unclassified slot, or an unsupported slot value type,
        raises.
    step : int
        Absolute step index reached.  The OMIP forcing indices are pure
        functions of ``(step, dt, n_rec)``, so persisting all three (see
        ``dt_seconds`` / ``n_forcing_records``) reproduces the forcing exactly.
    dt_seconds, n_forcing_records : optional
        Continuation fingerprint.  Recorded and, when the loader is given the
        resuming run's values, compared — a resume at a different dt or
        against a different forcing archive lands on different records.
    time_days : float
        Model day reached (provenance + resume logging).
    grid_type : str
        Grid identity of the writing run; a resume onto a different grid is
        refused (a cross-grid template can pass per-field shape checks by
        coincidence and be physically meaningless).
    ice_state : sea-ice state NamedTuple, optional
        Persisted UNMANGLED under ``ice_<field>`` keys — all slots, not the
        land-masked / category-aggregated pair the diagnostic snapshot writes.
    sha : str, optional
        Source revision, recorded for provenance.
    """
    out_path = Path(path)
    payload: dict[str, np.ndarray] = {}
    kinds = _encode_state(state, payload)
    ice_kinds: dict[str, dict] = {}
    if ice_state is not None:
        ice_kinds = _encode_state(ice_state, payload, prefix=_ICE_PREFIX)
    # Record what was DELIBERATELY excluded, so the archive itself documents
    # the decision (a reader can tell "absent by policy" from "lost").
    _excl = sorted(set(classify_restart_slots(state)[1])
                   | (set(classify_restart_slots(ice_state)[1])
                      if ice_state is not None else set()))
    payload["_excluded"] = np.asarray(json.dumps(_excl))
    payload["_format"] = np.asarray(int(_RUN_RESTART_FORMAT))
    payload["_step"] = np.asarray(int(step))
    payload["_time_days"] = np.asarray(float(time_days))
    payload["_grid_type"] = np.asarray(str(grid_type))
    payload["_slot_kinds"] = np.asarray(json.dumps(kinds, sort_keys=True))
    payload["_ice_slot_kinds"] = np.asarray(
        json.dumps(ice_kinds, sort_keys=True))
    # --- continuation fingerprint + exact inventory (codex r1 HIGH) --------
    # `step` alone does NOT pin the forcing: _idx_t(step, dt, n_rec) needs dt
    # and the record count too, so a 900 s parent resumed at 1800 s reads a
    # different record while passing every other check.  x64 pins the
    # precision the archive was integrated at.
    payload["_state_class"] = np.asarray(type(state).__name__)
    payload["_inventory"] = np.asarray(
        json.dumps(_state_inventory(state), sort_keys=True))
    payload["_ice_class"] = np.asarray(
        type(ice_state).__name__ if ice_state is not None else "")
    payload["_ice_inventory"] = np.asarray(json.dumps(
        _state_inventory(ice_state) if ice_state is not None else {},
        sort_keys=True))
    payload["_x64"] = np.asarray(bool(jax.config.jax_enable_x64))
    if dt_seconds is not None:
        payload["_dt_seconds"] = np.asarray(float(dt_seconds))
    if n_forcing_records is not None:
        payload["_n_forcing_records"] = np.asarray(int(n_forcing_records))
    if sha is not None:
        payload["_sha"] = np.asarray(str(sha))
    return _atomic_savez(out_path, payload)


def run_restart_metadata(path: str | Path) -> dict[str, Any]:
    """Return ``{format, step, time_days, grid_type, sha, slots, ice_slots,
    excluded}``.

    Cheap header read (no state template needed) for launchers that need the
    resume step / day before building the model.
    """
    in_path = Path(path)
    out: dict[str, Any] = {}
    with np.load(in_path, allow_pickle=False) as f:
        if "_slot_kinds" not in f.files:
            raise ValueError(
                f"{in_path} is not a run restart (no '_slot_kinds' manifest); "
                "it is probably a diagnostic snapshot or a legacy save_restart "
                "archive, which do not carry the full integrator state.")
        for key in ("_format", "_step", "_time_days", "_grid_type", "_sha"):
            if key in f.files:
                val = f[key]
                out[key[1:]] = (str(val) if val.dtype.kind in ("U", "S")
                                else val.item())
        for key in ("_dt_seconds", "_n_forcing_records", "_x64",
                    "_state_class", "_ice_class"):
            if key in f.files:
                val = f[key]
                out[key[1:]] = (str(val) if val.dtype.kind in ("U", "S")
                                else val.item())
        out["inventory"] = (json.loads(str(f["_inventory"]))
                            if "_inventory" in f.files else {})
        out["ice_inventory"] = (json.loads(str(f["_ice_inventory"]))
                                if "_ice_inventory" in f.files else {})
        out["slots"] = json.loads(str(f["_slot_kinds"]))
        out["ice_slots"] = json.loads(str(f["_ice_slot_kinds"])
                                      ) if "_ice_slot_kinds" in f.files else {}
        # Slots the writer deliberately did NOT persist (policy DIAGNOSTIC), so
        # a reader can distinguish "absent by design" from "lost".
        out["excluded"] = (json.loads(str(f["_excluded"]))
                           if "_excluded" in f.files else [])
    return out


def load_run_restart(path: str | Path, template_state, *,
                     ice_template=None,
                     grid_type: str | None = None,
                     dt_seconds: float | None = None,
                     n_forcing_records: int | None = None) -> tuple:
    """Resume from a :func:`save_run_restart` archive.

    Returns ``(state, ice_state, meta)`` where ``ice_state`` is ``None`` when
    the archive holds no sea ice, and ``meta`` is
    :func:`run_restart_metadata`'s dict.

    Every slot recorded in the archive's manifest MUST be restored; a missing
    array, an unknown kind, a grid-type mismatch, or an ice-state
    present/absent mismatch is a hard error.  Silence here would mean resuming
    a 60-day-spun-up ocean under a cold-start carry.
    """
    in_path = Path(path)
    meta = run_restart_metadata(in_path)
    fmt = int(meta.get("format", 0))
    if fmt != _RUN_RESTART_FORMAT:
        raise ValueError(
            f"load_run_restart: {in_path} has format version {fmt}, this build "
            f"reads version {_RUN_RESTART_FORMAT}.")
    if grid_type is not None and meta.get("grid_type") != grid_type:
        raise ValueError(
            f"load_run_restart: restart grid_type {meta.get('grid_type')!r} "
            f"does not match the run's grid_type {grid_type!r} ({in_path}); "
            "reconstructing a restart from another grid's template is "
            "physically meaningless even when the shapes happen to match.")

    # --- continuation fingerprint (codex r1 HIGH) -------------------------
    # The resumed leg must integrate the SAME system the parent did.  dt and
    # the forcing-record count are checked because the OMIP forcing index is
    # _idx_t(step, dt, n_rec): a parent at dt=900 resumed at dt=1800 lands on
    # a different record while every other check passes.
    def _require_same(label, saved, current, why):
        if saved is None or current is None:
            return
        if saved != current:
            raise ValueError(
                f"load_run_restart: {in_path} was written with {label}="
                f"{saved!r} but this run has {label}={current!r}.  {why}  "
                "Resuming across that change is a new experiment, not a "
                "continuation — start a fresh run, or match the parent.")

    _require_same("dt_seconds", meta.get("dt_seconds"),
                  float(dt_seconds) if dt_seconds is not None else None,
                  "The forcing index _idx_t(step, dt, n_rec) depends on dt, so "
                  "the resumed leg would read different CORE-II records.")
    _require_same("n_forcing_records", meta.get("n_forcing_records"),
                  int(n_forcing_records) if n_forcing_records is not None
                  else None,
                  "The forcing index wraps modulo the record count.")
    _require_same("x64", meta.get("x64"), bool(jax.config.jax_enable_x64),
                  "The archive was integrated at a different float precision.")
    # Internal consistency: time_days must be step*dt (a mismatch means the
    # writer's two provenance fields disagree, i.e. one of them is untrusted).
    _saved_dt = meta.get("dt_seconds")
    if _saved_dt:
        _expect_days = float(meta["step"]) * float(_saved_dt) / 86400.0
        if abs(_expect_days - float(meta["time_days"])) > 1e-6:
            raise ValueError(
                f"load_run_restart: {in_path} records step={meta['step']} and "
                f"dt={_saved_dt} s (=> day {_expect_days:.6f}) but "
                f"time_days={meta['time_days']}; the provenance is "
                "inconsistent, so neither value can be trusted to resume.")

    with np.load(in_path, allow_pickle=False) as f:
        loaded = {k: f[k] for k in f.files if k not in _RUN_RESERVED_KEYS}

    def _rebuild(template, kinds, prefix, inventory, saved_class, what):
        # --- EXACT layout, not a subset (codex r1 HIGH) --------------------
        # Without this, a manifest that simply OMITS a slot leaves the fresh
        # template's cold-start value in place and the resume is silently not
        # a continuation; and a 3-field SeaIceState archive would load into a
        # 12-field DynamicSeaIceState template with nine fields cold.
        cls = type(template).__name__
        if saved_class and saved_class != cls:
            raise ValueError(
                f"load_run_restart: {in_path} holds a {saved_class} {what} but "
                f"this run builds a {cls}.  These are different state layouts "
                "(e.g. the 3-field slab SeaIceState vs the 12-field "
                "DynamicSeaIceState); loading one into the other would leave "
                "the missing fields at their cold-start values.")
        if inventory:
            declared = set(_iter_state_fields(template))
            if set(inventory) != declared:
                raise ValueError(
                    f"load_run_restart: {in_path} records {what} slots "
                    f"{sorted(set(inventory) ^ declared)} that differ from "
                    f"this build's {cls}; the state layout changed between "
                    "the writing build and this one.")
            missing = sorted(n for n, lab in inventory.items()
                             if lab == _INV_PERSISTED and n not in kinds)
            if missing:
                raise ValueError(
                    f"load_run_restart: {in_path} declares {what} slot(s) "
                    f"{missing} as persisted but the manifest does not carry "
                    "them.  Refusing to resume with those cold-started.")
            # LIMITATION, stated plainly: this only fires when the resuming
            # run PRE-SEEDS the carry before loading.  run_omip_core2 builds a
            # rest state with every optional slot None and loads before the
            # first step, so a gate flipped ON between legs is NOT caught here
            # — closing that needs a resolved-config fingerprint, which is
            # listed as not-done (a naive config hash false-aborts a
            # --visc-schedule chain, whose model is rebuilt mid-leg).
            live = sorted(n for n, lab in inventory.items()
                          if lab == _INV_ABSENT
                          and getattr(template, n, None) is not None)
            if live:
                raise ValueError(
                    f"load_run_restart: {what} slot(s) {live} are populated in "
                    f"this run but were UNSET in {in_path}.  A carry that the "
                    "parent leg did not have cannot be continued — the gate "
                    "was turned on between legs, which is a new experiment.")
        return _rebuild_slots(template, kinds, prefix)

    def _rebuild_slots(template, kinds, prefix):
        # An archive that persisted a DIAGNOSTIC slot came from a writer with a
        # different policy; restoring it resurrects a stale value and can flip
        # an optional slot None->Field, breaking a scan carry's treedef.  This
        # is the run_omip asymmetry (write-but-drop) made impossible.
        diag = sorted(n for n in kinds
                      if _SLOT_POLICY.get(n) == _SLOT_DIAGNOSTIC)
        if diag:
            raise ValueError(
                f"load_run_restart: {in_path} persists diagnostic slot(s) "
                f"{diag}, which this build recomputes every step and refuses "
                "to restore (a stale diagnostic is not a continuation). "
                "Regenerate the restart with a matching build.")
        known = set(_iter_state_fields(template))
        unknown = sorted(set(kinds) - known)
        if unknown:
            raise ValueError(
                f"load_run_restart: {in_path} carries slot(s) {unknown} that "
                f"{type(template).__name__} does not define — the restart was "
                "written by a build with a different state layout.  Refusing "
                "to resume with those carries dropped.")
        replace_kw = {
            name: _decode_slot(prefix + name, entry, loaded, in_path)
            for name, entry in kinds.items()
        }
        out = template._replace(**replace_kw)
        # Mechanical non-drop check: every recorded slot is non-None on the
        # rebuilt state.  Cheap, and it turns any future decode regression
        # into a loud failure instead of a cold-started carry.
        dropped = sorted(n for n in kinds if getattr(out, n) is None)
        if dropped:
            raise ValueError(
                f"load_run_restart: slot(s) {dropped} were recorded in "
                f"{in_path} but are None after the rebuild (decode bug).")
        # Shape guard: a slot the template ALREADY populates has a known shape
        # (all prognostics; all 12 sea-ice fields).  A mismatch means the
        # resuming run was built at a different resolution / vertical levels /
        # ITD category count — the _replace would succeed and the model would
        # crash (or, worse, broadcast) many lines later.
        for name in kinds:
            ref, new = getattr(template, name), getattr(out, name)
            ref_d = ref.data if isinstance(ref, Field) else ref
            new_d = new.data if isinstance(new, Field) else new
            if not (hasattr(ref_d, "shape") and hasattr(new_d, "shape")):
                continue
            if tuple(ref_d.shape) != tuple(new_d.shape):
                raise ValueError(
                    f"load_run_restart: slot {name!r} in {in_path} has shape "
                    f"{tuple(new_d.shape)} but this run's "
                    f"{type(template).__name__} expects {tuple(ref_d.shape)}. "
                    "The resume was built at a different resolution / level "
                    "count / sea-ice category count than the leg that wrote "
                    "the restart.")
        # Static-geometry guard (see _STATIC_GEOMETRY_SLOTS): the resuming run
        # must have rebuilt the SAME mesh/bathymetry, else the archive's masks
        # would run against a different grid's metrics.
        for name in _STATIC_GEOMETRY_SLOTS:
            ref = getattr(template, name, None)
            new = getattr(out, name, None)
            if not (isinstance(ref, Field) and isinstance(new, Field)):
                continue
            # Relative tolerance, not exact equality: the mesh/bathymetry build
            # is host-side numpy and can differ in the last bit between nodes,
            # which must NOT abort a legitimate chained leg.  Any REAL change
            # (a different --min-levels, smoothing pass count or mask) moves
            # values by many orders of magnitude more than this, and a 0-vs-1
            # mask flip is caught because atol is 0.
            if (ref.data.shape != new.data.shape
                    or not bool(jnp.allclose(ref.data, new.data,
                                             rtol=1.0e-9, atol=0.0))):
                raise ValueError(
                    f"load_run_restart: static geometry slot {name!r} in "
                    f"{in_path} differs from the geometry this run built "
                    "(shape or values).  The resume is NOT a continuation — "
                    "rebuild with the same mesh / bathymetry / masking "
                    "arguments as the leg that wrote the restart.")
        return out

    state = _rebuild(template_state, meta["slots"], "",
                     meta.get("inventory"), meta.get("state_class"), "ocean")

    ice_kinds = meta["ice_slots"]
    if ice_kinds and ice_template is None:
        raise ValueError(
            f"load_run_restart: {in_path} holds a sea-ice state "
            f"({len(ice_kinds)} slots) but no ice_template was supplied — "
            "resume the run with prognostic sea ice enabled, or the pack "
            "would silently restart from the cold-start seed.")
    if ice_template is not None and not ice_kinds:
        raise ValueError(
            f"load_run_restart: prognostic sea ice is enabled for this run but "
            f"{in_path} holds no ice state; the pack would cold-start at the "
            "seed temperature/salinity and dump a large spurious surface flux "
            "on step 1.  Re-run the parent leg with sea ice, or start fresh.")
    ice_state = (_rebuild(ice_template, ice_kinds, _ICE_PREFIX,
                          meta.get("ice_inventory"), meta.get("ice_class"),
                          "sea-ice")
                 if ice_kinds else None)
    return state, ice_state, meta


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
    "save_run_restart", "load_run_restart", "run_restart_metadata",
    "classify_restart_slots", "validate_restart_policy",
]
