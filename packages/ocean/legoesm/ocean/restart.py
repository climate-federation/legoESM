"""Restart harness for the legoESM ocean dycores.

Saves the full ocean prognostic state to disk as a numpy ``.npz``
archive and loads it back into an empty state container of the
matching grid.

SCOPE OF THE GUARANTEE -- read this before requoting it (codex r6/r7)
=====================================================================
WHAT IS DELIVERED.  The RUN pair (:func:`save_run_restart` /
:func:`load_run_restart`) persists every PROGNOSTIC and STATIC slot --
including the integrator carries that are not ``Field``s -- the
unmangled sea-ice state, and the ABSOLUTE step counter.  Arrays
round-trip losslessly through the ``.npz``, and the run LOADER refuses a
dtype demotion outright (``_decode_slot._as_jnp``), so an f64 archive
read back under ``JAX_ENABLE_X64`` unset raises instead of silently
resuming at f32.  Structurally, the loader demands an EXACT slot layout:
manifest, inventory and payload key names must agree in both directions,
unknown metadata keys are refused, and a diagnostic-persisting archive is
rejected.

EVERY CHECK IS CONDITIONAL ON WHAT THE CALLER SUPPLIES.  ``grid_type``,
``dt_seconds``, ``n_forcing_records`` and ``config_fingerprint`` are all
OPTIONAL arguments of :func:`load_run_restart`; each check runs only when
its argument is given.  ``run_omip_core2`` supplies all four, so its
resumes are fully checked; a direct caller that omits them gets
correspondingly less.  (The asymmetric case IS refused: supplying a value
the archive does not carry raises rather than silently skipping.)

WHAT IS NOT DELIVERED -- a cryptographically strict or bit-identical
continuation.  This is a guarded RECOVERY resume.  It detects
configuration DRIFT, archive TRUNCATION/CORRUPTION and STRUCTURAL
tampering (a slot renamed, relabelled, removed, resized, retyped, or
smuggled into the reserved metadata namespace).  It is NOT a proof that
the resumed leg reproduces an uninterrupted run bit for bit.  Concretely,
and by design:

* THERE IS NO PAYLOAD CHECKSUM.  An edit to the VALUES of a persisted
  array that keeps its shape and dtype -- overwriting ``T`` with a
  different temperature field, say -- passes every check and resumes
  silently (codex r7 HIGH).  Structural tampering is caught; value
  tampering is not.  Closing it needs a per-array digest or a signature
  over the whole archive, which this format does not carry;
* a SOURCE-REVISION mismatch WARNS and resumes (``run_omip_core2``):
  chaining a multi-day production run across a bug fix is a legitimate
  and expected workflow, so the model code may differ between legs;
* EXTERNAL INPUTS (the CORE-II forcing archive, mesh, WOA, sea-ice IC)
  are pinned by RESOLVED PATH, not by a content hash -- a file mutated
  in place at the same path passes every check here;
* the caller's config fingerprint is a digest of ``repr()``, which
  ELIDES the interior of a large array-valued setting;
* the static-geometry check uses ``rtol=1e-9`` -- deliberately, so a
  host-side mesh rebuild's last-bit jitter cannot false-abort a valid
  leg -- not exact equality;
* a COHERENTLY FORGED archive (inventory, manifest and fingerprint all
  rewritten together) is out of scope: detecting that needs a signature,
  which this format does not carry;
* nothing here pins the accelerator, the XLA version or the reduction
  order, so even a byte-identical state can step to different last bits
  on different hardware or a different backend.

FORCING REPRODUCTION IS CONDITIONAL ON THE CALLER (codex r7).  Every OMIP
forcing index is a pure function of ``(step, dt, n_records)``, but
``dt_seconds`` / ``n_forcing_records`` are OPTIONAL arguments on both
sides.  ``run_omip_core2`` always supplies them, so its resumes are
checked; a direct caller that omits them gets the step counter only.  The
loader does refuse the asymmetric case -- a caller that supplies a value
the archive lacks raises rather than silently skipping the check.

THE LEGACY PAIR IS WEAKER.  :func:`save_restart` / :func:`load_restart`
serialise ``Field``-valued slots ONLY (non-``Field`` carries are dropped
silently) and rebuild via a bare ``jnp.asarray``, which DEMOTES f64 to
f32 when ``JAX_ENABLE_X64`` is unset.  Use the run pair to resume an
integration; the legacy pair is for offline analysis snapshots.

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
    # Field-valued slots come back np.array_equal -- PROVIDED
    # JAX_ENABLE_X64 matches the writing run (this legacy loader does
    # NOT check that; the run loader does).

For RESUMING a driver mid-integration use the run-restart pair instead
(:func:`save_run_restart` / :func:`load_run_restart`): it persists every
PROGNOSTIC and STATIC state slot — including the non-``Field`` carries
``save_restart`` drops silently (rigid-lid ``psi``/``dpsi``/``dpsin``, the
``bt_hist`` tuple) — plus the unmangled sea-ice state and the absolute step
counter, which together with ``dt`` and the forcing-record count determine
every OMIP forcing index.  Slots labelled
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
# METADATA NAMESPACE of the RUN restart archive: every key written by
# ``save_run_restart`` that is NOT a state array carries a leading underscore
# (``_format``, ``_step``, ``_time_days``, ``_grid_type``, ``_x64``,
# ``_dt_seconds``, ``_n_forcing_records``, ``_config_fingerprint``, ``_parent``,
# ``_sha``, ``_state_class``, ``_ice_class``, ``_inventory``,
# ``_ice_inventory``, ``_slot_kinds``, ``_ice_slot_kinds``, ``_excluded``).
# A NamedTuple field name can never start with ``_`` (Python forbids it), so
# the prefix is a collision-free reserved namespace — the reader partitions on
# it rather than on an explicit list, which would go stale the moment a
# metadata key is added and would then make that key look like an orphan
# payload array.  ``run_restart_metadata`` enforces the REQUIRED subset.
#
# ONE list, used by BOTH sides: the writer asserts it emits nothing else, and
# the reader rejects any underscore key that is not in it.  A prefix-only rule
# was not airtight (codex r5 HIGH) — a payload array renamed to ``_tke`` was
# silently dropped as "metadata" and its carry cold-started.  A prefix-only
# rule in the other direction went stale the moment a metadata key was added.
# Sharing the set makes both failure modes impossible, and the save-time
# assertion means a future key that is forgotten here fails immediately, in
# the test suite, rather than at a resume months later.
_RUN_METADATA_KEYS: frozenset[str] = frozenset({
    "_format", "_step", "_time_days", "_grid_type", "_sha",
    "_slot_kinds", "_ice_slot_kinds", "_excluded",
    "_dt_seconds", "_n_forcing_records", "_x64",
    "_state_class", "_ice_class", "_inventory", "_ice_inventory",
    "_config_fingerprint", "_parent",
})
# Metadata that EVERY v2 archive must carry (the optional remainder is the
# caller-supplied fingerprint triple + _sha).
_RUN_METADATA_REQUIRED: tuple[str, ...] = (
    "_format", "_step", "_time_days", "_grid_type", "_x64",
    "_state_class", "_ice_class", "_inventory", "_ice_inventory",
    "_excluded", "_slot_kinds", "_ice_slot_kinds",
)
# Bumped when the on-disk layout changes incompatibly.
# 3: every manifest entry carries the payload's shape + dtype, so the loader
#    can cross-check the array against an independent record instead of relying
#    on the resuming template (whose optional carries are None) — codex r8.
_RUN_RESTART_FORMAT: int = 3
# Upper bound on a tuple-valued carry's element count.  The manifest's `n`
# drives key-name expansion BEFORE any array is read, so a hand-edited archive
# with a huge n would otherwise allocate that many strings.  bt_hist (6) is the
# largest tuple carry in the model; the bound is generous.
_MAX_TUPLE_CARRY: int = 64
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
# bug that broke the SPECTRAL model's restart in this repo before (#1310, an
# uncheckpointed mass-fixer anchor).  That is the failure mode this partition
# closes; it is not a claim that this format yields a bit-identical
# continuation — see the module docstring's SCOPE OF THE GUARANTEE.
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

    Unlike :func:`save_restart` (every prognostic field, so the arrays
    round-trip exactly), this writes ONLY what the de Boyer Montegut /
    Treguier (2023)
    MLD diagnostic and the OMIP-NEMO comparison consume: ``T``, ``S``,
    ``land_mask`` and ``H_bathy`` from the state, the horizontal grid
    (``lat_T``, ``lon_T`` in DEGREES), and the reference level-centre depths
    (``z_center_ref``, positive-down) the scorers use to build the per-level
    wet mask ``z_center_ref < H_bathy``.  ``u``/``v``/``eta`` are included
    when present (``v`` is absent on MPAS; ``eta`` lets an OFFLINE tool re-seed
    a state from this snapshot without a barotropic-adjustment shock).

    THIS IS NOT A RESTART.  ``--restart-from`` REJECTS a snapshot outright — it
    carries no ``_slot_kinds`` manifest, so :func:`load_run_restart` cannot
    tell a dropped carry from an absent one.  Re-seeding from a snapshot is an
    explicit offline workflow that cold-starts every integrator carry; use
    :func:`save_run_restart` to resume an integration.

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
# every forcing index in the OMIP drivers is a pure function of ``(step, dt,
# n_records)`` (``_idx_t``, ``_runoff_month_idx``): persisting the step
# reproduces the forcing exactly PROVIDED the other two match, which is what
# the optional ``dt_seconds``/``n_forcing_records`` checks are for.


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
    # SHAPE + DTYPE ARE PART OF THE MANIFEST (format 3, codex r8 HIGH).
    # Without them the loader could only compare a restored slot against the
    # TEMPLATE, and the template's optional carries are None in the resuming
    # run (run_omip_core2 loads into an all-None rest state), so their shape
    # check was skipped entirely and a wrong-shaped `tke` loaded.  The dtype
    # check was weaker still: ``_as_jnp`` only asserts JAX does not DEMOTE what
    # the archive holds, so an f64 ``T`` rewritten as f32 passed.  Recording
    # both here makes the manifest an independent record to cross-check the
    # payload against — the same cheap-tamper level as the orphan check, not a
    # defence against a coordinated rewrite of manifest AND array.
    def _spec(arr) -> dict:
        a = np.asarray(arr)
        return {"shape": list(a.shape), "dtype": str(a.dtype)}

    if isinstance(value, Field):
        payload[name] = np.asarray(value.data)
        return {"kind": _KIND_FIELD, "meta": _field_meta(value),
                **_spec(value.data)}
    # EXACT tuple, not any sequence: a list or a NamedTuple would be restored
    # as a plain tuple, silently changing the pytree treedef (codex r1 LOW).
    if type(value) is tuple:
        metas: list = []
        specs: list = []
        for i, elem in enumerate(value):
            key = f"{name}{_TUPLE_SEP}{i}"
            if isinstance(elem, Field):
                payload[key] = np.asarray(elem.data)
                metas.append(_field_meta(elem))
                specs.append(_spec(elem.data))
            elif hasattr(elem, "shape") and hasattr(elem, "dtype"):
                payload[key] = np.asarray(elem)
                metas.append(None)
                specs.append(_spec(elem))
            else:
                raise TypeError(
                    f"save_run_restart: carry slot {name!r} element {i} has "
                    f"unsupported type {type(elem).__name__}; the restart "
                    "format serialises Fields, arrays and tuples of them.  "
                    "Extend _encode_slot/_decode_slot rather than letting the "
                    "carry be dropped.")
        return {"kind": _KIND_TUPLE, "n": len(value), "elems": metas,
                "specs": specs}
    # Arrays only — NOT bare Python scalars, which would come back as a JAX
    # 0-D array and change the leaf type (codex r1 LOW).
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        payload[name] = np.asarray(value)
        return {"kind": _KIND_ARRAY, **_spec(value)}
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

    def _checked(key, spec):
        """The payload array, cross-checked against the manifest's own record.

        The MANIFEST is the independent witness (codex r8 HIGH): the template
        cannot supply an expected shape for a carry the resuming run leaves
        ``None``, and ``_as_jnp`` only proves JAX did not demote what the file
        holds — neither notices an array swapped for a different shape, or an
        f64 field rewritten as f32.  Comparing the array to the shape/dtype the
        writer recorded closes the cheap-edit case.
        """
        arr = _need(key)
        want_shape = tuple(spec["shape"])
        if tuple(arr.shape) != want_shape:
            raise ValueError(
                f"load_run_restart: {path} array {key!r} has shape "
                f"{tuple(arr.shape)} but its manifest entry records "
                f"{want_shape}.  The payload and the manifest disagree; the "
                "archive is corrupt or was edited.")
        if str(arr.dtype) != spec["dtype"]:
            raise ValueError(
                f"load_run_restart: {path} array {key!r} has dtype "
                f"{arr.dtype} but its manifest entry records "
                f"{spec['dtype']!r}.  Resuming at a different precision than "
                "the writing run is not a continuation, and the disagreement "
                "means one of the two records was edited.")
        return _as_jnp(arr)

    kind = entry.get("kind")
    if kind == _KIND_FIELD:
        return _field_from_meta(_checked(name, entry), entry.get("meta"), name)
    if kind == _KIND_ARRAY:
        # Raw-array carries (the rigid-lid psi/dpsi/dpsin quintet) have no
        # Field metadata by construction.
        return _checked(name, entry)
    if kind == _KIND_TUPLE:
        n = int(entry["n"])
        metas = entry.get("elems") or [None] * n
        specs = entry["specs"]
        out = []
        for i in range(n):
            data = _checked(f"{name}{_TUPLE_SEP}{i}", specs[i])
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
                     config_fingerprint: str | None = None,
                     parent: str | None = None,
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
    config_fingerprint : str, optional
        Digest of the writing run's RESOLVED configuration.  When the loader is
        given the resuming run's digest, a mismatch is refused: dt / n_rec /
        x64 alone do not pin the viscosity, EOS, mixing, barotropic, sea-ice or
        host-loop forcing settings, all of which change step N+1.
    parent : str, optional
        Path of the restart this leg was resumed FROM (lineage); empty for a
        fresh run.
    sha : str, optional
        Source revision, recorded for provenance.
    """
    out_path = Path(path)
    # Namespace guard (codex r5 LOW): the sea-ice slots are stored under an
    # ``ice_`` prefix, so an ocean slot literally named ``ice_<x>`` would alias
    # the ice payload for ``<x>``.  No such slot exists today; fail loudly if
    # one is ever added rather than silently corrupting one of the two.
    _clash = sorted(n for n in _iter_state_fields(state)
                    if n.startswith(_ICE_PREFIX))
    if _clash:
        raise ValueError(
            f"save_run_restart: ocean slot(s) {_clash} start with "
            f"{_ICE_PREFIX!r}, which is the sea-ice payload namespace; their "
            "arrays would alias the ice state's.  Rename the slot or change "
            "_ICE_PREFIX.")
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
    if config_fingerprint is not None:
        payload["_config_fingerprint"] = np.asarray(str(config_fingerprint))
    # Lineage: which restart this leg was resumed FROM (codex r2 LOW).
    payload["_parent"] = np.asarray(str(parent) if parent else "")
    if sha is not None:
        payload["_sha"] = np.asarray(str(sha))
    # Drift gate: every metadata key this writer emits must be declared in
    # _RUN_METADATA_KEYS, or the reader would treat it as an orphan payload.
    _undeclared = sorted(k for k in payload
                         if k.startswith("_") and k not in _RUN_METADATA_KEYS)
    if _undeclared:
        raise ValueError(
            f"save_run_restart: metadata key(s) {_undeclared} are not declared "
            "in _RUN_METADATA_KEYS; add them there so the reader does not "
            "mistake them for orphan payload arrays.")
    return _atomic_savez(out_path, payload)


def _validate_manifest_schema(kinds: Any, what: str, in_path: Path) -> None:
    """Structural schema for a slot manifest — checked BEFORE any array read.

    An unknown ``kind`` used to survive every key-name and layout check and was
    only rejected inside :func:`_decode_slot`, i.e. AFTER the entire payload had
    been inflated (codex r7 MEDIUM).  A malformed TUPLE entry was worse: the
    key-name expansion does ``int(entry["n"])``, so a missing or non-numeric
    ``n`` surfaced as a bare ``KeyError``/``ValueError`` with no diagnosis of
    what was actually wrong with the archive.
    """
    if not isinstance(kinds, dict):
        raise ValueError(
            f"{in_path}: the {what} slot manifest decodes to "
            f"{type(kinds).__name__}, expected an object; the archive is "
            "corrupt or hand-edited.")
    known = (_KIND_FIELD, _KIND_ARRAY, _KIND_TUPLE)

    def _check_spec(where, spec):
        """A shape/dtype record must be usable BEFORE any array is inflated."""
        if not isinstance(spec, dict):
            raise ValueError(
                f"{in_path}: {where} carries no shape/dtype record (got "
                f"{type(spec).__name__}); this build cannot cross-check the "
                "payload against the manifest without one.")
        shape = spec.get("shape")
        if (not isinstance(shape, list)
                or any(not isinstance(d, int) or isinstance(d, bool) or d < 0
                       for d in shape)):
            raise ValueError(
                f"{in_path}: {where} records shape {shape!r}; expected a list "
                "of non-negative integers.")
        if not isinstance(spec.get("dtype"), str) or not spec["dtype"]:
            raise ValueError(
                f"{in_path}: {where} records dtype {spec.get('dtype')!r}; "
                "expected a non-empty dtype string.")

    def _check_field_meta(where, meta):
        """``_field_from_meta`` unpacks EXACTLY five entries (codex r8)."""
        if meta is None:
            return
        if not isinstance(meta, list) or len(meta) != 5:
            raise ValueError(
                f"{in_path}: {where} carries Field metadata {meta!r}; expected "
                "a 5-element [name, dims, units, long_name, staggering] list. "
                "Rebuilding would fail mid-unpack, after the payload had "
                "already been inflated.")
        if not isinstance(meta[1], list) or any(not isinstance(d, str)
                                                for d in meta[1]):
            raise ValueError(
                f"{in_path}: {where} declares dims {meta[1]!r}; expected a "
                "list of strings (they become the Field's pytree aux-data).")

    for name, entry in kinds.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"{in_path}: {what} manifest slot {name!r} decodes to "
                f"{type(entry).__name__}, expected an object with a 'kind'.")
        kind = entry.get("kind")
        if kind not in known:
            raise ValueError(
                f"{in_path}: {what} manifest slot {name!r} declares kind "
                f"{kind!r}, which this build cannot decode (known: "
                f"{', '.join(known)}).  The archive was written by an "
                "incompatible writer or hand-edited; refusing to resume "
                "rather than inflate a payload we cannot rebuild.")
        where = f"{what} manifest slot {name!r}"
        if kind != _KIND_TUPLE:
            _check_spec(where, entry)
            if kind == _KIND_FIELD:
                _check_field_meta(where, entry.get("meta"))
            continue
        n = entry.get("n")
        # BOUNDED (codex r8): the key expansion below materialises n names, so
        # an unbounded n is a cheap denial of service on a hand-edited archive.
        # No state slot is a tuple of more than a handful of arrays (bt_hist,
        # the largest, is 6).
        if (not isinstance(n, int) or isinstance(n, bool)
                or not 0 <= n <= _MAX_TUPLE_CARRY):
            raise ValueError(
                f"{in_path}: {where} is a tuple carry with element count "
                f"{n!r}; expected an integer in [0, {_MAX_TUPLE_CARRY}].")
        elems = entry.get("elems")
        if elems is not None and (not isinstance(elems, list)
                                  or len(elems) != n):
            got = len(elems) if isinstance(elems, list) else repr(elems)
            raise ValueError(
                f"{in_path}: {where} declares n={n} but carries {got} "
                "element-metadata entries; the two records disagree, so "
                "neither can be trusted.")
        for i, m in enumerate(elems or []):
            _check_field_meta(f"{where} element {i}", m)
        specs = entry.get("specs")
        if not isinstance(specs, list) or len(specs) != n:
            got = len(specs) if isinstance(specs, list) else repr(specs)
            raise ValueError(
                f"{in_path}: {where} declares n={n} but carries {got} "
                "shape/dtype records; refusing to resume without them.")
        for i, s in enumerate(specs):
            _check_spec(f"{where} element {i}", s)


def _parse_run_metadata(f, in_path: Path) -> dict[str, Any]:
    """Extract + validate the header from an ALREADY-OPEN ``NpzFile``.

    Split out so :func:`load_run_restart` can read the header and the payload
    from ONE open of the archive.  Two opens raced with the atomic
    checkpoint replacement the writer performs (codex r5 HIGH): a cadence write
    landing between them yielded state from archive B under archive A's step
    and fingerprint, which silently breaks the continuation.
    """
    files = set(f.files)
    if "_slot_kinds" not in files:
        raise ValueError(
            f"{in_path} is not a run restart (no '_slot_kinds' manifest); "
            "it is probably a diagnostic snapshot or a legacy save_restart "
            "archive, which do not carry the full integrator state.")
    absent = [k for k in _RUN_METADATA_REQUIRED if k not in files]
    if absent:
        raise ValueError(
            f"{in_path} is missing required run-restart metadata {absent}; "
            "it is truncated, hand-edited, or was written by an older format "
            "retagged as the current one.  Refusing to resume: without the "
            "inventory the exact-layout checks that prevent a silently "
            "cold-started carry cannot run.")
    # UNKNOWN underscore keys are refused rather than ignored: silently
    # dropping them is exactly how a payload array renamed to ``_tke`` escaped
    # the orphan check (codex r5 HIGH).
    unknown = sorted(k for k in files
                     if k.startswith("_") and k not in _RUN_METADATA_KEYS)
    if unknown:
        raise ValueError(
            f"{in_path} carries unrecognised metadata key(s) {unknown}.  The "
            "archive is corrupt or was edited to smuggle a payload array into "
            "the reserved namespace; refusing to resume.")

    # HEADER SCHEMA (codex r6 LOW rank guard, tightened to a full type/range
    # schema at codex r7).  Every metadata value this writer emits is a 0-d
    # ``np.asarray(scalar)`` of a KNOWN dtype class:
    #  * rank: ``ndarray.item()`` also accepts any SIZE-ONE array and ``str()``
    #    of one silently yields ``"['x']"``, so a payload array reshaped to
    #    (1,) and renamed to a declared metadata key would be read as that
    #    scalar (and would simultaneously vanish from the orphan check, which
    #    skips the underscore namespace);
    #  * dtype/range: rank alone still accepted a 0-d FLOAT ``_step`` such as
    #    2.5.  With ``_time_days`` edited to match, every other check passed
    #    and ``run_omip_core2`` then truncated it with ``int()`` — resuming the
    #    state at one step and the forcing index at another.  ``_step`` and the
    #    record count are integral by construction, so demand it.
    def _scalar(key: str, kinds: str, *, lo=None, integral: bool = False):
        val = f[key]
        if val.ndim != 0:
            raise ValueError(
                f"{in_path}: metadata key {key!r} has shape {val.shape}, but "
                "every header field is written as a 0-d scalar.  A size-one "
                "array here means the archive was edited (a payload array "
                "renamed into the reserved metadata namespace); refusing to "
                "resume.")
        if val.dtype.kind not in kinds:
            raise ValueError(
                f"{in_path}: metadata key {key!r} has dtype {val.dtype} "
                f"(kind {val.dtype.kind!r}), but this field is written with a "
                f"dtype of kind {kinds!r}.  The archive was edited or written "
                "by an incompatible writer; refusing to resume.")
        if val.dtype.kind in "fc" and not np.isfinite(val):
            raise ValueError(
                f"{in_path}: metadata key {key!r} is {val.item()!r}, which is "
                "not finite.  Refusing to resume on an unusable header.")
        if integral and val.dtype.kind == "f" and float(val) != int(val):
            raise ValueError(
                f"{in_path}: metadata key {key!r} is {val.item()!r}, but it "
                "counts whole steps/records.  A fractional value would be "
                "TRUNCATED by the driver, placing the state at one step and "
                "the forcing index at another; refusing to resume.")
        if lo is not None and float(val) < lo:
            raise ValueError(
                f"{in_path}: metadata key {key!r} is {val.item()!r}, below the "
                f"minimum {lo}.  Refusing to resume on an impossible header.")
        return val

    # key -> (allowed dtype kinds, minimum, must-be-integral)
    _HDR = {
        "_format": ("iu", 1, True),
        "_step": ("iuf", 0, True),
        "_n_forcing_records": ("iuf", 1, True),
        "_time_days": ("fiu", 0, False),
        "_dt_seconds": ("fiu", None, False),   # >0 enforced just below
        "_x64": ("b", None, False),
        "_grid_type": ("US", None, False),
        "_sha": ("US", None, False),
        "_state_class": ("US", None, False),
        "_ice_class": ("US", None, False),
        "_config_fingerprint": ("US", None, False),
        "_parent": ("US", None, False),
    }
    out: dict[str, Any] = {}
    for key, (kinds, lo, integral) in _HDR.items():
        if key in files:
            val = _scalar(key, kinds, lo=lo, integral=integral)
            out[key[1:]] = (str(val) if val.dtype.kind in ("U", "S")
                            else val.item())
    # dt must be strictly positive: 0 would make the step<->day cross-check
    # below vacuous and the forcing index undefined.
    if "dt_seconds" in out and float(out["dt_seconds"]) <= 0.0:
        raise ValueError(
            f"{in_path}: metadata key '_dt_seconds' is {out['dt_seconds']!r}; "
            "a non-positive timestep cannot have produced this archive.")
    out["inventory"] = json.loads(str(_scalar("_inventory", "US")))
    out["ice_inventory"] = json.loads(str(_scalar("_ice_inventory", "US")))
    out["slots"] = json.loads(str(_scalar("_slot_kinds", "US")))
    out["ice_slots"] = json.loads(str(_scalar("_ice_slot_kinds", "US")))
    # Slots the writer deliberately did NOT persist (policy DIAGNOSTIC), so a
    # reader can distinguish "absent by design" from "lost".
    out["excluded"] = json.loads(str(_scalar("_excluded", "US")))
    # The JSON blobs must decode to the CONTAINER the readers assume: a bare
    # string or list here would sail through `.items()`-free code paths and
    # only fail much later, with a confusing message.
    for _k, _want in (("inventory", dict), ("ice_inventory", dict),
                      ("slots", dict), ("ice_slots", dict),
                      ("excluded", list)):
        if not isinstance(out[_k], _want):
            raise ValueError(
                f"{in_path}: metadata '_{_k}' decodes to "
                f"{type(out[_k]).__name__}, expected {_want.__name__}; the "
                "archive is corrupt or hand-edited.")
    _validate_manifest_schema(out["slots"], "ocean", in_path)
    _validate_manifest_schema(out["ice_slots"], "sea-ice", in_path)
    # A no-ice archive must be consistent across all three ice records
    # (codex r5 LOW): an empty manifest with a populated class, or vice versa,
    # means one of them was edited.
    _ice_present = (bool(out["ice_slots"]), bool(out["ice_inventory"]),
                    bool(out.get("ice_class")))
    if len(set(_ice_present)) != 1:
        raise ValueError(
            f"{in_path} has inconsistent sea-ice records "
            f"(slots={_ice_present[0]}, inventory={_ice_present[1]}, "
            f"class={_ice_present[2]}); the archive was edited.")
    return out


def run_restart_metadata(path: str | Path) -> dict[str, Any]:
    """Return the archive header.

    Keys: ``format``, ``step``, ``time_days``, ``grid_type``, ``x64``,
    ``state_class``, ``ice_class``, ``inventory``, ``ice_inventory``,
    ``slots``, ``ice_slots``, ``excluded``, ``parent``, and — when the writer
    supplied them — ``dt_seconds``, ``n_forcing_records``,
    ``config_fingerprint``, ``sha``.

    Cheap header read (no state template needed) for launchers that need the
    resume step / day before building the model.  :func:`load_run_restart`
    does NOT call this: it parses the header from its own single open so the
    header and the payload cannot come from two different archives.
    """
    in_path = Path(path)
    with np.load(in_path, allow_pickle=False) as f:
        return _parse_run_metadata(f, in_path)


# --------------------------------------------------------------------------
# HEADER-ONLY validation.
#
# Split out of ``load_run_restart`` (codex r6 MEDIUM) so every check that can
# be answered from the archive HEADER or from the payload KEY NAMES runs before
# a single array is decompressed.  Reading ``NpzFile.files`` costs only the zip
# central directory; ``f[key]`` inflates the member.  These functions therefore
# take ``meta`` / key names and NEVER an array.
# --------------------------------------------------------------------------

def _expected_payload_keys(kinds: dict, prefix: str) -> set[str]:
    """Payload array names a slot manifest implies (tuples expand per element)."""
    out: set[str] = set()
    for name, entry in kinds.items():
        if entry.get("kind") == _KIND_TUPLE:
            out.update(f"{prefix}{name}{_TUPLE_SEP}{i}"
                       for i in range(int(entry["n"])))
        else:
            out.add(prefix + name)
    return out


def _validate_run_header(meta: dict, in_path: Path, *,
                         grid_type: str | None,
                         dt_seconds: float | None,
                         n_forcing_records: int | None,
                         config_fingerprint: str | None) -> None:
    """Refuse a wrong-format / wrong-grid / wrong-configuration archive."""
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
        if current is None:
            return                      # caller opted out of this check
        if saved is None:
            # The CALLER asked for this check but the archive has no value:
            # a stripped/older archive must NOT silently bypass it (codex r3).
            raise ValueError(
                f"load_run_restart: {in_path} records no {label}, but this run "
                f"supplied {current!r} to check against.  The archive predates "
                "the continuation fingerprint or was stripped; refusing to "
                "resume, because the check that would catch a changed "
                "configuration cannot run.")
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
    # RESOLVED-CONFIG identity (codex r2 HIGH): dt/n_rec/x64 do not pin the
    # viscosity, EOS, mixing, barotropic or sea-ice settings, nor which forcing
    # archive is mounted — all of which change step N+1.  The caller supplies a
    # digest of its RESOLVED configuration; run_omip_core2 computes it ONCE at
    # setup so a --visc-schedule leg (whose model is rebuilt mid-run with a
    # different A_h) still matches its sibling legs.
    _require_same("config_fingerprint", meta.get("config_fingerprint"),
                  config_fingerprint,
                  "The resolved model / sea-ice / forcing configuration "
                  "differs from the leg that wrote the restart.")
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


def _validate_payload_keys(meta: dict, payload_keys, in_path: Path) -> None:
    """The manifests and the archive's array NAMES must agree exactly.

    ORPHAN PAYLOAD (codex r4 HIGH): every raw array in the archive must be
    referenced by one of the two manifests.  Without this, leaving a ``tke``
    array in the npz while relabelling the slot ``absent`` and deleting its
    manifest entry passes every inventory/manifest check — and the carry
    silently cold-starts.  That is a cheap corruption/tamper case, distinct
    from a fully coordinated rewrite (which needs a signature to detect), so
    it is worth closing rather than filing under "forgery".

    The MISSING direction (a manifest entry with no array) is checked here too
    rather than deep inside ``_decode_slot``, so a truncated archive is also
    refused before the surviving members are decompressed.
    """
    referenced = (_expected_payload_keys(meta["slots"], "")
                  | _expected_payload_keys(meta["ice_slots"], _ICE_PREFIX))
    present = set(payload_keys)
    orphans = sorted(present - referenced)
    if orphans:
        raise ValueError(
            f"load_run_restart: {in_path} contains array(s) {orphans} that no "
            "manifest entry references.  The archive is corrupt or was edited "
            "to hide a carry (relabelling a slot while leaving its data "
            "behind); refusing to resume rather than silently ignore them.")
    absent = sorted(referenced - present)
    if absent:
        raise ValueError(
            f"load_run_restart: {in_path} is missing array(s) {absent} that "
            "its manifest declares; the archive is truncated.  Refusing to "
            "resume with those carries cold-started.")


def _validate_run_layout(template, kinds: dict, inventory, saved_class,
                         what: str, in_path: Path) -> None:
    """Manifest/inventory/state-class layout checks (no payload array read)."""
    # Diagnostic-persisted check FIRST: it is the most specific diagnosis
    # of a policy-mismatched archive, and the generic inventory
    # cross-check below would otherwise mask it with a vaguer message.
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
    # An EMPTY or non-mapping inventory used to skip every layout check
    # (codex r3): it is a required key, so anything but a populated mapping
    # is a corrupt archive.
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError(
            f"load_run_restart: {in_path} has an empty or malformed "
            f"{what} inventory ({inventory!r}); without it the exact-layout "
            "checks cannot run and a carry could resume cold-started.")
    declared = set(_iter_state_fields(template))
    if set(inventory) != declared:
        raise ValueError(
            f"load_run_restart: {in_path} records {what} slots "
            f"{sorted(set(inventory) ^ declared)} that differ from "
            f"this build's {cls}; the state layout changed between "
            "the writing build and this one.")
    bad = sorted(f"{n}={lab}" for n, lab in inventory.items()
                 if lab not in (_INV_PERSISTED, _INV_DIAGNOSTIC,
                                _INV_ABSENT))
    if bad:
        raise ValueError(
            f"load_run_restart: {in_path} has invalid {what} inventory "
            f"label(s) {bad}; the archive is corrupt or hand-edited.")
    # Cross-check BOTH ways: the manifest and the inventory must agree
    # on exactly which slots were written (codex r2 HIGH).
    extra = sorted(n for n in kinds
                   if inventory.get(n) != _INV_PERSISTED)
    if extra:
        raise ValueError(
            f"load_run_restart: {in_path} carries {what} slot(s) "
            f"{extra} in its manifest that the inventory does not "
            "label 'persisted'; the two records disagree, so neither "
            "can be trusted.")
    missing = sorted(n for n, lab in inventory.items()
                     if lab == _INV_PERSISTED and n not in kinds)
    if missing:
        raise ValueError(
            f"load_run_restart: {in_path} declares {what} slot(s) "
            f"{missing} as persisted but the manifest does not carry "
            "them.  Refusing to resume with those cold-started.")
    # SCOPE: this only fires when the resuming run PRE-SEEDS the
    # carry before loading.  run_omip_core2 builds a rest state with
    # every optional slot None and loads before the first step, so it
    # is the CONFIG FINGERPRINT — not this check — that catches a gate
    # flipped ON between legs there (the driver hashes every
    # non-excluded CLI setting plus the resolved model/ice configs).
    # What neither catches is a deliberately forged archive whose
    # inventory and fingerprint were both rewritten coherently; this
    # format defends against drift and corruption, not forgery.
    live = sorted(n for n, lab in inventory.items()
                  if lab == _INV_ABSENT
                  and getattr(template, n, None) is not None)
    if live:
        raise ValueError(
            f"load_run_restart: {what} slot(s) {live} are populated in "
            f"this run but were UNSET in {in_path}.  A carry that the "
            "parent leg did not have cannot be continued — the gate "
            "was turned on between legs, which is a new experiment.")


def _rebuild_slots(template, kinds: dict, prefix: str, loaded: dict,
                   in_path: Path):
    """Decode the payload into ``template`` and check shapes / geometry.

    The ONLY stage that touches a payload array; everything that can reject an
    archive from its header runs before this (see :func:`_validate_run_header`
    / :func:`_validate_run_layout`).
    """
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


def load_run_restart(path: str | Path, template_state, *,
                     ice_template=None,
                     grid_type: str | None = None,
                     dt_seconds: float | None = None,
                     n_forcing_records: int | None = None,
                     config_fingerprint: str | None = None) -> tuple:
    """Resume from a :func:`save_run_restart` archive.

    Returns ``(state, ice_state, meta)`` where ``ice_state`` is ``None`` when
    the archive holds no sea ice, and ``meta`` is
    :func:`run_restart_metadata`'s dict.

    Every slot recorded in the archive's manifest MUST be restored; a missing
    array, an unknown kind, a grid-type mismatch, or an ice-state
    present/absent mismatch is a hard error.  Silence here would mean resuming
    a 60-day-spun-up ocean under a cold-start carry.

    These guards detect configuration DRIFT, archive TRUNCATION/CORRUPTION and
    STRUCTURAL tampering (a slot renamed, relabelled, removed or smuggled into
    the reserved metadata namespace).  THERE IS NO PAYLOAD CHECKSUM: an edit to
    a persisted array's VALUES that keeps its shape and dtype resumes silently.
    They do NOT certify a bit-identical continuation — see the module
    docstring's SCOPE OF THE GUARANTEE before quoting them as one.
    """
    in_path = Path(path)
    # ONE open for header AND payload (codex r5 HIGH): the writer replaces the
    # archive atomically at every cadence, so two opens could mix archives —
    # state from archive B under archive A's step and fingerprint.
    #
    # FAIL-FAST ORDERING WITHIN that one open (codex r6 MEDIUM).  ``NpzFile``
    # decompresses on ACCESS (``f[k]``), so materialising the payload dict up
    # front inflated the entire multi-GB state off disk only to reject it on a
    # one-line header mismatch.  Everything that needs the HEADER alone, and
    # everything that needs only the payload KEY NAMES (``f.files`` is read
    # from the zip directory, not the members), now runs FIRST; the arrays are
    # touched on the last line of the block.  The single open is preserved.
    with np.load(in_path, allow_pickle=False) as f:
        meta = _parse_run_metadata(f, in_path)
        payload_keys = [k for k in f.files if not k.startswith("_")]
        _validate_run_header(
            meta, in_path, grid_type=grid_type, dt_seconds=dt_seconds,
            n_forcing_records=n_forcing_records,
            config_fingerprint=config_fingerprint)
        _validate_payload_keys(meta, payload_keys, in_path)
        _validate_run_layout(template_state, meta["slots"],
                             meta.get("inventory"), meta.get("state_class"),
                             "ocean", in_path)
        ice_kinds = meta["ice_slots"]
        if ice_kinds and ice_template is None:
            raise ValueError(
                f"load_run_restart: {in_path} holds a sea-ice state "
                f"({len(ice_kinds)} slots) but no ice_template was supplied — "
                "resume the run with prognostic sea ice enabled, or the pack "
                "would silently restart from the cold-start seed.")
        if ice_template is not None and not ice_kinds:
            raise ValueError(
                f"load_run_restart: prognostic sea ice is enabled for this run "
                f"but {in_path} holds no ice state; the pack would cold-start "
                "at the seed temperature/salinity and dump a large spurious "
                "surface flux on step 1.  Re-run the parent leg with sea ice, "
                "or start fresh.")
        if ice_kinds:
            _validate_run_layout(ice_template, ice_kinds,
                                 meta.get("ice_inventory"),
                                 meta.get("ice_class"), "sea-ice", in_path)
        # LAST LINE OF THE BLOCK: every check above is header / key-name only,
        # so a wrong-config or corrupt archive is rejected without paying for
        # the decompression.
        loaded = {k: f[k] for k in payload_keys}

    # Payload-dependent reconstruction (decode + shape + static geometry).
    state = _rebuild_slots(template_state, meta["slots"], "", loaded, in_path)
    ice_state = (_rebuild_slots(ice_template, ice_kinds, _ICE_PREFIX, loaded,
                                in_path)
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
