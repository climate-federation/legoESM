"""Restart / checkpoint I/O for offline land-only runs (Phase C).

Saves and loads the full :class:`~legoesm.land.state.MultiLayerLandState` (and
optional canopy 30-day EMA ``TgC``) as a compressed ``.npz`` so a long spin-up
can span multiple queue submissions.  The driver auto-saves at the end of every
run under a MODEL-TIME-STAMPED name (``restart_<YEAR>_d<DDD>h<HH>.npz``) so a
directory of end-states is an audit trail (chronological on ``ls``) — no
overwrites::

    # first run: cold start, saves e.g. ``restart_1921_d000h00.npz``
    python scripts/run/run_lmip_biophys.py --year 1920 --n-steps 8760 \\
        --output $SCRATCH/spinup_yr01

    # second run: warm start off the previous end-state
    python scripts/run/run_lmip_biophys.py --year 1921 --n-steps 8760 \\
        --restart-from $SCRATCH/spinup_yr01/restart_1921_d000h00.npz \\
        --output $SCRATCH/spinup_yr02

Biophysics-only restarts (this branch's driver runs ``carbon.scheme="none"``).
Carbon-pool restarts are additive (append ``carbon_*`` fields to the same npz)
when we enable DALEC in a later push.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

# Relative tolerance for comparing a restart's soil layer thicknesses against the
# consuming run's.  Loose on purpose: both sides are the SAME geometric series
# recomputed, possibly one in single and one in double precision, while any real
# column difference is a fraction of the depth rather than a rounding difference.
_SOIL_DZ_RTOL = 1e-5

# Restart format version — bump when the payload schema changes so old
# checkpoints refuse to load rather than silently corrupt a run.
#   v1: core prognostic fields + optional TgC.
#   v2: adds the optional elevation-band / ponding water reservoirs
#       (surface_water, snow_bands, snow_age_bands, ice_bands) so a warm start
#       does not silently zero that water mass.  v1 files still load (they carry
#       none of the optional reservoirs — the legacy single-column case).
_RESTART_VERSION = 2
_SUPPORTED_VERSIONS = (1, 2)

# Only multilayer runs use this path today.  Slab is a natural next addition
# using the same version-tagged .npz layout.
_MULTILAYER_FIELDS = (
    "T_soil", "psi_soil", "theta_soil",
    "runoff_surface", "runoff_subsurface",
    "snow_depth", "snow_age",
)

# Optional prognostic WATER reservoirs (arrays or None).  Present only when the
# corresponding feature is enabled (elevation-band snow, surface ponding); each
# holds real mass, so it must round-trip or the warm start leaks it.  Serialised
# like ``TgC``: written only when not None, grafted on merge only when both the
# restart and the template carry it.
_MULTILAYER_OPTIONAL_ARRAY_FIELDS = (
    "surface_water", "snow_bands", "snow_age_bands", "ice_bands",
    # Intercepted canopy-water store (present iff interception is enabled); real
    # mass, so it must round-trip or the warm start leaks it.
    "W_canopy",
)


def save_land_restart(
    path,
    state,
    *,
    land_mode: str,
    t_end_s: float,
    n_steps_completed: int,
    metadata: dict[str, Any] | None = None,
    soil_grid=None,
) -> Path:
    """Write ``state`` and its bookkeeping to a compressed ``.npz`` restart file.

    Parameters
    ----------
    path : path-like
        Output ``.npz`` path; parent directory must exist.
    state : MultiLayerLandState
        The end-of-run state.  ``state.TgC`` is written when not ``None``.
    land_mode : str
        ``"multilayer"`` (only supported today).  Recorded so a slab-mode driver
        can't accidentally load a multilayer restart.
    t_end_s : float
        Model time reached at the end of the run, seconds since the run's
        ``year_start`` Jan 1 (noleap).  The next chained run uses this to know
        where to pick up in the forcing time axis.
    n_steps_completed : int
        Total number of land steps taken in the producing run (audit only).
    metadata : dict, optional
        Informational only (git SHA, grid_type, resolution, dt, CLI args, …);
        serialised as JSON alongside the arrays.  Not consumed on load.
    soil_grid : SoilGridConfig, optional
        The vertical soil grid this state lives on.  Its layer INTERFACES are
        written so a continuation run can refuse a grid the profile does not
        belong to.  The layer COUNT alone does not identify a grid: ten layers
        over 3 m and ten over 6.4 m have identical array shapes, so without
        this a warm start silently reinterprets the temperature and moisture
        profile at the wrong depths.

    Returns
    -------
    Path
        The written file's :class:`Path`.
    """
    if land_mode != "multilayer":
        raise NotImplementedError(f"restart for land_mode={land_mode!r} not implemented")

    payload = {
        "restart_version": np.array(_RESTART_VERSION, dtype=np.int32),
        "land_mode": np.array(land_mode, dtype="U16"),
        "t_end_s": np.array(float(t_end_s), dtype=np.float64),
        "n_steps_completed": np.array(int(n_steps_completed), dtype=np.int64),
        "metadata_json": np.array(json.dumps(metadata or {}), dtype="U65536"),
    }
    if soil_grid is not None:
        from legoesm.land.soil_grid import make_soil_grid
        payload["soil_z_interface"] = np.asarray(
            make_soil_grid(soil_grid).z_interface, dtype=np.float64)
    # The CLM-ML canopy carries a nested mlcanopy pytree, not a plain array; it
    # has no serialiser yet, so refuse loudly rather than silently drop it.
    if getattr(state, "canopy_state", None) is not None:
        raise NotImplementedError(
            "save_land_restart cannot yet serialise the CLM-ML canopy_state "
            "(nested pytree); restart support for the multilayer canopy scheme "
            "is not implemented."
        )

    for field in _MULTILAYER_FIELDS:
        payload[field] = np.asarray(getattr(state, field))
    tgc = getattr(state, "TgC", None)
    if tgc is not None:
        payload["TgC"] = np.asarray(tgc)
    # Optional water reservoirs — write each only when the feature is active.
    for field in _MULTILAYER_OPTIONAL_ARRAY_FIELDS:
        val = getattr(state, field, None)
        if val is not None:
            payload[field] = np.asarray(val)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out), **payload)
    return out


def soil_dz_matches(got, want) -> bool:
    """Do two soil columns describe the same layer thicknesses [m]?

    THE single comparison, so a caller checking a restart before the run and the
    loader checking it during the run can never disagree about what "the same
    column" means (codex).  See ``_SOIL_DZ_RTOL`` for why the tolerance is loose.
    """
    a = np.asarray(got, dtype=np.float64).reshape(-1)
    b = np.asarray(want, dtype=np.float64).reshape(-1)
    return a.shape == b.shape and bool(
        np.allclose(a, b, rtol=_SOIL_DZ_RTOL, atol=0.0))


def load_land_restart_soil_dz(path):
    """Return a restart's recorded soil layer thicknesses [m], or ``None``.

    Reads only that one array, so a caller can check a restart belongs to its
    soil column WITHOUT loading the state — which matters under MPI, where the
    state is loaded only by ranks that own land while this check must give the
    same answer everywhere.  ``None`` means the file predates the recording.
    """
    data = np.load(str(path), allow_pickle=False)
    if "soil_z_interface" not in data.files:
        return None
    # ONE stamp on disk, read two ways: the archive records layer INTERFACES,
    # and the thicknesses are their differences. Two records of one fact can
    # disagree; this reader kept its own key after the writer moved to
    # interfaces, which made it return "no stamp" for every file and silently
    # switched this check off.
    return np.diff(np.asarray(data["soil_z_interface"], dtype=np.float64).reshape(-1))


def load_land_restart(
    path,
    *,
    expected_land_mode: str,
    expected_ncol: int,
    expected_n_layers: int | None = None,
    expected_soil_grid=None,
    require_soil_grid: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Load a ``.npz`` restart and return ``(state, meta)``.

    ``state`` is a :class:`~legoesm.land.state.MultiLayerLandState` reconstructed
    from the file (with ``TgC=None`` when the file predates the EMA field).
    ``meta`` is a dict with ``t_end_s``, ``n_steps_completed``, ``land_mode``,
    ``restart_version``, and the informational ``metadata`` dict.

    Raises
    ------
    ValueError
        If the version is unknown, land_mode mismatches, or ncol / n_layers
        don't match the current grid (silent shape mismatch would corrupt the
        continuation run).  Also when ``expected_soil_grid`` is given and the
        file records DIFFERENT layer interfaces: ten layers over 3 m and ten
        over 6.4 m have the same array shapes, so the layer count alone cannot
        tell them apart and the profile would be reinterpreted at the wrong
        depths.  A file written before the interfaces were recorded cannot be
        checked; that warns rather than raising, because the published
        initial states predate the stamp — unless ``require_soil_grid`` is
        set, which turns the unstamped case into a hard error.  Pass it when
        the run is on a column that is NOT the historical default: an old file
        carrying no interfaces is then almost certainly on the other one, and
        warning about the file most people will load is not a check.
    """
    # Import here so importing this module doesn't drag the full land state class
    # (avoids a circular-import risk with land/__init__).
    from legoesm.land.state import MultiLayerLandState

    data = np.load(str(path), allow_pickle=False)
    version = int(data["restart_version"])
    if version not in _SUPPORTED_VERSIONS:
        raise ValueError(
            f"restart version {version} in {path} is not supported by this "
            f"loader (supported {_SUPPORTED_VERSIONS}); the schema has changed."
        )
    land_mode = str(data["land_mode"])
    if land_mode != expected_land_mode:
        raise ValueError(
            f"restart is for land_mode={land_mode!r} but this run uses "
            f"land_mode={expected_land_mode!r}"
        )

    T = jnp.asarray(data["T_soil"])
    if T.shape[0] != expected_ncol:
        raise ValueError(
            f"restart ncol={T.shape[0]} != current grid ncol={expected_ncol}"
        )
    if expected_n_layers is not None and T.shape[1] != expected_n_layers:
        raise ValueError(
            f"restart n_layers={T.shape[1]} != current config n_layers="
            f"{expected_n_layers}"
        )

    if expected_soil_grid is not None:
        from legoesm.land.soil_grid import make_soil_grid
        want = np.asarray(make_soil_grid(expected_soil_grid).z_interface,
                          dtype=np.float64)
        if "soil_z_interface" not in data.files:
            _unstamped = (
                f"{path} records no soil-layer interfaces, so its vertical "
                f"grid cannot be checked against this run's "
                f"({want[-1]:.4g} m over {len(want) - 1} layers).")
            if require_soil_grid:
                raise ValueError(
                    _unstamped + " This run is on a column that is not the "
                    "historical default, so an unstamped file is almost "
                    "certainly on a different one; refusing rather than "
                    "warning. Re-save the restart from a run that stamps it, "
                    "or drop require_soil_grid if you know the column matches.")
            warnings.warn(
                f"{path} records no soil-layer interfaces, so its vertical "
                f"grid cannot be checked against this run's "
                f"({want[-1]:.4g} m over {len(want) - 1} layers). Written "
                f"before the geometry was stamped; if it came from a "
                f"different soil column its profile is being reinterpreted "
                f"at the wrong depths.", RuntimeWarning, stacklevel=2)
        else:
            got = np.asarray(data["soil_z_interface"], dtype=np.float64)
            # Loose on purpose (``_SOIL_DZ_RTOL``): both sides are the SAME
            # geometric series recomputed, possibly one in single and one in
            # double precision, while any real column difference is a fraction
            # of the depth rather than a rounding difference.
            if got.shape != want.shape or not np.allclose(
                    got, want, rtol=_SOIL_DZ_RTOL, atol=0.0):
                raise ValueError(
                    f"restart {path} was written on a soil column of "
                    f"{got[-1]:.6g} m in {len(got) - 1} layers, but this run "
                    f"uses {want[-1]:.6g} m in {len(want) - 1}. The layer "
                    f"count matches, so the arrays would load without "
                    f"complaint and the temperature and moisture profile "
                    f"would be read at the wrong depths.")

    optional = {
        field: jnp.asarray(data[field])
        for field in _MULTILAYER_OPTIONAL_ARRAY_FIELDS
        if field in data.files
    }
    state = MultiLayerLandState(
        T_soil=T,
        psi_soil=jnp.asarray(data["psi_soil"]),
        theta_soil=jnp.asarray(data["theta_soil"]),
        runoff_surface=jnp.asarray(data["runoff_surface"]),
        runoff_subsurface=jnp.asarray(data["runoff_subsurface"]),
        snow_depth=jnp.asarray(data["snow_depth"]),
        snow_age=jnp.asarray(data["snow_age"]),
        TgC=jnp.asarray(data["TgC"]) if "TgC" in data.files else None,
        **optional,
    )
    meta = {
        "restart_version": version,
        "land_mode": land_mode,
        "t_end_s": float(data["t_end_s"]),
        "n_steps_completed": int(data["n_steps_completed"]),
        "metadata": json.loads(str(data["metadata_json"])) if "metadata_json" in data.files else {},
    }
    return state, meta


def merge_land_restart_into_template(loaded, template):
    """Return ``template`` with its prognostic fields replaced by ``loaded``'s.

    A restart round-trips the core prognostic fields (``_MULTILAYER_FIELDS`` +
    ``TgC``) and the optional water reservoirs
    (``_MULTILAYER_OPTIONAL_ARRAY_FIELDS``: ``surface_water``, ``snow_bands``,
    ``snow_age_bands``, ``ice_bands``).  ``step_multilayer_land`` populates the
    optional fields as arrays, so feeding a bare loaded state straight into a
    ``lax.scan`` (the coupled-AMIP segment, or a chained spin-up) raises a carry
    input/output pytree-structure mismatch.  Building from a freshly-initialised
    ``template`` (canonical structure) and grafting the restart's prognostic
    columns onto it fixes the structure while keeping the equilibrated values.

    Each optional reservoir is grafted only when BOTH the restart and the
    template carry it (same feature enabled on both ends); a v1 restart (no
    reservoirs) or a bands-off template falls back to the template's value, so
    no mass is invented and pytree structure is preserved.

    Mirrors the land-ml CHECKPOINT restore in ``model_driver`` (``template.
    _replace(**fields)``), with a shape check per field so a resolution /
    soil-layer skew fails loudly rather than silently reshaping.
    """
    fields = {}
    for name in _MULTILAYER_FIELDS:
        arr = getattr(loaded, name)
        ref = getattr(template, name)
        if ref is not None and hasattr(arr, "shape") and arr.shape != ref.shape:
            raise ValueError(
                f"land restart field '{name}' has shape {tuple(arr.shape)}, "
                f"expected {tuple(ref.shape)} (resolution / soil-layer skew)")
        fields[name] = arr
    if getattr(loaded, "TgC", None) is not None:
        fields["TgC"] = loaded.TgC
    for name in _MULTILAYER_OPTIONAL_ARRAY_FIELDS:
        arr = getattr(loaded, name, None)
        ref = getattr(template, name, None)
        if arr is None or ref is None:
            continue  # feature off on one end -> keep template structure, invent no mass
        if hasattr(arr, "shape") and hasattr(ref, "shape") and arr.shape != ref.shape:
            raise ValueError(
                f"land restart field '{name}' has shape {tuple(arr.shape)}, "
                f"expected {tuple(ref.shape)} (band-count / resolution skew)")
        fields[name] = arr
    return template._replace(**fields)


__all__ = [
    "save_land_restart", "load_land_restart",
    "merge_land_restart_into_template",
]
