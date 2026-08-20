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
    soil_dz=None,
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
    soil_dz : array-like, optional
        The producing run's soil layer thicknesses [m], ``(n_layers,)``.  Written
        so a consumer can refuse a restart whose column has the same LAYER COUNT
        but different layer DEPTHS — the loader's ncol/n_layers checks let such a
        file through, and its temperature / moisture profile would then be read
        at the wrong depths without any error.  Omit only for a producer that
        genuinely has no soil grid.

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
    if soil_dz is not None:
        payload["soil_dz"] = np.asarray(soil_dz, dtype=np.float64).reshape(-1)
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
    if "soil_dz" not in data.files:
        return None
    return np.asarray(data["soil_dz"], dtype=np.float64).reshape(-1)


def load_land_restart(
    path,
    *,
    expected_land_mode: str,
    expected_ncol: int,
    expected_n_layers: int | None = None,
    expected_soil_dz=None,
    soil_dz_rtol: float = _SOIL_DZ_RTOL,
    require_soil_dz: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Load a ``.npz`` restart and return ``(state, meta)``.

    ``state`` is a :class:`~legoesm.land.state.MultiLayerLandState` reconstructed
    from the file (with ``TgC=None`` when the file predates the EMA field).
    ``meta`` is a dict with ``t_end_s``, ``n_steps_completed``, ``land_mode``,
    ``restart_version``, and the informational ``metadata`` dict.

    ``expected_soil_dz`` (the consuming run's layer thicknesses [m]) closes the
    gap the shape checks leave open: two columns can share ``n_layers`` and still
    span different DEPTHS (8 layers over 3 m at growth 1.5 vs 8 over 6.375 m at
    growth 2), in which case every soil temperature and moisture value would be
    read at the wrong depth with no error anywhere.  A restart written before the
    thicknesses were recorded carries none, so it can only be WARNED about — pass
    ``require_soil_dz`` to make that unverifiable case a hard error instead, which
    a run whose column is NOT the historical default should always do.

    ``soil_dz_rtol`` defaults to the module's ``_SOIL_DZ_RTOL``; see the note
    there for why it is deliberately loose.

    Raises
    ------
    ValueError
        If the version is unknown, land_mode mismatches, or ncol / n_layers /
        soil layer thicknesses don't match the current grid (a silent mismatch
        would corrupt the continuation run).
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
    if expected_soil_dz is not None:
        _want_dz = np.asarray(expected_soil_dz, dtype=np.float64).reshape(-1)
        if "soil_dz" in data.files:
            _got_dz = np.asarray(data["soil_dz"], dtype=np.float64).reshape(-1)
            if not (soil_dz_matches(_got_dz, _want_dz)
                    if soil_dz_rtol == _SOIL_DZ_RTOL
                    else (_got_dz.shape == _want_dz.shape and np.allclose(
                        _got_dz, _want_dz, rtol=soil_dz_rtol, atol=0.0))):
                raise ValueError(
                    f"restart soil column does not match this run: restart layer "
                    f"thicknesses {_got_dz.tolist()} m (total "
                    f"{float(_got_dz.sum()):.4f} m) vs current "
                    f"{_want_dz.tolist()} m (total {float(_want_dz.sum()):.4f} m). "
                    "The soil profile would be read at the wrong depths. Re-run "
                    "the land spin-up on this run's soil column."
                )
        elif require_soil_dz:
            raise ValueError(
                f"restart {path} predates soil-column recording, so its layer "
                f"depths cannot be verified against this run's column "
                f"({_want_dz.tolist()} m, total {float(_want_dz.sum()):.4f} m). "
                "This run's column is not the historical default, so an older "
                "restart is most likely on the WRONG one and its soil profile "
                "would be read at the wrong depths. Re-run the land spin-up on "
                "this run's column."
            )
        else:
            logger.warning(
                "Land restart %s predates soil-column recording, so its layer "
                "depths CANNOT be verified against this run's column (%s m, "
                "total %.4f m). If it was spun up on a different column the soil "
                "profile is being read at the wrong depths. Re-run the spin-up to "
                "get a checkable restart.",
                path, _want_dz.tolist(), float(_want_dz.sum()),
            )

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
