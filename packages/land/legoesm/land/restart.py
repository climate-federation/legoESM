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
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

# Restart format version — bump when the payload schema changes so old
# checkpoints refuse to load rather than silently corrupt a run.
_RESTART_VERSION = 1

# Only multilayer runs use this path today.  Slab is a natural next addition
# using the same version-tagged .npz layout.
_MULTILAYER_FIELDS = (
    "T_soil", "psi_soil", "theta_soil",
    "runoff_surface", "runoff_subsurface",
    "snow_depth", "snow_age",
)


def save_land_restart(
    path,
    state,
    *,
    land_mode: str,
    t_end_s: float,
    n_steps_completed: int,
    metadata: dict[str, Any] | None = None,
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
    for field in _MULTILAYER_FIELDS:
        payload[field] = np.asarray(getattr(state, field))
    tgc = getattr(state, "TgC", None)
    if tgc is not None:
        payload["TgC"] = np.asarray(tgc)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out), **payload)
    return out


def load_land_restart(
    path,
    *,
    expected_land_mode: str,
    expected_ncol: int,
    expected_n_layers: int | None = None,
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
        continuation run).
    """
    # Import here so importing this module doesn't drag the full land state class
    # (avoids a circular-import risk with land/__init__).
    from legoesm.land.state import MultiLayerLandState

    data = np.load(str(path), allow_pickle=False)
    version = int(data["restart_version"])
    if version != _RESTART_VERSION:
        raise ValueError(
            f"restart version {version} in {path} is not supported by this "
            f"loader (expected {_RESTART_VERSION}); the schema has changed."
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

    state = MultiLayerLandState(
        T_soil=T,
        psi_soil=jnp.asarray(data["psi_soil"]),
        theta_soil=jnp.asarray(data["theta_soil"]),
        runoff_surface=jnp.asarray(data["runoff_surface"]),
        runoff_subsurface=jnp.asarray(data["runoff_subsurface"]),
        snow_depth=jnp.asarray(data["snow_depth"]),
        snow_age=jnp.asarray(data["snow_age"]),
        TgC=jnp.asarray(data["TgC"]) if "TgC" in data.files else None,
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

    A restart round-trips only the core prognostic fields (``_MULTILAYER_FIELDS``
    + ``TgC``); the OPTIONAL structural fields (``surface_water``,
    ``snow_bands``, ``ice_bands``, ``canopy_state``, …) come back at their
    NamedTuple ``None`` defaults.  But ``step_multilayer_land`` populates those
    as arrays, so feeding a bare loaded state straight into a ``lax.scan`` (the
    coupled-AMIP segment, or a chained spin-up) raises a carry input/output
    pytree-structure mismatch.  Building from a freshly-initialised ``template``
    (which has the canonical structure) and grafting the restart's prognostic
    columns onto it fixes the structure while keeping the equilibrated values.

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
    return template._replace(**fields)


__all__ = [
    "save_land_restart", "load_land_restart",
    "merge_land_restart_into_template",
]
