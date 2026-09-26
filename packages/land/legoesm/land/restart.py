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

Biophysics restarts by default; the prognostic-carbon lane
(``physics.carbon_prognostic``) additionally writes ``carbon_*`` pool fields
plus ``soil_frozen_fraction`` into the same npz (purely additive, so a
biophysics-only reader is unaffected)
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


def _is_default_soil_column(dz) -> bool:
    """Is this the historical default soil column the published states use?"""
    from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
    ref = np.asarray(make_soil_grid(SoilGridConfig()).dz, dtype=np.float64)
    return soil_dz_matches(np.asarray(dz, dtype=np.float64).reshape(-1), ref)


def _soil_dz_from(soil_grid=None, soil_dz=None, *, what: str):
    """The column as layer THICKNESSES [m], from either spelling, or ``None``.

    Two independent fixes for the same defect landed either side of a merge:
    one identifies a soil column by its layer INTERFACES, the other by its
    layer THICKNESSES. They are the same fact -- the interfaces are the running
    sum of the thicknesses, exactly -- so both spellings are accepted
    everywhere and reduced HERE to one form before anything is compared.
    Giving both is allowed only when they agree; disagreeing is a caller bug,
    not something to silently prefer one of.
    """
    if soil_grid is None and soil_dz is None:
        return None
    from_grid = None
    if soil_grid is not None:
        from legoesm.land.soil_grid import make_soil_grid
        from_grid = np.asarray(make_soil_grid(soil_grid).dz,
                               dtype=np.float64).reshape(-1)
    from_dz = (None if soil_dz is None
               else np.asarray(soil_dz, dtype=np.float64).reshape(-1))
    if from_grid is not None and from_dz is not None:
        if not soil_dz_matches(from_grid, from_dz):
            raise ValueError(
                f"{what}: the soil grid and the layer thicknesses describe "
                f"DIFFERENT columns ({from_grid.tolist()} m vs "
                f"{from_dz.tolist()} m). Pass one, or pass two that agree.")
    return from_grid if from_grid is not None else from_dz


def _recorded_soil_dz(data, path="<archive>"):
    """A restart archive's column as thicknesses [m], or ``None`` if unstamped.

    Reads either stamp, because files written by either lane are in the wild:
    ``soil_dz`` directly, or the differences of ``soil_z_interface``.

    A file carrying BOTH must have them AGREE.  The writer here derives both
    from one column at one moment so they cannot drift, but a file is not
    always written by this writer: an archive can be rewritten by a
    post-processing tool that updates one key and not the other, and a column
    built from interfaces rather than from thicknesses does not round-trip
    bitwise.  Two records of one fact that disagree are a corrupt file, so this
    refuses rather than picking a winner -- picking one is how the wrong soil
    profile gets used with no error, which is the defect this stamp exists to
    prevent (GLM-5.2).
    """
    dz = iz = None
    if "soil_dz" in data.files:
        dz = np.asarray(data["soil_dz"], dtype=np.float64).reshape(-1)
    if "soil_z_interface" in data.files:
        z = np.asarray(data["soil_z_interface"], dtype=np.float64).reshape(-1)
        # The relation is ENFORCED, not assumed: interfaces are n+1 depths
        # running from the surface downwards. A stamp that is not that is not
        # a column this reader can interpret, and guessing at one is worse
        # than saying so.
        # The SURFACE interface must be zero, not merely non-negative: the
        # thicknesses are the differences, so a column translated bodily
        # downwards ([1.0, 1.003, ..., 4.0] for a 3 m column) differences to
        # exactly the right thicknesses and would be accepted -- putting every
        # state value 1 m from where it belongs (codex).
        if (z.size < 2 or not np.all(np.diff(z) > 0.0)
                or abs(float(z[0])) > _SOIL_DZ_RTOL * float(z[-1] - z[0])):
            raise ValueError(
                f"{path}: soil_z_interface is not a set of increasing layer "
                f"interface depths starting AT the surface (got {z.tolist()}). "
                f"The soil column cannot be read from it.")
        iz = np.diff(z)
    if dz is not None and iz is not None and not soil_dz_matches(dz, iz):
        raise ValueError(
            f"{path} carries TWO soil-column stamps that disagree: layer "
            f"thicknesses {dz.tolist()} m versus interfaces implying "
            f"{iz.tolist()} m. One of them has been rewritten independently "
            f"of the other, so neither can be trusted to say which column "
            f"this state belongs to.")
    if dz is not None and (dz.size == 0 or not np.all(dz > 0.0)):
        raise ValueError(
            f"{path}: soil_dz is not a set of positive layer thicknesses "
            f"(got {dz.tolist()}).")
    return dz if dz is not None else iz


def save_land_restart(
    path,
    state,
    *,
    land_mode: str,
    t_end_s: float,
    n_steps_completed: int,
    metadata: dict[str, Any] | None = None,
    soil_grid=None,
    soil_dz=None,
    carbon_state=None,
    soil_frozen_fraction=None,
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
    soil_dz : array-like, optional
        The vertical soil column this state lives on, in either spelling: the
        grid itself, or its layer thicknesses [m].  Recorded so a continuation
        run can refuse a column the profile does not belong to.  The layer
        COUNT alone does not identify one: ten layers over 3 m and ten over
        6.4 m have identical array shapes, so without this a warm start
        silently reinterprets the temperature and moisture profile at the
        wrong depths.  BOTH stamps are written, from the one column given, so
        a reader that knows either spelling can check the file and the two can
        never disagree.

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
    _dz = _soil_dz_from(soil_grid, soil_dz, what="save_land_restart")
    if _dz is not None:
        payload["soil_dz"] = _dz
        # The interfaces are the running sum, so the two stamps are one fact
        # written twice rather than two facts that can drift.
        payload["soil_z_interface"] = np.concatenate(
            [np.zeros(1, dtype=np.float64), np.cumsum(_dz)])
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

    # --- optional prognostic carbon (additive; absent = biophysics-only) -----
    # Written as flat ``carbon_<field>`` arrays so a biophysics-only reader is
    # unaffected, and the permafrost phi alongside them: phi is NOT part of the
    # land state, so without it a resumed run loses the SOM protection that the
    # seeded high-latitude carbon was equilibrated under and the pools decay.
    if carbon_state is not None:
        for field in carbon_state._fields:
            payload[f"carbon_{field}"] = np.asarray(getattr(carbon_state, field))
    if soil_frozen_fraction is not None:
        payload["soil_frozen_fraction"] = np.asarray(soil_frozen_fraction)

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
    if a.shape != b.shape:
        return False
    # Relative only, no absolute floor. A reviewer argued one was needed:
    # recovering thicknesses by differencing interfaces was said to carry an
    # error of order eps x TOTAL depth, which on a 3 m column would be 1.4e-5
    # relative on a 2.6 cm top layer and would REFUSE a valid file. MEASURED
    # instead, over three columns including the thinnest top layer this model
    # builds: the worst relative error is 7.9e-8, a hundredfold inside the
    # tolerance. The argument assumed the surface interface carries the whole
    # column's rounding; it does not -- the shallow interfaces are themselves
    # small, so differencing near the surface subtracts small numbers and the
    # error scales with the LOCAL depth, not the total. A floor would have been
    # a knob that never binds.
    return bool(np.allclose(a, b, rtol=_SOIL_DZ_RTOL, atol=0.0))


def load_land_restart_soil_dz(path):
    """Return a restart's recorded soil layer thicknesses [m], or ``None``.

    Reads only that one array, so a caller can check a restart belongs to its
    soil column WITHOUT loading the state — which matters under MPI, where the
    state is loaded only by ranks that own land while this check must give the
    same answer everywhere.  ``None`` means the file predates the recording.
    """
    return _recorded_soil_dz(np.load(str(path), allow_pickle=False), path)


def load_land_restart(
    path,
    *,
    expected_land_mode: str,
    expected_ncol: int,
    expected_n_layers: int | None = None,
    expected_soil_grid=None,
    expected_soil_dz=None,
    require_soil_grid: bool = False,
    require_soil_dz: bool = False,
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
        initial states predate the stamp — unless ``require_soil_grid`` (or
        its ``require_soil_dz`` spelling) is set, which turns the unstamped
        case into a hard error.  Pass it when the run is on a column that is
        NOT the historical default: an old file carrying no stamp is then
        almost certainly on the other one, and warning about the file most
        people will load is not a check.

    The column may be given either as the grid (``expected_soil_grid``) or as
    its layer thicknesses (``expected_soil_dz``); both are reduced to
    thicknesses and compared once, so the two spellings cannot disagree about
    what "the same column" means.
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

    _want_dz = _soil_dz_from(expected_soil_grid, expected_soil_dz,
                             what="load_land_restart")
    if _want_dz is not None:
        # WHOSE column decides, and it is not a calibration flag. Three callers
        # asked for the strict behaviour by passing a preset's name, which left
        # every OTHER non-default column -- any run that sets its own layer
        # count or depth -- accepting an unstamped file and reading its profile
        # at the wrong depths. Both reviewers found this independently. The
        # predicate is a property of the column itself: a run on the historical
        # default may load an unstamped file, because the published initial
        # states are on that column and predate the stamp; a run on any other
        # column may not, because an unstamped file is then almost certainly on
        # the default one. The explicit flags remain, to force refusal on the
        # default column too.
        if not _is_default_soil_column(_want_dz):
            require_soil_grid = True
        _total = float(_want_dz.sum())
        _got_dz = _recorded_soil_dz(data, path)
        if _got_dz is None:
            _unstamped = (
                f"{path} records no soil column, so its layer depths cannot "
                f"be checked against this run's ({_want_dz.tolist()} m, total "
                f"{_total:.4g} m over {len(_want_dz)} layers).")
            if require_soil_grid or require_soil_dz:
                raise ValueError(
                    _unstamped + " This run's column is not the historical "
                    "default, so an unstamped file is almost certainly on a "
                    "different one and its soil profile would be read at the "
                    "wrong depths. Re-run the land spin-up on this run's "
                    "column, or drop the require flag if you know it matches.")
            warnings.warn(
                _unstamped + " Written before the column was stamped; if it "
                "came from a different one its profile is being reinterpreted "
                "at the wrong depths.", RuntimeWarning, stacklevel=2)
        elif not soil_dz_matches(_got_dz, _want_dz):
            raise ValueError(
                f"restart {path} was written on a soil column of "
                f"{_got_dz.tolist()} m (total {float(_got_dz.sum()):.6g} m in "
                f"{len(_got_dz)} layers), but this run uses "
                f"{_want_dz.tolist()} m (total {_total:.6g} m in "
                f"{len(_want_dz)}). The layer count alone cannot tell those "
                f"apart, so the arrays would load without complaint and the "
                f"temperature and moisture profile would be read at the wrong "
                f"depths.")

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
    # Prognostic carbon, when the writer had it.  Returned through ``meta`` so the
    # (state, meta) contract every existing caller unpacks is unchanged; both
    # entries are None for a biophysics-only restart.
    from legoesm.land.carbon.config import CarbonState
    _cfields = CarbonState._fields
    carbon = None
    if all(f"carbon_{f}" in data.files for f in _cfields):
        carbon = CarbonState(**{f: jnp.asarray(data[f"carbon_{f}"]) for f in _cfields})
    elif any(f"carbon_{f}" in data.files for f in _cfields):
        _missing = [f for f in _cfields if f"carbon_{f}" not in data.files]
        raise ValueError(
            f"{path}: restart carries SOME carbon pools but is missing {_missing}; "
            "refusing to resume from a partially-written carbon state.")
    meta = {
        "restart_version": version,
        "land_mode": land_mode,
        "t_end_s": float(data["t_end_s"]),
        "n_steps_completed": int(data["n_steps_completed"]),
        "metadata": json.loads(str(data["metadata_json"])) if "metadata_json" in data.files else {},
        "carbon_state": carbon,
        "soil_frozen_fraction": (jnp.asarray(data["soil_frozen_fraction"])
                                 if "soil_frozen_fraction" in data.files else None),
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
