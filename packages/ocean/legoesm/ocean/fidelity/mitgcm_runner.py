"""Offline-reference loader for MITgcm oracle runs.

The counterpart to :mod:`legoesm.ocean.fidelity.veros_runner`, but with a
fundamentally different execution model. Veros is a JAX model that the runner
*steps in-process*; MITgcm is a Fortran model that is built and run **separately**
(its own toolchain + MPI), dumping diagnostics to disk in ``mdsio`` format. So
this "runner" does not run anything — it **loads** a reference archive that a
prior MITgcm run (or the Slice-2 generator under ``scripts/data/``) produced, and
packages it as a :class:`MitgcmResult` ready for the state bridge / ``compare``.

Reference layout (one directory per case)::

    <reference_root>/<case>/
        U.0000000010.{meta,data}      # prognostic field dumps (per iteration)
        V.0000000010.{meta,data}
        Eta.0000000010.{meta,data}
        XC.{meta,data}  YC.{meta,data}  RC.{meta,data}  DRF.{meta,data}
        Depth.{meta,data}  hFacC.{meta,data}  hFacW.{meta,data}  hFacS.{meta,data}

``<reference_root>`` resolves to ``$LEGOESM_OCEAN_FIDELITY_MITGCM_REF`` if set,
else the ocean-fidelity cache's ``mitgcm/`` subdir. Each case lists the
prognostic and grid fields it expects; unknown cases raise (no silent default).

This module reads mdsio with the dependency-free
:mod:`legoesm.ocean.fidelity.mitgcm_io` reader by default. For multi-tile output
pass ``prefer_mitgcmutils=True`` to delegate tile-gluing to ``MITgcmutils.rdmds``
(lazy import; actionable error if absent).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from legoesm.ocean.fidelity import cache as _cache
from legoesm.ocean.fidelity import mitgcm_io


@dataclass(frozen=True)
class CaseSpec:
    """Static description of a MITgcm reference case.

    ``prognostic_fields`` are the per-iteration dumps to load (keyed by MITgcm
    field name); ``grid_fields`` are the time-independent grid descriptors.
    ``default_delta_t_s`` seeds ``times_s = iters * deltaT`` when the caller does
    not override it. ``reference`` documents the verification experiment.
    """

    name: str
    prognostic_fields: tuple[str, ...]
    grid_fields: tuple[str, ...]
    default_delta_t_s: float
    reference: str


# The grid descriptors MITgcm writes for a rectilinear ocean config. A case may
# legitimately lack some (e.g. a single-level run has trivial DRF); the loader
# treats grid fields as best-effort and records which were found.
_DEFAULT_GRID_FIELDS: tuple[str, ...] = (
    "XC", "YC", "RC", "DRF", "DXG", "DYG", "RAC", "Depth",
    "hFacC", "hFacW", "hFacS",
)

# Registered cases. New oracle cases land here (and get a reference-generation
# recipe under scripts/data/). Dispatch is hardened: an unknown case_name raises
# rather than silently falling back (CLAUDE.md dispatch rule).
KNOWN_CASES: dict[str, CaseSpec] = {
    "barotropic_gyre": CaseSpec(
        name="barotropic_gyre",
        # Homogeneous (tempStepping/saltStepping = .FALSE.) wind-driven gyre:
        # the only prognostic state is the barotropic flow + free surface.
        prognostic_fields=("U", "V", "Eta"),
        grid_fields=_DEFAULT_GRID_FIELDS,
        default_delta_t_s=1200.0,  # input/data deltaT for tutorial_barotropic_gyre
        reference="MITgcm verification/tutorial_barotropic_gyre "
                  "(Cartesian beta-plane Stommel/Munk gyre; 62x62x1, 20 km, "
                  "viscAh=400, f0=1e-4, beta=1e-11).",
    ),
}

DEFAULT_REFERENCE_ENV = "LEGOESM_OCEAN_FIDELITY_MITGCM_REF"


class MitgcmReferenceError(RuntimeError):
    """Raised when a MITgcm reference archive is missing or incomplete."""


@dataclass(frozen=True)
class MitgcmResult:
    """Loaded MITgcm reference snapshot bundle.

    Mirrors ``veros_runner.VerosResult`` so the comparison layer is
    oracle-agnostic.

    Attributes
    ----------
    case_name : str
        Registered case (key of :data:`KNOWN_CASES`).
    iters : np.ndarray
        Iteration numbers loaded (1-D, int).
    times_s : np.ndarray
        Model time in seconds for each iteration (``iters * deltaT``).
    variables : dict[str, np.ndarray]
        Prognostic fields keyed by **MITgcm** field name (``U``/``V``/``Eta``…).
        When one iteration is loaded each array has its native field shape
        (``(nz, ny, nx)`` or ``(ny, nx)``); for multiple iterations the first
        axis is time. Translate names with
        ``concept_registry.oracle_to_canonical("mitgcm")`` (the bridge does).
    grid_metadata : dict[str, Any]
        Grid descriptors that were found (``XC``, ``YC``, ``RC``, ``Depth``,
        ``hFacC`` …) plus derived scalars (``nx``, ``ny``, ``nz``, ``delta_t_s``).
    provenance : dict[str, Any]
        Reference dir, iterations, reader used, per-field source mtimes.
    """

    case_name: str
    iters: np.ndarray
    times_s: np.ndarray
    variables: dict[str, np.ndarray]
    grid_metadata: dict[str, Any]
    provenance: dict[str, Any]


def reference_root() -> Path:
    """Return the root holding per-case MITgcm reference archives.

    ``$LEGOESM_OCEAN_FIDELITY_MITGCM_REF`` overrides; otherwise the
    ocean-fidelity cache ``mitgcm/`` subdir is used (where the Slice-2 generator
    writes reference output).
    """
    explicit = os.environ.get(DEFAULT_REFERENCE_ENV)
    if explicit:
        root = Path(explicit).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        return root
    return _cache.sub("mitgcm")


def available_cases() -> tuple[str, ...]:
    """Return the registered MITgcm case names."""
    return tuple(sorted(KNOWN_CASES))


def _resolve_case_dir(case_name: str, ref_dir: Path | str | None) -> Path:
    if ref_dir is not None:
        case_dir = Path(ref_dir).expanduser()
    else:
        case_dir = reference_root() / case_name
    if not case_dir.is_dir():
        raise MitgcmReferenceError(
            f"MITgcm reference for case {case_name!r} not found at {case_dir}. "
            f"Set ${DEFAULT_REFERENCE_ENV} or pass ref_dir=, and generate the "
            f"reference with the Slice-2 generator (scripts/data/) — see "
            f"{KNOWN_CASES[case_name].reference}."
        )
    return case_dir


def _discover_iterations(case_dir: Path, field: str) -> list[int]:
    """Iteration numbers present for ``field`` (from ``<field>.<iter>.meta``)."""
    iters: set[int] = set()
    for meta in case_dir.glob(f"{field}.*.meta"):
        stem = meta.name[len(field) + 1 : -len(".meta")]
        # Skip tiled names (``field.<iter>.001.001.meta``) — those need rdmds.
        if stem.isdigit():
            iters.add(int(stem))
    return sorted(iters)


def _read_field(
    case_dir: Path, field: str, iteration: int | None, *, prefer_mitgcmutils: bool
) -> np.ndarray:
    if prefer_mitgcmutils:
        rd = _require_mitgcmutils()
        prefix = str(case_dir / field)
        if iteration is None:
            return np.asarray(rd(prefix))
        return np.asarray(rd(prefix, iteration))
    arr, _meta = mitgcm_io.read_mds(case_dir / field, iteration=iteration)
    return arr


def _require_mitgcmutils():
    """Lazy ``MITgcmutils.rdmds``; actionable error when missing."""
    try:
        from MITgcmutils import rdmds  # type: ignore
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "prefer_mitgcmutils=True needs the MITgcmutils package "
            "(`pip install MITgcmutils`). The dependency-free reader "
            "(prefer_mitgcmutils=False) handles single global-file mdsio, but "
            "not multi-tile output."
        ) from exc
    return rdmds


def load_mitgcm_reference(
    case_name: str,
    *,
    ref_dir: Path | str | None = None,
    iterations: int | tuple[int, ...] | None = None,
    delta_t_s: float | None = None,
    grid_fields: tuple[str, ...] | None = None,
    prefer_mitgcmutils: bool = False,
) -> MitgcmResult:
    """Load a registered MITgcm reference case into a :class:`MitgcmResult`.

    Parameters
    ----------
    case_name : str
        One of :func:`available_cases`. Unknown names raise ``ValueError``.
    ref_dir : path, optional
        Explicit case directory; defaults to ``reference_root()/<case_name>``.
    iterations : int | tuple[int], optional
        Iteration number(s) to load. ``None`` loads the latest available.
    delta_t_s : float, optional
        Timestep (seconds) for ``times_s``; defaults to the case spec value.
    grid_fields : tuple[str], optional
        Grid descriptors to load; defaults to the case spec list.
    prefer_mitgcmutils : bool, default False
        Use ``MITgcmutils.rdmds`` (handles multi-tile output) instead of the
        built-in single-file reader.
    """
    if case_name not in KNOWN_CASES:
        raise ValueError(
            f"Unknown MITgcm case {case_name!r}; available: {available_cases()}"
        )
    spec = KNOWN_CASES[case_name]
    case_dir = _resolve_case_dir(case_name, ref_dir)
    dt = float(delta_t_s if delta_t_s is not None else spec.default_delta_t_s)

    # Resolve the iteration list from the first prognostic field's dumps.
    probe_field = spec.prognostic_fields[0]
    if iterations is None:
        available = _discover_iterations(case_dir, probe_field)
        if not available:
            raise MitgcmReferenceError(
                f"no iteration dumps for {probe_field!r} found in {case_dir} "
                f"(looked for {probe_field}.<iter>.meta)."
            )
        iter_list: list[int] = [available[-1]]
    elif isinstance(iterations, int):
        iter_list = [iterations]
    else:
        iter_list = [int(i) for i in iterations]

    variables: dict[str, np.ndarray] = {}
    src_mtimes: dict[str, float] = {}
    for field in spec.prognostic_fields:
        per_iter = []
        for it in iter_list:
            per_iter.append(
                _read_field(case_dir, field, it, prefer_mitgcmutils=prefer_mitgcmutils)
            )
            meta_path = case_dir / f"{field}.{it:010d}.meta"
            if meta_path.exists():
                src_mtimes[meta_path.name] = meta_path.stat().st_mtime
        variables[field] = per_iter[0] if len(per_iter) == 1 else np.stack(per_iter)

    # Grid metadata (best-effort: a case may lack some descriptors).
    requested_grid = grid_fields if grid_fields is not None else spec.grid_fields
    grid_metadata: dict[str, Any] = {}
    for field in requested_grid:
        meta_path = case_dir / f"{field}.meta"
        if meta_path.exists():
            arr, _m = mitgcm_io.read_mds(case_dir / field)
            grid_metadata[field] = arr
            src_mtimes[meta_path.name] = meta_path.stat().st_mtime

    # Derive grid scalars. Use the DEEPEST-rank prognostic field for the spatial
    # shape so a 2-D field (e.g. Eta) listed first does not force nz=1 while a
    # sibling 3-D field carries the real vertical extent.
    spatial_samples = [
        (v[0] if len(iter_list) > 1 else v)  # strip a stacked time axis
        for v in variables.values()
    ]
    spatial = max(spatial_samples, key=lambda a: a.ndim)
    if spatial.ndim >= 2:
        grid_metadata["nx"] = int(spatial.shape[-1])
        grid_metadata["ny"] = int(spatial.shape[-2])
        grid_metadata["nz"] = int(spatial.shape[-3]) if spatial.ndim >= 3 else 1
    # The vertical coordinate descriptor, when present, is authoritative for nz.
    for zfield in ("RC", "DRF"):
        if zfield in grid_metadata:
            grid_metadata["nz"] = int(np.asarray(grid_metadata[zfield]).size)
            break
    grid_metadata["delta_t_s"] = dt

    iters_arr = np.asarray(iter_list, dtype=np.int64)
    times_s = iters_arr.astype(np.float64) * dt

    provenance = {
        "case_name": case_name,
        "reference_dir": str(case_dir),
        "iterations": iter_list,
        "delta_t_s": dt,
        "reader": "MITgcmutils.rdmds" if prefer_mitgcmutils else "mitgcm_io.read_mds",
        "source_mtimes": src_mtimes,
        "reference": spec.reference,
    }

    return MitgcmResult(
        case_name=case_name,
        iters=iters_arr,
        times_s=times_s,
        variables=variables,
        grid_metadata=grid_metadata,
        provenance=provenance,
    )
