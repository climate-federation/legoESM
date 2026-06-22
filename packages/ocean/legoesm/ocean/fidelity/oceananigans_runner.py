"""Offline-reference loader for Oceananigans (Julia) oracle runs.

Counterpart to :mod:`legoesm.ocean.fidelity.mitgcm_runner`. Like MITgcm (and
unlike the in-process Veros runner), Oceananigans runs **out-of-process**: it is
a Julia model executed in an isolated depot (``JULIA_DEPOT_PATH=/tmp/ocn_depot``)
that dumps diagnostics to disk via ``NetCDFOutputWriter``. So this "runner" does
not run anything — it **loads** a reference archive that a prior run (the
generator under ``scripts/data/``) produced, and packages it as an
:class:`OceananigansResult` ready for the state bridge / ``compare``.

Reference layout (one directory per case)::

    <reference_root>/<case>/
        <case>.nc            # NetCDFOutputWriter dump (time series of fields)

``<reference_root>`` resolves to ``$LEGOESM_OCEAN_FIDELITY_OCEANANIGANS_REF`` if
set, else the ocean-fidelity cache's ``oceananigans/`` subdir. Each case lists
the prognostic and grid fields it expects; **unknown cases raise** (no silent
default) — the CLAUDE.md dispatch rule, mirrored from ``mitgcm_runner``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from legoesm.ocean.fidelity import cache as _cache
from legoesm.ocean.fidelity import oceananigans_io


@dataclass(frozen=True)
class OceananigansCaseSpec:
    """Static description of an Oceananigans reference case.

    ``prognostic_fields`` are the Oceananigans field names to load from the
    NetCDF; ``grid_fields`` are the coordinate vectors the bridge needs.
    ``cyclic_x`` records the zonal topology (``Periodic`` -> the C-grid bridge
    wraps the east face; ``Bounded`` -> wall). ``default_delta_t_s`` is only a
    fallback for ``times_s`` when the file carries no ``time`` coordinate.
    """

    name: str
    prognostic_fields: tuple[str, ...]
    grid_fields: tuple[str, ...]
    default_delta_t_s: float
    cyclic_x: bool
    reference: str


# Coordinate vectors a LatitudeLongitudeGrid case writes. A case lists the
# subset its dumped fields need; the loader records which were found.
_DEFAULT_GRID_FIELDS: tuple[str, ...] = ("xC", "xF", "yC", "yF", "zC", "zF")

# Registered cases. New cases land here (and get a reference-generation deck
# under scripts/data/). Dispatch is hardened: an unknown case_name raises.
KNOWN_CASES: dict[str, OceananigansCaseSpec] = {
    "barotropic_gyre": OceananigansCaseSpec(
        name="barotropic_gyre",
        # Homogeneous wind-driven gyre: barotropic flow (velocities). The 2-D
        # free surface eta is a follow-up (v0.110.x NetCDFWriter z-coord quirk).
        prognostic_fields=("u", "v"),
        grid_fields=_DEFAULT_GRID_FIELDS,
        default_delta_t_s=0.0,
        cyclic_x=False,  # longitude (-30, 30) Bounded
        reference="Oceananigans validation/barotropic_gyre "
                  "(LatLonGrid 60x60x1, lon(-30,30) lat(15,75) z(-4000,0), "
                  "ImplicitFreeSurface g=0.1, HydrostaticSphericalCoriolis "
                  "EnstrophyConserving, cos wind stress, linear bottom drag). "
                  "Isolates the barotropic solver + spherical Coriolis.",
    ),
    "bickley_jet": OceananigansCaseSpec(
        name="bickley_jet",
        # Barotropic shear instability: isolates the eddy-regime vector-invariant
        # momentum advection. Deterministic twin (same IC both codes); compare the
        # relative vorticity zeta. u,v at faces; zeta at the FF vertex.
        prognostic_fields=("u", "v", "zeta"),
        grid_fields=_DEFAULT_GRID_FIELDS,
        default_delta_t_s=0.0,
        cyclic_x=True,  # zonally periodic jet
        reference="Oceananigans validation/bickley_jet "
                  "(unstable Bickley shear jet; isolates the eddy-regime "
                  "vector-invariant momentum advection / inverse cascade).",
    ),
}

DEFAULT_REFERENCE_ENV = "LEGOESM_OCEAN_FIDELITY_OCEANANIGANS_REF"


class OceananigansReferenceError(RuntimeError):
    """Raised when an Oceananigans reference archive is missing or incomplete."""


@dataclass(frozen=True)
class OceananigansResult:
    """Loaded Oceananigans reference snapshot bundle.

    Mirrors ``MitgcmResult`` / ``VerosResult`` so the comparison layer is
    oracle-agnostic.

    Attributes
    ----------
    case_name : str
        The registered case name.
    times_s : np.ndarray
        Snapshot times in seconds (the file's ``time`` coordinate, or
        ``arange(n)*default_delta_t_s`` if absent).
    variables : dict[str, np.ndarray]
        Prognostic fields keyed by their **Oceananigans** name (translated to
        canonical legoESM names later by the state bridge), on-disk axis order.
    coords : dict[str, np.ndarray]
        Coordinate vectors found (``xC, xF, yC, yF, zC, zF``).
    grid_metadata : dict
        ``nx/ny/nz`` + the dimension sizes + ``cyclic_x``.
    provenance : dict
        Reference dir, file, dimension sizes, global attrs.
    """

    case_name: str
    times_s: np.ndarray
    variables: dict
    coords: dict
    grid_metadata: dict
    provenance: dict


def reference_root() -> Path:
    """Return the Oceananigans reference root (env override else cache subdir)."""
    explicit = os.environ.get(DEFAULT_REFERENCE_ENV)
    if explicit:
        root = Path(explicit).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        return root
    return _cache.sub("oceananigans")


def available_cases() -> tuple[str, ...]:
    """Return the registered case names."""
    return tuple(KNOWN_CASES)


def load_oceananigans_reference(
    case_name: str,
    *,
    ref_dir: str | Path | None = None,
    delta_t_s: float | None = None,
) -> OceananigansResult:
    """Load an Oceananigans reference archive for *case_name*.

    Parameters
    ----------
    case_name
        A key of :data:`KNOWN_CASES`. An unknown name raises ``ValueError``
        (dispatch hardening — no silent fallback).
    ref_dir
        Override the per-case directory (default
        ``reference_root()/<case_name>``).
    delta_t_s
        Override the case's ``default_delta_t_s`` (used only if the NetCDF has
        no ``time`` coordinate).

    Returns
    -------
    OceananigansResult

    Raises
    ------
    ValueError
        If *case_name* is not registered.
    OceananigansReferenceError
        If the reference directory or NetCDF file is missing.
    """
    if case_name not in KNOWN_CASES:
        raise ValueError(
            f"Unknown Oceananigans case {case_name!r}; "
            f"registered cases: {available_cases()}"
        )
    spec = KNOWN_CASES[case_name]

    case_dir = Path(ref_dir) if ref_dir is not None else (
        reference_root() / case_name
    )
    if not case_dir.is_dir():
        raise OceananigansReferenceError(
            f"Oceananigans reference dir for {case_name!r} not found: "
            f"{case_dir}. Generate it with "
            f"scripts/data/generate_oceananigans_{case_name}_reference.py "
            f"(or set ${DEFAULT_REFERENCE_ENV})."
        )

    nc_path = case_dir / f"{case_name}.nc"
    if not nc_path.is_file():
        # Best-effort: accept any single .nc in the dir.
        candidates = sorted(case_dir.glob("*.nc"))
        if not candidates:
            raise OceananigansReferenceError(
                f"No NetCDF (*.nc) reference found in {case_dir} for "
                f"{case_name!r}."
            )
        nc_path = candidates[0]

    raw = oceananigans_io.read_oceananigans_netcdf(
        nc_path, variables=spec.prognostic_fields,
    )

    times = raw.times
    if times.size == 0:
        # No time coordinate: synthesise from the leading axis + delta_t.
        dt = spec.default_delta_t_s if delta_t_s is None else delta_t_s
        any_field = next(iter(raw.variables.values()))
        times = np.arange(any_field.shape[0], dtype=float) * dt

    sizes = raw.sizes

    def _first(*names, default=None):
        for n in names:
            if n in sizes:
                return sizes[n]
        return default

    # Version-agnostic CENTRE-point counts. Old scheme: xC/yC/zC; v0.110.x
    # LatitudeLongitudeGrid: lon centre ``λ_caa``, lat centre ``φ_aca``, depth
    # centre ``z_aac`` (Face variants ``λ_faa``/``φ_afa``/``z_aaf`` are N+1).
    grid_metadata = {
        "nx": _first("xC", "λ_caa", "xF", "λ_faa"),
        "ny": _first("yC", "φ_aca", "yF", "φ_afa"),
        "nz": _first("zC", "z_aac", "zF", "z_aaf", default=1),
        "sizes": sizes,
        "cyclic_x": spec.cyclic_x,
    }
    provenance = {
        "case_dir": str(case_dir),
        "nc_file": str(nc_path),
        "sizes": sizes,
        "attrs": raw.attrs,
        "grid_fields_found": tuple(raw.coords),
    }
    return OceananigansResult(
        case_name=case_name,
        times_s=times,
        variables=raw.variables,
        coords=raw.coords,
        grid_metadata=grid_metadata,
        provenance=provenance,
    )
