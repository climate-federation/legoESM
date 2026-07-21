"""LES-truth suite: enumerable LES reference cases for tuning/comparing SCM
turbulence closures (docs/atmosphere/les_suite/LES_SUITE.md).

Public API is the case registry + default catalog; the LES→SCM forcing/truth
bridge and the scoring assembly are separate follow-ups.
"""
from __future__ import annotations

from .bridge import (
    BridgeError,
    LESReferenceArtifact,
    LESTruth,
    artifact_to_scm_forcing,
    diagnostic_truth,
    load_artifact,
    prognostic_truth,
    save_artifact,
    total_turbulent_flux,
)
from .catalog import (
    ANCHOR_CASES,
    dry_shear_buoyancy_grid,
    register_default_catalog,
)
from .counter_gradient import (
    CounterGradientResult,
    centered_dtheta_dz,
    counter_gradient_diagnostic,
    diagnose_truth,
)
from .registry import (
    CI_MARKERS,
    REGIMES,
    SGS_CHOICES,
    LESCase,
    LESGrid,
    RegistryError,
    clear_registry,
    get_case,
    get_cases_for_regime,
    list_cases,
    list_regimes,
    register_case,
)
from .score import (
    DiagnosticScore,
    PrognosticScore,
    diagnostic_flux_score,
    prognostic_profile_score,
)

__all__ = [
    "ANCHOR_CASES",
    "CI_MARKERS",
    "BridgeError",
    "CounterGradientResult",
    "DiagnosticScore",
    "LESCase",
    "LESGrid",
    "LESReferenceArtifact",
    "LESTruth",
    "PrognosticScore",
    "REGIMES",
    "RegistryError",
    "SGS_CHOICES",
    "artifact_to_scm_forcing",
    "centered_dtheta_dz",
    "clear_registry",
    "counter_gradient_diagnostic",
    "diagnose_truth",
    "diagnostic_flux_score",
    "diagnostic_truth",
    "dry_shear_buoyancy_grid",
    "get_case",
    "get_cases_for_regime",
    "list_cases",
    "list_regimes",
    "load_artifact",
    "prognostic_profile_score",
    "prognostic_truth",
    "register_case",
    "save_artifact",
    "register_default_catalog",
    "total_turbulent_flux",
]
