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
from .emit import (
    build_reference_artifact,
    horizontal_mean,
    resolved_vertical_flux,
    sgs_vertical_scalar_flux_mean,
)
from .intercomparison import (
    CBL_ENVELOPE,
    CBLDiagnostics,
    GateResult,
    MetricBand,
    cbl_diagnostics,
    evaluate_cbl_gate,
    gate_from_cbl_profiles,
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
from .scm_coupling import (
    T_from_theta,
    interp_profile,
    regrid_truth,
    theta_from_temperature,
)
from .scm_runner import (
    build_cbl_scm_from_artifact,
    scm_final_theta_on,
    scm_les_final_loss,
)
from .score import (
    DiagnosticScore,
    PrognosticScore,
    diagnostic_flux_score,
    prognostic_profile_score,
)
from .scorecard import (
    Scorecard,
    ScorecardError,
    assemble_scorecard,
    coefficient_spreads,
    rank_closures_per_regime,
    render_markdown,
)
from .sigma_les import (
    SigmaLES,
    SigmaLESError,
    sigma_les_prognostic,
)

__all__ = [
    "ANCHOR_CASES",
    "CBL_ENVELOPE",
    "CBLDiagnostics",
    "CI_MARKERS",
    "SigmaLES",
    "SigmaLESError",
    "sigma_les_prognostic",
    "BridgeError",
    "CounterGradientResult",
    "DiagnosticScore",
    "GateResult",
    "LESCase",
    "LESGrid",
    "LESReferenceArtifact",
    "LESTruth",
    "MetricBand",
    "PrognosticScore",
    "REGIMES",
    "RegistryError",
    "SGS_CHOICES",
    "Scorecard",
    "ScorecardError",
    "T_from_theta",
    "assemble_scorecard",
    "coefficient_spreads",
    "rank_closures_per_regime",
    "render_markdown",
    "artifact_to_scm_forcing",
    "interp_profile",
    "regrid_truth",
    "theta_from_temperature",
    "build_cbl_scm_from_artifact",
    "build_reference_artifact",
    "cbl_diagnostics",
    "centered_dtheta_dz",
    "clear_registry",
    "horizontal_mean",
    "resolved_vertical_flux",
    "scm_final_theta_on",
    "scm_les_final_loss",
    "sgs_vertical_scalar_flux_mean",
    "counter_gradient_diagnostic",
    "diagnose_truth",
    "diagnostic_flux_score",
    "diagnostic_truth",
    "dry_shear_buoyancy_grid",
    "evaluate_cbl_gate",
    "gate_from_cbl_profiles",
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
