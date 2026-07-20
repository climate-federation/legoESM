"""LES-truth suite: enumerable LES reference cases for tuning/comparing SCM
turbulence closures (docs/atmosphere/les_suite/LES_SUITE.md).

Public API is the case registry + default catalog; the LES→SCM forcing/truth
bridge and the scoring assembly are separate follow-ups.
"""
from __future__ import annotations

from .catalog import (
    ANCHOR_CASES,
    dry_shear_buoyancy_grid,
    register_default_catalog,
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

__all__ = [
    "ANCHOR_CASES",
    "CI_MARKERS",
    "LESCase",
    "LESGrid",
    "REGIMES",
    "RegistryError",
    "SGS_CHOICES",
    "clear_registry",
    "dry_shear_buoyancy_grid",
    "get_case",
    "get_cases_for_regime",
    "list_cases",
    "list_regimes",
    "register_case",
    "register_default_catalog",
]
