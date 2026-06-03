"""FidelityCase registry mapping ``(tier, case_name)`` to evaluator callables.

Per-tier metric files (under ``tests/ocean/fidelity/`` and Layer B report
hooks) register their cases at import time via :func:`register_case`. The
registry is then consulted by:

* **Layer A** — pytest fixtures iterate ``get_cases_for_tier(N)`` to drive
  one test per registered case at tier N.
* **Layer B** — :mod:`scripts.ocean_fidelity.build_fidelity_report` walks
  every tier in :func:`list_tiers` and emits one report subsection per case.

The registry is process-global; :func:`clear_registry` is exposed for tests
that need a clean slate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional

CIMarker = Literal["fast", "nightly", "manual_only"]

MetricFn = Callable[[Any], float]
ReferenceFn = Callable[[Any], float]
PlotFn = Callable[[Any, Path], None]


@dataclass(frozen=True)
class FidelityCase:
    """A single tier+case binding to evaluator callables.

    Attributes
    ----------
    tier : int (0..8)
        Tier index this case contributes to.
    name : str
        Case name (matches a registered ocean experiment, e.g.
        ``"inertia_gravity_wave"``).
    metric_fn : Callable
        Maps an artifact dict (loaded from
        ``results/ocean/<case>/latlon/<res>/``) to a scalar measured value.
    reference_fn : Callable
        Maps the same artifact dict to the analytical / literature /
        external reference value the measured metric is compared against.
    plotters_fn : Callable, optional
        Writes diagnostic PNGs under ``<artifact_dir>/fidelity/``. Called
        only by Layer B.
    ci_marker : ``"fast"`` | ``"nightly"`` | ``"manual_only"``
        Drives pytest marker assignment in Layer A.
    description : str
        Free-text explanation used in Layer B subsection headers.
    """

    tier: int
    name: str
    metric_fn: MetricFn
    reference_fn: ReferenceFn
    plotters_fn: Optional[PlotFn] = None
    ci_marker: CIMarker = "fast"
    description: str = ""


TIER_REGISTRY: dict[tuple[int, str], FidelityCase] = {}
TIER_RANGE: tuple[int, int] = (0, 8)


class RegistryError(ValueError):
    """Raised when a registry operation violates the contract."""


def register_case(case: FidelityCase) -> None:
    """Register a :class:`FidelityCase`. Duplicate ``(tier, name)`` errors."""
    if not isinstance(case, FidelityCase):
        raise RegistryError(f"expected FidelityCase, got {type(case).__name__}")
    lo, hi = TIER_RANGE
    if not (lo <= case.tier <= hi):
        raise RegistryError(f"tier {case.tier} not in [{lo}, {hi}]")
    if case.ci_marker not in ("fast", "nightly", "manual_only"):
        raise RegistryError(f"unknown ci_marker {case.ci_marker!r}")
    key = (case.tier, case.name)
    if key in TIER_REGISTRY:
        raise RegistryError(
            f"case ({case.tier}, {case.name!r}) already registered"
        )
    TIER_REGISTRY[key] = case


def get_case(tier: int, name: str) -> FidelityCase:
    try:
        return TIER_REGISTRY[(tier, name)]
    except KeyError:
        raise RegistryError(f"no case ({tier}, {name!r}) registered") from None


def get_cases_for_tier(tier: int) -> list[FidelityCase]:
    """Return all cases registered for ``tier``, sorted by name."""
    return sorted(
        (c for (t, _), c in TIER_REGISTRY.items() if t == tier),
        key=lambda c: c.name,
    )


def list_tiers() -> list[int]:
    """Return the sorted list of tier indices that have at least one case."""
    return sorted({tier for tier, _ in TIER_REGISTRY.keys()})


def clear_registry() -> None:
    """Empty the registry. Test-only — production code must not call this."""
    TIER_REGISTRY.clear()
