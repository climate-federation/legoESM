"""LESCase registry — enumerable, parameterized cases for the LES-truth suite.

The LES suite (``docs/atmosphere/les_suite/LES_SUITE.md``) tunes and compares SCM
turbulence closures against LES reference runs. This module is the enumerable
*catalog* of those reference runs — the atmosphere-LES analogue of
:mod:`legoesm.ocean.fidelity.registry` (``FidelityCase``), turning the scattered
``scripts/run/run_*_les.py`` / ``run_spectral_{cbl,sbl}.py`` CLIs into data.

A :class:`LESCase` records what *defines* a reference run for the science
questions (``LES_SUITE.md`` §1): its physical **regime**, the **regime-axis
coordinates** that are the independent variables of Q1 (local→nonlocal flux
threshold) and Q3 (inter-regime coefficient spread) — the surface kinematic heat
flux ``w'θ'_s`` and the geostrophic wind ``|U_g|`` — the **grid/resolution**, the
**SGS variants** that make up the D7 LES error bar, the **driver** that runs it,
and the published **reference** it is validated against.

Scope boundary (deliberately NOT here): this registry is pure data + selection.
It does not *run* the LES (the drivers do) and it does not extract SCM forcing
from LES output (that is the follow-up ``bridge.py``). Keeping it side-effect-free
mirrors ``ocean/fidelity/registry.py`` and keeps it CPU-/import-cheap.

The registry is process-global; :func:`clear_registry` is exposed for tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Physical boundary-layer regimes the suite spans (LES_SUITE.md D2/D3). Dry axes
# are the confound-free core; the two moist non-precipitating regimes add the
# cloud response. Precipitating (RICO) is out of scope (D2).
Regime = Literal[
    "dry_convective",   # free/mixed convective BL (Nieuwstadt CBL anchor)
    "dry_stable",       # stably stratified, sheared BL (GABLS1 SBL anchor)
    "dry_neutral",      # near-neutral, shear-dominated BL
    "shallow_cumulus",  # non-precipitating shallow Cu (BOMEX anchor)
    "stratocumulus",    # nocturnal Sc (DYCOMS-II RF01 anchor)
]
REGIMES: tuple[Regime, ...] = (
    "dry_convective", "dry_stable", "dry_neutral",
    "shallow_cumulus", "stratocumulus",
)

# LES-side subgrid closures selectable on the plane cores (LES_SUITE.md D7 spread
# uses {lasd, smagorinsky, vreman} to bound σ_LES). Kept distinct from the SCM
# ``TurbulenceConfig.scheme`` closures under study.
SgsClosure = Literal["lasd", "smagorinsky", "vreman", "amd"]
SGS_CHOICES: tuple[SgsClosure, ...] = ("lasd", "smagorinsky", "vreman", "amd")

CiMarker = Literal["fast", "nightly", "manual_only"]
CI_MARKERS: tuple[CiMarker, ...] = ("fast", "nightly", "manual_only")

# The incompressible pseudo-spectral plane core is the suite's truth core (D1).
LesCore = Literal["spectral"]
CORES: tuple[LesCore, ...] = ("spectral",)


class RegistryError(ValueError):
    """Raised when a registry operation or LESCase violates the contract."""


@dataclass(frozen=True)
class LESGrid:
    """Doubly-periodic plane grid + timestep for one LES reference run."""

    nx: int
    ny: int
    nz: int
    Lx_m: float
    Ly_m: float
    Lz_m: float
    dt_s: float

    @property
    def label(self) -> str:
        """Cubed-grid label used in output paths / matrix selection (``96x96x96``)."""
        return f"{self.nx}x{self.ny}x{self.nz}"

    def validate(self) -> None:
        for name, v in (
            ("nx", self.nx), ("ny", self.ny), ("nz", self.nz),
            ("Lx_m", self.Lx_m), ("Ly_m", self.Ly_m), ("Lz_m", self.Lz_m),
            ("dt_s", self.dt_s),
        ):
            if not (isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0):
                raise RegistryError(f"LESGrid.{name} must be a positive number, got {v!r}")


@dataclass(frozen=True)
class LESCase:
    """One LES reference run in the suite catalog.

    Attributes
    ----------
    name : str
        Globally unique case name (also the output-dir stem).
    regime : Regime
        Physical BL regime (drives the Q1/Q3 grouping).
    core : LesCore
        LES dynamical core (``"spectral"`` — the truth core, D1).
    grid : LESGrid
        Resolution + domain + timestep.
    duration_hours : float
        Physical integration length (spin-up + averaging window).
    surface_theta_flux_K_m_s : float | None
        Prescribed surface kinematic heat flux ``w'θ'_s`` [K m/s] — the Q1 buoyancy
        axis coordinate for dry cases. ``None`` for cases whose surface is
        interactive / SST-driven (the moist anchors).
    geostrophic_wind_m_s : float | None
        Geostrophic wind magnitude ``|U_g|`` [m/s] — the shear axis coordinate.
        ``None`` for pure free convection (no imposed mean wind).
    sgs_variants : tuple[SgsClosure, ...]
        SGS closures to run for this case. One entry = a single production run;
        the D7 error-bar subset carries the full ``("lasd","smagorinsky","vreman")``.
    driver : str
        ``scripts/run/<driver>.py`` basename that implements this regime's setup.
    reference : str
        Published intercomparison / oracle the run is validated against.
    ci_marker : CiMarker
        Cost tier for any harness that iterates the catalog.
    description : str
        Free-text.
    provisional_axis : bool
        ``True`` while the ``(flux, wind)`` axis values are placeholders pending the
        gate-0 CBL calibration (LES_SUITE.md §8/§9); real anchors set ``False``.
    """

    name: str
    regime: Regime
    core: LesCore
    grid: LESGrid
    duration_hours: float
    surface_theta_flux_K_m_s: float | None
    geostrophic_wind_m_s: float | None
    sgs_variants: tuple[SgsClosure, ...]
    driver: str
    reference: str
    ci_marker: CiMarker = "nightly"
    description: str = ""
    provisional_axis: bool = False

    def validate(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise RegistryError(f"LESCase.name must be a non-empty str, got {self.name!r}")
        if self.regime not in REGIMES:
            raise RegistryError(f"{self.name}: unknown regime {self.regime!r}; known {REGIMES}")
        if self.core not in CORES:
            raise RegistryError(f"{self.name}: unknown core {self.core!r}; known {CORES}")
        if self.ci_marker not in CI_MARKERS:
            raise RegistryError(f"{self.name}: unknown ci_marker {self.ci_marker!r}")
        if not self.sgs_variants:
            raise RegistryError(f"{self.name}: sgs_variants must be non-empty")
        bad_sgs = [s for s in self.sgs_variants if s not in SGS_CHOICES]
        if bad_sgs:
            raise RegistryError(f"{self.name}: unknown sgs_variants {bad_sgs}; known {SGS_CHOICES}")
        if len(set(self.sgs_variants)) != len(self.sgs_variants):
            raise RegistryError(f"{self.name}: duplicate sgs_variants {self.sgs_variants}")
        if not (isinstance(self.duration_hours, (int, float)) and self.duration_hours > 0):
            raise RegistryError(f"{self.name}: duration_hours must be > 0, got {self.duration_hours!r}")
        for axis in ("surface_theta_flux_K_m_s", "geostrophic_wind_m_s"):
            v = getattr(self, axis)
            if v is not None and not isinstance(v, (int, float)):
                raise RegistryError(f"{self.name}: {axis} must be a number or None, got {v!r}")
        if self.geostrophic_wind_m_s is not None and self.geostrophic_wind_m_s < 0:
            raise RegistryError(f"{self.name}: geostrophic_wind_m_s must be >= 0")
        self.grid.validate()


# --- process-global registry ------------------------------------------------
LES_CASE_REGISTRY: dict[str, LESCase] = {}


def register_case(case: LESCase) -> LESCase:
    """Validate and register ``case``; duplicate ``name`` is an error."""
    if not isinstance(case, LESCase):
        raise RegistryError(f"expected LESCase, got {type(case).__name__}")
    case.validate()
    if case.name in LES_CASE_REGISTRY:
        raise RegistryError(f"case {case.name!r} already registered")
    LES_CASE_REGISTRY[case.name] = case
    return case


def get_case(name: str) -> LESCase:
    try:
        return LES_CASE_REGISTRY[name]
    except KeyError:
        raise RegistryError(
            f"no case {name!r} registered; known={sorted(LES_CASE_REGISTRY)}"
        ) from None


def get_cases_for_regime(regime: Regime) -> list[LESCase]:
    """All registered cases in ``regime``, sorted by name."""
    if regime not in REGIMES:
        raise RegistryError(f"unknown regime {regime!r}; known {REGIMES}")
    return sorted(
        (c for c in LES_CASE_REGISTRY.values() if c.regime == regime),
        key=lambda c: c.name,
    )


def list_regimes() -> list[Regime]:
    """Regimes that have at least one registered case, in canonical order."""
    present = {c.regime for c in LES_CASE_REGISTRY.values()}
    return [r for r in REGIMES if r in present]


def list_cases() -> list[LESCase]:
    """All registered cases, sorted by name."""
    return sorted(LES_CASE_REGISTRY.values(), key=lambda c: c.name)


def clear_registry() -> None:
    """Empty the registry. Test-only — production code must not call this."""
    LES_CASE_REGISTRY.clear()
