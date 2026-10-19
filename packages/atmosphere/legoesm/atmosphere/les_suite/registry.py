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
# cloud response; RICO (precipitating shallow Cu) runs in the shallow_cumulus regime.
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


SurfaceDrag = Literal["neutral", "most"]
SURFACE_DRAGS: tuple[SurfaceDrag, ...] = ("neutral", "most")


@dataclass(frozen=True)
class LESSetup:
    """Full published setup of a case the generic dry builder runs.

    Every scientific input is a field here, so nothing falls back to a CLI
    default: the case is the record. Profiles are piecewise-linear knots in
    height, interpolated to the LES levels (held constant beyond the end knots).

    Attributes
    ----------
    lat_deg : float
        Latitude [deg] — the Coriolis parameter ``f = 2Ω sin φ`` (sign kept, so a
        southern-hemisphere case rotates the right way).
    z0_m : float
        Momentum roughness length [m].
    theta_ref_K : float
        Reference θ of the Boussinesq buoyancy term [K].
    theta_knots : tuple of (z [m], θ [K])
        Initial potential-temperature profile. Empty when ``dephy_case`` supplies it.
    geostrophic_knots : tuple of (z [m], u_g [m/s], v_g [m/s])
        Geostrophic wind profile (constant in time). Empty when ``dephy_case``
        supplies it.
    wind_init_knots : tuple of (z [m], u [m/s], v [m/s])
        Initial wind; empty ⇒ the geostrophic profile.
    z0h_m : float | None
        Thermal roughness [m] of the prescribed-surface-temperature MOST layer;
        ``None`` ⇒ ``z0_m``.
    surface_drag : "neutral" | "most"
        Wall drag under a prescribed surface heat FLUX (the core's
        ``sfc_flux_drag``); the published cases use Monin–Obukhov similarity.
    surface_theta_knots : tuple of (t [s], θ_s [K])
        Prescribed surface potential temperature series (MOST BC), linear in time.
        Empty ⇒ flux- or cooling-rate-forced from the LESCase fields.
    sponge_bottom_m, sponge_tau_s : float | None
        Rayleigh sponge from ``sponge_bottom_m`` to the lid with timescale
        ``sponge_tau_s`` (cosine ramp, the suite's SBL sponge form). ``None`` = no
        sponge.
    perturb_K, perturb_top_m : float
        Amplitude [K] of the random θ seed noise and the height [m] below which it
        is applied.
    dephy_case : str | None
        Name in ``forcing.scm.dephy_scm.DEPHY_SCM_CASES`` whose file supplies the
        initial θ/u/v, the geostrophic wind, the surface forcing, z0 and latitude.
    pending : tuple[str, ...]
        Values in this setup the user has NOT approved yet, one line each (value
        and why). A case with any pending item cannot emit a production artifact
        — the driver refuses it except as a ``--smoke`` run.
    approved : tuple[str, ...]
        Formerly pending items the user approved (recorded with the setup, so
        every artifact carries them as provenance).
    """

    lat_deg: float
    z0_m: float
    theta_ref_K: float
    perturb_K: float
    perturb_top_m: float
    theta_knots: tuple[tuple[float, float], ...] = ()
    geostrophic_knots: tuple[tuple[float, float, float], ...] = ()
    wind_init_knots: tuple[tuple[float, float, float], ...] = ()
    z0h_m: float | None = None
    surface_drag: SurfaceDrag = "most"
    surface_theta_knots: tuple[tuple[float, float], ...] = ()
    sponge_bottom_m: float | None = None
    sponge_tau_s: float | None = None
    dephy_case: str | None = None
    pending: tuple[str, ...] = ()
    approved: tuple[str, ...] = ()

    def validate(self, name: str) -> None:
        if not (-90.0 <= self.lat_deg <= 90.0):
            raise RegistryError(f"{name}: setup.lat_deg must be in [-90, 90]")
        for fld in ("z0_m", "theta_ref_K", "perturb_top_m"):
            if not getattr(self, fld) > 0:
                raise RegistryError(f"{name}: setup.{fld} must be > 0")
        if self.perturb_K < 0:
            raise RegistryError(f"{name}: setup.perturb_K must be >= 0")
        if self.z0h_m is not None and not self.z0h_m > 0:
            raise RegistryError(f"{name}: setup.z0h_m must be > 0 or None")
        if self.surface_drag not in SURFACE_DRAGS:
            raise RegistryError(f"{name}: unknown setup.surface_drag "
                                f"{self.surface_drag!r}; known {SURFACE_DRAGS}")
        if (self.sponge_bottom_m is None) != (self.sponge_tau_s is None):
            raise RegistryError(f"{name}: set both sponge_bottom_m and sponge_tau_s or "
                                "neither")
        if self.sponge_tau_s is not None and not self.sponge_tau_s > 0:
            raise RegistryError(f"{name}: setup.sponge_tau_s must be > 0")
        if self.dephy_case is None:
            if len(self.theta_knots) < 1 or len(self.geostrophic_knots) < 1:
                raise RegistryError(f"{name}: a setup without dephy_case needs "
                                    "theta_knots and geostrophic_knots")
        elif (self.theta_knots or self.geostrophic_knots or self.wind_init_knots
              or self.surface_theta_knots):
            raise RegistryError(f"{name}: dephy_case supplies the profiles and the "
                                "surface forcing; do not also give knots")
        for label, knots, width in (("theta_knots", self.theta_knots, 2),
                                    ("geostrophic_knots", self.geostrophic_knots, 3),
                                    ("wind_init_knots", self.wind_init_knots, 3),
                                    ("surface_theta_knots", self.surface_theta_knots, 2)):
            if any(len(k) != width for k in knots):
                raise RegistryError(f"{name}: every {label} entry needs {width} values")
            x = [k[0] for k in knots]
            if any(b <= a for a, b in zip(x, x[1:])):
                raise RegistryError(f"{name}: {label} must be strictly increasing")


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
    surface_cooling_rate_K_per_h : float | None
        Prescribed surface-temperature COOLING rate [K/h] for a stable case forced
        the published way (GABLS1: ``T_s(t) = θ₀ − rate·t`` with the surface flux
        and drag from Monin–Obukhov similarity). Mutually exclusive with
        ``surface_theta_flux_K_m_s``; ``None`` = flux-forced.
    provisional_axis : bool
        ``True`` while the ``(flux, wind)`` axis values are placeholders pending the
        gate-0 CBL calibration (LES_SUITE.md §8/§9); real anchors set ``False``.
    setup : LESSetup | None
        Full published setup for the generic dry builder; ``None`` = the legacy
        anchor/grid builders driven by the flux/U_g fields and CLI values.
    held_out : bool
        ``True`` = held out of tuning and evolution, scored only for
        generalisation (the user's 2026-10-15 split: Ayotte, Sullivan–Patton,
        GABLS3, GABLS4, neutral Ekman, RICO held out).
    sponge_base_frac : float | None
        Base of the LES Rayleigh sponge as a fraction of the domain height, for a
        case WITHOUT a ``setup`` (its driver reads it from here; a setup case
        carries ``setup.sponge_bottom_m``). ``None`` = no sponge. See
        :meth:`sponge_base_m`.
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
    surface_cooling_rate_K_per_h: float | None = None
    setup: LESSetup | None = None
    held_out: bool = False
    sponge_base_frac: float | None = None

    def sponge_base_m(self, lz_m: float | None = None) -> float | None:
        """Height [m] of the LES sponge base (``None`` = no sponge): the setup's
        ``sponge_bottom_m``, else ``sponge_base_frac`` of the domain height
        ``lz_m`` (default the catalog grid's)."""
        if self.setup is not None:
            return self.setup.sponge_bottom_m
        if self.sponge_base_frac is None:
            return None
        return self.sponge_base_frac * (self.grid.Lz_m if lz_m is None else lz_m)

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
            raise RegistryError(
                f"{self.name}: duration_hours must be > 0, got {self.duration_hours!r}"
            )
        for axis in ("surface_theta_flux_K_m_s", "geostrophic_wind_m_s"):
            v = getattr(self, axis)
            if v is not None and not isinstance(v, (int, float)):
                raise RegistryError(f"{self.name}: {axis} must be a number or None, got {v!r}")
        rate = self.surface_cooling_rate_K_per_h
        if rate is not None:
            if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not rate > 0:
                raise RegistryError(
                    f"{self.name}: surface_cooling_rate_K_per_h must be > 0, got {rate!r}")
            if self.surface_theta_flux_K_m_s is not None:
                raise RegistryError(
                    f"{self.name}: set surface_cooling_rate_K_per_h OR "
                    "surface_theta_flux_K_m_s, not both")
        if self.geostrophic_wind_m_s is not None and self.geostrophic_wind_m_s < 0:
            raise RegistryError(f"{self.name}: geostrophic_wind_m_s must be >= 0")
        if self.setup is not None:
            if not isinstance(self.setup, LESSetup):
                raise RegistryError(f"{self.name}: setup must be an LESSetup")
            self.setup.validate(self.name)
            n_forcing = sum((self.surface_theta_flux_K_m_s is not None,
                             self.surface_cooling_rate_K_per_h is not None,
                             bool(self.setup.surface_theta_knots),
                             self.setup.dephy_case is not None))
            if n_forcing != 1:
                raise RegistryError(
                    f"{self.name}: a setup case needs exactly one surface forcing "
                    "(flux, cooling rate, surface θ series or a DEPHY file), got "
                    f"{n_forcing}")
        frac = self.sponge_base_frac
        if frac is not None:
            if self.setup is not None:
                raise RegistryError(f"{self.name}: a setup case carries its sponge in "
                                    "setup.sponge_bottom_m, not sponge_base_frac")
            if isinstance(frac, bool) or not isinstance(frac, (int, float)) or not (
                    0.0 < frac < 1.0):
                raise RegistryError(
                    f"{self.name}: sponge_base_frac must be in (0, 1), got {frac!r}")
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
