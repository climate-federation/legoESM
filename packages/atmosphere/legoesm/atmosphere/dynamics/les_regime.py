"""LES regime selection + resolution config for the column spin-off (gap #4).

Stage 5 of ``docs/COMPARE_REANALYSIS.md`` (§2 "LES regime configurable per
column chosen from the diagnosis"): pick whether a flagged column's spun-off LES
runs in a **shallow** (boundary-layer / shallow-convection) or **deep**
(deep-convection) regime from its environment tags, and supply the matching
LES-resolution grid parameters.  The regime sets the dx / domain / model-top so
the LES resolves the relevant eddies at an affordable cost:

* **shallow** — dx ~ 25–100 m, domain ~ 5–10 km, top ~ 3–4 km.
* **deep** — dx ~ 100–500 m, domain ~ 50–100 km, top ~ 20 km.

Selection is by CAPE (the manifest's environment tag from
:func:`legoesm.training.column_manifest.compute_column_environment`): a column
with CAPE above ``deep_cape_threshold_J_kg`` gets the deep regime.  The regime
dispatch RAISES on an unknown selection (CLAUDE.md dispatch hardening) — a typo
must never silently pick a default LES box.

These are pure config helpers (no JAX); the column-LES driver consumes them to
build the plane grid + stretched height coordinate (the heavy LES run itself
reuses the ``run_les_plane`` machinery on the plane NH dycore).
"""

from __future__ import annotations

import math
from typing import NamedTuple

_REGIMES = ("shallow", "deep")

# Default CAPE split [J/kg] between shallow and deep convection.  ~1000 J/kg is
# the conventional moderate-instability threshold above which deep convection is
# expected; it is a config field, overridable per campaign.
_DEEP_CAPE_THRESHOLD_J_KG = 1000.0


class LESResolutionConfig(NamedTuple):
    """Plane-LES grid parameters for one regime (feeds ``create_plane_grid`` +
    ``create_stretched_height_coordinate``)."""

    dx_m: float          # horizontal grid spacing [m]
    nx: int              # zonal cells
    ny: int              # meridional cells
    nlev: int            # vertical levels
    domain_top_m: float  # model top H [m]
    dz_sfc_m: float      # surface-layer thickness for the stretched grid [m]


class LESRegimeConfig(NamedTuple):
    """Regime split + per-regime resolution for the column LES spin-off."""

    deep_cape_threshold_J_kg: float = _DEEP_CAPE_THRESHOLD_J_KG
    # Horizontal domain = nx·dx must contain several of the regime's largest
    # eddies: shallow 128·50 m = 6.4 km (BOMEX-class); deep 256·200 m = 51.2 km.
    shallow: LESResolutionConfig = LESResolutionConfig(
        dx_m=50.0, nx=128, ny=128, nlev=60, domain_top_m=4000.0, dz_sfc_m=25.0
    )
    deep: LESResolutionConfig = LESResolutionConfig(
        dx_m=200.0, nx=256, ny=256, nlev=80, domain_top_m=20000.0, dz_sfc_m=100.0
    )


def validate_les_resolution(resolution: LESResolutionConfig) -> None:
    """Reject non-physical resolution params.

    All counts/lengths must be positive, and ``dz_sfc·nlev < domain_top`` so the
    stretched height coordinate has room to stretch (a hard requirement of
    :func:`legoesm.grids.vertical.create_stretched_height_coordinate`).
    """
    for name in ("dx_m", "domain_top_m", "dz_sfc_m"):
        if getattr(resolution, name) <= 0.0:
            raise ValueError(
                f"LESResolutionConfig.{name} must be > 0, got "
                f"{getattr(resolution, name)}."
            )
    for name in ("nx", "ny", "nlev"):
        if getattr(resolution, name) <= 0:
            raise ValueError(
                f"LESResolutionConfig.{name} must be > 0, got "
                f"{getattr(resolution, name)}."
            )
    if resolution.dz_sfc_m * resolution.nlev >= resolution.domain_top_m:
        raise ValueError(
            f"dz_sfc_m·nlev = {resolution.dz_sfc_m * resolution.nlev} must be < "
            f"domain_top_m = {resolution.domain_top_m} so the stretched height "
            f"coordinate has room to stretch."
        )


def validate_regime_config(config: LESRegimeConfig) -> None:
    """Reject a non-physical regime config (negative CAPE split, bad boxes)."""
    if not math.isfinite(config.deep_cape_threshold_J_kg) or (
        config.deep_cape_threshold_J_kg < 0.0
    ):
        raise ValueError(
            f"deep_cape_threshold_J_kg must be finite and >= 0, got "
            f"{config.deep_cape_threshold_J_kg}."
        )
    validate_les_resolution(config.shallow)
    validate_les_resolution(config.deep)


def select_les_regime(
    cape_J_kg: float, config: LESRegimeConfig = LESRegimeConfig()
) -> str:
    """Return ``"deep"`` if ``cape_J_kg >= deep_cape_threshold_J_kg`` else ``"shallow"``.

    A non-finite CAPE (NaN/inf) raises rather than silently defaulting — CAPE is
    a diagnosed environment tag, so a non-finite value signals upstream bad data
    that must not pick an LES box by accident.
    """
    cape = float(cape_J_kg)
    if not math.isfinite(cape):
        raise ValueError(
            f"select_les_regime: non-finite CAPE {cape_J_kg!r}; the environment "
            f"tag must be finite (upstream diagnosis error)."
        )
    return "deep" if cape >= config.deep_cape_threshold_J_kg else "shallow"


def les_resolution_for_regime(
    regime: str, config: LESRegimeConfig = LESRegimeConfig()
) -> LESResolutionConfig:
    """Return the resolution config for ``regime`` (raises on an unknown regime)."""
    if regime == "shallow":
        return config.shallow
    if regime == "deep":
        return config.deep
    raise ValueError(
        f"Unknown LES regime {regime!r}; choose from {_REGIMES}."
    )


def les_resolution_for_column(
    cape_J_kg: float, config: LESRegimeConfig = LESRegimeConfig()
) -> tuple[str, LESResolutionConfig]:
    """Convenience: select the regime from CAPE and return ``(regime, resolution)``.

    The whole regime config is validated (CAPE split + both boxes), so a
    mis-configured threshold or regime box fails loudly before the (expensive)
    LES is built.
    """
    validate_regime_config(config)
    regime = select_les_regime(cape_J_kg, config)
    resolution = les_resolution_for_regime(regime, config)
    return regime, resolution
