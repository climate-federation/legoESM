"""Bioclimatic climate-zone classification for the ESA -> CLM5 PFT crosswalk.

ESA CCI PFTs are leaf-form x phenology x growth-form but carry **no climate
zone**; CLM5's 17 PFTs split trees/shrubs/grass into tropical / temperate /
boreal (and C3-arctic grass).  To map ESA -> CLM5 we therefore need, per
gridcell, a climate-zone label derived from a temperature climatology.

This module computes the two diagnostics the standard CLM ``mksurfdata`` rules
use — **coldest-month mean temperature** (``Tc``) and **growing degree-days**
(``GDD``, base 5 C) — from a monthly 2 m air-temperature climatology (e.g. ERA5),
and classifies each cell as tropical / temperate / boreal.

Thresholds (Bonan et al., 2002; CLM ``mksurfdata`` / Lawrence & Chase, 2007):
  - broadleaf **tropical** if ``Tc >= +15.5 C``;
  - **temperate** vs **boreal** (needleleaf, deciduous, shrubs, grass) at
    ``Tc = -19 C`` combined with ``GDD >= 1200`` for temperate; otherwise boreal.

The functions are pure NumPy and unit-tested on synthetic climatologies; the
ERA5 read that supplies the monthly temperature field is a separate I/O step.

References
----------
- Bonan, G. B., Levis, S., Kergoat, L., and Oleson, K. W. (2002): Landscapes as
  patches of plant functional types: An integrating concept for climate and
  ecosystem models. Global Biogeochem. Cycles, 16(2), 1021.
  https://doi.org/10.1029/2000GB001360
- Lawrence, P. J. and Chase, T. N. (2007): Representing a new MODIS consistent
  land surface in the Community Land Model (CLM 3.0). J. Geophys. Res., 112,
  G01023. https://doi.org/10.1029/2006JG000168
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from legoesm import constants

# Non-leap days per calendar month (for the monthly-mean GDD approximation).
_DAYS_PER_MONTH = np.array(
    [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.float64
)


class BioclimateConfig(NamedTuple):
    """Climate-zone thresholds for the ESA -> CLM5 PFT crosswalk.

    Defaults follow CLM ``mksurfdata`` (Bonan et al., 2002).  Temperatures carry
    a ``_C`` suffix (degrees Celsius); ``GDD`` is dimensionless degree-days.
    """

    tc_tropical_min_C: float = 15.5     # broadleaf tropical if Tc >= this
    tc_boreal_max_C: float = -19.0      # temperate requires Tc >= this ...
    gdd_temperate_min: float = 1200.0   # ... AND GDD >= this; else boreal
    gdd_base_C: float = 5.0             # GDD accumulation base temperature


class ClimateZones(NamedTuple):
    """Mutually-exclusive climate-zone boolean masks (each shape ``S``).

    ``tropical | temperate | boreal`` partitions every cell exactly once, so a
    source PFT fraction multiplied by these masks is conserved.
    """

    tropical: np.ndarray
    temperate: np.ndarray
    boreal: np.ndarray


def coldest_month_temp_C(
    t2m_monthly_K: np.ndarray, *, month_axis: int = 0
) -> np.ndarray:
    """Coldest-month mean temperature ``Tc`` [C] from a monthly climatology [K]."""
    t = np.asarray(t2m_monthly_K, dtype=np.float64)
    return np.min(t, axis=month_axis) - constants.T_freeze


def growing_degree_days(
    t2m_monthly_K: np.ndarray,
    config: BioclimateConfig = BioclimateConfig(),
    *,
    month_axis: int = 0,
) -> np.ndarray:
    """Growing degree-days (base ``gdd_base_C``) from a monthly climatology.

    Monthly-mean approximation: ``sum_m max(0, T_m - base) * days_in_month`` —
    the standard estimate used by the CLM land-cover->PFT crosswalk when only
    monthly means are available.
    """
    t_C = np.asarray(t2m_monthly_K, dtype=np.float64) - constants.T_freeze
    excess = np.clip(t_C - config.gdd_base_C, 0.0, None)
    # Move the month axis to front, weight each month by its day count, and sum.
    e = np.moveaxis(excess, month_axis, 0)
    if e.shape[0] != _DAYS_PER_MONTH.shape[0]:
        raise ValueError(
            f"expected 12 months on axis {month_axis}, got {e.shape[0]}."
        )
    d = _DAYS_PER_MONTH.reshape((-1,) + (1,) * (e.ndim - 1))
    return np.sum(e * d, axis=0)


def classify_climate_zones(
    tc_C: np.ndarray, gdd: np.ndarray, config: BioclimateConfig = BioclimateConfig()
) -> ClimateZones:
    """Partition cells into tropical / temperate / boreal from ``Tc`` and ``GDD``.

    - tropical: ``Tc >= tc_tropical_min_C``
    - temperate: not tropical, ``Tc >= tc_boreal_max_C`` **and** ``GDD >= gdd_temperate_min``
    - boreal: everything else (cold and/or short growing season)
    """
    tc_C = np.asarray(tc_C, dtype=np.float64)
    gdd = np.asarray(gdd, dtype=np.float64)
    tropical = tc_C >= config.tc_tropical_min_C
    temperate = (~tropical) & (tc_C >= config.tc_boreal_max_C) & (gdd >= config.gdd_temperate_min)
    boreal = ~(tropical | temperate)
    return ClimateZones(tropical=tropical, temperate=temperate, boreal=boreal)


def climate_zones_from_t2m(
    t2m_monthly_K: np.ndarray,
    config: BioclimateConfig = BioclimateConfig(),
    *,
    month_axis: int = 0,
) -> ClimateZones:
    """Convenience: monthly 2 m temperature climatology [K] -> climate zones."""
    tc = coldest_month_temp_C(t2m_monthly_K, month_axis=month_axis)
    gdd = growing_degree_days(t2m_monthly_K, config, month_axis=month_axis)
    return classify_climate_zones(tc, gdd, config)
