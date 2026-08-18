"""Shared constants and private helpers for the boundary-data consumer.

These are intentionally module-private (``_`` prefix on the few callables and
on the file itself).  Public API lives in :mod:`~legoesm.land.boundary_data`'s
top-level re-exports.
"""

from __future__ import annotations

import numpy as np

from legoesm.land.surface_params import CLM5_PFT_NAMES, N_PFT_CLM5
from legoesm.land.canopy.config import (
    PFT_VCMAX25_C3,
    PFT_VCMAX25_C4,
    PFT_AERO_PARAMS,
    PFT_CANOPY_HEIGHT,
)


# Zone index into the 3-element biome lists: [boreal, temperate, tropical].
_BOREAL, _TEMPERATE, _TROPICAL = 0, 1, 2


# --- Canopy / Ball-Berry column defaults (CLM5 tech note; Ball-Berry 1987;
#     Bonan 2019 two-leaf canopy) — fill CanopyLandParams columns the surfdata /
#     PFT tables do not prescribe per-column. ---
CI_DEFAULT = 0.75          # clumping index [-]
KN_DEFAULT = 0.3           # nitrogen extinction coefficient [-]
ALF_DEFAULT = 0.3          # quantum yield [mol CO2 / mol photon]
M_C3 = 9.0                 # Ball-Berry slope, C3 [-]
M_C4 = 4.0                 # Ball-Berry slope, C4 [-]
B0_C3 = 0.01               # Ball-Berry intercept, C3 [mol m-2 s-1]
B0_C4 = 0.04               # Ball-Berry intercept, C4 [mol m-2 s-1]
TGC_DEFAULT_C = 25.0       # growth-temperature default for Vcmax acclimation [°C]
HC_MIN_M = 0.1             # minimum canopy height [m]
EMISS_VEG = 0.97           # vegetated-surface emissivity [-]
EMISS_BARE = 0.96          # bare-soil emissivity [-]
RZ0M_BARE = 0.01           # bare-surface z0m/hc ratio [-]
ALB_VIS_BARE = 0.2         # bare fallback visible albedo [-]
ALB_NIR_BARE = 0.3         # bare fallback NIR albedo [-]

# --- Glacier ice-surface albedo (snow/ice; CLM glacier landunit) ---
GLACIER_ALB_VIS = 0.70
GLACIER_ALB_NIR = 0.50
GLACIER_ALBEDO_DEFAULT = 0.6   # broadband for the SEB / slab path

# --- Soil-texture fallback where HWSD has no soil (sandy default) ---
FALLBACK_SAND_PCT = 92.0
FALLBACK_CLAY_PCT = 3.0

# --- Nominal top-layer wetness for soil-colour albedo before state exists ---
THETA_TOP_DEFAULT = 0.2


# CLM5 PFT name -> (biome key, climate-zone index, is_c4).  ``None`` biome = bare.
_CLM5_TO_BIOME: dict[str, tuple] = {
    "bare_soil": (None, _TEMPERATE, False),
    "needleleaf_evergreen_temperate": ("ENF", _TEMPERATE, False),
    "needleleaf_evergreen_boreal": ("ENF", _BOREAL, False),
    "needleleaf_deciduous_boreal": ("DNF", _BOREAL, False),
    "broadleaf_evergreen_tropical": ("EBF", _TROPICAL, False),
    "broadleaf_evergreen_temperate": ("EBF", _TEMPERATE, False),
    "broadleaf_deciduous_tropical": ("DBF", _TROPICAL, False),
    "broadleaf_deciduous_temperate": ("DBF", _TEMPERATE, False),
    "broadleaf_deciduous_boreal": ("DBF", _BOREAL, False),
    "broadleaf_evergreen_shrub": ("SHR", _TEMPERATE, False),
    "broadleaf_deciduous_temperate_shrub": ("SHR", _TEMPERATE, False),
    "broadleaf_deciduous_boreal_shrub": ("SHR", _BOREAL, False),
    "c3_arctic_grass": ("GRA", _BOREAL, False),
    "c3_grass": ("GRA", _TEMPERATE, False),
    "c4_grass": ("GRA", _TEMPERATE, True),
    "crop_c3": ("CRO", _TEMPERATE, False),
    "crop_c4": ("CRO", _TEMPERATE, True),
}


def pft_lookup_arrays() -> dict[str, np.ndarray]:
    """Per-CLM5-PFT (length-17) lookups built from the canopy biome tables."""
    vc3 = np.zeros(N_PFT_CLM5); vc4 = np.zeros(N_PFT_CLM5)
    rz0m = np.zeros(N_PFT_CLM5); rd = np.zeros(N_PFT_CLM5)
    hc = np.zeros(N_PFT_CLM5); fc4 = np.zeros(N_PFT_CLM5); is_veg = np.zeros(N_PFT_CLM5)
    for i, name in enumerate(CLM5_PFT_NAMES):
        biome, zone, c4 = _CLM5_TO_BIOME[name]
        if biome is None:                      # bare soil: no canopy
            continue
        is_veg[i] = 1.0
        fc4[i] = 1.0 if c4 else 0.0
        if c4:
            vc4[i] = PFT_VCMAX25_C4.get(biome, PFT_VCMAX25_C4["GRA"])[zone]
        else:
            vc3[i] = PFT_VCMAX25_C3[biome][zone]
        rz0m[i] = PFT_AERO_PARAMS[biome]["rz0m"]
        rd[i] = PFT_AERO_PARAMS[biome]["rd"]
        hc[i] = PFT_CANOPY_HEIGHT[biome]
    return {"vc3": vc3, "vc4": vc4, "rz0m": rz0m, "rd": rd,
            "hc": hc, "fc4": fc4, "is_veg": is_veg}


def cover1d(a):
    """A cover fraction as ``(ncol,)`` (squeeze a leading single-year axis)."""
    a = np.asarray(a)
    return a[0] if a.ndim == 2 else a
