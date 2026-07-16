"""NEMO ``ln_rnf_depth_ini`` runoff spread-depth map (sbcrnf.F90).

NEMO computes, ONCE at initialisation, a per-cell depth over which river
runoff is spread, proportional to the cell's climatological runoff
maximum::

    h_rnf = (rn_dep_max / rn_rnf_max) · max_month(rnf)   where rnf > 0
    h_rnf = 1 m                                          elsewhere
    h_rnf = min(h_rnf, local bottom depth)

(sbcrnf.F90 "depth of runoff computed once from max value of runoff";
ORCA1 namelist: ``rn_dep_max = 150``, ``rn_rnf_max = 0.05`` kg/m²/s.)
The Amazon (rnf ≈ rn_rnf_max) spreads over the full 150 m while small
Arctic rivers (rnf ~ 1e-3) stay in the top ~3 m — a FLAT 150 m spread
dilutes their surface plumes through the shelf column and leaves the
Siberian/Beaufort/Hudson SSS several PSU too salty (the day-90 faithful
runs' northern SSS residual).

The map feeds
:func:`legoesm.ocean.freshwater.runoff_spread_virtual_salt_tendency_3d`
via the model config's ``runoff_depth_spread_map`` (its fractional
per-level weights also cap the spread at the wet column depth, matching
NEMO's bottom clamp exactly on any vertical grid).
"""

from __future__ import annotations

import numpy as np

__physics_contract__ = {
    "units": {"runoff_monthly": "kg/m^2/s", "H_bathy": "m",
              "return": "m"},
    "signs": {"runoff_monthly": "positive into the ocean",
              "return": "positive spread depth below the surface"},
    "conserves": ["none"],
    "differentiable": False,
    "reference": "NEMO 5.0.1 sbcrnf.F90 ln_rnf_depth_ini "
                 "(ORCA1: rn_dep_max=150, rn_rnf_max=0.05)",
    "idealized_test": "tests/ocean/unit/test_runoff_depth_map.py — F90 "
                      "transliteration equality + Amazon/Arctic contrast",
}

# --- NEMO ORCA1 namelist values (namsbc_rnf) ---
_RN_DEP_MAX_DEFAULT = 150.0     # depth over which runoff is spread [m]
_RN_RNF_MAX_DEFAULT = 0.05      # max of the runoff climatology [kg/m^2/s]
_H_RNF_DRY_DEFAULT = 1.0        # h_rnf where the climatology is zero [m]


def nemo_runoff_depth_map(
    runoff_monthly: np.ndarray,
    H_bathy: np.ndarray,
    *,
    dep_max: float = _RN_DEP_MAX_DEFAULT,
    rnf_max: float = _RN_RNF_MAX_DEFAULT,
    h_dry: float = _H_RNF_DRY_DEFAULT,
) -> np.ndarray:
    """Per-cell runoff spread depth [m], NEMO ``ln_rnf_depth_ini``.

    Parameters
    ----------
    runoff_monthly : (12, ...) or (...,)
        Runoff climatology [kg/m²/s]; the month axis (if present) is
        reduced with ``max`` exactly like sbcrnf's read loop.
    H_bathy : (...)
        Local bottom depth [m, positive] — the ``min(h_rnf, gdept(mbkt))``
        clamp. (The freshwater helper's fractional weights re-apply the
        wet-column cap at run time; clamping here too keeps the stored
        map physical.)
    dep_max, rnf_max, h_dry : float
        ``rn_dep_max``, ``rn_rnf_max``, and the no-runoff floor (1 m).
    """
    rnf = np.asarray(runoff_monthly, dtype=np.float64)
    if rnf.ndim == np.asarray(H_bathy).ndim + 1:
        rnf_max_clim = rnf.max(axis=0)
    elif rnf.shape == np.asarray(H_bathy).shape:
        rnf_max_clim = rnf
    else:
        raise ValueError(
            f"runoff_monthly shape {rnf.shape} is neither (12, *bathy) nor "
            f"*bathy for H_bathy shape {np.asarray(H_bathy).shape}")
    if dep_max <= 0.0 or rnf_max <= 0.0:
        raise ValueError(
            f"dep_max ({dep_max}) and rnf_max ({rnf_max}) must be > 0")
    h = np.where(rnf_max_clim > 0.0,
                 (dep_max / rnf_max) * rnf_max_clim, h_dry)
    h = np.minimum(h, np.maximum(np.asarray(H_bathy, dtype=np.float64),
                                 h_dry))
    return np.maximum(h, h_dry)
