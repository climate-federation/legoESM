"""Interface between legoESM and the CLM-ML-JAX multilayer canopy model.

This module provides :func:`compute_clm_ml_canopy_fluxes`, which translates
legoESM's ``AtmToSurface`` forcing into the ``mlcanopy_type`` input
container, calls ``MLCanopyFluxes``, and maps the output back to a
``SurfaceFluxOutput``.

All imports of ``clm_ml_jax`` / ``multilayer_canopy`` are **lazy** (inside
function bodies) so that the rest of legoESM continues to import cleanly
without the optional ``canopy`` extra installed.

Variable-unit conventions
-------------------------
- legoESM ``psi_soil``: matric potential [m], negative for unsaturated
- CLM ``smp_l``:         matric potential [mm], negative for unsaturated
- legoESM ``K_unsat``:  hydraulic conductivity [m/s]
- CLM ``hk_l``:          hydraulic conductivity [mm/s]
- CLM ``swskyb``, ``swskyd``: direct/diffuse SW per waveband [W/m²]
- CLM patch/column/gridcell indices: 1-based (index 0 unused)
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.land.canopy.config import CLMMLCanopyConfig
from legoesm.land.canopy.sif import SIFConfig, multilayer_canopy_sif
from legoesm.land.canopy.state import CanopyState
from legoesm.land.surface_scheme import SurfaceFluxOutput
from legoesm.thermo import saturation_specific_humidity

# CLM-ML unfilled array elements carry spval = 1e36; treat anything above this
# guard (or non-finite) as invalid padding and drop it from the SIF sum.
_SPVAL_GUARD = 1.0e30

if TYPE_CHECKING:
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.land.config import MultiLayerLandConfig

# ---------------------------------------------------------------------------
# Physics contract
# ---------------------------------------------------------------------------

__physics_contract__ = {
    "units": {
        "shflx": "W/m² (positive upward, sensible heat to atmosphere)",
        "lhflx": "W/m² (positive upward, latent heat to atmosphere)",
        "G_soil": "W/m² (positive downward into soil)",
        "lw_up": "W/m² (positive upward, outgoing longwave)",
        "T_canopy_air": "K (PAI-weighted mean within-canopy air temperature)",
        "gpp": "gC/m²/s (gross primary production, zero in darkness)",
    },
    "signs": {
        "shflx": "positive = atmosphere gains heat",
        "lhflx": "positive = atmosphere gains moisture-equivalent energy",
        "G_soil": "positive = soil gains heat",
        "lw_up": "positive = upward emission from surface",
    },
    "conserves": "energy (Rnet = shflx + lhflx + G_soil + stflx_air + stflx_veg per timestep)",
    # Differentiable via the JAX-native diff path, GATED on
    # ``CLMMLCanopyConfig.differentiable=True`` (single column): the interface
    # passes a GridInfo (``grid=``) so ``MLCanopyFluxes`` runs ``lax.scan`` +
    # ``jax.checkpoint`` and the forcing→flux map (incl. trainable Vcmax25/g1) is
    # on the ``jax.grad`` tape.  Production default (``differentiable=False``)
    # stays forward-only (host-syncing checks, no tape).  Verified: forward/diff
    # flux parity ~1e-14 and FD grad rel_err <0.01% (TestCLMMLDifferentiability).
    "differentiable": True,
    "reference": "Bonan et al. (2021), GMD, CLM-ML v2",
    "idealized_test": "tests/land/unit/test_canopy.py::TestCLMMLInterface",
}

# ---------------------------------------------------------------------------
# Module-level CLM initialization guard and topology cache
# ---------------------------------------------------------------------------

# Minimum canopy-top geometry height [m].  A prescribed/climatology ``htop`` can
# be 0 on a bare or surfdata-uncovered column; CLM-ML then derives
# ``hbot = hbot_frac * htop = 0`` and the ``hbot < htop`` layering invariant
# collapses to a zero-thickness canopy.  Floor ``htop`` to this small positive
# value so the geometry stays valid — LAI is 0 on those columns, so the canopy
# contributes no fluxes regardless of the nominal height.  Matches the two-leaf
# path's ``HC_MIN_M`` (0.1 m) in ``boundary_data/_internals``.
_HTOP_GEOM_MIN_M: float = 0.1

_CLM_INITIALIZED: bool = False

# Topology cache: avoid re-running _setup_clm_topology when the grid has not
# changed between timesteps.  Key = (ncol, lat[0], lon[0]) — sufficient to
# detect a new grid allocation; full lat/lon arrays are not hashed here for
# performance.  Reset when any element changes.
_last_topology_key: tuple | None = None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _ensure_clm_initialized() -> None:
    """Call CLM phase-1 initialization exactly once."""
    global _CLM_INITIALIZED
    if _CLM_INITIALIZED:
        return
    from clm_src_main.clm_varpar import clm_varpar_init
    from offline_driver import clmSoilOptionMod

    # Use CLM4.5 physics so nlevsoi=10 matches legoESM's default 10-layer soil.
    clmSoilOptionMod.clm_phys = "CLM4_5"
    clm_varpar_init()

    # Initialize MLpftcon and psihat look-up tables.
    from multilayer_canopy.MLCanopyTurbulenceMod import LookupPsihatINI
    from clm_src_main import pftconMod
    from multilayer_canopy import MLpftconMod

    pftconMod.pftcon = pftconMod.Init()
    MLpftconMod.MLpftcon = MLpftconMod.Init()
    LookupPsihatINI()

    # Initialize orbital parameters for year 2001 (non-leap, 365 days).
    # Must match the start_date_ymd=20010101 epoch used in _setup_clm_time so
    # that CLM's shr_orb_cosz gets consistent orbital geometry (year 2000 is
    # a leap year and shifts caldays after Feb 28 by one day).
    from clm_share.shr_orb_mod import shr_orb_params
    import clm_src_utils.clm_varorb as _varorb

    # shr_orb_params returns (eccen, obliq_deg, mvelp_deg, obliqr, lambm0, mvelpp)
    eccen, _obliq, _mvelp, obliqr, lambm0, mvelpp = shr_orb_params(2001)  # coeff-ok: non-leap reference year 2001 for CLM orbital parameters epoch
    _varorb.eccen = float(eccen)
    _varorb.obliqr = float(obliqr)
    _varorb.mvelpp = float(mvelpp)
    _varorb.lambm0 = float(lambm0)

    _CLM_INITIALIZED = True


def _setup_clm_topology(
    ncol: int,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    dz_soil: np.ndarray,
    z_soil: np.ndarray,
    z_ref: float,
    pft_clm: int = 7,
) -> None:
    """Set up CLM module-level topology singletons for ``ncol`` columns.

    Uses a 1:1 mapping: patch index ``p`` = column index ``c`` = gridcell
    index ``g`` = legoESM column ``i + 1`` (1-based).

    Parameters
    ----------
    ncol : int
        Number of legoESM columns.
    lat_deg : np.ndarray
        Latitude of each column [degrees], shape ``(ncol,)``.
    lon_deg : np.ndarray
        Longitude of each column [degrees, -180..180 or 0..360],
        shape ``(ncol,)``.  Used by the internal solar zenith calculation.
    dz_soil : np.ndarray
        Soil layer thicknesses [m], shape ``(n_layers,)``.
    z_soil : np.ndarray
        Soil layer mid-point depths from surface [m], shape ``(n_layers,)``.
    z_ref : float
        Atmospheric reference height [m].
    pft_clm : int
        CLM PFT index (1-based) applied to all columns.  Controls Vcmax25
        and plant hydraulic parameters via the MLpftcon lookup table.
    """
    from clm_src_main import ColumnType as _col_mod
    from clm_src_main import GridcellType as _grc_mod
    from clm_src_main.ColumnType import column_type
    from clm_src_main.PatchType import patch
    from clm_src_main.clm_varpar import nlevsno, nlevgrnd, nlevsoi
    from clm_src_main.clm_varcon import ispval

    n_layers = len(dz_soil)

    # ---- patch ----
    # patch arrays: 1-based, index 0 unused
    col_arr = np.full(ncol + 1, ispval, dtype=np.int32)
    gc_arr = np.full(ncol + 1, ispval, dtype=np.int32)
    itype_arr = np.full(ncol + 1, ispval, dtype=np.int32)
    for i in range(ncol):
        p = i + 1  # 1-based patch index
        col_arr[p] = p       # column = patch (1:1)
        gc_arr[p] = p        # gridcell = patch (1:1)
        itype_arr[p] = pft_clm  # CLM PFT from config (was hardcoded to 13)

    patch.column = jnp.array(col_arr, dtype=jnp.int32)
    patch.gridcell = jnp.array(gc_arr, dtype=jnp.int32)
    patch.itype = jnp.array(itype_arr, dtype=jnp.int32)

    # ---- col ----
    # Shape: (ncol+1, nlevsno+nlevgrnd+1)
    nth = nlevsno + nlevgrnd + 1
    snl_np = np.zeros(ncol + 1, dtype=np.int32)  # no snow
    snl_np[0] = ispval
    dz_np = np.full((ncol + 1, nth), np.nan)
    z_np = np.full((ncol + 1, nth), np.nan)
    zi_np = np.full((ncol + 1, nth), np.nan)
    nbedrock_np = np.full(ncol + 1, ispval, dtype=np.int32)

    # Soil layers in CLM convention:
    # Python index j (1-based) → Fortran layer j → depth from surface
    # nlevsno=5 in CLM4.5, so Python j=6..15 are soil layers 1..10 for CLM4.5
    # BUT in standalone mode snl=0, so Fortran layer 1 = Python index nlevsno+1 = 6
    # _GetCLMVar: j = int(snl[c]) + 1 = 1 → Python index 1 (no snow offset applied)
    # This seems inconsistent... let me use the simpler j=1..n_layers directly.
    # _GetCLMVar line: j = int(_snl_np[c]) + 1 = 1, then t_soisno_col[c, 1]
    # So the convention in standalone CLM-ML-JAX is: j=1 is the top soil layer,
    # regardless of the snow layer allocation. The snl offset is not applied here.

    for i in range(ncol):
        c = i + 1  # 1-based column index
        for j in range(1, n_layers + 1):
            dz_np[c, j] = float(dz_soil[j - 1])
            z_np[c, j] = float(z_soil[j - 1])
        # Fill remaining CLM soil layers (n_layers+1..nlevsoi) with the deepest
        # legoESM layer rather than leaving them as NaN.  SoilResistance reads
        # dz[c, j] for all j=1..nlevsoi; NaN propagates to gradients via jnp.where.
        for j in range(n_layers + 1, nlevsoi + 1):
            dz_np[c, j] = float(dz_soil[-1])
            z_np[c, j] = float(z_soil[-1]) + float(dz_soil[-1]) * (j - n_layers)
        # Interface depths: zi[c, 0] = 0 (surface), zi[c, j] = cumulative depth
        zi_np[c, 0] = 0.0
        cum = 0.0
        for j in range(1, nlevsoi + 1):
            cum += dz_np[c, j]
            zi_np[c, j] = cum
        nbedrock_np[c] = n_layers

    _col_mod.col = column_type(
        snl=jnp.array(snl_np, dtype=jnp.int32),
        dz=jnp.array(dz_np, dtype=jnp.float64),
        z=jnp.array(z_np, dtype=jnp.float64),
        zi=jnp.array(zi_np, dtype=jnp.float64),
        nbedrock=jnp.array(nbedrock_np, dtype=jnp.int32),
    )

    # ---- grc ----
    from clm_src_main.GridcellType import GridcellType as gridcell_type
    latdeg_np = np.full(ncol + 1, 0.0, dtype=np.float64)
    londeg_np = np.full(ncol + 1, 0.0, dtype=np.float64)
    for i in range(ncol):
        g = i + 1
        latdeg_np[g] = float(lat_deg[i]) if i < len(lat_deg) else 0.0
        londeg_np[g] = float(lon_deg[i]) if i < len(lon_deg) else 0.0
    _grc_mod.grc = gridcell_type(
        latdeg=jnp.array(latdeg_np, dtype=jnp.float64),
        londeg=jnp.array(londeg_np, dtype=jnp.float64),
    )


def _setup_clm_time(dt: float, doy: float, step_count: int) -> None:
    """Configure the CLM time manager so that ``get_curr_calday(0) ≈ doy + 1``.

    CLM calday convention: calday 1.000 = 0Z on Jan 1.  So if legoESM
    provides ``doy`` as a 0-based float (0.0 = Jan 1, 120.0 = May 1 in a
    non-leap year), the correct CLM calday is ``doy + 1``.

    We encode this by pinning ``start_date_ymd = 20000101`` (calday=1) and
    choosing ``itim`` such that::

        get_curr_calday(0) = 1 + itim * dt / 86400 ≈ doy + 1
        → itim = round(doy * 86400 / dt)

    This fixes the previous bug where ``itim = step_count`` (1, 2, 3 …)
    caused the internal calday to always be near January 1 regardless of the
    actual simulation date, producing wrong solar zenith angles and hence
    wrong sun/shade fractions and per-layer radiation profiles.
    """
    import clm_src_utils.clm_time_manager as _tm

    _tm.dtstep = int(dt)
    # Encode actual day-of-year into itim so that get_curr_calday returns
    # the correct calendar day for solar zenith computation.
    _tm.itim = max(1, round(doy * 86400.0 / max(dt, 1.0)))
    # Use year 2001 (non-leap) as the reference epoch for both start_date and
    # curr_date.  Year 2000 is a leap year (366 days); the CLM time manager's
    # get_curr_date() recomputes curr_date_ymd from itim*dtstep using isleap(),
    # so keeping start_date_ymd=20000101 caused all caldays after Feb 28 to be
    # 1 day behind (Apr 30 instead of May 1 for doy=120).
    _tm.start_date_ymd = 20010101  # coeff-ok: CLM time epoch: Jan 1 2001 (non-leap year, 365 days)
    _tm.start_date_tod = 0
    # Derive curr_date_ymd from doy (non-leap 2001 epoch).
    import datetime as _dt
    _jan1 = _dt.date(2001, 1, 1)  # coeff-ok: CLM time epoch: Jan 1 2001 (non-leap year, 365 days)
    _curr = _jan1 + _dt.timedelta(days=int(doy))
    _tm.curr_date_ymd = int(_curr.strftime("%Y%m%d"))
    _tm.curr_date_tod = int((doy * 86400.0) % 86400)


def _compute_cos_zenith(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    doy: float,
) -> np.ndarray:
    """Compute cosine of solar zenith angle per column.

    .. note::
        **Legacy function** — used only in unit tests
        (``tests/land/unit/test_canopy.py::TestSolarGeometry``).
        The production path uses :func:`_compute_virtual_lon_deg` with
        CLM's Kepler ``shr_orb_cosz`` for round-trip consistency.
        Do NOT delete this function without updating the test class.

    Uses the Spencer (1971) declination formula.  Accuracy is ±0.01 in
    cos_zen, sufficient for the Weiss–Norman clearness-index partition.

    Parameters
    ----------
    lat_deg : np.ndarray  shape (ncol,), latitude in degrees
    lon_deg : np.ndarray  shape (ncol,), longitude in degrees (-180..180 or 0..360)
    doy     : float, 0-based day of year (0.0 = Jan 1 00:00 UTC)

    Returns
    -------
    cos_zen : np.ndarray  shape (ncol,), clamped to [0, 1]
    """
    lat_r = np.deg2rad(lat_deg)
    # Solar declination — Spencer (1971).
    # doy is 0-based (0.0 = Jan 1 00:00), so Spencer's d_n = doy + 1 and
    # B = 2π*(d_n-1)/365 = 2π*doy/365.  The previous (doy-1) was wrong by 1 day.
    B = 2.0 * np.pi * doy / 365.0  # coeff-ok: Julian year length (365 days, non-leap 2001 epoch)
    decl = (0.006918                # coeff-ok: Spencer (1971, J. Appl. Meteorol.) declination polynomial
            - 0.399912 * np.cos(B)  # coeff-ok: Spencer (1971) declination polynomial
            + 0.070257 * np.sin(B)  # coeff-ok: Spencer (1971) declination polynomial
            - 0.006758 * np.cos(2 * B)  # coeff-ok: Spencer (1971) declination polynomial
            + 0.000907 * np.sin(2 * B)  # coeff-ok: Spencer (1971) declination polynomial
            - 0.002697 * np.cos(3 * B)  # coeff-ok: Spencer (1971) declination polynomial
            + 0.00148  * np.sin(3 * B))  # coeff-ok: Spencer (1971) declination polynomial
    # Fractional time of day (UTC hours from doy fractional part)
    frac = doy % 1.0           # 0.0 = midnight, 0.5 = noon UTC
    utc_hour = frac * 24.0     # coeff-ok: exact hours-per-day conversion (24 h/day)
    # Local solar time hour angle (degrees, 0=noon)
    lon_norm = np.where(lon_deg > 180.0, lon_deg - 360.0, lon_deg)
    ha_deg = (utc_hour - 12.0) * 15.0 + lon_norm  # coeff-ok: exact degrees-per-hour (360°/24h=15°/h)
    ha_r = np.deg2rad(ha_deg)
    cos_zen = (np.sin(lat_r) * np.sin(decl)
               + np.cos(lat_r) * np.cos(decl) * np.cos(ha_r))
    return np.maximum(cos_zen, 0.0)


def _compute_virtual_lon_deg(
    cos_zen_forcing: np.ndarray,
    lat_deg: np.ndarray,
    caldaym1: float,
) -> np.ndarray:
    """Invert shr_orb_cosz to find a longitude that reproduces cos_zen_forcing.

    Uses CLM's own Kepler orbital mechanics (``shr_orb_decl``) and the same
    ``caldaym1`` that ``MLCanopyFluxesMod._MLCanopyForcing`` will use, so the
    inversion is exact rather than approximated by Spencer (1971).

    Must be called after ``_ensure_clm_initialized()`` (sets ``clm_varorb``
    orbital parameters used by ``shr_orb_decl``).

    The CLM formula (``shr_orb_cosz``):
        cosz = sin(lat)*sin(decl) - cos(lat)*cos(decl)*cos(jday_frac*2π + lon)
    Inversion:
        lon  = arccos((sin(lat)*sin(decl) − cosz) / (cos(lat)*cos(decl)))
               − jday_frac * 2π

    Both arccos branches produce the same cosz when substituted back, so the
    principal branch (arccos → [0, π]) is unambiguous.  The arccos result is
    well-defined even at night (``cos_arg`` is clipped to [−1, 1]) so no
    nighttime guard is needed.

    Parameters
    ----------
    cos_zen_forcing : np.ndarray  shape (ncol,), ≥ 0 (clamp before calling)
    lat_deg : np.ndarray  shape (ncol,), latitude [°]
    caldaym1 : float
        CLM calendar day at the *beginning* of the current timestep.
        Equals ``get_curr_calday(offset=-dt)`` after ``_setup_clm_time``.
        Computed as ``1 + (itim - 1) * dt / 86400`` where
        ``itim = max(1, round(doy * 86400 / dt))``.

    Returns
    -------
    lon_deg : np.ndarray  shape (ncol,), virtual longitude [°, −180..180]
    """
    from clm_share.shr_orb_mod import shr_orb_decl
    import clm_src_utils.clm_varorb as _varorb

    lat_r = np.deg2rad(lat_deg)

    # CLM Kepler orbital mechanics — exact match with _MLCanopyForcing
    declinm1, _ = shr_orb_decl(caldaym1, _varorb.eccen, _varorb.mvelpp,
                                _varorb.lambm0, _varorb.obliqr)
    decl = float(declinm1)

    jday_frac = caldaym1 % 1.0  # fractional part of caldaym1 → [0, 1)

    sin_lat, cos_lat = np.sin(lat_r), np.cos(lat_r)
    sin_decl = np.full_like(lat_r, np.sin(decl))
    cos_decl = np.full_like(lat_r, np.cos(decl))

    denom = cos_lat * cos_decl
    denom_safe = np.where(np.abs(denom) > 1e-6, denom,
                          np.where(denom >= 0, 1e-6, -1e-6))
    cos_arg = np.clip((sin_lat * sin_decl - cos_zen_forcing) / denom_safe, -1.0, 1.0)

    lon_rad = np.arccos(cos_arg) - jday_frac * 2.0 * np.pi
    lon_rad = ((lon_rad + np.pi) % (2.0 * np.pi)) - np.pi  # → [−π, π]
    return np.degrees(lon_rad)


def _estimate_beam_fraction(
    sw_down: jnp.ndarray,
    cos_zen: np.ndarray,
) -> jnp.ndarray:
    """Estimate direct-beam fraction from clearness index (Erbs et al. 1982).

    Clearness index  kt = SW_down / (S0 * cos_zen)  where S0 = 1361 W/m².
    Erbs et al. (1982, Solar Energy 28:293-302) give the *diffuse* fraction Id/I:

        kt ≤ 0.22:  Id/I = 1 - 0.09*kt
        0.22 < kt ≤ 0.80:
            Id/I = 0.9511 - 0.1604*kt + 4.388*kt² - 16.638*kt³ + 12.336*kt⁴
        kt > 0.80:  Id/I = 0.165

    Direct fraction: f_dir = 1 - Id/I, clamped to [0, 1].
    Returns f_dir per column.

    JAX-native: ``sw_down`` is kept as a traced ``jnp`` array so that
    ``d(f_dir)/d(sw_down)`` (via the clearness index ``kt``) stays on the
    ``jax.grad`` tape.  ``cos_zen`` is solar geometry (a non-differentiated
    constant); passing a NumPy array is fine — ``jnp`` ops upcast it.
    """
    S0 = constants.S_0  # solar constant [W/m²]
    sw = jnp.asarray(sw_down, dtype=jnp.float64)
    cos_zen_clamped = jnp.maximum(jnp.asarray(cos_zen, dtype=jnp.float64), 0.01)  # coeff-ok: minimum cos_zen floor to avoid division by zero in clearness index
    sw_toa = S0 * cos_zen_clamped
    kt = jnp.where(sw > 1.0, jnp.minimum(sw / sw_toa, 1.0), 0.0)

    # Erbs et al. (1982, Solar Energy 28:293-302) diffuse-fraction polynomial
    id_over_i_low = 1.0 - 0.09 * kt  # coeff-ok: Erbs et al. (1982, Solar Energy 28:293) low-kt regime
    id_over_i_mid = (0.9511 - 0.1604 * kt + 4.388 * kt**2  # coeff-ok: Erbs et al. (1982) mid-kt polynomial
                     - 16.638 * kt**3 + 12.336 * kt**4)     # coeff-ok: Erbs et al. (1982) mid-kt polynomial
    id_over_i_high = jnp.full_like(kt, 0.165)  # coeff-ok: Erbs et al. (1982) high-kt (clear-sky) limit
    id_over_i = jnp.where(kt <= 0.22, id_over_i_low,  # coeff-ok: Erbs et al. (1982) kt regime threshold
                jnp.where(kt <= 0.80, id_over_i_mid, id_over_i_high))  # coeff-ok: Erbs et al. (1982) kt regime threshold
    f_dir = jnp.clip(1.0 - id_over_i, 0.0, 1.0)

    # At night (sw_down < 1 W/m²) force beam fraction to zero
    return jnp.where(sw < 1.0, 0.0, f_dir)


def _sw_partition(
    sw_down: jnp.ndarray,
    f_vis: float = 0.46,  # coeff-ok: observation-based VIS fraction (Weiss & Norman 1985; ~0.46 climatological mean)
    f_dir: float = -1.0,
    cos_zen: np.ndarray | None = None,
    f_dir_fallback: float = 0.30,  # coeff-ok: Erbs et al. (1982) overcast-sky beam fraction fallback; prefer CLMMLCanopyConfig.f_dir_noclearness_fallback
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Partition total downwelling SW into direct/diffuse × VIS/NIR bands.

    Parameters
    ----------
    sw_down : jnp.ndarray
        Total downwelling shortwave [W/m²], shape ``(ncol,)``.
    f_vis : float
        Fraction of total SW in the visible (PAR) band [0.4–0.7 µm].
        Observation-based climatological value is 0.46 (not 0.50).
    f_dir : float
        Direct-beam fraction of total SW.
        -1.0  → auto-estimated from ``cos_zen`` via clearness index
                 (Weiss & Norman 1985; Erbs et al. 1982).  This is the
                 physically correct default for ESM applications.
        0–1   → fixed override (use only for idealised runs).
    cos_zen : np.ndarray | None
        Cosine of solar zenith angle per column, shape ``(ncol,)``.
        Required when ``f_dir < 0``.  Ignored otherwise.
    f_dir_fallback : float
        Beam fraction used when ``f_dir < 0`` but ``cos_zen`` is unavailable.
        Provided from ``CLMMLCanopyConfig.f_dir_noclearness_fallback``.

    Returns
    -------
    swskyb_vis, swskyb_nir : jnp.ndarray
        Direct beam SW in VIS and NIR bands [W/m²].
    swskyd_vis, swskyd_nir : jnp.ndarray
        Diffuse SW in VIS and NIR bands [W/m²].

    JAX-native: ``sw_down`` is kept traced end-to-end so ``d(swsky*)/d(sw_down)``
    flows on the ``jax.grad`` tape (needed for differentiability w.r.t. the SW
    forcing).  ``f_vis``/``f_dir``/``f_dir_fallback`` are static Python floats, so
    the ``if f_dir < 0.0`` branch is resolved at trace time (not a traced select).
    """
    sw = jnp.asarray(sw_down, dtype=jnp.float64)

    if f_dir < 0.0:
        # Physics-based estimate using clearness index
        if cos_zen is None:
            # Fallback: assume overcast (conservative, no zenith info)
            f_dir_arr = jnp.full_like(sw, f_dir_fallback)
        else:
            f_dir_arr = _estimate_beam_fraction(sw, cos_zen)
    else:
        f_dir_arr = jnp.full_like(sw, float(f_dir))

    vis = f_vis * sw
    nir = (1.0 - f_vis) * sw
    swskyb_vis = f_dir_arr * vis
    swskyb_nir = f_dir_arr * nir
    swskyd_vis = (1.0 - f_dir_arr) * vis
    swskyd_nir = (1.0 - f_dir_arr) * nir
    return swskyb_vis, swskyb_nir, swskyd_vis, swskyd_nir


def _build_stubs(
    ncol: int,
    forcing: AtmToSurface,
    canopy_config: CLMMLCanopyConfig,
    land_config: "MultiLayerLandConfig",
    land_params: Any | None,
    T_soil_top: jnp.ndarray,
    psi_soil: jnp.ndarray | None,
    theta_soil: jnp.ndarray | None,
    dz_soil: np.ndarray,
    soil_hydraulics: Any | None,
    cos_zen: np.ndarray | None = None,
    T_soil_all: jnp.ndarray | None = None,
    t_a10_prior: jnp.ndarray | None = None,
    lai_override: jnp.ndarray | None = None,
) -> dict[str, Any]:
    """Build minimal CLM input stub objects from legoESM state.

    All stubs are simple Python namespaces; only the fields accessed by
    ``_GetCLMVar``, ``initVerticalStructure``, ``SoilResistance``, and the
    soil relative humidity block are populated.

    Returns
    -------
    dict with keys: atm2lnd, wateratm2lndbulk, surfalb, soilstate,
                    waterstatebulk, temperature, frictionvel, canopystate,
                    energyflux, waterfluxbulk, solarabs, waterdiagnosticbulk
    """
    from clm_src_main.clm_varpar import nlevsoi, ivis, inir, nlevgrnd, nlevsno
    # NOTE: ivis=1, inir=2 in CLM

    np_ = ncol + 1  # 1-based patch dimension

    # ---- SW partitioning ----
    swskyb_vis, swskyb_nir, swskyd_vis, swskyd_nir = _sw_partition(
        forcing.sw_down,
        f_vis=float(canopy_config.f_vis),
        f_dir=float(canopy_config.f_dir),
        cos_zen=cos_zen,
        f_dir_fallback=float(canopy_config.f_dir_noclearness_fallback),
    )

    # shape (np_, numrad+1) = (np_, 3); CLM uses ivis=1, inir=2
    numrad = 2
    swskyb_col = np.zeros((np_, numrad + 1))
    swskyd_grc = np.zeros((np_, numrad + 1))
    swskyb_col_jax = jnp.zeros((np_, numrad + 1), dtype=jnp.float64)
    swskyd_grc_jax = jnp.zeros((np_, numrad + 1), dtype=jnp.float64)

    for i in range(ncol):
        p = i + 1  # 1-based
        swskyb_col_jax = swskyb_col_jax.at[p, ivis].set(swskyb_vis[i])
        swskyb_col_jax = swskyb_col_jax.at[p, inir].set(swskyb_nir[i])
        swskyd_grc_jax = swskyd_grc_jax.at[p, ivis].set(swskyd_vis[i])
        swskyd_grc_jax = swskyd_grc_jax.at[p, inir].set(swskyd_nir[i])

    # ---- CO2 and O2 partial pressures ----
    # forc_pco2[g] = CO2 partial pressure [Pa]
    # co2_ppmv × p_surface × 1e-6 → Pa
    # In _GetCLMVar: co2ref_cur = forc_pco2[g] / pbot * 1e6 → umol/mol (round-trips)
    co2_pa = forcing.co2_ppmv * forcing.p_surface * 1.0e-6  # Pa
    o2_pa = canopy_config.o2ref * 1.0e-3 * forcing.p_surface  # coeff-ok: exact unit conversion mmol/mol → mol/mol (1e-3); Pa = mol/mol × p_surface

    # ---- 1-D forcing arrays (1-based) ----
    def _pad1(arr):
        """Pad array to shape (ncol+1,) with 0 at index 0."""
        out = jnp.zeros(np_, dtype=jnp.float64)
        return out.at[1:].set(arr.astype(jnp.float64))

    forc_u = _pad1(forcing.u_lowest)
    forc_v = _pad1(forcing.v_lowest)
    forc_pco2 = _pad1(co2_pa)
    forc_po2 = _pad1(o2_pa)
    forc_solad_col = swskyb_col_jax   # (np_, 3)  direct SW
    forc_solai_grc = swskyd_grc_jax   # (np_, 3)  diffuse SW
    forc_t = _pad1(forcing.T_lowest)
    forc_pbot = _pad1(forcing.p_surface)
    forc_lwrad = _pad1(forcing.lw_down)
    forc_q = _pad1(forcing.q_lowest)
    forc_rain = _pad1(forcing.precip_total - forcing.precip_snow)
    forc_snow = _pad1(forcing.precip_snow)

    # ---- Ground albedo ----
    # Use land_params or config fallback
    # albgrd_col / albgri_col are the SUB-CANOPY SOIL (ground) spectral albedos,
    # used as the bottom boundary in the CLM-ML two-stream RT solver.
    # These must NOT be set from land_params.albedo_veg (vegetation broadband
    # albedo) — doing so was a prior bug.  Use configurable soil albedo defaults:
    # CLM4.5 lookup for loam soil: VIS ≈ 0.10, NIR ≈ 0.20.
    albgrd_vis = float(canopy_config.albgrd_vis_default)
    albgrd_nir = float(canopy_config.albgrd_nir_default)
    albgrd_col = jnp.zeros((np_, numrad + 1), dtype=jnp.float64)
    albgri_col = jnp.zeros((np_, numrad + 1), dtype=jnp.float64)
    for i in range(ncol):
        c = i + 1
        albgrd_col = albgrd_col.at[c, ivis].set(albgrd_vis)
        albgrd_col = albgrd_col.at[c, inir].set(albgrd_nir)
        albgri_col = albgri_col.at[c, ivis].set(albgrd_vis)
        albgri_col = albgri_col.at[c, inir].set(albgrd_nir)

    # ---- Soil state ----
    # Use nlevgrnd (total ground layers = 15 in CLM4.5) as the allocation size,
    # not nlevsoi (10 soil layers only).  MLSoilTemperatureMod writes to
    # thk[c, j] for j=1..nlevgrnd and would OOB on a nlevsoi+1-sized array.
    # Similarly t_soisno_col is declared with shape nlevsno+nlevgrnd in the type.
    # Both arrays are filled only for j=1..nlevsoi; deeper layers keep default.
    n_layers = len(dz_soil)
    smp_l_col = jnp.zeros((np_, nlevgrnd + 1), dtype=jnp.float64)   # [mm]
    hk_l_col = jnp.zeros((np_, nlevgrnd + 1), dtype=jnp.float64)    # [mm/s]
    rootfr_patch = jnp.zeros((np_, nlevgrnd + 1), dtype=jnp.float64)
    h2osoi_ice = jnp.zeros((np_, nlevgrnd + 1), dtype=jnp.float64)   # no ice
    # Soil thermal conductivity [W/m/K] — CLM4.5 Table 3.3 moist loam default.
    # Physical range 0.2–2.0 W/m/K; 0.9–1.5 for wetter/sandier soils.
    thk_col = jnp.full((np_, nlevgrnd + 1),
                       float(canopy_config.thk_soil_default_W_m_K),
                       dtype=jnp.float64)

    root_depth = float(land_config.root_depth)
    z_centers = np.array([float(z) for z in dz_soil], dtype=np.float64).cumsum() - dz_soil / 2

    # Root fraction exponential profile (same as legoESM multilayer_land.py)
    root_frac_np = np.exp(-z_centers / root_depth)
    root_frac_np = root_frac_np / root_frac_np.sum()

    # Pad root fraction to nlevsoi layers and renormalize so sum == 1.0.
    # When n_layers < nlevsoi, the deepest legoESM fraction is replicated into
    # extra CLM layers; without renormalization sum(rootfr) > 1 and btran is
    # overweighted.
    n_clm_soil = nlevsoi
    rootfr_padded = np.zeros(n_clm_soil, dtype=np.float64)
    for j in range(n_clm_soil):
        jl = min(j, n_layers - 1)
        rootfr_padded[j] = float(root_frac_np[jl])
    rootfr_total = rootfr_padded.sum()
    if rootfr_total > 0:
        rootfr_padded /= rootfr_total  # guarantee sum == 1.0

    for i in range(ncol):
        p = i + 1  # 1-based patch = column
        # Fill all nlevsoi layers so that layers beyond n_layers don't stay at
        # smp=0 mm (saturated) which biases btran via the weighted-average.
        for j in range(1, nlevsoi + 1):
            if psi_soil is not None and j - 1 < psi_soil.shape[1]:
                # psi [m] → smp_l [mm]; preserve sign (negative for unsaturated)
                smp_l_col = smp_l_col.at[p, j].set(psi_soil[i, j - 1] * 1000.0)
            else:
                smp_l_col = smp_l_col.at[p, j].set(float(canopy_config.smp_default_mm))

            if (soil_hydraulics is not None and psi_soil is not None
                    and theta_soil is not None and j - 1 < theta_soil.shape[1]):
                from legoesm.land.soil_hydraulics import hydraulic_conductivity
                K = hydraulic_conductivity(psi_soil[i, j - 1], theta_soil[i, j - 1],
                                           soil_hydraulics)
                # Keep K traced (no float()) so d(hk)/d(psi,theta) stays on the
                # jax.grad tape; numerically identical to the prior float() cast.
                hk_l_col = hk_l_col.at[p, j].set(K * 1000.0)  # m/s → mm/s
            else:
                hk_l_col = hk_l_col.at[p, j].set(float(canopy_config.hk_default_mm_s))

            rootfr_patch = rootfr_patch.at[p, j].set(rootfr_padded[j - 1])

    # ---- Soil evaporative resistance (Sellers-Lockwood formula) ----
    # The CLM-ML SurfaceResistanceMod uses rs = exp(8.206 - 4.255*Se) [s/m]
    # where Se = effective saturation.  For a silty clay loam with porosity 0.464
    # and the default smp = -50000 mm = -0.49 MPa, Se ≈ 0.15, so
    # rs ≈ exp(8.206 - 4.255*0.15) ≈ exp(7.568) ≈ 1927 s/m.
    # For saturated soil (smp ≈ 0): rs ≈ exp(8.206 - 4.255) ≈ 52 s/m.
    # We use 2000 s/m as the default to match the moderate-stress config default,
    # replacing the previous 100 s/m (matched to saturated soil only).
    _rs_base = float(canopy_config.soilresis_default_s_m)
    soilresis_col = jnp.full(np_, _rs_base, dtype=jnp.float64)

    # ---- Temperature ----
    # t_a10_patch is the 10-day running mean canopy air temperature [K].
    # On first call (canopy_state=None) we have no prior state, so we fall back
    # to instantaneous T_lowest.  The caller must extract t_a10 from canopy_state
    # and pass it here on subsequent steps via the t_a10_prior argument.
    t_a10_patch = _pad1(t_a10_prior if t_a10_prior is not None else forcing.T_lowest)
    # Fill all nlevsoi soil temperature layers.  Previously only layer 1 was set;
    # layers 2–10 = 0 K corrupted thermal gradient and soil heat flux.
    # t_soisno_col shape must be (np_, nlevgrnd+1) to match CLM-ML type declaration.
    t_soisno_col = jnp.zeros((np_, nlevgrnd + 1), dtype=jnp.float64)
    for i in range(ncol):
        c = i + 1
        # No float() casts below: soil temperature is kept traced so
        # d(flux)/d(T_soil) / d(T_soil_top) flows on the jax.grad tape.  Values
        # are identical to the prior float() path in the eager (production) mode.
        if T_soil_all is not None and T_soil_all.shape[1] >= 1:
            n_fill = min(T_soil_all.shape[1], nlevsoi)
            for j in range(1, n_fill + 1):
                t_soisno_col = t_soisno_col.at[c, j].set(T_soil_all[i, j - 1])
            # Layers n_fill+1..nlevsoi: repeat deepest legoESM layer
            deepest_T = T_soil_all[i, n_fill - 1]
            for j in range(n_fill + 1, nlevgrnd + 1):
                t_soisno_col = t_soisno_col.at[c, j].set(deepest_T)
        else:
            # Fallback: fill all layers with surface soil temperature
            for j in range(1, nlevgrnd + 1):
                t_soisno_col = t_soisno_col.at[c, j].set(T_soil_top[i])

    # ---- canopystate ----
    htop_patch = jnp.zeros(np_, dtype=jnp.float64)
    elai_patch = jnp.zeros(np_, dtype=jnp.float64)
    esai_patch = jnp.zeros(np_, dtype=jnp.float64)
    for i in range(ncol):
        p = i + 1
        htop_v = (float(land_params.htop[i]) if land_params is not None and land_params.htop is not None
                  else 5.0)  # coeff-ok: 5 m fallback canopy height (CLM4.5 DBF-temperate default)
        # A prescribed/climatology htop can be 0 on a bare or uncovered column;
        # floor it so hbot = hbot_frac*htop stays < htop (valid CLM-ML layering).
        # LAI is 0 there, so the nominal height changes no canopy flux.
        htop_v = max(htop_v, _HTOP_GEOM_MIN_M)
        # LAI precedence: prognostic ``lai_override`` (C_fol / LCMA from the
        # DifferLand carbon pool) > prescribed ``LandSurfaceParams.LAI``
        # climatology > scalar fallback.  Canopy STRUCTURE (htop/SAI) stays
        # prescribed either way (the carbon cycle produces no allometric map).
        if lai_override is not None:
            lai_v = float(lai_override[i])
        elif land_params is not None and land_params.LAI is not None:
            lai_v = float(land_params.LAI[i])
        else:
            lai_v = 2.0  # coeff-ok: LAI=2 fallback (no prescribed/prognostic LAI)
        sai_v  = (float(land_params.SAI[i]) if land_params is not None and land_params.SAI is not None
                  else 0.5)
        htop_patch = htop_patch.at[p].set(htop_v)
        elai_patch = elai_patch.at[p].set(lai_v)
        esai_patch = esai_patch.at[p].set(sai_v)

    # ---- frictionvel ----
    forc_hgt_u_patch = jnp.full(np_, float(land_config.z_ref), dtype=jnp.float64)

    # ---- Build stub namespaces ----
    atm2lnd = SimpleNamespace(
        forc_u_grc=forc_u,
        forc_v_grc=forc_v,
        forc_pco2_grc=forc_pco2,
        forc_po2_grc=forc_po2,
        forc_solad_downscaled_col=forc_solad_col,
        forc_solai_grc=forc_solai_grc,
        forc_t_downscaled_col=forc_t,
        forc_pbot_downscaled_col=forc_pbot,
        forc_lwrad_downscaled_col=forc_lwrad,
    )
    wateratm2lndbulk = SimpleNamespace(
        forc_q_downscaled_col=forc_q,
        forc_rain_downscaled_col=forc_rain,
        forc_snow_downscaled_col=forc_snow,
    )
    surfalb = SimpleNamespace(
        albgrd_col=albgrd_col,
        albgri_col=albgri_col,
    )
    soilstate = SimpleNamespace(
        smp_l_col=smp_l_col,
        hk_l_col=hk_l_col,
        rootfr_patch=rootfr_patch,
        soilresis_col=soilresis_col,
        thk_col=thk_col,
    )
    waterstatebulk = SimpleNamespace(
        h2osoi_ice_col=h2osoi_ice,
    )
    temperature = SimpleNamespace(
        t_a10_patch=t_a10_patch,
        t_soisno_col=t_soisno_col,
    )
    frictionvel = SimpleNamespace(
        forc_hgt_u_patch=forc_hgt_u_patch,
    )
    canopystate = SimpleNamespace(
        htop_patch=htop_patch,
        elai_patch=elai_patch,
        esai_patch=esai_patch,
    )
    # Output receiver stubs (MLCanopyFluxes writes to these only when mlcan_to_clm=1)
    energyflux = SimpleNamespace(
        taux_patch=jnp.zeros(np_),
        tauy_patch=jnp.zeros(np_),
        eflx_lh_tot_patch=jnp.zeros(np_),
        eflx_sh_tot_patch=jnp.zeros(np_),
        eflx_lwrad_out_patch=jnp.zeros(np_),
    )
    waterfluxbulk = SimpleNamespace(
        qflx_evap_tot_patch=jnp.zeros(np_),
    )
    solarabs = SimpleNamespace(
        fsa_patch=jnp.zeros(np_),
    )
    waterdiagnosticbulk = SimpleNamespace(
        q_ref2m_patch=jnp.zeros(np_),
    )

    return dict(
        atm2lnd=atm2lnd,
        wateratm2lndbulk=wateratm2lndbulk,
        surfalb=surfalb,
        soilstate=soilstate,
        waterstatebulk=waterstatebulk,
        temperature=temperature,
        frictionvel=frictionvel,
        canopystate=canopystate,
        energyflux=energyflux,
        waterfluxbulk=waterfluxbulk,
        solarabs=solarabs,
        waterdiagnosticbulk=waterdiagnosticbulk,
    )


def _init_mlcanopy(ncol: int, stubs: dict, canopy_config: CLMMLCanopyConfig) -> Any:
    """Allocate and cold-start a fresh ``mlcanopy_type`` instance.

    Called only on the first legoESM step (``canopy_state.mlcanopy is None``).
    Sets ``htop``, ``hbot``, and beta-distribution shape parameters for the
    vertical PAD profile, then calls ``init_cold`` for leaf water potential
    and intercepted water initialisation.
    """
    from multilayer_canopy.MLCanopyFluxesType import create_mlcanopy, init_cold
    from multilayer_canopy.MLclm_varpar import nlevmlcan
    from clm_src_main.clm_varcon import spval

    mlcanopy = create_mlcanopy(1, ncol)

    # Set canopy geometry for each patch.
    ztop = mlcanopy.ztop_canopy
    zbot = mlcanopy.zbot_canopy
    # getPADparameters assigns PFT-default beta params only when pbeta < 0.
    # create_mlcanopy initializes them to spval (1e36), so force them negative
    # to trigger the PFT lookup in getPADparameters.
    pbeta_lai = jnp.full_like(mlcanopy.pbeta_lai_canopy, -1.0)
    pbeta_sai = jnp.full_like(mlcanopy.pbeta_sai_canopy, -1.0)

    canopy = stubs["canopystate"]
    for i in range(ncol):
        p = i + 1
        htop_v = float(canopy.htop_patch[p])
        hbot_v = canopy_config.hbot_frac * htop_v  # hbot_frac from CLMMLCanopyConfig (default 0.1)

        ztop = ztop.at[p].set(htop_v)
        zbot = zbot.at[p].set(hbot_v)

    # Initialize root_biomass_canopy to a physical value.
    # create_mlcanopy initializes it to spval=1e36, which causes SoilResistance
    # to compute rld ≈ 1e37 m/m³ and a soil-root conductance of ~1e33 — the
    # plant resistance then dominates correctly but layer-wise soil ET is wrong.
    # Default: 300 g/m² (temperate deciduous tree, Jackson et al. 1997).
    root_biomass = jnp.full_like(mlcanopy.root_biomass_canopy,
                                  float(canopy_config.root_biomass_default_g_m2))
    # root_biomass_canopy is initialized to spval for p=0; keep that convention
    # and only fill 1-based patch indices.
    for i in range(ncol):
        root_biomass = root_biomass.at[i + 1].set(
            float(canopy_config.root_biomass_default_g_m2)
        )

    mlcanopy = mlcanopy._replace(
        ztop_canopy=ztop,
        zbot_canopy=zbot,
        pbeta_lai_canopy=pbeta_lai,
        pbeta_sai_canopy=pbeta_sai,
        root_biomass_canopy=root_biomass,
    )

    # Call init_cold to set initial leaf water potential and intercepted water.
    # Signature: init_cold(mlcanopy_inst, begp, endp)
    mlcanopy = init_cold(mlcanopy, 1, ncol)

    return mlcanopy


def _extract_clm_ml_sif(
    mlcanopy: Any, ncol: int, sif_cfg: SIFConfig,
) -> jnp.ndarray | None:
    """Top-of-canopy SIF from the CLM-ML per-(layer, leaf) photosynthesis.

    Reads the ``mlcanopy_inst`` per-(patch, layer, leaf) arrays ``je_leaf``
    (the model's native electron-transport rate) and ``apar_leaf`` (absorbed
    PAR per unit leaf area), plus the per-(patch, layer) ``dpai_profile`` /
    ``fracsun_profile`` — the CLM-ml ``MLCanopyFluxesType`` names the
    ``multilayer_canopy`` JAX port mirrors faithfully.  SIF is a **pure
    consumer** of the Bonan model's Farquhar solution: it uses ``je_leaf``
    directly and never re-inverts ``je`` from ``An``/``Ci`` (nor re-derives
    ``Gamma*``).  Sums the per-leaf SIF weighted by leaf area (``dpai *
    fracsun`` sunlit, ``dpai * (1 - fracsun)`` shaded) the SAME way CLM-ML sums
    ``anet_leaf * dpai`` into canopy GPP, via the shared
    :func:`~legoesm.land.canopy.sif.multilayer_canopy_sif` core.

    hasattr-guarded: any missing field returns ``None`` (SIF is opt-in and must
    never crash the flux path if a port version renames a field).  Unfilled
    layers/leaves carry ``spval = 1e36`` and are masked to zero leaf area with a
    sanitised ``je``/``apar`` so they drop out of the sum cleanly.
    """
    required = ("je_leaf", "apar_leaf", "dpai_profile", "fracsun_profile")
    if not all(hasattr(mlcanopy, f) for f in required):
        return None

    # Patch axis is 1-based here: the legoESM interface builds mlcanopy via
    # create_mlcanopy(1, ncol) (begp=1), so column i (0..ncol-1) lives at array
    # index i+1 and index 0 is the unused pad — matching every sibling read in
    # _extract_surface_fluxes.  The layer (1:nlevmlcan) and leaf (1:nleaf) blocks
    # likewise skip index 0.  Leaf order: il=1 sunlit, il=2 shaded.
    def _lv(name):  # (ncol, nlev, nleaf)
        return jnp.stack([getattr(mlcanopy, name)[i + 1, 1:, 1:] for i in range(ncol)])

    def _pr(name):  # (ncol, nlev)
        return jnp.stack([getattr(mlcanopy, name)[i + 1, 1:] for i in range(ncol)])

    je, apar = _lv("je_leaf"), _lv("apar_leaf")
    dpai, fracsun = _pr("dpai_profile"), _pr("fracsun_profile")

    # Per-(layer, leaf) leaf-area weight from the layer PAI and sunlit fraction.
    leaf_area = jnp.stack([dpai * fracsun, dpai * (1.0 - fracsun)], axis=-1)  # (ncol,nlev,nleaf)

    # Mask unfilled / spval elements: zero their leaf area and sanitise je/apar
    # so nothing non-finite reaches the SIF core.
    valid = (
        jnp.isfinite(je) & jnp.isfinite(apar) & jnp.isfinite(leaf_area)
        & (jnp.abs(je) < _SPVAL_GUARD) & (apar < _SPVAL_GUARD) & (jnp.abs(leaf_area) < _SPVAL_GUARD)
    )
    je = jnp.where(valid, je, 0.0)
    apar = jnp.where(valid, jnp.maximum(apar, 0.0), 0.0)
    leaf_area = jnp.where(valid, jnp.maximum(leaf_area, 0.0), 0.0)

    def _flat(a):  # (ncol, nlev*nleaf)
        return a.reshape(a.shape[0], -1)

    return multilayer_canopy_sif(_flat(je), _flat(apar), _flat(leaf_area), sif_cfg)


def _extract_surface_fluxes(
    mlcanopy: Any,
    ncol: int,
    forcing: AtmToSurface,
    land_config: "MultiLayerLandConfig",
    land_params: Any | None,
    canopy_config: CLMMLCanopyConfig | None = None,
) -> SurfaceFluxOutput:
    """Map ``mlcanopy_type`` output fields to a ``SurfaceFluxOutput``.

    Sign conventions
    ----------------
    - ``shflx_canopy``: sensible heat flux [W/m²], positive into atmosphere
    - ``lhflx_canopy``: latent heat flux [W/m²], positive into atmosphere
    - ``gsoi_soil``:    ground heat flux [W/m²], positive into soil (CLM)
    - ``lwup_canopy``:  upwelling LW [W/m²]
    - ``gppveg_canopy``: GPP [µmol CO₂/m²/s]
    """
    from clm_src_main.clm_varpar import ivis, inir

    # Per-column arrays → (ncol,)
    shflx = jnp.stack([mlcanopy.shflx_canopy[i + 1] for i in range(ncol)])
    lhflx = jnp.stack([mlcanopy.lhflx_canopy[i + 1] for i in range(ncol)])
    G_soil = jnp.stack([mlcanopy.gsoi_soil[i + 1] for i in range(ncol)])
    lw_up = jnp.stack([mlcanopy.lwup_canopy[i + 1] for i in range(ncol)])
    # GPP: µmol CO2/m²/s → gC/m²/s  (standard atomic weight of C = constants.M_C g/mol)
    _M_C_g_per_mol = constants.M_C
    gpp_umol = jnp.stack([mlcanopy.gppveg_canopy[i + 1] for i in range(ncol)])
    gpp = gpp_umol * (_M_C_g_per_mol * 1.0e-6)  # µmol/m²/s → gC/m²/s

    # Friction velocity → momentum flux components
    ustar = jnp.stack([mlcanopy.ustar_canopy[i + 1] for i in range(ncol)])
    rho_a = forcing.rho_lowest  # moist air density, includes virtual-T correction
    tau_total = rho_a * ustar ** 2  # [N/m²]
    # Split tau between x and y proportional to wind components
    u = forcing.u_lowest
    v = forcing.v_lowest
    ws = jnp.sqrt(u ** 2 + v ** 2 + 1.0e-6)
    tau_x = tau_total * u / ws
    tau_y = tau_total * v / ws

    # Net radiation: absorbed SW + net LW
    # rnet_canopy is the full-column net radiation (canopy + soil).
    # swveg_canopy and swsoi_soil are indexed (patch, band) where band
    # indices ivis=1, inir=2; index 0 holds spval and must be skipped.
    rnet = jnp.stack([mlcanopy.rnet_canopy[i + 1] for i in range(ncol)])
    sw_net = jnp.stack(
        [
            mlcanopy.swveg_canopy[i + 1, ivis]
            + mlcanopy.swveg_canopy[i + 1, inir]
            + mlcanopy.swsoi_soil[i + 1, ivis]
            + mlcanopy.swsoi_soil[i + 1, inir]
            for i in range(ncol)
        ]
    )
    lw_net = rnet - sw_net

    # Canopy-mean albedo: 1 - SW_absorbed / SW_down
    sw_down = forcing.sw_down
    albedo = jnp.where(sw_down > 1.0, 1.0 - sw_net / (sw_down + 1.0e-6), 0.15)  # coeff-ok: nighttime/low-light albedo fallback (~0.15 broadband for vegetated surface)

    # Surface temperature: use soil surface T (updated by CLM-ML)
    T_surface = jnp.stack([mlcanopy.tg_soil[i + 1] for i in range(ncol)])

    # q_surface: surface specific humidity [kg/kg].
    # CLM-ML computes rhg_soil = exp(psis*Mw/(R*Tg)) (Philip formula) for the
    # soil surface relative humidity.  Using qsat(Tg) ignores this and
    # overestimates soil evaporation by ~30% at smp=-50000 mm (rhg≈0.70).
    # Use rhg_soil from mlcanopy if available; fall back to qsat only if absent.
    q_sat_surface = saturation_specific_humidity(T_surface, forcing.p_surface)
    if hasattr(mlcanopy, "rhg_soil"):
        rhg = jnp.stack([mlcanopy.rhg_soil[i + 1] for i in range(ncol)])
        # Clamp rhg to [0, 1] — spval=1e36 indicates uninitialised
        rhg = jnp.clip(rhg, 0.0, 1.0)
        q_surface = rhg * q_sat_surface
    else:
        q_surface = q_sat_surface

    # Emissivity and roughness from canopy
    emissivity = jnp.full(ncol, float(land_config.emissivity_land))
    z0 = jnp.stack([mlcanopy.z0m_canopy[i + 1] for i in range(ncol)])

    # Optional solar-induced fluorescence (passive TOC diagnostic).  Static gate
    # on the config leaf; per-(layer,leaf) sum shares the two-leaf/big-leaf core.
    sif = None
    sif_cfg = getattr(canopy_config, "sif", None) if canopy_config is not None else None
    if sif_cfg is not None:
        sif = _extract_clm_ml_sif(mlcanopy, ncol, sif_cfg)

    # Canopy heat storage: needed for energy balance closure in the coupler.
    # Rnet = SH + LH + G_soil + stflx_air + stflx_veg
    # Extract from mlcanopy if the fields exist (they are always computed by
    # MLCanopyFluxes but named differently in old CLM-ML-JAX versions).
    if hasattr(mlcanopy, "stflx_air_canopy"):
        stflx_air = jnp.stack([mlcanopy.stflx_air_canopy[i + 1] for i in range(ncol)])
    else:
        stflx_air = None
    if hasattr(mlcanopy, "stflx_veg_canopy"):
        stflx_veg = jnp.stack([mlcanopy.stflx_veg_canopy[i + 1] for i in range(ncol)])
    else:
        stflx_veg = None

    # T_canopy_air: aerodynamic exchange temperature for SH coupling to atmosphere.
    # taveg_canopy is the PAI-weighted mean within-canopy air temperature — the
    # best available proxy for the canopy exchange node Tc in H = rho*cp*(Tc-Ta)/Ra.
    # (tair_profile is not written back to the output NamedTuple by MLCanopyFluxes.)
    if hasattr(mlcanopy, "taveg_canopy"):
        T_canopy_air = jnp.stack([mlcanopy.taveg_canopy[i + 1] for i in range(ncol)])
        # Guard against cold-start spval=1e36 (no PAI → taveg=0, or uninitialized)
        T_canopy_air = jnp.where(
            (T_canopy_air > 1e30) | (T_canopy_air < 100.0),
            T_surface, T_canopy_air,
        )
    else:
        T_canopy_air = T_surface  # fallback: tg_soil

    # Below-canopy GROUND latent heat [W/m²]: CLM-ML's soil-surface evaporation
    # (``lhsoi_soil``, throttled by the Philip rhg_soil humidity), separate from
    # the leaf transpiration + canopy-water evaporation folded into ``lhflx``.
    # Exposing it lets the multilayer driver route the ground component to
    # snowpack sublimation (L_s) over snow while leaf transpiration (LE_canopy =
    # lhflx − LE_soil) draws soil water at L_v — the same phase-split the two-leaf
    # scheme gets.  ``lhsoi_soil`` is a CORE MLSoilFluxes output (always present
    # in a compatible clm-ml-jax), so read it directly: a missing field raises
    # loudly here rather than silently reverting the driver to the pre-F13 all-
    # L_v-soil routing (which would drain soil water for ground evaporation that
    # should sublimate from the snowpack).
    LE_soil = jnp.stack([mlcanopy.lhsoi_soil[i + 1] for i in range(ncol)])
    LE_canopy = lhflx - LE_soil

    return SurfaceFluxOutput(
        shflx=shflx,
        lhflx=lhflx,
        tau_x=tau_x,
        tau_y=tau_y,
        sw_net=sw_net,
        lw_net=lw_net,
        lw_up=lw_up,
        G_soil=G_soil,
        T_surface=T_surface,
        q_surface=q_surface,
        albedo=albedo,
        emissivity=emissivity,
        z0=z0,
        gpp=gpp,
        sif=sif,
        T_canopy_air=T_canopy_air,
        stomatal_ratio=jnp.ones(ncol),
        stflx_air=stflx_air,
        stflx_veg=stflx_veg,
        LE_canopy=LE_canopy,
        LE_soil=LE_soil,
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def extract_clm_ml_grid_info(canopy_state: CanopyState, patch: int = 1) -> Any:
    """Extract the concrete ``GridInfo(p, ncan, ntop, nbot)`` from a warm state.

    Call this ONCE on a concrete (non-traced) warm-start ``CanopyState`` — e.g.
    the state returned by the first forward step — and thread the result through
    every subsequent differentiable step via ``compute_clm_ml_canopy_fluxes(...,
    grid_info=...)``.  This keeps the single-site structural integers
    (``ncan``/``ntop``/``nbot``) CONCRETE even when the carried
    ``canopy_state.mlcanopy`` becomes a ``jax.grad`` tracer in a multi-step
    rollout (otherwise ``int(tracer)`` raises ``ConcretizationTypeError``).

    Parameters
    ----------
    canopy_state:
        A concrete warm-started state (``canopy_state.mlcanopy`` populated by a
        prior forward step).  Must NOT be a tracer.
    patch:
        1-based patch index (single-site diff mode uses ``1``).

    Returns
    -------
    GridInfo
        ``multilayer_canopy.MLclm_varctl.GridInfo`` with concrete Python ints.
    """
    from multilayer_canopy.MLclm_varctl import GridInfo
    if canopy_state is None or canopy_state.mlcanopy is None:
        raise ValueError(
            "extract_clm_ml_grid_info needs a warm-started canopy_state whose "
            "mlcanopy is populated; got None. Run one forward step first."
        )
    m = canopy_state.mlcanopy
    try:
        return GridInfo(
            p=int(patch),
            ncan=int(m.ncan_canopy[patch]),
            ntop=int(m.ntop_canopy[patch]),
            nbot=int(m.nbot_canopy[patch]),
        )
    except jax.errors.ConcretizationTypeError as exc:  # pragma: no cover - guard
        raise RuntimeError(
            "extract_clm_ml_grid_info must be called on a CONCRETE canopy_state "
            "(outside jax.grad tracing), not on a traced carry."
        ) from exc


def compute_clm_ml_canopy_fluxes(
    T_soil_top: jnp.ndarray,
    forcing: AtmToSurface,
    canopy_config: CLMMLCanopyConfig,
    land_config: "MultiLayerLandConfig",
    land_params: Any | None,
    w_frac_rz: jnp.ndarray,
    wind_speed: jnp.ndarray,
    canopy_state: CanopyState | None,
    dt: float,
    T_soil: jnp.ndarray | None = None,
    psi_soil: jnp.ndarray | None = None,
    theta_soil: jnp.ndarray | None = None,
    lat: jnp.ndarray | None = None,
    lon: jnp.ndarray | None = None,
    doy: float = 0.0,
    lai_override: jnp.ndarray | None = None,
    vcmaxpft_jax: jnp.ndarray | None = None,
    g1_medlyn_jax: jnp.ndarray | None = None,
    grid_info: Any | None = None,
) -> tuple[SurfaceFluxOutput, CanopyState]:
    """Compute canopy fluxes via the CLM-ML-JAX multilayer canopy model.

    Parameters
    ----------
    T_soil_top:
        Soil surface temperature [K], shape ``(ncol,)``.
    forcing:
        Atmospheric forcing from the coupler.
    canopy_config:
        Static CLM-ML-JAX configuration (``CLMMLCanopyConfig``).
    land_config:
        Parent ``MultiLayerLandConfig`` (for fallback soil parameters).
    land_params:
        Per-column ``LandSurfaceParams`` (or ``None`` for config defaults).
        Must supply ``LAI``, ``SAI``, ``htop`` for the canopy scheme.
    w_frac_rz:
        Root-zone soil-moisture stress fraction [0–1], shape ``(ncol,)``.
        **Not forwarded to CLM-ML-JAX.** CLM-ML computes its own hydraulic
        water-stress (btran) internally from the ``smp_l_col`` derived from
        ``psi_soil``.  Both scalars represent the same physical state, so no
        inconsistency arises; the legoESM diagnostic value is simply unused.
    wind_speed:
        Scalar wind speed [m/s], shape ``(ncol,)``.
        **Not forwarded to CLM-ML-JAX.** CLM-ML derives wind speed internally
        from ``forc_u_grc = forcing.u_lowest`` and ``forc_v_grc = forcing.v_lowest``.
        Pass this argument only if the caller wants to document the intended
        wind forcing; the value is ignored.
    canopy_state:
        Previous-step canopy state.  ``None`` or ``mlcanopy is None``
        triggers cold-start allocation.
    dt:
        Timestep [s].
    T_soil:
        Full soil temperature profile [K], shape ``(ncol, n_layers)``.
    psi_soil:
        Soil matric potential [m], shape ``(ncol, n_layers)``.
    theta_soil:
        Volumetric water content [m³/m³], shape ``(ncol, n_layers)``.
    lat:
        Latitude [degrees], shape ``(ncol,)`` or ``None``.
    lon:
        Longitude [degrees], shape ``(ncol,)`` or ``None``.
        Used for solar zenith angle and SW partitioning.  Pass ``None``
        to default to 0° (Greenwich); this is incorrect for most sites.
    doy:
        Day of year (0-based float, e.g. 120.0 = May 1 in a non-leap year).
    vcmaxpft_jax:
        Optional trainable per-PFT Vcmax25 override [µmol/m²/s], shape
        ``(mxpft+1,)`` — replaces the module-global ``MLpftcon.vcmaxpft`` lookup
        so ``jax.grad`` can flow into Vcmax25.  Injected as a TRACED leaf from
        the loss (SegmentForcing doctrine); ``None`` keeps the PFT default.
    g1_medlyn_jax:
        Optional trainable per-PFT Medlyn ``g1`` override [kPa^0.5], same shape
        contract as ``vcmaxpft_jax``.  Only active when the canopy stomatal model
        is Medlyn (``MLclm_varctl.gs_type == 0``); inert under the default WUE
        conductance (``gs_type == 2``).  ``None`` keeps the PFT default.
    grid_info:
        Optional concrete ``GridInfo(p, ncan, ntop, nbot)`` structural constants
        for the differentiable path.  REQUIRED for a MULTI-STEP differentiated
        rollout: when a ``jax.grad`` loss unrolls ≥2 canopy steps and carries the
        returned ``CanopyState`` forward, ``canopy_state.mlcanopy`` is a tracer, so
        the structural ints cannot be read from it — extract them once from the
        (concrete) warm-start state via :func:`extract_clm_ml_grid_info` and pass
        the same object each step.  ``None`` (single warm step whose state is a
        captured constant) reads the ints off the concrete template.

    Returns
    -------
    surface_out : SurfaceFluxOutput
        Canopy surface energy balance fluxes for the legoESM coupler.
    new_canopy_state : CanopyState
        Updated prognostic state to carry forward to the next step.
    """
    # ---- Phase-1 CLM initialization (once) ----
    _ensure_clm_initialized()

    # Lazy imports of CLM-ML-JAX entry point
    from multilayer_canopy.MLCanopyFluxesMod import MLCanopyFluxes
    from clm_src_main.decompMod import bounds_type
    from legoesm.land.soil_grid import make_soil_grid

    ncol = T_soil_top.shape[0]

    # ---- Differentiable-mode gate (static, resolved here — never traced) ----
    # ``differentiable=True`` opts a training run into the JAX-native diff path
    # (``grid=`` + ``lax.scan``).  It requires (a) a WARM canopy_state whose
    # ``mlcanopy`` already carries the vertical structure (ncan/ntop/nbot) — the
    # first cold-start step always runs forward-only to build it — and (b) a
    # single column: the diff path reads one concrete ``(ncan, ntop, nbot)``.
    # A multi-column diff request is a hard error (vmap over columns instead of
    # silently degrading to the forward path), matching the dispatch-hardening
    # rule.
    _want_diff = bool(getattr(canopy_config, "differentiable", False))
    if _want_diff and ncol != 1:
        raise ValueError(
            "CLMMLCanopyConfig.differentiable=True is single-column only "
            f"(ncol == 1); got ncol={ncol}. vmap the interface over columns for "
            "multi-column differentiation (M3)."
        )
    _warm_started = canopy_state is not None and canopy_state.mlcanopy is not None
    _diff_mode = _want_diff and _warm_started  # ncol == 1 guaranteed above

    # Solar geometry (lat / lon / doy / cos_zenith) is a NON-differentiated
    # static input BY DESIGN: it is consumed by host-side CLM orbital setup
    # (_setup_clm_time / _setup_clm_topology / shr_orb_cosz), not by the traced
    # canopy physics.  You do not train through the Sun's position — it is fixed
    # by lat/lon/time.  If a caller differentiates the WHOLE AtmToSurface pytree
    # (so cos_zenith becomes a tracer), the host ``np.array(forcing.cos_zenith)``
    # below would raise a cryptic TracerArrayConversionError.  Detect it here and
    # fail with an actionable message instead.  (Differentiating the physical
    # forcing leaves — T_lowest, sw_down, q, u, v, lw_down, p, co2 — is fully
    # supported; only geometry must stay concrete.)
    if _want_diff:
        for _geo_name, _geo_val in (
            ("forcing.cos_zenith", forcing.cos_zenith),
            ("lat", lat),
            ("lon", lon),
        ):
            if _geo_val is not None and isinstance(_geo_val, jax.core.Tracer):
                raise ValueError(
                    f"CLM-ML diff mode: {_geo_name} is a jax tracer, but solar "
                    "geometry (lat/lon/doy/cos_zenith) is a NON-differentiated static "
                    "input (host-side CLM orbital setup). Differentiate only the "
                    "physical forcing leaves and keep geometry concrete (close over "
                    "it, or jax.lax.stop_gradient it before the grad boundary)."
                )

    # ---- Build soil grid data ----
    grid = make_soil_grid(land_config.soil_grid)
    dz_soil = np.array(grid.dz, dtype=np.float64)     # (n_layers,)
    z_soil = np.array(grid.z_node, dtype=np.float64)   # (n_layers,)
    n_layers = len(dz_soil)

    # ---- Latitude / longitude arrays ----
    if lat is not None:
        lat_deg = np.array(lat, dtype=np.float64)
    else:
        lat_deg = np.zeros(ncol, dtype=np.float64)

    # ---- CLM global state setup ----
    # _setup_clm_time must precede _compute_virtual_lon_deg because the latter
    # calls shr_orb_decl with orbital params set by _ensure_clm_initialized and
    # needs caldaym1 derived from the itim we are about to write.
    z_ref = float(land_config.z_ref)
    _setup_clm_time(dt, doy, 0)
    # caldaym1 matches what _MLCanopyForcing computes internally:
    #   get_curr_calday(offset=-int(dtime_clm)) = 1 + (itim-1) * dt / 86400
    # This is distinct from doy%1 by exactly one CLM step (dt/86400 days).
    _itim = max(1, round(doy * 86400.0 / max(dt, 1.0)))
    _caldaym1 = 1.0 + (_itim - 1) * dt / 86400.0

    if lon is not None:
        lon_deg = np.array(lon, dtype=np.float64)
        # Use CLM's own Kepler shr_orb_cosz at caldaym1 — same declination and
        # phase as CLM's internal solar_zen_forcing, so SW partitioning is
        # consistent with beam extinction kb=0.5/coszen in MLCanopyFluxes.
        from clm_share.shr_orb_mod import shr_orb_cosz as _clm_cosz, shr_orb_decl as _clm_decl
        import clm_src_utils.clm_varorb as _varorb_loc
        _declinm1, _ = _clm_decl(_caldaym1, _varorb_loc.eccen, _varorb_loc.mvelpp,
                                  _varorb_loc.lambm0, _varorb_loc.obliqr)
        _pi = np.pi
        cos_zen = np.maximum(np.array([
            float(_clm_cosz(_caldaym1,
                            float(lat_deg[i]) * _pi / 180.0,
                            float(lon_deg[i]) * _pi / 180.0,
                            float(_declinm1)))
            for i in range(ncol)
        ], dtype=np.float64), 0.0)
    else:
        # Use coupler-provided cos_zenith directly — avoids 100× error in beam
        # extinction (kb=0.5/coszen) that occurs when lon defaults to 0° (Greenwich).
        # Then invert the CLM shr_orb_cosz formula (using CLM's own Kepler declination
        # and caldaym1) so that CLM's internal solar_zen_forcing[p] reproduces the
        # forcing.  Both arccos branches give the same cosz, so the virtual longitude
        # is numerically correct even if not geographically meaningful.
        cos_zen = np.maximum(np.array(forcing.cos_zenith, dtype=np.float64), 0.0)
        lon_deg = _compute_virtual_lon_deg(cos_zen, lat_deg, _caldaym1)

    _topo_key = (ncol, float(lat_deg[0]) if len(lat_deg) > 0 else 0.0,
                 float(lon_deg[0]) if len(lon_deg) > 0 else 0.0)
    global _last_topology_key
    if _topo_key != _last_topology_key:
        _setup_clm_topology(ncol, lat_deg, lon_deg, dz_soil, z_soil, z_ref,
                            pft_clm=int(canopy_config.pft_clm))
        _last_topology_key = _topo_key

    # ---- Propagate 10-day running mean temperature for Vcmax acclimation ----
    # MLCanopyFluxes copies t_a10_patch into tacclim_forcing on output — reading
    # it back would apply the filter twice per step (effective ~20d, not 10d).
    # Instead we carry the running mean explicitly in CanopyState.t_a10_arr and
    # update it here:
    #   T_a10_new = (1 - alpha) * T_a10_old + alpha * T_lowest
    #   alpha = dt / (10 * 86400)  (10-day e-folding, CLM default)
    alpha = min(dt / (10.0 * 86400.0), 1.0)
    if _diff_mode:
        # Traced running mean: t_a10 is the Vcmax temperature-acclimation state,
        # a real function of T_lowest, so keep it on the jax.grad tape (a host
        # np.array(forcing.T_lowest) would raise on a tracer — scope item A).
        T_low = jnp.asarray(forcing.T_lowest, dtype=jnp.float64)
        if canopy_state is not None and canopy_state.t_a10_arr is not None:
            t_a10_prev = jnp.asarray(canopy_state.t_a10_arr, dtype=jnp.float64)
            t_a10_now = (1.0 - alpha) * t_a10_prev + alpha * T_low
        else:
            t_a10_now = T_low
        t_a10_prior = t_a10_now
    else:
        T_lowest_np = np.array(forcing.T_lowest, dtype=np.float64)
        if canopy_state is not None and canopy_state.t_a10_arr is not None:
            t_a10_prev = np.array(canopy_state.t_a10_arr, dtype=np.float64)
            t_a10_now = ((1.0 - alpha) * t_a10_prev + alpha * T_lowest_np).astype(np.float64)
        else:
            # Cold start: initialize to instantaneous T (will converge in ~10 days)
            t_a10_now = T_lowest_np.copy()
        t_a10_prior = jnp.array(t_a10_now)

    # ---- Build stub CLM instances ----
    soil_hyd = land_config.hydraulics
    stubs = _build_stubs(
        ncol, forcing, canopy_config, land_config, land_params,
        T_soil_top, psi_soil, theta_soil, dz_soil, soil_hyd,
        cos_zen=cos_zen,
        T_soil_all=T_soil,
        t_a10_prior=t_a10_prior,
        lai_override=lai_override,
    )

    # ---- Allocate / retrieve mlcanopy_type ----
    if canopy_state is None or canopy_state.mlcanopy is None:
        mlcanopy = _init_mlcanopy(ncol, stubs, canopy_config)
    else:
        mlcanopy = canopy_state.mlcanopy

    # ---- Decomposition bounds ----
    bounds = bounds_type(begg=1, endg=ncol, begl=1, endl=ncol,
                         begc=1, endc=ncol, begp=1, endp=ncol)

    # ---- filter_exposedvegp: all columns (1-based) ----
    filter_exposedvegp = list(range(1, ncol + 1))
    num_exposedvegp = ncol

    # ---- Set MLclm_varctl global settings ----
    import multilayer_canopy.MLclm_varctl as _ml_ctl
    _ml_ctl.runge_kutta_type = canopy_config.runge_kutta_type
    _ml_ctl.met_type = canopy_config.met_type
    # dtime_ml: sub-step length. Must divide dt evenly.
    _ml_ctl.dtime_ml = dt / max(1, canopy_config.num_ml_steps)
    _ml_ctl.mlcan_to_clm = 0  # we read output directly from mlcanopy_type
    # DIFFERENTIABLE_MODE is a static intent flag (currently vestigial upstream —
    # every physics module switches on the ``grid`` argument, not this global —
    # but set it to match the mode so any future read is consistent).
    _ml_ctl.DIFFERENTIABLE_MODE = bool(_diff_mode)

    # ---- Build GridInfo for the differentiable path ----
    # Structural ints must be concrete Python ints extracted BEFORE jax.grad
    # tracing (int() on a tracer raises ConcretizationTypeError).  The warm-start
    # template ``mlcanopy`` is a captured constant under jax.grad, so these reads
    # are concrete.  Mirrors make_clm_ml_forward (MLCanopyFluxesMod.py:2098).
    if _diff_mode:
        from multilayer_canopy.MLclm_varctl import GridInfo
        # Capability guard: the differentiable path is only CORRECT with a
        # clm-ml-jax build whose ``_CanopyFluxesDiagnostics`` runs in diff mode
        # (``grid=`` parameter).  Older builds return from ``MLCanopyFluxes``
        # BEFORE diagnostics in diff mode, leaving the canopy-integrated outputs
        # (shflx/lhflx/gpp/rnet/…) stale — a silent wrong-answer.  clm-ml-jax is
        # not on PyPI (local install), so we cannot pin a version; probe the
        # capability directly and fail LOUDLY instead.
        import inspect as _inspect
        from multilayer_canopy import MLCanopyFluxesMod as _mlmod
        if "grid" not in _inspect.signature(_mlmod._CanopyFluxesDiagnostics).parameters:
            raise RuntimeError(
                "CLMMLCanopyConfig.differentiable=True requires a clm-ml-jax build "
                "whose _CanopyFluxesDiagnostics accepts grid= (runs canopy-flux "
                "diagnostics on the jax.grad tape). The installed clm-ml-jax returns "
                "before diagnostics in differentiable mode, which would leave shflx/"
                "lhflx/gpp/rnet stale. Update clm-ml-jax to a revision including the "
                "differentiable-diagnostics fix (adds grid= to _CanopyFluxesDiagnostics)."
            )
        _p = int(filter_exposedvegp[0])
        # Structural ints (ncan/ntop/nbot) must be CONCRETE Python ints — the diff
        # path reads them at trace time.  Two sources:
        #  (1) caller-supplied ``grid_info`` (REQUIRED for a multi-step
        #      differentiated rollout: there the carried ``canopy_state.mlcanopy``
        #      is itself a tracer, so reading ints off it would raise); or
        #  (2) the warm template ``mlcanopy`` when it is concrete (single warm
        #      step whose state is a captured constant under jax.grad).
        # ``dpai_profile.shape`` is static, so the range check works either way.
        _ncan_max = int(mlcanopy.dpai_profile.shape[1])
        if grid_info is not None:
            _ncan_p = int(grid_info.ncan)
            _ntop_p = int(grid_info.ntop)
            _nbot_p = int(grid_info.nbot)
        else:
            try:
                _ncan_p = int(mlcanopy.ncan_canopy[_p])
                _ntop_p = int(mlcanopy.ntop_canopy[_p])
                _nbot_p = int(mlcanopy.nbot_canopy[_p])
            except jax.errors.ConcretizationTypeError as exc:
                raise RuntimeError(
                    "CLM-ML diff mode: the canopy_state.mlcanopy structural ints "
                    "(ncan/ntop/nbot) are TRACED — this happens when a differentiated "
                    "loss unrolls MULTIPLE canopy steps and carries the returned state "
                    "as the jax.grad tape's carry. Extract the concrete structural "
                    "ints once from the warm-start state and thread them through every "
                    "step via grid_info=extract_clm_ml_grid_info(state0)."
                ) from exc
        if not (1 <= _ncan_p <= _ncan_max):
            raise ValueError(
                "CLM-ML diff mode needs a warm-started canopy_state whose vertical "
                f"structure is initialised; got ncan={_ncan_p} (valid 1..{_ncan_max}). "
                "Run one forward (differentiable=False or cold-start) step first."
            )
        grid = GridInfo(p=_p, ncan=_ncan_p, ntop=_ntop_p, nbot=_nbot_p)
    else:
        grid = None

    # ---- Call MLCanopyFluxes ----
    # Only forward the diff-mode / trainable-param kwargs when they are actually
    # used, so the DEFAULT forward-only path keeps the exact call signature the
    # interface has always used (``_o2ref_py`` only) and does NOT depend on the
    # newer ``grid=``/``vcmaxpft_jax=``/``g1_MED_jax=`` upstream API surface
    # (Codex P1: an older clm-ml-jax would otherwise TypeError on every call,
    # including production forward-only runs).  Diff mode is already gated by the
    # capability guard above.
    _opt_kwargs: dict[str, Any] = {}
    if grid is not None:
        _opt_kwargs["grid"] = grid
    if vcmaxpft_jax is not None:
        _opt_kwargs["vcmaxpft_jax"] = vcmaxpft_jax
    if g1_medlyn_jax is not None:
        _opt_kwargs["g1_MED_jax"] = g1_medlyn_jax
    mlcanopy_new = MLCanopyFluxes(
        bounds=bounds,
        num_exposedvegp=num_exposedvegp,
        filter_exposedvegp=filter_exposedvegp,
        atm2lnd_inst=stubs["atm2lnd"],
        canopystate_inst=stubs["canopystate"],
        soilstate_inst=stubs["soilstate"],
        temperature_inst=stubs["temperature"],
        waterstatebulk_inst=stubs["waterstatebulk"],
        waterfluxbulk_inst=stubs["waterfluxbulk"],
        energyflux_inst=stubs["energyflux"],
        frictionvel_inst=stubs["frictionvel"],
        surfalb_inst=stubs["surfalb"],
        solarabs_inst=stubs["solarabs"],
        mlcanopy_inst=mlcanopy,
        wateratm2lndbulk_inst=stubs["wateratm2lndbulk"],
        waterdiagnosticbulk_inst=stubs["waterdiagnosticbulk"],
        _o2ref_py=float(canopy_config.o2ref),
        **_opt_kwargs,
    )

    # ---- Extract SurfaceFluxOutput ----
    surface_out = _extract_surface_fluxes(
        mlcanopy_new, ncol, forcing, land_config, land_params,
        canopy_config=canopy_config)

    return surface_out, CanopyState(mlcanopy=mlcanopy_new, t_a10_arr=t_a10_now)
