"""
JAX/Python translation of the CLM multilayer canopy soil temperature module.

Simplified from CLM5 for standalone (uncoupled) MLCanopy runs.
Not called when CLMml is actively coupled to CLM — in that case CLM
provides soil temperature via SoilTemperatureMod.

Original Fortran module: MLSoilTemperatureMod
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array

from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.SoilStateType import soilstate_type
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.TemperatureType import temperature_type
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.WaterDiagnosticBulkType import waterdiagnosticbulk_type
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.WaterStateBulkType import waterstatebulk_type
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.WaterType import water_type
from legoesm.land.canopy.clm_ml_backend.clm_src_main.abortutils import endrun
from legoesm.land.canopy.clm_ml_backend.clm_src_main.ColumnType import col
from legoesm.land.canopy.clm_ml_backend.clm_src_main.decompMod import bounds_type
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyFluxesType import mlcanopy_type
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLMathToolsMod import tridiag

# Threshold for thin surface layer — Fortran line 33
_thin_sfclayer: float = 1.0e-6


# ---------------------------------------------------------------------------
# SoilTemperature
# ---------------------------------------------------------------------------


def SoilTemperature(
    bounds: bounds_type,
    num_nolakec: int,
    filter_nolakec: np.ndarray,
    soilstate_inst: soilstate_type,
    temperature_inst: temperature_type,
    waterdiagnosticbulk_inst: waterdiagnosticbulk_type,
    waterstatebulk_inst: waterstatebulk_type,
    water_inst: water_type,
    mlcanopy_inst: mlcanopy_type,
) -> tuple:
    """
    Compute soil temperature via implicit tridiagonal solver.

    Mirrors Fortran subroutine ``SoilTemperature`` (lines 39-151).

    Standalone-only: not called when CLMml is coupled to CLM (which
    provides soil temperature).  In standalone mode p == c so the
    ``pfilter`` mapping is the identity.

    **Algorithm** (Fortran lines 96-151):

    1. Build ``pfilter`` (p→c identity map for standalone).
    2. Save ``tssbef = t_soisno`` for energy conservation check.
    3. Compute thermal conductivity and heat capacity via
       :func:`SoilThermProp`.
    4. Assemble tridiagonal system:

       - Layer 1 (top): ``gsoi`` as upper boundary condition, zero
         flux below.
       - Layers 2 … nlevgrnd-1: interior implicit diffusion.
       - Layer nlevgrnd (bottom): zero heat flux lower boundary.

    5. Solve with :func:`MLMathToolsMod.tridiag`.
    6. Energy conservation check: ``|gsoi - edif| < 1e-6 W/m²``.

    Args:
        bounds:                   Index bounds.
        num_nolakec:              Number of non-lake column filter points.
        filter_nolakec:           Column filter indices (1-based, length
                                  ≥ num_nolakec).
        soilstate_inst:           Soil state (in/out).
        temperature_inst:         Temperature state (in/out).
        waterdiagnosticbulk_inst: Diagnostic water bulk state (in/out).
        waterstatebulk_inst:      Water state bulk (in/out).
        water_inst:               Water instance (in/out).
        mlcanopy_inst:            Multilayer canopy instance (in/out).

    Returns:
        Tuple ``(temperature_inst, soilstate_inst)`` with updated
        ``t_soisno_col`` and thermal properties.
    """
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import nlevgrnd
    from legoesm.land.canopy.clm_ml_backend.clm_src_utils.clm_time_manager import get_step_size

    endc = bounds.endc
    nc = endc + 1
    # State arrays (t_soisno_col, h2osoi_liq_col, z, etc.) use direct soil-layer
    # indexing j = 1..nlevgrnd (same as SoilInit).  No nlevsno offset needed.
    nth = nlevgrnd + 1  # local tk/cv arrays: index 1..nlevgrnd

    # ------------------------------------------------------------------
    # Aliases — mirror Fortran associate block (lines 76-80)
    # ------------------------------------------------------------------
    z = col.z  # (nc, nlevsno+nlevgrnd) using direct index
    t_soisno = temperature_inst.t_soisno_col  # JAX array — updated immutably with .at[].set()
    gsoi = mlcanopy_inst.gsoi_soil  # (nc,)

    # ------------------------------------------------------------------
    # pfilter: p→c identity map — Fortran lines 90-93
    # Integer indices only; not part of the differentiable computation.
    # ------------------------------------------------------------------
    pfilter = np.zeros(nc, dtype=np.int32)
    for fc in range(1, num_nolakec + 1):
        c = int(filter_nolakec[fc - 1])
        pfilter[c] = c

    # ------------------------------------------------------------------
    # CLM timestep — Fortran line 96
    # ------------------------------------------------------------------
    dtime = get_step_size()

    # ------------------------------------------------------------------
    # Save soil temperature for energy conservation — Fortran lines 99-104
    # tssbef(c,j) = t_soisno(c,j) for j in 1:nlevgrnd
    # Only used in the energy-conservation error check (concrete values OK).
    # ------------------------------------------------------------------
    tssbef = np.zeros((nc, nlevgrnd + 1), dtype=np.float64)
    for j in range(1, nlevgrnd + 1):
        for fc in range(1, num_nolakec + 1):
            c = int(filter_nolakec[fc - 1])
            tssbef[c, j] = float(t_soisno[c, j])

    # ------------------------------------------------------------------
    # Thermal conductivity and heat capacity — Fortran lines 107-110
    # ------------------------------------------------------------------
    tk = jnp.zeros((nc, nth), dtype=jnp.float64)
    cv = jnp.zeros((nc, nth), dtype=jnp.float64)
    tk_h2osfc = jnp.zeros(nc, dtype=jnp.float64)

    soilstate_inst, temperature_inst, tk, cv, tk_h2osfc = SoilThermProp(
        bounds,
        num_nolakec,
        filter_nolakec,
        tk,
        cv,
        tk_h2osfc,
        temperature_inst,
        waterdiagnosticbulk_inst,
        waterstatebulk_inst,
        water_inst,
        soilstate_inst,
    )

    # ------------------------------------------------------------------
    # Tridiagonal matrix assembly — Fortran lines 113-143
    # Arrays indexed 1:nlevgrnd (index 0 unused)
    # Use direct soil-layer index j (matching SoilInit convention)
    # ------------------------------------------------------------------
    atri = jnp.zeros((nc, nlevgrnd + 1), dtype=jnp.float64)
    btri = jnp.zeros((nc, nlevgrnd + 1), dtype=jnp.float64)
    ctri = jnp.zeros((nc, nlevgrnd + 1), dtype=jnp.float64)
    rtri = jnp.zeros((nc, nlevgrnd + 1), dtype=jnp.float64)

    for j in range(1, nlevgrnd + 1):
        for fc in range(1, num_nolakec + 1):
            c = int(filter_nolakec[fc - 1])

            fact = dtime / cv[c, j]

            if j == 1:
                # Top layer: gsoi upper boundary — Fortran lines 118-123
                dzp = z[c, j + 1] - z[c, j]
                atri = atri.at[c, j].set(0.0)
                btri = btri.at[c, j].set(1.0 + fact * tk[c, j] / dzp)
                ctri = ctri.at[c, j].set(-fact * tk[c, j] / dzp)
                rtri = rtri.at[c, j].set(t_soisno[c, j] + fact * gsoi[pfilter[c]])

            elif j <= nlevgrnd - 1:
                # Interior layers — Fortran lines 125-132
                dzm = z[c, j] - z[c, j - 1]
                dzp = z[c, j + 1] - z[c, j]
                atri = atri.at[c, j].set(-fact * tk[c, j - 1] / dzm)
                btri = btri.at[c, j].set(1.0 + fact * (tk[c, j - 1] / dzm + tk[c, j] / dzp))
                ctri = ctri.at[c, j].set(-fact * tk[c, j] / dzp)
                rtri = rtri.at[c, j].set(t_soisno[c, j])

            elif j == nlevgrnd:
                # Bottom layer: zero heat flux — Fortran lines 134-140
                dzm = z[c, j] - z[c, j - 1]
                atri = atri.at[c, j].set(-fact * tk[c, j - 1] / dzm)
                btri = btri.at[c, j].set(1.0 + fact * tk[c, j - 1] / dzm)
                ctri = ctri.at[c, j].set(0.0)
                rtri = rtri.at[c, j].set(t_soisno[c, j])

    # ------------------------------------------------------------------
    # Solve tridiagonal system — Fortran lines 145-149
    # tridiag() expects 1-based arrays of size (n+1,): index 0 unused,
    # data at indices 1..n.  Pass full slices [0:nlevgrnd+1].
    # ------------------------------------------------------------------
    for fc in range(1, num_nolakec + 1):
        c = int(filter_nolakec[fc - 1])
        u_sol = tridiag(
            atri[c, 0 : nlevgrnd + 1],
            btri[c, 0 : nlevgrnd + 1],
            ctri[c, 0 : nlevgrnd + 1],
            rtri[c, 0 : nlevgrnd + 1],
            nlevgrnd,
        )
        # u_sol has shape (nlevgrnd+1,) with data at indices 1..nlevgrnd
        for j in range(1, nlevgrnd + 1):
            t_soisno = t_soisno.at[c, j].set(u_sol[j])

    # ------------------------------------------------------------------
    # Energy conservation check — Fortran lines 151-159
    # |gsoi - edif| < 1e-6 W/m2
    # Concrete float comparisons are acceptable for this error-check path.
    # ------------------------------------------------------------------
    for fc in range(1, num_nolakec + 1):
        c = int(filter_nolakec[fc - 1])
        edif = 0.0
        for j in range(1, nlevgrnd + 1):
            edif += float(cv[c, j]) * (float(t_soisno[c, j]) - tssbef[c, j]) / dtime
        if abs(float(gsoi[pfilter[c]]) - edif) >= 1.0e-6:
            endrun(msg="ERROR: MLSoilTemperatureMod: soil temperature energy conservation error")

    # Write updated t_soisno back to temperature_inst
    temperature_inst = temperature_inst._replace(t_soisno_col=t_soisno)

    return temperature_inst, soilstate_inst


# ---------------------------------------------------------------------------
# SoilThermProp
# ---------------------------------------------------------------------------


def SoilThermProp(
    bounds: bounds_type,
    num_nolakec: int,
    filter_nolakec: np.ndarray,
    tk: Array,
    cv: Array,
    tk_h2osfc: Array,
    temperature_inst: temperature_type,
    waterdiagnosticbulk_inst: waterdiagnosticbulk_type,
    waterstatebulk_inst: waterstatebulk_type,
    water_inst: water_type,
    soilstate_inst: soilstate_type,
) -> tuple:
    """
    Compute thermal conductivities and heat capacities of snow/soil layers.

    Mirrors Fortran subroutine ``SoilThermProp`` (lines 153-282),
    adapted from CLM5.

    **Soil thermal conductivity** (Farouki 1981, Fortran lines 209-234):

    For each soil layer j in 1:nlevgrnd:

    .. code-block:: none

        satw = min(1, (liq/denh2o + ice/denice) / (dz*watsat))
        if satw > 1e-7:
            fl    = liq_vol / (liq_vol + ice_vol)
            dksat = tkmg * tkwat^(fl*watsat) * tkice^((1-fl)*watsat)
            if T >= tfrz: dke = max(0, log10(satw) + 1)
            else:         dke = satw
            thk = dke*dksat + (1-dke)*tkdry
        else:
            thk = tkdry
        if j > nbedrock: thk = thk_bedrock

    **Snow thermal conductivity** (Jordan 1991, Fortran lines 236-241):

    .. code-block:: none

        bw   = (ice + liq) / (frac_sno * dz)
        thk  = tkair + (7.75e-5*bw + 1.105e-6*bw²) * (tkice - tkair)

    **Interface conductivity** (Fortran lines 244-256):

    .. code-block:: none

        tk(j) = thk(j)*thk(j+1)*(z(j+1)-z(j))
                / (thk(j)*(z(j+1)-zi(j)) + thk(j+1)*(zi(j)-z(j)))
        tk(nlevgrnd) = 0   (zero-flux bottom)

    **h2osfc conductivity** (Fortran lines 258-264):

    .. code-block:: none

        zh2osfc    = 1e-3 * 0.5 * h2osfc
        tk_h2osfc  = tkwat * thk(1) * (z(1) + zh2osfc)
                     / (tkwat * z(1) + thk(1) * zh2osfc)

    **Soil heat capacity** (de Vries 1963, Fortran lines 267-280):

    .. code-block:: none

        cv(j) = csol*(1-watsat)*dz + ice*cpice + liq*cpliq
        if j > nbedrock: cv(j) = csol_bedrock * dz
        if j==1 and no snow layers and h2osno>0: cv(1) += cpice*h2osno

    **Snow heat capacity** (Fortran lines 282-292):

    .. code-block:: none

        cv(j) = max(thin_sfclayer,
                    (cpliq*liq(j) + cpice*ice(j)) / frac_sno)

    Args:
        bounds:                   Index bounds.
        num_nolakec:              Number of non-lake column filter points.
        filter_nolakec:           Column filter indices (1-based).
        tk:                       Output thermal conductivity at layer
                                  interface (W/m/K), shape
                                  ``(nc, nlevsno+nlevgrnd+1)``.
        cv:                       Output heat capacity (J/m²/K), same shape.
        tk_h2osfc:                Output h2osfc thermal conductivity
                                  (W/m/K), shape ``(nc,)``.
        temperature_inst:         Temperature state.
        waterdiagnosticbulk_inst: Diagnostic water bulk state.
        waterstatebulk_inst:      Water state bulk.
        water_inst:               Water instance.
        soilstate_inst:           Soil state (``thk_col`` updated in place
                                  as a new NamedTuple).

    Returns:
        Tuple ``(soilstate_inst, temperature_inst)`` with updated
        ``thk_col`` inside ``soilstate_inst``.
    """
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import (
        cpice,
        cpliq,
        csol_bedrock,
        denh2o,
        denice,
        tfrz,
        thk_bedrock,
        tkice,
        tkwat,
    )
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import nlevgrnd

    # ------------------------------------------------------------------
    # Aliases — mirror Fortran associate block (lines 175-203)
    # ------------------------------------------------------------------
    # Re-read col from the ColumnType module so we always get the version
    # updated by initVertical (the module-level import binding is stale).
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.ColumnType import col as _fresh_col

    nbedrock = _fresh_col.nbedrock
    snl = _fresh_col.snl
    dz = _fresh_col.dz
    zi = _fresh_col.zi
    z = _fresh_col.z
    t_soisno = temperature_inst.t_soisno_col
    h2osfc = waterstatebulk_inst.h2osfc_col
    h2osno = water_inst.h2osno_col
    h2osoi_liq = waterstatebulk_inst.h2osoi_liq_col
    h2osoi_ice = waterstatebulk_inst.h2osoi_ice_col
    tkmg = soilstate_inst.tkmg_col
    tkdry = soilstate_inst.tkdry_col
    csol = soilstate_inst.csol_col
    watsat = soilstate_inst.watsat_col

    # Work arrays — JAX, updated via .at[c, j].set(...)
    # thk: per-layer thermal conductivity (W/m/K)
    # tk, cv, tk_h2osfc: passed in as zero-filled JAX arrays, filled here
    thk = soilstate_inst.thk_col  # start from existing JAX array

    # ------------------------------------------------------------------
    # Indexing note:
    #   All state arrays (h2osoi_liq, h2osoi_ice, t_soisno, dz, z, zi)
    #   are populated by SoilInit / initVertical at DIRECT indices j=1..nlevgrnd
    #   (no nlevsno offset).  thk_col is allocated with full nlevsno+nlevgrnd+1
    #   size but we write to it at j=1..nlevgrnd directly as well.
    #   tk and cv are local (nc, nlevgrnd+1) arrays used by SoilTemperature
    #   which also uses direct j indexing.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Soil thermal conductivity — Fortran lines 208-234
    # Loop j = 1 : nlevgrnd (soil only — no snow in standalone mode)
    # ------------------------------------------------------------------
    for j in range(1, nlevgrnd + 1):
        for fc in range(1, num_nolakec + 1):
            c = int(filter_nolakec[fc - 1])

            # h2osoi_liq/ice are stored at [c, j-1] (clmDataMod convention: first
            # soil layer at index 0). Using [c, j] reads the next layer's moisture.
            liq_j = h2osoi_liq[c, j - 1]
            ice_j = h2osoi_ice[c, j - 1]
            liq_vol = liq_j / denh2o
            ice_vol = ice_j / denice
            dz_j = dz[c, j]
            ws_j = watsat[c, j]  # watsat indexed 1:nlevgrnd

            # Safe denominator for satw (ws_j could be 0 for bedrock)
            ws_j_safe = jnp.where(ws_j > 0.0, ws_j, 1.0)
            satw_raw = (liq_vol + ice_vol) / (dz_j * ws_j_safe)
            satw = jnp.minimum(1.0, satw_raw)

            # Wet-path (satw > 1e-7): compute fl, dksat, dke, thk_wet
            # Safe fl denominator: liq_vol/dz + ice_vol/dz could be zero
            fl_num = liq_vol / dz_j
            fl_denom = liq_vol / dz_j + ice_vol / dz_j
            fl_denom_safe = jnp.where(fl_denom > 0.0, fl_denom, 1.0)
            fl = fl_num / fl_denom_safe

            dksat = tkmg[c, j] * tkwat ** (fl * ws_j) * tkice ** ((1.0 - fl) * ws_j)

            # dke depends on temperature: log10(satw)+1 (thawed) or satw (frozen)
            # Use satw_safe to avoid log10(0) in the thawed branch
            satw_safe = jnp.where(satw > 0.0, satw, 1.0e-10)
            dke_thawed = jnp.maximum(0.0, jnp.log10(satw_safe) + 1.0)
            dke_frozen = satw
            dke = jnp.where(t_soisno[c, j] >= tfrz, dke_thawed, dke_frozen)

            thk_wet = dke * dksat + (1.0 - dke) * tkdry[c, j]
            thk_dry = tkdry[c, j]

            # Select wet or dry path based on satw
            thk_val = jnp.where(satw > 1.0e-7, thk_wet, thk_dry)

            # Bedrock override (static Python comparison — j and nbedrock[c] are concrete)
            if j > int(nbedrock[c]):
                thk_val = jnp.asarray(thk_bedrock)

            thk = thk.at[c, j].set(thk_val)

    # Snow thermal conductivity — skipped in standalone mode (snl=0)

    # ------------------------------------------------------------------
    # Interface thermal conductivity — Fortran lines 244-256
    # tk(c,j) for j = 1 : nlevgrnd-1 (soil only), then tk(c,nlevgrnd)=0
    # tk uses direct j indexing (matches SoilTemperature)
    # ------------------------------------------------------------------
    for fc in range(1, num_nolakec + 1):
        c = int(filter_nolakec[fc - 1])

        for j in range(1, nlevgrnd):  # j = 1 .. nlevgrnd-1
            thk_j = thk[c, j]
            thk_jp1 = thk[c, j + 1]
            z_j = z[c, j]
            z_jp1 = z[c, j + 1]
            zi_j = zi[c, j]
            # Safe denominator for interface conductivity formula
            denom = thk_j * (z_jp1 - zi_j) + thk_jp1 * (zi_j - z_j)
            denom_safe = jnp.where(jnp.abs(denom) > 0.0, denom, 1.0e-10)
            tk_val = thk_j * thk_jp1 * (z_jp1 - z_j) / denom_safe
            tk = tk.at[c, j].set(tk_val)

        tk = tk.at[c, nlevgrnd].set(0.0)

    # ------------------------------------------------------------------
    # h2osfc thermal conductivity — Fortran lines 258-264
    # z[c, 1] and thk[c, 1] use direct index 1 (top soil layer)
    # ------------------------------------------------------------------
    for fc in range(1, num_nolakec + 1):
        c = int(filter_nolakec[fc - 1])
        zh2osfc = 1.0e-3 * (0.5 * h2osfc[c])
        z1 = z[c, 1]
        thk1 = thk[c, 1]
        denom = tkwat * z1 + thk1 * zh2osfc
        # Avoid division by zero; fall back to tkwat when denom≈0
        denom_safe = jnp.where(jnp.abs(denom) > 0.0, denom, 1.0)
        tk_h2osfc_val = jnp.where(
            jnp.abs(denom) > 0.0,
            tkwat * thk1 * (z1 + zh2osfc) / denom_safe,
            jnp.asarray(tkwat),
        )
        tk_h2osfc = tk_h2osfc.at[c].set(tk_h2osfc_val)

    # ------------------------------------------------------------------
    # Soil heat capacity (de Vries 1963) — Fortran lines 267-280
    # cv uses direct j indexing (matches SoilTemperature)
    # ------------------------------------------------------------------
    for j in range(1, nlevgrnd + 1):
        for fc in range(1, num_nolakec + 1):
            c = int(filter_nolakec[fc - 1])
            # h2osoi_liq/ice stored at [c, j-1] (clmDataMod convention)
            cv_val = (
                csol[c, j] * (1.0 - watsat[c, j]) * dz[c, j]
                + h2osoi_ice[c, j - 1] * cpice
                + h2osoi_liq[c, j - 1] * cpliq
            )

            # Bedrock override (static Python comparison)
            if j > int(nbedrock[c]):
                cv_val = jnp.asarray(csol_bedrock) * dz[c, j]

            # Top layer: add snow heat capacity if no active snow layers
            if j == 1:
                snl_c = int(snl[c])
                if snl_c + 1 == 1:
                    cv_val = cv_val + jnp.where(
                        h2osno[c] > 0.0,
                        cpice * h2osno[c],
                        jnp.zeros(()),
                    )

            cv = cv.at[c, j].set(cv_val)

    # Snow heat capacity — skipped in standalone mode (snl=0)

    # Write thk back to soilstate_inst NamedTuple (thk_col has full size,
    # but only j=1..nlevgrnd elements were updated).
    soilstate_inst = soilstate_inst._replace(thk_col=thk)
    # bw unchanged (no snow in standalone) — leave waterdiagnosticbulk_inst as-is

    return (soilstate_inst, temperature_inst, tk, cv, tk_h2osfc)
