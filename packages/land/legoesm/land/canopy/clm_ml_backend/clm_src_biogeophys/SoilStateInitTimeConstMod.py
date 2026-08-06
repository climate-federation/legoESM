"""
JAX translation of SoilStateInitTimeConstMod Fortran module.

Sets hydraulic and thermal properties for soil layers, including
root fraction profiles, organic matter adjustments, and mineral/organic
mixing rules for porosity, suction, conductivity, and heat capacity.

Original Fortran module: SoilStateInitTimeConstMod
Fortran lines 1-230
"""

from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.SoilStateType import soilstate_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.abortutils import endrun  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import csol_bedrock  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varctl import iulog  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import nlevgrnd, nlevsoi  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.decompMod import bounds_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.PatchType import patch  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.offline_driver.SoilTexMod import (
    bsw_tex,
    clay_tex,
    hksat_tex,
    ntex,
    sand_tex,
    smpsat_tex,
    soil_tex,
    watsat_tex,
)  # noqa: F401

# ---------------------------------------------------------------------------
# Module-level constants (Fortran lines 58-65)
# ---------------------------------------------------------------------------

organic_max: float = 130.0  # Organic matter content where soil acts like peat (kg/m3)
zsapric: float = 0.5  # Depth (m) that organic matter takes on characteristics of sapric peat
pcalpha: float = 0.5  # Percolation threshold
pcbeta: float = 0.139  # Percolation exponent
m_to_cm: float = 1.0e2  # Conversion factor m -> cm


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def SoilStateInitTimeConst(
    bounds: bounds_type,
    soilstate_inst: soilstate_type,
) -> soilstate_type:
    """
    Initialize time-constant soil hydraulic and thermal properties.

    Mirrors Fortran subroutine ``SoilStateInitTimeConst`` (lines 32-225).

    Computes for each column and soil layer:

    - Root fraction profiles (Jackson 1996 beta method), adjusted for
      bedrock depth.
    - Percent sand, clay, and organic matter per layer.
    - Hydraulic properties (porosity, suction, conductivity, Clapp-
      Hornberger b) for mineral soil, then adjusted for organic matter
      using percolation-theory mixing rules.
    - Thermal properties (dry conductivity, mineral conductivity,
      heat capacity) for mineral/organic mixtures, with bedrock values
      for layers below ``nlevsoi``.

    Args:
        bounds: Decomposition bounds for the local MPI task, providing
            ``begp``, ``endp``, ``begc``, and ``endc``.
        soilstate_inst: Soil state container to be populated. All
            output fields are written and returned in a new instance.

    Returns:
        Updated :class:`soilstate_type` with all time-constant soil
        properties initialised.
    """
    # IMPORTANT: several modules rebind module-level singletons during initialization
    # (e.g., ColumnType.col and pftconMod.pftcon). Avoid stale direct imports by
    # reading these from their modules at call time.
    import legoesm.land.canopy.clm_ml_backend.clm_src_main.ColumnType as ColumnType
    import legoesm.land.canopy.clm_ml_backend.clm_src_main.pftconMod as pftconMod
    import legoesm.land.canopy.clm_ml_backend.offline_driver.TowerDataMod as TowerDataMod

    col = ColumnType.col
    pftcon = pftconMod.pftcon

    tower_num = TowerDataMod.tower_num
    tower_tex = TowerDataMod.tower_tex
    tower_clay = TowerDataMod.tower_clay
    tower_sand = TowerDataMod.tower_sand
    tower_organic = TowerDataMod.tower_organic

    # ------------------------------------------------------------------
    # Unpack output arrays (Fortran associate block, lines 70-86)
    # ------------------------------------------------------------------
    rootfr = soilstate_inst.rootfr_patch  # Fraction of roots in each soil layer
    cellsand = soilstate_inst.cellsand_col  # Soil layer percent sand
    cellclay = soilstate_inst.cellclay_col  # Soil layer percent clay
    cellorg = soilstate_inst.cellorg_col  # Soil layer organic matter (kg/m3)
    watsat = soilstate_inst.watsat_col  # Volumetric water content at saturation (porosity)
    sucsat = soilstate_inst.sucsat_col  # Suction at saturation (mm)
    hksat = soilstate_inst.hksat_col  # Hydraulic conductivity at saturation (mm H2O/s)
    bsw = soilstate_inst.bsw_col  # Clapp and Hornberger "b" parameter
    tkmg = soilstate_inst.tkmg_col  # Thermal conductivity, soil minerals (W/m/K)
    tkdry = soilstate_inst.tkdry_col  # Thermal conductivity, dry soil (W/m/K)
    csol = soilstate_inst.csol_col  # Heat capacity, soil solids (J/m3/K)

    # ------------------------------------------------------------------
    # Root fraction profiles — Fortran lines 90-111
    # ------------------------------------------------------------------
    for p in range(bounds.begp, bounds.endp + 1):
        c = int(patch.column[p])

        beta = float(pftcon.rootprof_beta[int(patch.itype[p])])

        if not (0.0 < beta < 1.0):
            endrun(
                msg=(
                    "ERROR: SoilStateInitTimeConst: invalid root profile beta. "
                    f"patch.itype={int(patch.itype[p])}, beta={beta}"
                )
            )

        # Jackson (1996) root profile — Fortran lines 95-97
        for j in range(1, nlevsoi + 1):  # j = 1, nlevsoi
            rootfr = rootfr.at[p, j].set(
                beta ** (float(col.zi[c, j - 1]) * m_to_cm)
                - beta ** (float(col.zi[c, j]) * m_to_cm)
            )

        # Bedrock layers have no roots — Fortran lines 99-101
        for j in range(nlevsoi + 1, nlevgrnd + 1):  # j = nlevsoi+1, nlevgrnd
            rootfr = rootfr.at[p, j].set(0.0)

        # Adjust roots for depth of soil — Fortran lines 103-108
        nb = int(col.nbedrock[c])
        surplus = sum(float(rootfr[p, j]) for j in range(nb + 1, nlevsoi + 1))
        for j in range(1, nb + 1):  # j = 1, nbedrock(c)
            rootfr = rootfr.at[p, j].add(surplus / float(nb))
        for j in range(nb + 1, nlevsoi + 1):  # rootfr(p,nbedrock+1:nlevsoi) = 0
            rootfr = rootfr.at[p, j].set(0.0)

    # ------------------------------------------------------------------
    # Soil hydraulic and thermal properties — Fortran lines 115-220
    # ------------------------------------------------------------------
    for c in range(bounds.begc, bounds.endc + 1):

        # Organic matter fraction — Fortran line 118
        om_frac = float(tower_organic[tower_num]) / organic_max

        # ------------------------------------------------------------------
        # Determine clay and sand — Fortran lines 122-145
        # ------------------------------------------------------------------
        if float(tower_clay[tower_num]) >= 0.0 and float(tower_sand[tower_num]) >= 0.0:
            # Tower site clay/sand provided directly — Fortran lines 124-126
            tex = 0
            clay = float(tower_clay[tower_num])
            sand = float(tower_sand[tower_num])

        else:
            # Look up soil texture class — Fortran lines 130-145
            tex = 0
            for m in range(ntex):  # m = 0, 1, ..., 10
                if tower_tex[tower_num] == soil_tex[m]:
                    tex = m + 1  # keep tex 1-based
                    break

            if tex == 0:
                print(
                    f"{iulog}  ERROR: SoilStateInitTimeConst: "
                    f"soil type = {tower_tex[tower_num]} not found for c = {c}"
                )
                endrun()

            clay = float(clay_tex[tex]) * 100.0  # fraction -> percent
            sand = float(sand_tex[tex]) * 100.0  # fraction -> percent

        # --------------------------------------------------------------
        # Per-layer properties — Fortran lines 147-218
        # --------------------------------------------------------------
        for j in range(1, nlevgrnd + 1):  # j = 1, nlevgrnd

            # Deep soil: zero organic fraction — Fortran lines 151-152
            if float(col.z[c, j]) > 0.5:
                om_frac = 0.0

            # sand / clay / organic matter for soil layers — Fortran lines 154-159
            if j <= nlevsoi:
                cellsand = cellsand.at[c, j].set(sand)
                cellclay = cellclay.at[c, j].set(clay)
                cellorg = cellorg.at[c, j].set(om_frac * organic_max)

            # ----------------------------------------------------------
            # Mineral hydraulic properties — Fortran lines 163-172
            # ----------------------------------------------------------
            if tex == 0:
                # Sand/clay based (CLM5 method)
                watsat = watsat.at[c, j].set(0.489 - 0.00126 * sand)
                sucsat = sucsat.at[c, j].set(10.0 * (10.0 ** (1.88 - 0.0131 * sand)))
                hksat = hksat.at[c, j].set(0.0070556 * (10.0 ** (-0.884 + 0.0153 * sand)))
                bsw = bsw.at[c, j].set(2.91 + 0.159 * clay)
            else:
                # Clapp and Hornberger (1978) texture class
                watsat = watsat.at[c, j].set(float(watsat_tex[tex]))
                sucsat = sucsat.at[c, j].set(-float(smpsat_tex[tex]))
                hksat = hksat.at[c, j].set(float(hksat_tex[tex]) / 60.0)  # mm/min -> mm/s
                bsw = bsw.at[c, j].set(float(bsw_tex[tex]))

            # ----------------------------------------------------------
            # Organic matter adjustments to hydraulic properties
            # Fortran lines 174-179
            # ----------------------------------------------------------
            z_cj = float(col.z[c, j])
            z_ratio = z_cj / zsapric

            om_watsat = max(0.93 - 0.1 * z_ratio, 0.83)
            om_sucsat = min(10.3 - 0.2 * z_ratio, 10.1)
            om_hksat = max(0.28 - 0.2799 * z_ratio, float(hksat[c, j]))
            om_b = min(2.7 + 9.3 * z_ratio, 12.0)

            watsat = watsat.at[c, j].set(
                (1.0 - om_frac) * float(watsat[c, j]) + om_watsat * om_frac
            )
            sucsat = sucsat.at[c, j].set(
                (1.0 - om_frac) * float(sucsat[c, j]) + om_sucsat * om_frac
            )
            bsw = bsw.at[c, j].set((1.0 - om_frac) * float(bsw[c, j]) + om_frac * om_b)

            # ----------------------------------------------------------
            # Percolating fraction of organic soil — Fortran lines 182-188
            # ----------------------------------------------------------
            if om_frac > pcalpha:
                perc_norm = (1.0 - pcalpha) ** (-pcbeta)
                perc_frac = perc_norm * (om_frac - pcalpha) ** pcbeta
            else:
                perc_frac = 0.0

            # ----------------------------------------------------------
            # Series conductivity mixing — Fortran lines 191-203
            # ----------------------------------------------------------
            uncon_frac = (1.0 - om_frac) + (1.0 - perc_frac) * om_frac

            if om_frac < 1.0:
                uncon_hksat = uncon_frac / (
                    (1.0 - om_frac) / float(hksat[c, j]) + ((1.0 - perc_frac) * om_frac) / om_hksat
                )
            else:
                uncon_hksat = 0.0

            hksat = hksat.at[c, j].set(uncon_frac * uncon_hksat + (perc_frac * om_frac) * om_hksat)

            # ----------------------------------------------------------
            # Thermal properties — Fortran lines 205-218
            # om_frac_therm: use om_frac (pre-CLM5 used 0.02)
            # ----------------------------------------------------------
            om_frac_therm = om_frac  # Fortran line 207 (active branch)

            # Dry thermal conductivity (W/m/K) — Fortran lines 210-216
            om_tkdry = 0.05
            if j <= nlevsoi:
                bulk_dens_min = 2700.0 * (1.0 - float(watsat[c, j]))
                tkdry_min = (0.135 * bulk_dens_min + 64.7) / (2700.0 - 0.947 * bulk_dens_min)
                tkdry = tkdry.at[c, j].set(
                    (1.0 - om_frac_therm) * tkdry_min + om_frac_therm * om_tkdry
                )
            else:
                bulk_dens_min = 2700.0
                tkdry_min = (0.135 * bulk_dens_min + 64.7) / (2700.0 - 0.947 * bulk_dens_min)
                tkdry = tkdry.at[c, j].set(tkdry_min)

            # Soil solids thermal conductivity (W/m/K) — Fortran lines 219-228
            om_tksol = 0.25
            if j <= nlevsoi:
                tksol_min = (8.80 * sand + 2.92 * clay) / (sand + clay)
                tkm = (1.0 - om_frac_therm) * tksol_min + om_frac_therm * om_tksol
                tkmg = tkmg.at[c, j].set(tkm ** (1.0 - float(watsat[c, j])))
            else:
                tkmg = tkmg.at[c, j].set(3.0)

            # Heat capacity, soil solids (J/m3/K) — Fortran lines 231-241
            om_cvsol = 2.5e6
            if tex == 0:
                cvsol = ((2.128 * sand + 2.385 * clay) / (sand + clay)) * 1.0e6
            else:
                cvsol = 1.926e6

            if j <= nlevsoi:
                csol = csol.at[c, j].set((1.0 - om_frac_therm) * cvsol + om_frac_therm * om_cvsol)
            else:
                csol = csol.at[c, j].set(csol_bedrock)

    # ------------------------------------------------------------------
    # Write all updated arrays back into the immutable state container
    # ------------------------------------------------------------------
    return soilstate_inst._replace(
        rootfr_patch=rootfr,
        cellsand_col=cellsand,
        cellclay_col=cellclay,
        cellorg_col=cellorg,
        watsat_col=watsat,
        sucsat_col=sucsat,
        hksat_col=hksat,
        bsw_col=bsw,
        tkmg_col=tkmg,
        tkdry_col=tkdry,
        csol_col=csol,
    )
