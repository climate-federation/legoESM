"""
JAX translation of MLinitVerticalMod Fortran module.

Initialize multilayer canopy vertical structure and profiles.
Provides three public routines:

- :func:`initVerticalStructure`: define layer heights, thicknesses,
  and normalised leaf/stem area index fractions from beta distributions.
- :func:`initVerticalProfiles`: initialise within-canopy wind, temperature,
  vapour pressure, CO2, intercepted water, leaf water potential, and
  leaf temperature profiles from the atmospheric forcing.
- :func:`getPADparameters`: set beta-distribution shape parameters for
  the plant area density profile from PFT defaults when not already
  provided by the tower site.

Original Fortran module: MLinitVerticalMod
Fortran lines 1-260
"""

from typing import Sequence

import jax.numpy as jnp

from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.CanopyStateType import canopystate_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.FrictionVelocityMod import frictionvel_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.abortutils import endrun  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.atm2lndType import atm2lnd_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varctl import iulog  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import mxpft  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.decompMod import bounds_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.PatchType import patch  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.wateratm2lndBulkType import wateratm2lndbulk_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyFluxesType import mlcanopy_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varcon import mmdry, mmh2o  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varctl import (  # noqa: F401
    dpai_min,
    dz_param,
    dz_short,
    dz_tall,
    nlayer_above,
    nlayer_within,
)
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varpar import isha, isun, nlevmlcan  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLMathToolsMod import beta_distribution_cdf  # noqa: F401

# ---------------------------------------------------------------------------
# Public: define canopy layer vertical structure
# ---------------------------------------------------------------------------


def initVerticalStructure(
    bounds: bounds_type,
    num_filter: int,
    filter_patch: Sequence[int],
    canopystate_inst: canopystate_type,
    frictionvel_inst: frictionvel_type,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Define canopy layer vertical structure from the reference height,
    canopy top height, and plant area density beta distribution.

    Mirrors Fortran subroutine ``initVerticalStructure`` (lines 24-175).

    Layer discretization (Fortran lines 66-98)
    -------------------------------------------
    Two modes are supported, selected by ``nlayer_within`` /
    ``nlayer_above`` in ``MLclm_varctl``:

    **Explicit layer counts** (both > 0):
        ``ntop = nlayer_within``, ``nabove = nlayer_above``;
        ``dz_within = ztop / ntop``, ``dz_above = ztop_to_zref / nabove``.

    **Height-increment mode** (either == 0):
        ``dz_within = dz_tall`` if ``ztop > dz_param`` else ``dz_short``;
        ``ntop = nint(ztop / dz_within)``; ``dz_above = dz_within``;
        ``nabove = nint(ztop_to_zref / dz_above)``.

    Interface heights (Fortran lines 100-120)
    ------------------------------------------
    ``zw(p, 0..ncan)`` are the layer-interface heights, with
    ``zw(p,0) = 0`` (ground) and ``zw(p,ncan) = zref``. Layer scalar
    heights ``zs`` are midpoints; layer thicknesses ``dz = zw[ic] - zw[ic-1]``.

    Plant area density profiles (Fortran lines 122-167)
    ----------------------------------------------------
    Leaf and stem area fractions per layer are computed from the
    2-parameter beta distribution CDF evaluated at the normalised bottom
    and top heights of each within-canopy layer. Layers with
    ``dlai + dsai < dpai_min`` are zeroed and their area is redistributed
    proportionally across the remaining layers. The lowest layer with
    plant area sets ``nbot``; above-canopy layers get zero area.
    All profiles are normalised to unit LAI / SAI.

    Args:
        bounds: Decomposition bounds (``begp``, ``endp``).
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter array (1-based values).
        canopystate_inst: Canopy state container supplying ``htop_patch``
            (read-only).
        frictionvel_inst: Friction velocity container supplying
            ``forc_hgt_u_patch`` (read-only).
        mlcanopy_inst: Multilayer canopy container; all geometry and
            profile-fraction fields are populated and returned.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    # Unpack output arrays (Fortran associate block, lines 46-63)
    forc_hgt_u = frictionvel_inst.forc_hgt_u_patch
    htop = canopystate_inst.htop_patch
    pbeta_lai = mlcanopy_inst.pbeta_lai_canopy
    pbeta_sai = mlcanopy_inst.pbeta_sai_canopy

    zref = mlcanopy_inst.zref_forcing
    ztop = mlcanopy_inst.ztop_canopy
    zbot = mlcanopy_inst.zbot_canopy
    ncan = mlcanopy_inst.ncan_canopy
    ntop = mlcanopy_inst.ntop_canopy
    nbot = mlcanopy_inst.nbot_canopy
    dlai_frac = mlcanopy_inst.dlai_frac_profile
    dsai_frac = mlcanopy_inst.dsai_frac_profile
    zs = mlcanopy_inst.zs_profile
    zw = mlcanopy_inst.zw_profile
    dz = mlcanopy_inst.dz_profile

    unit_lai: float = 1.0  # Unit LAI — Fortran line 43
    unit_sai: float = 1.0  # Unit SAI — Fortran line 44

    # Working arrays for unscaled LAI/SAI per layer (local, not stored)
    n_patch = bounds.endp - bounds.begp + 1
    dlai = jnp.zeros((n_patch + bounds.begp, nlevmlcan + 1), dtype=jnp.float64)
    dsai = jnp.zeros((n_patch + bounds.begp, nlevmlcan + 1), dtype=jnp.float64)

    for fp in range(1, num_filter + 1):  # Fortran: do fp = 1, num_filter
        p = int(filter_patch[fp - 1])

        # Atmospheric reference height and canopy top — Fortran lines 66-68
        zref = zref.at[p].set(float(forc_hgt_u[p]))
        ztop = ztop.at[p].set(float(htop[p]))
        ztop_to_zref = float(zref[p]) - float(ztop[p])

        # ------------------------------------------------------------------
        # Layer counts and thicknesses — Fortran lines 70-98
        # ------------------------------------------------------------------
        if nlayer_within > 0 and nlayer_above > 0:
            # Explicit layer counts — Fortran lines 72-78
            _ntop = nlayer_within
            dz_within = float(ztop[p]) / float(_ntop)
            nabove = nlayer_above
            _ncan = _ntop + nabove
            dz_above = ztop_to_zref / float(nabove)

        elif nlayer_within == 0 or nlayer_above == 0:
            # Height-increment mode — Fortran lines 80-98
            if float(ztop[p]) > dz_param:
                dz_within = dz_tall
            else:
                dz_within = dz_short

            _ntop = round(float(ztop[p]) / dz_within)  # Fortran: nint
            dz_within = float(ztop[p]) / float(_ntop)
            dz_above = dz_within
            nabove = round(ztop_to_zref / dz_above)  # Fortran: nint
            _ncan = _ntop + nabove
            dz_above = ztop_to_zref / float(nabove)

        else:
            endrun(msg=" ERROR: initVerticalStructure: invalid canopy specification")
            _ntop = 1
            _ncan = 2
            dz_within = 1.0
            dz_above = 1.0
            nabove = 1

        if _ncan > nlevmlcan:
            endrun(msg=" ERROR: initVerticalStructure: ncan > nlevmlcan")

        ntop = ntop.at[p].set(_ntop)
        ncan = ncan.at[p].set(_ncan)

        # ------------------------------------------------------------------
        # Interface heights (zw) — Fortran lines 100-120
        # ------------------------------------------------------------------

        # Within-canopy: ic = ntop down to 0 — Fortran lines 103-110
        zw = zw.at[p, _ntop].set(float(ztop[p]))
        for ic in range(_ntop - 1, -1, -1):  # Fortran: do ic = ntop-1, 0, -1
            zw = zw.at[p, ic].set(float(zw[p, ic + 1]) - dz_within)

        # Guard: zw(p,0) must be zero — Fortran lines 112-116
        if abs(float(zw[p, 0])) > 1.0e-10:
            endrun(msg=" ERROR: initVerticalStructure: zw(p,0) improperly defined")
        zw = zw.at[p, 0].set(max(float(zw[p, 0]), 0.0))  # negative → 0
        zw = zw.at[p, 0].set(min(float(zw[p, 0]), 0.0))  # positive → 0

        # Above-canopy: ic = ncan down to ntop+1 — Fortran lines 118-121
        zw = zw.at[p, _ncan].set(float(zref[p]))
        for ic in range(_ncan - 1, _ntop, -1):  # Fortran: do ic = ncan-1, ntop+1, -1
            zw = zw.at[p, ic].set(float(zw[p, ic + 1]) - dz_above)

        # Layer thickness and scalar height — Fortran lines 123-129
        for ic in range(1, _ncan + 1):  # Fortran: do ic = 1, ncan
            dz = dz.at[p, ic].set(float(zw[p, ic]) - float(zw[p, ic - 1]))
            zs = zs.at[p, ic].set(0.5 * (float(zw[p, ic]) + float(zw[p, ic - 1])))

        # ------------------------------------------------------------------
        # Beta-distribution LAI profile — Fortran lines 131-147
        # ------------------------------------------------------------------
        for ic in range(1, _ntop + 1):
            zrel_bot = min(float(zw[p, ic - 1]) / float(ztop[p]), 1.0)
            zrel_top = min(float(zw[p, ic]) / float(ztop[p]), 1.0)
            cdf_bot = beta_distribution_cdf(
                float(pbeta_lai[p, 1]), float(pbeta_lai[p, 2]), zrel_bot
            )
            cdf_top = beta_distribution_cdf(
                float(pbeta_lai[p, 1]), float(pbeta_lai[p, 2]), zrel_top
            )
            dlai = dlai.at[p, ic].set((cdf_top - cdf_bot) * unit_lai)

        # Beta-distribution SAI profile — Fortran lines 149-155
        for ic in range(1, _ntop + 1):
            zrel_bot = min(float(zw[p, ic - 1]) / float(ztop[p]), 1.0)
            zrel_top = min(float(zw[p, ic]) / float(ztop[p]), 1.0)
            cdf_bot = beta_distribution_cdf(
                float(pbeta_sai[p, 1]), float(pbeta_sai[p, 2]), zrel_bot
            )
            cdf_top = beta_distribution_cdf(
                float(pbeta_sai[p, 1]), float(pbeta_sai[p, 2]), zrel_top
            )
            dsai = dsai.at[p, ic].set((cdf_top - cdf_bot) * unit_sai)

        # PAI sum check — Fortran lines 157-160
        pai_sum = sum(float(dlai[p, ic]) + float(dsai[p, ic]) for ic in range(1, _ntop + 1))
        if abs(pai_sum - (unit_lai + unit_sai)) > 1.0e-6:
            endrun(
                msg=" ERROR: initVerticalStructure: plant area profile does not sum to canopy total"
            )

        # ------------------------------------------------------------------
        # Zero layers below dpai_min and redistribute — Fortran lines 162-185
        # ------------------------------------------------------------------
        lai_miss = 0.0
        sai_miss = 0.0
        for ic in range(1, _ntop + 1):
            if float(dlai[p, ic]) + float(dsai[p, ic]) < dpai_min:
                lai_miss += float(dlai[p, ic])
                sai_miss += float(dsai[p, ic])
                dlai = dlai.at[p, ic].set(0.0)
                dsai = dsai.at[p, ic].set(0.0)

        if lai_miss > 0.0:
            lai_sum = sum(float(dlai[p, ic]) for ic in range(1, _ntop + 1))
            for ic in range(1, _ntop + 1):
                dlai = dlai.at[p, ic].set(
                    float(dlai[p, ic]) + lai_miss * (float(dlai[p, ic]) / lai_sum)
                )

        if sai_miss > 0.0:
            sai_sum = sum(float(dsai[p, ic]) for ic in range(1, _ntop + 1))
            for ic in range(1, _ntop + 1):
                dsai = dsai.at[p, ic].set(
                    float(dsai[p, ic]) + sai_miss * (float(dsai[p, ic]) / sai_sum)
                )

        # ------------------------------------------------------------------
        # Find lowest leaf/stem layer (nbot) — Fortran lines 187-196
        # ------------------------------------------------------------------
        _nbot = 0
        for ic in range(_ntop, 0, -1):  # Fortran: do ic = ntop, 1, -1
            if float(dlai[p, ic]) + float(dsai[p, ic]) > 0.0:
                _nbot = ic
        if _nbot == 0:
            endrun(msg=" ERROR: initVerticalStructure: nbot not defined")
        nbot = nbot.at[p].set(_nbot)
        zbot = zbot.at[p].set(float(zw[p, _nbot - 1]))  # bottom of layer nbot

        # Post-redistribution sum checks — Fortran lines 198-205
        lai_sum = sum(float(dlai[p, ic]) for ic in range(1, _ntop + 1))
        if abs(lai_sum - unit_lai) > 1.0e-6:
            endrun(
                msg=" ERROR: initVerticalStructure: leaf area profile does not sum to canopy total after redistribution"
            )
        sai_sum = sum(float(dsai[p, ic]) for ic in range(1, _ntop + 1))
        if abs(sai_sum - unit_sai) > 1.0e-6:
            endrun(
                msg=" ERROR: initVerticalStructure: stem area profile does not sum to canopy total after redistribution"
            )

        # Zero above-canopy layers — Fortran lines 207-210
        for ic in range(_ntop + 1, _ncan + 1):
            dlai = dlai.at[p, ic].set(0.0)
            dsai = dsai.at[p, ic].set(0.0)

        # Normalise profiles — Fortran lines 212-215
        for ic in range(1, _ncan + 1):
            dlai_frac = dlai_frac.at[p, ic].set(float(dlai[p, ic]) / unit_lai)
            dsai_frac = dsai_frac.at[p, ic].set(float(dsai[p, ic]) / unit_sai)

        # Check no zero-PAI layers between nbot and ntop — Fortran lines 217-222
        iflag = 0
        for ic in range(_nbot, _ntop + 1):
            if float(dlai_frac[p, ic]) + float(dsai_frac[p, ic]) <= 0.0:
                iflag = 1
        if iflag == 1:
            endrun(msg=" ERROR: initVerticalStructure: canopy layer has zero plant area index")

    return mlcanopy_inst._replace(
        zref_forcing=zref,
        ztop_canopy=ztop,
        zbot_canopy=zbot,
        ncan_canopy=ncan,
        ntop_canopy=ntop,
        nbot_canopy=nbot,
        dlai_frac_profile=dlai_frac,
        dsai_frac_profile=dsai_frac,
        zs_profile=zs,
        zw_profile=zw,
        dz_profile=dz,
    )


# ---------------------------------------------------------------------------
# Public: initialise vertical profiles and canopy states
# ---------------------------------------------------------------------------


def initVerticalProfiles(
    num_filter: int,
    filter_patch: Sequence[int],
    atm2lnd_inst: atm2lnd_type,
    wateratm2lndbulk_inst: wateratm2lndbulk_type,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Initialise within-canopy vertical profiles and leaf state variables
    from the current atmospheric forcing.

    Mirrors Fortran subroutine ``initVerticalProfiles`` (lines 178-235).

    For each patch and each aboveground layer ``ic = 1, ncan``:

    .. code-block:: none

        wind(p,ic) = sqrt(forc_u^2 + forc_v^2)
        tair(p,ic) = forc_t(c)
        eair(p,ic) = forc_q * forc_pbot / (mmh2o/mmdry + (1-mmh2o/mmdry)*forc_q)
        cair(p,ic) = forc_pco2 / forc_pbot * 1e6          [umol/mol]
        tleaf(p,ic,isun) = tleaf(p,ic,isha) = forc_t(c)
        lwp(p,ic,isun)   = lwp(p,ic,isha)   = -0.1        [MPa]
        h2ocan(p,ic)     = 0

    Soil surface temperature is set to ``forc_t(c)``.

    Args:
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter array (1-based values).
        atm2lnd_inst: Atmosphere-to-land forcing container (read-only).
        wateratm2lndbulk_inst: Bulk atm-to-land water forcing container
            (read-only).
        mlcanopy_inst: Multilayer canopy container; profile fields and
            leaf state variables are updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    # Unpack input forcing (Fortran associate block, lines 198-213)
    forc_u = atm2lnd_inst.forc_u_grc
    forc_v = atm2lnd_inst.forc_v_grc
    forc_pco2 = atm2lnd_inst.forc_pco2_grc
    forc_t = atm2lnd_inst.forc_t_downscaled_col
    forc_q = wateratm2lndbulk_inst.forc_q_downscaled_col
    forc_pbot = atm2lnd_inst.forc_pbot_downscaled_col

    ncan = mlcanopy_inst.ncan_canopy
    tg = mlcanopy_inst.tg_soil
    wind = mlcanopy_inst.wind_profile
    tair = mlcanopy_inst.tair_profile
    eair = mlcanopy_inst.eair_profile
    cair = mlcanopy_inst.cair_profile
    h2ocan = mlcanopy_inst.h2ocan_profile
    lwp = mlcanopy_inst.lwp_leaf
    tleaf = mlcanopy_inst.tleaf_leaf

    for fp in range(1, num_filter + 1):  # Fortran: do fp = 1, num_filter
        p = int(filter_patch[fp - 1])
        c = int(patch.column[p])
        g = int(patch.gridcell[p])

        for ic in range(1, int(ncan[p]) + 1):  # Fortran: do ic = 1, ncan(p)

            # Wind speed — Fortran line 219
            wind_val = float(jnp.sqrt(float(forc_u[g]) ** 2 + float(forc_v[g]) ** 2))
            wind = wind.at[p, ic].set(wind_val)

            # Air temperature — Fortran line 220
            tair = tair.at[p, ic].set(float(forc_t[c]))

            # Vapour pressure (Pa) from specific humidity — Fortran line 221
            q = float(forc_q[c])
            pb = float(forc_pbot[c])
            e = q * pb / (mmh2o / mmdry + (1.0 - mmh2o / mmdry) * q)
            eair = eair.at[p, ic].set(e)

            # CO2 mole fraction (umol/mol) — Fortran line 222
            cair = cair.at[p, ic].set(float(forc_pco2[g]) / pb * 1.0e6)

            # Leaf temperature and water potential — Fortran line 224
            tleaf = tleaf.at[p, ic, isun].set(float(forc_t[c]))
            tleaf = tleaf.at[p, ic, isha].set(float(forc_t[c]))
            lwp = lwp.at[p, ic, isun].set(-0.1)
            lwp = lwp.at[p, ic, isha].set(-0.1)

            # Canopy intercepted water — Fortran line 225
            h2ocan = h2ocan.at[p, ic].set(0.0)

        # Soil surface temperature — Fortran line 228
        tg = tg.at[p].set(float(forc_t[c]))

    return mlcanopy_inst._replace(
        tg_soil=tg,
        wind_profile=wind,
        tair_profile=tair,
        eair_profile=eair,
        cair_profile=cair,
        h2ocan_profile=h2ocan,
        lwp_leaf=lwp,
        tleaf_leaf=tleaf,
    )


# ---------------------------------------------------------------------------
# Public: set plant area density beta-distribution parameters from PFT defaults
# ---------------------------------------------------------------------------


def getPADparameters(
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Set beta-distribution shape parameters for the plant area density
    (PAD) vertical profile from PFT defaults.

    Mirrors Fortran subroutine ``getPADparameters`` (lines 237-260).

    Parameters are only assigned when not already set to positive values
    (tower-site overrides from ``TowerDataMod`` take precedence). The
    guard condition matches the Fortran check:

    .. code-block:: none

        if (pbeta_lai(p,1) < 0 .or. pbeta_lai(p,2) < 0 .or.
            pbeta_sai(p,1) < 0 .or. pbeta_sai(p,2) < 0)

    PFT default values (Fortran lines 249-263):

    .. code-block:: none

        PFT 1        (needle ET): lai=[11.5, 3.5]
        PFT 2-3      (needle EB): lai=[3.5,  2.0]
        PFT 4-5      (BL ET):     lai=[3.5,  2.0]
        PFT 6-8      (BL DT):     lai=[3.5,  2.0]
        PFT 9-11     (shrub):     lai=[3.5,  2.0]
        PFT 12-16    (grass/crop):lai=[2.5,  2.5]
        sai: same as lai for all PFTs

    Args:
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter array (1-based values).
        mlcanopy_inst: Multilayer canopy container; ``pbeta_lai_canopy``
            and ``pbeta_sai_canopy`` are updated where necessary.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    # ------------------------------------------------------------------
    # PFT default beta parameters — Fortran lines 249-263
    # Shape (mxpft+1, 2): index 0 unused; columns 1 and 2 are the two params.
    # ------------------------------------------------------------------
    _def = -999.0
    pbeta_lai_pft: list[list[float]] = [[_def, _def]] * (mxpft + 1)
    pbeta_sai_pft: list[list[float]] = [[_def, _def]] * (mxpft + 1)

    pbeta_lai_pft = pbeta_lai_pft.copy()  # make mutable

    # 1 => needleleaf_evergreen_temperate_tree — Fortran line 251
    pbeta_lai_pft[1] = [11.5, 3.5]

    # 2-3 => needleleaf_evergreen/deciduous_boreal_tree — Fortran line 254
    for i in range(2, 4):
        pbeta_lai_pft[i] = [3.5, 2.0]

    # 4-5 => broadleaf_evergreen_tropical/temperate_tree — Fortran line 257
    for i in range(4, 6):
        pbeta_lai_pft[i] = [3.5, 2.0]

    # 6-8 => broadleaf_deciduous trees — Fortran line 260
    for i in range(6, 9):
        pbeta_lai_pft[i] = [3.5, 2.0]

    # 9-11 => broadleaf shrubs — Fortran line 263
    for i in range(9, 12):
        pbeta_lai_pft[i] = [3.5, 2.0]

    # 12-16 => grasses and crops — Fortran line 266
    for i in range(12, 17):
        pbeta_lai_pft[i] = [2.5, 2.5]

    # SAI: same as LAI for all PFTs — Fortran line 268
    pbeta_sai_pft = [row[:] for row in pbeta_lai_pft]

    # ------------------------------------------------------------------
    # Assign to patches where parameters have not been set — Fortran lines 272-279
    # ------------------------------------------------------------------
    pbeta_lai = mlcanopy_inst.pbeta_lai_canopy
    pbeta_sai = mlcanopy_inst.pbeta_sai_canopy

    for fp in range(1, num_filter + 1):  # Fortran: do fp = 1, num_filter
        p = int(filter_patch[fp - 1])
        pft = int(patch.itype[p])

        # Use PFT defaults only when any parameter is < 0 — Fortran lines 275-279
        if (
            float(pbeta_lai[p, 1]) < 0.0
            or float(pbeta_lai[p, 2]) < 0.0
            or float(pbeta_sai[p, 1]) < 0.0
            or float(pbeta_sai[p, 2]) < 0.0
        ):
            pbeta_lai = pbeta_lai.at[p, 1].set(pbeta_lai_pft[pft][0])
            pbeta_lai = pbeta_lai.at[p, 2].set(pbeta_lai_pft[pft][1])
            pbeta_sai = pbeta_sai.at[p, 1].set(pbeta_sai_pft[pft][0])
            pbeta_sai = pbeta_sai.at[p, 2].set(pbeta_sai_pft[pft][1])

    return mlcanopy_inst._replace(
        pbeta_lai_canopy=pbeta_lai,
        pbeta_sai_canopy=pbeta_sai,
    )
