"""
JAX translation of MLFluxProfileSolutionMod Fortran module.

Source/sink fluxes for leaves and soil, and concentration profiles
for the multilayer canopy model (CLM/CTSM).

Original Fortran module: MLFluxProfileSolutionMod
Fortran lines 1-430
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from legoesm.land.canopy.clm_ml_backend.clm_src_main.abortutils import endrun  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varctl import iulog  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyFluxesType import mlcanopy_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varctl import dtime_ml, flux_profile_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varpar import isha, isun, nleaf, nlevmlcan  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLLeafFluxesMod import LeafFluxes  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLMathToolsMod import tridiag_2eq  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLSoilFluxesMod import SoilFluxes  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLWaterVaporMod import LatVap, SatVap  # noqa: F401

# ---------------------------------------------------------------------------
# Debug flag — set True to run ErrorCheck01 / ErrorCheck02 after every
# FluxProfileSolution call.  When False (default / production) the checks
# are skipped entirely, eliminating the 9 np.asarray() host-device syncs
# that those checks require.
# ---------------------------------------------------------------------------
DEBUG_FPS_CHECKS: bool = False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def FluxProfileSolution(
    num_filter: int,
    filter: Array,
    mlcanopy_inst: mlcanopy_type,
    grid=None,
) -> mlcanopy_type:
    """
    Source/sink fluxes for leaves and soil, and concentration profiles.

    Mirrors Fortran subroutine ``FluxProfileSolution`` (lines 27-84).
    Dispatches to ``WellMixed`` or ``ImplicitFluxProfileSolution`` based
    on ``flux_profile_type``.

    Args:
        num_filter: Number of patches in the filter.
            Fortran: ``integer, intent(in) :: num_filter``.
        filter: 1-D array of patch indices to process.
            Fortran: ``integer, intent(in) :: filter(:)``.
        mlcanopy_inst: Multilayer canopy state container (immutable
            NamedTuple); updated fields are returned in a new instance.

    Returns:
        Updated :class:`mlcanopy_type` after flux/profile calculation.

    Raises:
        RuntimeError: If ``flux_profile_type`` is 0 or -1 (WellMixed
            legacy path, not supported) or unrecognised.
    """
    co2ref = mlcanopy_inst.co2ref_forcing  # Atmospheric CO2 at ref height (umol/mol)
    ncan = mlcanopy_inst.ncan_canopy  # Number of aboveground layers
    cair = mlcanopy_inst.cair_profile  # Canopy layer atmospheric CO2 (umol/mol)

    if flux_profile_type in (0, -1):
        # Legacy well-mixed path — Fortran lines 52-58
        mlcanopy_inst = WellMixed(num_filter, filter, mlcanopy_inst)
        endrun(msg=" ERROR: WellMixed option not valid")

    elif flux_profile_type == 1:
        # Implicit flux-profile solution — Fortran lines 62-74
        for fp in range(num_filter):
            if grid is not None:
                p = grid.p
                n = grid.ncan
            else:
                p = int(filter[fp])
                n = int(ncan[p])
            mlcanopy_inst = ImplicitFluxProfileSolution(p, mlcanopy_inst, n=n)

            # No profile for CO2: set every layer to reference value
            cair = cair.at[p, :n].set(co2ref[p])
            mlcanopy_inst = mlcanopy_inst._replace(cair_profile=cair)

    else:
        endrun(msg=" ERROR: FluxProfileSolution: flux_profile_type not valid")

    return mlcanopy_inst


# ---------------------------------------------------------------------------
# Private: implicit flux-profile solution
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=(1,))
def _implicit_fps_jit(
    p: "int | jnp.ndarray",
    n: int,
    mlcanopy_inst: mlcanopy_type,
):
    """JIT-compiled core of ImplicitFluxProfileSolution.

    ``n`` is a static Python int so all Python ``range(1, n+1)`` loops and
    ``slice(1, n+1)`` index expressions are concrete at trace time.  ``p`` is
    DYNAMIC (arg 0, NOT static): it may be a TRACED lax.scan-over-columns index
    (S3), and it is used ONLY for gather/scatter (``field[p]`` / ``.at[p]``,
    incl. the ``SoilFluxes(p, ...)`` plain helper) — never as a slice bound or
    Python range limit.  Recompilation occurs only when the canopy layer count
    ``n`` changes (rare for a fixed site).

    Returns ``(mlcanopy_inst, aux)`` where ``aux`` is a tuple of arrays
    needed by ``ErrorCheck01`` / ``ErrorCheck02`` which run outside JIT
    using concrete numpy values.

    Mirrors Fortran subroutine ``ImplicitFluxProfileSolution``
    (lines 86-310).  Reference: Bonan et al. (2018) GMD 11, 1467-1496.
    """
    # ------------------------------------------------------------------
    # Unpack inputs from mlcanopy_inst  (Fortran associate block)
    # ------------------------------------------------------------------
    tair = mlcanopy_inst.tair_profile  # Canopy layer air temperature (K)
    eair = mlcanopy_inst.eair_profile  # Canopy layer vapor pressure (Pa)
    shair = mlcanopy_inst.shair_profile  # Canopy layer air SH flux (W/m2)
    etair = mlcanopy_inst.etair_profile  # Canopy layer air ET flux (mol H2O/m2/s)
    stair = mlcanopy_inst.stair_profile  # Canopy layer air storage flux (W/m2)

    # ------------------------------------------------------------------
    # Pre-extract all patch-p slices — JAX arrays used directly
    # ------------------------------------------------------------------
    _pref_p = mlcanopy_inst.pref_forcing[p]
    _rhomol_p = mlcanopy_inst.rhomol_forcing[p]
    _cpair_p = mlcanopy_inst.cpair_forcing[p]
    _tref_p = mlcanopy_inst.tref_forcing[p]
    _thref_p = mlcanopy_inst.thref_forcing[p]
    _eref_p = mlcanopy_inst.eref_forcing[p]
    _tg_bef_p = mlcanopy_inst.tg_bef_soil[p]
    _rhg_p = mlcanopy_inst.rhg_soil[p]
    _rnsoi_p = mlcanopy_inst.rnsoi_soil[p]
    _soilres_p = mlcanopy_inst.soilres_soil[p]
    _soil_t_p = mlcanopy_inst.soil_t_soil[p]
    _soil_dz_p = mlcanopy_inst.soil_dz_soil[p]
    _soil_tk_p = mlcanopy_inst.soil_tk_soil[p]
    _gac0_p = mlcanopy_inst.gac0_soil[p]
    # n is passed as a static Python int from the wrapper — do not re-derive

    # 1-D and 2-D slices — JAX arrays used directly
    _dz_p = mlcanopy_inst.dz_profile[p]
    _dpai_p = mlcanopy_inst.dpai_profile[p]
    _fwet_p = mlcanopy_inst.fwet_profile[p]
    _fdry_p = mlcanopy_inst.fdry_profile[p]
    _fracsun_p = mlcanopy_inst.fracsun_profile[p]
    _cpleaf_p = mlcanopy_inst.cpleaf_profile[p]
    _gac_p = mlcanopy_inst.gac_profile[p]
    _tair_bef_p = mlcanopy_inst.tair_bef_profile[p]
    _eair_bef_p = mlcanopy_inst.eair_bef_profile[p]
    _gbh_p = mlcanopy_inst.gbh_leaf[p]
    _gbv_p = mlcanopy_inst.gbv_leaf[p]
    _gs_p = mlcanopy_inst.gs_leaf[p]
    _rnleaf_p = mlcanopy_inst.rnleaf_leaf[p]
    _tleaf_bef_p = mlcanopy_inst.tleaf_bef_leaf[p]

    # ------------------------------------------------------------------
    # Timestep and latent heat — Fortran lines 155-158
    # ------------------------------------------------------------------
    dtime = dtime_ml
    lambda_ = LatVap(_tref_p)

    # ------------------------------------------------------------------
    # Ground temperature coefficients alpha0, beta0, delta0
    # Fortran lines 168-177
    # ------------------------------------------------------------------
    esat, desat = SatVap(_tg_bef_p)
    qsat0 = esat / _pref_p
    dqsat0 = desat / _pref_p

    gsw = (1.0 / _soilres_p) * _rhomol_p
    gs0 = _gac0_p * gsw / (_gac0_p + gsw)

    c02 = _soil_tk_p / _soil_dz_p
    c01 = -c02 * _soil_t_p

    den = _cpair_p * _gac0_p + lambda_ * _rhg_p * gs0 * dqsat0 + c02
    alpha0 = _cpair_p * _gac0_p / den
    beta0 = lambda_ * gs0 / den
    delta0 = (_rnsoi_p - lambda_ * _rhg_p * gs0 * (qsat0 - dqsat0 * _tg_bef_p) - c01) / den

    # ------------------------------------------------------------------
    # Leaf temperature coefficients — JAX arrays
    # Bonan et al. (2018) eqs. (10)-(13), (A1)-(A5); Fortran lines 187-247
    # ------------------------------------------------------------------
    gleaf_sh = jnp.zeros((nlevmlcan + 1, nleaf + 1))
    gleaf_et = jnp.zeros((nlevmlcan + 1, nleaf + 1))
    heatcap = jnp.zeros((nlevmlcan + 1, nleaf + 1))
    avail_energy = jnp.zeros((nlevmlcan + 1, nleaf + 1))
    dqsat_arr = jnp.zeros((nlevmlcan + 1, nleaf + 1))
    qsat_term = jnp.zeros((nlevmlcan + 1, nleaf + 1))
    alpha = jnp.zeros((nlevmlcan + 1, nleaf + 1))
    beta = jnp.zeros((nlevmlcan + 1, nleaf + 1))
    delta = jnp.zeros((nlevmlcan + 1, nleaf + 1))

    for ic in range(1, n + 1):
        active_ic = _dpai_p[ic] > 0.0
        for il in [isun, isha]:
            _gs_il = _gs_p[ic, il]
            _gbv_il = _gbv_p[ic, il]
            gsh_raw = 2.0 * _gbh_p[ic, il]
            # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
            gs_gbv_denom = jnp.maximum(_gs_il + _gbv_il, 1.0e-30)
            get_raw = _gs_il * _gbv_il / gs_gbv_denom * _fdry_p[ic] + _gbv_il * _fwet_p[ic]

            esat_l, desat_l = SatVap(_tleaf_bef_p[ic, il])
            qsat_l = esat_l / _pref_p
            dqsat_l = desat_l / _pref_p
            qsat_term_val = qsat_l - dqsat_l * _tleaf_bef_p[ic, il]

            hcap_raw = _cpleaf_p[ic]
            avail_raw = _rnleaf_p[ic, il]

            # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
            den_l = jnp.maximum(
                hcap_raw / dtime + gsh_raw * _cpair_p + get_raw * lambda_ * dqsat_l,
                1.0e-30,
            )
            alpha_val = gsh_raw * _cpair_p / den_l
            beta_val = get_raw * lambda_ / den_l
            delta_val = (
                avail_raw / den_l
                - lambda_ * get_raw * qsat_term_val / den_l
                + hcap_raw / dtime * _tleaf_bef_p[ic, il] / den_l
            )

            pai = (
                _dpai_p[ic] * _fracsun_p[ic] if il == isun else _dpai_p[ic] * (1.0 - _fracsun_p[ic])
            )
            gsh_ic = jnp.where(active_ic, gsh_raw * pai, 0.0)
            get_ic = jnp.where(active_ic, get_raw * pai, 0.0)
            hcap_ic = jnp.where(active_ic, hcap_raw * pai, 0.0)
            avail_ic = jnp.where(active_ic, avail_raw * pai, 0.0)

            gleaf_sh = gleaf_sh.at[ic, il].set(gsh_ic)
            gleaf_et = gleaf_et.at[ic, il].set(get_ic)
            heatcap = heatcap.at[ic, il].set(hcap_ic)
            avail_energy = avail_energy.at[ic, il].set(avail_ic)
            dqsat_arr = dqsat_arr.at[ic, il].set(jnp.where(active_ic, dqsat_l, 0.0))
            qsat_term = qsat_term.at[ic, il].set(jnp.where(active_ic, qsat_term_val, 0.0))
            alpha = alpha.at[ic, il].set(jnp.where(active_ic, alpha_val, 0.0))
            beta = beta.at[ic, il].set(jnp.where(active_ic, beta_val, 0.0))
            delta = delta.at[ic, il].set(jnp.where(active_ic, delta_val, 0.0))

    # ------------------------------------------------------------------
    # Tridiagonal coefficients — JAX arrays
    # Bonan et al. (2018) eqs. (16)-(17), (S10)-(S31); Fortran lines 258-316
    # ------------------------------------------------------------------
    rho_dz_over_dt = jnp.zeros(nlevmlcan + 1)
    a1 = jnp.zeros(nlevmlcan + 1)
    b11 = jnp.zeros(nlevmlcan + 1)
    b12 = jnp.zeros(nlevmlcan + 1)
    c1 = jnp.zeros(nlevmlcan + 1)
    d1 = jnp.zeros(nlevmlcan + 1)
    a2 = jnp.zeros(nlevmlcan + 1)
    b21 = jnp.zeros(nlevmlcan + 1)
    b22 = jnp.zeros(nlevmlcan + 1)
    c2 = jnp.zeros(nlevmlcan + 1)
    d2 = jnp.zeros(nlevmlcan + 1)

    for ic in range(1, n + 1):
        rdt_ic = _rhomol_p * _dz_p[ic] / dtime
        rho_dz_over_dt = rho_dz_over_dt.at[ic].set(rdt_ic)

        # ic == 1 and ic == n are Python ints — static comparisons, fine in jit
        gac_below_sh = _gac0_p if ic == 1 else _gac_p[ic - 1]
        gac_below_et = gs0 if ic == 1 else _gac_p[ic - 1]
        gac_ic = _gac_p[ic]

        a1_ic = -gac_below_sh
        b11_ic = (
            rdt_ic
            + gac_below_sh
            + gac_ic
            + gleaf_sh[ic, isun] * (1.0 - alpha[ic, isun])
            + gleaf_sh[ic, isha] * (1.0 - alpha[ic, isha])
        )
        b12_ic = -gleaf_sh[ic, isun] * beta[ic, isun] - gleaf_sh[ic, isha] * beta[ic, isha]
        c1_ic = -gac_ic
        d1_ic = (
            rdt_ic * _tair_bef_p[ic]
            + gleaf_sh[ic, isun] * delta[ic, isun]
            + gleaf_sh[ic, isha] * delta[ic, isha]
        )

        if ic == n:
            c1_ic = jnp.zeros(())
            d1_ic = d1_ic + gac_ic * _thref_p
        if ic == 1:
            a1_ic = jnp.zeros(())
            b11_ic = b11_ic - _gac0_p * alpha0
            b12_ic = b12_ic - _gac0_p * beta0
            d1_ic = d1_ic + _gac0_p * delta0

        a2_ic = -gac_below_et
        b21_ic = (
            -gleaf_et[ic, isun] * dqsat_arr[ic, isun] * alpha[ic, isun]
            - gleaf_et[ic, isha] * dqsat_arr[ic, isha] * alpha[ic, isha]
        )
        b22_ic = (
            rdt_ic
            + gac_below_et
            + gac_ic
            + gleaf_et[ic, isun] * (1.0 - dqsat_arr[ic, isun] * beta[ic, isun])
            + gleaf_et[ic, isha] * (1.0 - dqsat_arr[ic, isha] * beta[ic, isha])
        )
        c2_ic = -gac_ic
        d2_ic = (
            rdt_ic * (_eair_bef_p[ic] / _pref_p)
            + gleaf_et[ic, isun] * (dqsat_arr[ic, isun] * delta[ic, isun] + qsat_term[ic, isun])
            + gleaf_et[ic, isha] * (dqsat_arr[ic, isha] * delta[ic, isha] + qsat_term[ic, isha])
        )

        if ic == n:
            c2_ic = jnp.zeros(())
            d2_ic = d2_ic + gac_ic * (_eref_p / _pref_p)
        if ic == 1:
            a2_ic = jnp.zeros(())
            b21_ic = b21_ic - gs0 * _rhg_p * dqsat0 * alpha0
            b22_ic = b22_ic - gs0 * _rhg_p * dqsat0 * beta0
            d2_ic = d2_ic + gs0 * _rhg_p * (qsat0 + dqsat0 * (delta0 - _tg_bef_p))

        a1 = a1.at[ic].set(a1_ic)
        b11 = b11.at[ic].set(b11_ic)
        b12 = b12.at[ic].set(b12_ic)
        c1 = c1.at[ic].set(c1_ic)
        d1 = d1.at[ic].set(d1_ic)
        a2 = a2.at[ic].set(a2_ic)
        b21 = b21.at[ic].set(b21_ic)
        b22 = b22.at[ic].set(b22_ic)
        c2 = c2.at[ic].set(c2_ic)
        d2 = d2.at[ic].set(d2_ic)

    # ------------------------------------------------------------------
    # Tridiagonal solve for tair and eair (mol/mol) — Fortran lines 327-329
    # ------------------------------------------------------------------
    tair_p, eair_p = tridiag_2eq(
        a1[1 : n + 1],
        b11[1 : n + 1],
        b12[1 : n + 1],
        c1[1 : n + 1],
        d1[1 : n + 1],
        a2[1 : n + 1],
        b21[1 : n + 1],
        b22[1 : n + 1],
        c2[1 : n + 1],
        d2[1 : n + 1],
        n,
    )
    tair = tair.at[p, 1 : n + 1].set(jnp.stack(tair_p))
    eair = eair.at[p, 1 : n + 1].set(jnp.stack(eair_p))

    # ------------------------------------------------------------------
    # Soil surface temperature and vapor pressure — Fortran lines 332-333
    # ------------------------------------------------------------------
    _tair_solved = tair[p]  # JAX array, shape (nlevmlcan+1,)
    _eair_solved = eair[p]

    t0 = alpha0 * _tair_solved[1] + beta0 * _eair_solved[1] + delta0
    e0 = _rhg_p * (qsat0 + dqsat0 * (t0 - _tg_bef_p))

    # ------------------------------------------------------------------
    # Leaf temperature from implicit solution — Fortran lines 336-340
    # ------------------------------------------------------------------
    tleaf_implic = jnp.zeros((nlevmlcan + 1, nleaf + 1))
    for ic in range(1, n + 1):
        tleaf_implic = tleaf_implic.at[ic, isun].set(
            alpha[ic, isun] * _tair_solved[ic] + beta[ic, isun] * _eair_solved[ic] + delta[ic, isun]
        )
        tleaf_implic = tleaf_implic.at[ic, isha].set(
            alpha[ic, isha] * _tair_solved[ic] + beta[ic, isha] * _eair_solved[ic] + delta[ic, isha]
        )

    # ------------------------------------------------------------------
    # Convert water vapor from mol/mol to Pa — Fortran lines 343-346
    # ------------------------------------------------------------------
    eair = eair.at[p, 1 : n + 1].set(eair[p, 1 : n + 1] * _pref_p)
    e0 = e0 * _pref_p

    mlcanopy_inst = mlcanopy_inst._replace(
        tair_profile=tair,
        eair_profile=eair,
    )

    # ------------------------------------------------------------------
    # Leaf fluxes (per unit leaf area) — Fortran lines 352-354
    # Inlined from LeafFluxes; all inputs are JAX arrays
    # ------------------------------------------------------------------
    # eair in Pa
    _eair_lf_p = _eair_solved.at[1 : n + 1].set(_eair_solved[1 : n + 1] * _pref_p)

    _tleaf_new = jnp.zeros((nlevmlcan + 1, nleaf + 1))
    _stleaf_new = jnp.zeros((nlevmlcan + 1, nleaf + 1))
    _shleaf_new = jnp.zeros((nlevmlcan + 1, nleaf + 1))
    _lhleaf_new = jnp.zeros((nlevmlcan + 1, nleaf + 1))
    _evleaf_new = jnp.zeros((nlevmlcan + 1, nleaf + 1))
    _trleaf_new = jnp.zeros((nlevmlcan + 1, nleaf + 1))

    for ic in range(1, n + 1):
        tair_ic = _tair_solved[ic]
        eair_ic = _eair_lf_p[ic]
        active_ic = _dpai_p[ic] > 0.0
        for il in [isun, isha]:
            cpleaf_ic = _cpleaf_p[ic]
            fwet_ic = _fwet_p[ic]
            fdry_ic = _fdry_p[ic]
            gbh_ic = _gbh_p[ic, il]
            gbv_ic = _gbv_p[ic, il]
            gs_ic = _gs_p[ic, il]
            rnleaf_ic = _rnleaf_p[ic, il]
            tleaf_bef_ic = _tleaf_bef_p[ic, il]

            esat_l, desat_l = SatVap(tleaf_bef_ic)
            qsat_lf = esat_l / _pref_p
            dqsat_lf = desat_l / _pref_p

            # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
            gs_gbv_denom = jnp.maximum(gs_ic + gbv_ic, 1.0e-30)
            gleaf = gs_ic * gbv_ic / gs_gbv_denom
            gw = gleaf * fdry_ic + gbv_ic * fwet_ic

            num1 = 2.0 * _cpair_p * gbh_ic
            num2 = lambda_ * gw
            num3 = (
                rnleaf_ic
                - lambda_ * gw * (qsat_lf - dqsat_lf * tleaf_bef_ic)
                + cpleaf_ic / dtime * tleaf_bef_ic
            )
            # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
            den_lf = jnp.maximum(cpleaf_ic / dtime + num1 + num2 * dqsat_lf, 1.0e-30)
            tleaf_active = (num1 * tair_ic + num2 * (eair_ic / _pref_p) + num3) / den_lf
            tleaf_val = jnp.where(active_ic, tleaf_active, tair_ic)

            stleaf_val = jnp.where(active_ic, (tleaf_val - tleaf_bef_ic) * cpleaf_ic / dtime, 0.0)
            shleaf_val = jnp.where(active_ic, 2.0 * _cpair_p * (tleaf_val - tair_ic) * gbh_ic, 0.0)

            num1_flux = qsat_lf + dqsat_lf * (tleaf_val - tleaf_bef_ic) - eair_ic / _pref_p
            trleaf_val = jnp.where(active_ic, gleaf * fdry_ic * num1_flux, 0.0)
            evleaf_val = jnp.where(active_ic, gbv_ic * fwet_ic * num1_flux, 0.0)
            lhleaf_val = jnp.where(active_ic, (trleaf_val + evleaf_val) * lambda_, 0.0)

            err = jnp.where(active_ic, rnleaf_ic - shleaf_val - lhleaf_val - stleaf_val, 0.0)
            # JIT-compatible diagnostic — callback runs host-side with concrete value
            jax.debug.callback(
                lambda e: (
                    endrun(msg=" ERROR: LeafFluxes: energy balance error")
                    if abs(float(e)) > 1.0e-3
                    else None
                ),
                err,
            )

            _tleaf_new = _tleaf_new.at[ic, il].set(tleaf_val)
            _stleaf_new = _stleaf_new.at[ic, il].set(stleaf_val)
            _shleaf_new = _shleaf_new.at[ic, il].set(shleaf_val)
            _lhleaf_new = _lhleaf_new.at[ic, il].set(lhleaf_val)
            _evleaf_new = _evleaf_new.at[ic, il].set(evleaf_val)
            _trleaf_new = _trleaf_new.at[ic, il].set(trleaf_val)

    # Batch write-back: 6 bulk operations
    _sl_lf = slice(1, n + 1)
    mlcanopy_inst = mlcanopy_inst._replace(
        tleaf_leaf=mlcanopy_inst.tleaf_leaf.at[p, _sl_lf, :].set(_tleaf_new[_sl_lf, :]),
        stleaf_leaf=mlcanopy_inst.stleaf_leaf.at[p, _sl_lf, :].set(_stleaf_new[_sl_lf, :]),
        shleaf_leaf=mlcanopy_inst.shleaf_leaf.at[p, _sl_lf, :].set(_shleaf_new[_sl_lf, :]),
        lhleaf_leaf=mlcanopy_inst.lhleaf_leaf.at[p, _sl_lf, :].set(_lhleaf_new[_sl_lf, :]),
        evleaf_leaf=mlcanopy_inst.evleaf_leaf.at[p, _sl_lf, :].set(_evleaf_new[_sl_lf, :]),
        trleaf_leaf=mlcanopy_inst.trleaf_leaf.at[p, _sl_lf, :].set(_trleaf_new[_sl_lf, :]),
    )

    # ------------------------------------------------------------------
    # Soil fluxes — Fortran lines 358-360
    # ------------------------------------------------------------------
    mlcanopy_inst = SoilFluxes(p, mlcanopy_inst)

    # Re-read updated profiles (shair/etair/stair updated below)
    shair = mlcanopy_inst.shair_profile
    etair = mlcanopy_inst.etair_profile
    stair = mlcanopy_inst.stair_profile
    # tair/eair were not modified by LeafFluxes or SoilFluxes; reuse numpy
    _eair_final = _eair_lf_p
    _tair_final = _tair_solved

    # ------------------------------------------------------------------
    # Vertical sensible heat and water vapor fluxes between layers
    # Fortran lines 363-370
    # ------------------------------------------------------------------
    # tair[ic+1] for ic=1..n-1, then thref at ic=n
    _tair_next_inner = _tair_final[2 : n + 2]  # length n
    _tair_next_inner = _tair_next_inner.at[n - 1].set(_thref_p)
    _eair_next_inner = _eair_final[2 : n + 2]
    _eair_next_inner = _eair_next_inner.at[n - 1].set(_eref_p)

    _shair_new = jnp.zeros(nlevmlcan + 1)
    _etair_new = jnp.zeros(nlevmlcan + 1)
    _shair_new = _shair_new.at[1 : n + 1].set(
        -_cpair_p * (_tair_next_inner - _tair_final[1 : n + 1]) * _gac_p[1 : n + 1]
    )
    _etair_new = _etair_new.at[1 : n + 1].set(
        -(_eair_next_inner - _eair_final[1 : n + 1]) / _pref_p * _gac_p[1 : n + 1]
    )

    # ------------------------------------------------------------------
    # Canopy air storage flux — Fortran lines 373-379
    # ------------------------------------------------------------------
    storage_sh = jnp.zeros(nlevmlcan + 1)
    storage_et = jnp.zeros(nlevmlcan + 1)
    storage_sh = storage_sh.at[1 : n + 1].set(
        _cpair_p * (_tair_final[1 : n + 1] - _tair_bef_p[1 : n + 1]) * rho_dz_over_dt[1 : n + 1]
    )
    storage_et = storage_et.at[1 : n + 1].set(
        (_eair_final[1 : n + 1] - _eair_bef_p[1 : n + 1]) / _pref_p * rho_dz_over_dt[1 : n + 1]
    )
    _stair_new = jnp.zeros(nlevmlcan + 1)
    _stair_new = _stair_new.at[1 : n + 1].set(
        storage_sh[1 : n + 1] + storage_et[1 : n + 1] * lambda_
    )

    # Batch write-back
    _sl = slice(1, n + 1)
    mlcanopy_inst = mlcanopy_inst._replace(
        shair_profile=shair.at[p, _sl].set(_shair_new[_sl]),
        etair_profile=etair.at[p, _sl].set(_etair_new[_sl]),
        stair_profile=stair.at[p, _sl].set(_stair_new[_sl]),
    )

    # ------------------------------------------------------------------
    # Source fluxes from implicit solution — Fortran lines 387-398
    # ------------------------------------------------------------------
    shsrc = jnp.zeros(nlevmlcan + 1)
    etsrc = jnp.zeros(nlevmlcan + 1)
    stveg = jnp.zeros(nlevmlcan + 1)

    for ic in range(1, n + 1):
        active_ic = _dpai_p[ic] > 0.0
        shsrc_ic = jnp.zeros(())
        etsrc_ic = jnp.zeros(())
        stveg_ic = jnp.zeros(())
        for il in [isun, isha]:
            esat_l, desat_l = SatVap(_tleaf_bef_p[ic, il])
            sh_il = _cpair_p * (tleaf_implic[ic, il] - _tair_final[ic]) * gleaf_sh[ic, il]
            et_il = (
                (esat_l + desat_l * (tleaf_implic[ic, il] - _tleaf_bef_p[ic, il]) - _eair_final[ic])
                / _pref_p
                * gleaf_et[ic, il]
            )
            stv_il = heatcap[ic, il] * (tleaf_implic[ic, il] - _tleaf_bef_p[ic, il]) / dtime
            shsrc_ic = shsrc_ic + jnp.where(active_ic, sh_il, 0.0)
            etsrc_ic = etsrc_ic + jnp.where(active_ic, et_il, 0.0)
            stveg_ic = stveg_ic + jnp.where(active_ic, stv_il, 0.0)
        shsrc = shsrc.at[ic].set(shsrc_ic)
        etsrc = etsrc.at[ic].set(etsrc_ic)
        stveg = stveg.at[ic].set(stveg_ic)

    # Soil fluxes from implicit solution — Fortran lines 401-403
    sh0 = -_cpair_p * (_tair_final[1] - t0) * _gac0_p
    et0 = -(_eair_final[1] - e0) / _pref_p * gs0
    g0 = -_soil_tk_p / _soil_dz_p * _soil_t_p + _soil_tk_p / _soil_dz_p * t0

    # Return updated state + auxiliary data for error checks (run outside JIT)
    aux = (
        lambda_,
        shsrc,
        etsrc,
        stveg,
        tleaf_implic,
        sh0,
        et0,
        g0,
        t0,
        e0,
        avail_energy,
        storage_sh,
        storage_et,
    )
    return mlcanopy_inst, aux


def ImplicitFluxProfileSolution(
    p: int,
    mlcanopy_inst: mlcanopy_type,
    n: int = None,
) -> mlcanopy_type:
    """
    Implicit solution for source/sink fluxes and concentration profiles.

    Wrapper that:

    1. Extracts ``n = int(ncan[p])`` as a concrete Python int *before* the
       JIT boundary so it can serve as a static loop bound inside the kernel.
    2. Calls :func:`_implicit_fps_jit` (JIT-compiled with ``n`` and ``p``
       as static arguments).
    3. Optionally runs :func:`ErrorCheck01` and :func:`ErrorCheck02` when
       ``DEBUG_FPS_CHECKS`` is ``True``.  These checks are disabled by
       default because each call requires 9 ``np.asarray()`` host-device
       syncs that dominate the cost of the flux-profile step.
    """
    if n is None:
        n = int(mlcanopy_inst.ncan_canopy[p])  # concrete before JIT boundary
    mlcanopy_inst, aux = _implicit_fps_jit(p, n, mlcanopy_inst)
    if DEBUG_FPS_CHECKS:
        (
            lambda_,
            shsrc,
            etsrc,
            stveg,
            tleaf_implic,
            sh0,
            et0,
            g0,
            t0,
            e0,
            avail_energy,
            storage_sh,
            storage_et,
        ) = aux
        mlcanopy_inst = ErrorCheck01(
            p,
            lambda_,
            shsrc,
            etsrc,
            stveg,
            tleaf_implic,
            sh0,
            et0,
            g0,
            t0,
            e0,
            mlcanopy_inst,
        )
        mlcanopy_inst = ErrorCheck02(
            p,
            lambda_,
            avail_energy,
            shsrc,
            etsrc,
            stveg,
            storage_sh,
            storage_et,
            sh0,
            et0,
            g0,
            mlcanopy_inst,
        )
    return mlcanopy_inst


# ---------------------------------------------------------------------------
# Private: implicit solution error checks
# ---------------------------------------------------------------------------


def ErrorCheck01(
    p: int,
    lambda_: float,
    shsrc: Array,
    etsrc: Array,
    stveg: Array,
    tleaf_implic: Array,
    sh0: float,
    et0: float,
    g0: float,
    t0: float,
    e0: float,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Compare the implicit solution with ``LeafFluxes`` and ``SoilFluxes``.

    Mirrors Fortran subroutine ``ErrorCheck01`` (lines 315-390).

    Args:
        p: Patch index.
        lambda_: Latent heat of vaporization (J/mol).
        shsrc: Implicit-solution canopy-layer leaf SH flux (W/m2),
            shape ``(nlevmlcan,)``.
        etsrc: Implicit-solution canopy-layer leaf ET flux
            (mol H2O/m2/s), shape ``(nlevmlcan,)``.
        stveg: Implicit-solution canopy-layer leaf storage flux (W/m2),
            shape ``(nlevmlcan,)``.
        tleaf_implic: Implicit-solution leaf temperature (K),
            shape ``(nlevmlcan, nleaf)``.
        sh0: Implicit-solution ground SH flux (W/m2).
        et0: Implicit-solution ground ET flux (mol H2O/m2/s).
        g0: Implicit-solution soil heat flux (W/m2).
        t0: Implicit-solution soil surface temperature (K).
        e0: Implicit-solution soil surface vapor pressure (Pa).
        mlcanopy_inst: Multilayer canopy state container.

    Returns:
        Unchanged ``mlcanopy_inst`` (or aborts on error).
    """
    n = int(mlcanopy_inst.ncan_canopy[p])

    # Pre-extract all per-layer arrays as numpy — 7 syncs total vs ~120 in the loop
    _dpai = np.asarray(mlcanopy_inst.dpai_profile[p])
    _fracsun = np.asarray(mlcanopy_inst.fracsun_profile[p])
    _shleaf = np.asarray(mlcanopy_inst.shleaf_leaf[p])  # (nlevmlcan+1, nleaf+1)
    _trleaf = np.asarray(mlcanopy_inst.trleaf_leaf[p])
    _evleaf = np.asarray(mlcanopy_inst.evleaf_leaf[p])
    _stleaf = np.asarray(mlcanopy_inst.stleaf_leaf[p])
    _tleaf = np.asarray(mlcanopy_inst.tleaf_leaf[p])

    # Compare leaf fluxes — Fortran lines 358-383
    for ic in range(1, n + 1):
        if _dpai[ic] > 0.0:

            fs_ic = float(_fracsun[ic])
            dpai_ic = float(_dpai[ic])

            # Layer fluxes from LeafFluxes — Fortran lines 361-366
            shsrc_leaf = (
                float(_shleaf[ic, isun]) * fs_ic + float(_shleaf[ic, isha]) * (1.0 - fs_ic)
            ) * dpai_ic

            etsrc_leaf = (
                (float(_trleaf[ic, isun]) + float(_evleaf[ic, isun])) * fs_ic
                + (float(_trleaf[ic, isha]) + float(_evleaf[ic, isha])) * (1.0 - fs_ic)
            ) * dpai_ic

            stveg_leaf = (
                float(_stleaf[ic, isun]) * fs_ic + float(_stleaf[ic, isha]) * (1.0 - fs_ic)
            ) * dpai_ic

            # Error checks — Fortran lines 369-383
            if abs(shsrc[ic] - shsrc_leaf) > 0.001:
                endrun(msg=" ERROR: ImplicitFluxProfileSolution: Leaf sensible heat flux error")

            if abs(lambda_ * (etsrc[ic] - etsrc_leaf)) > 0.001:
                endrun(msg=" ERROR: ImplicitFluxProfileSolution: Leaf latent heat flux error")

            if abs(stveg[ic] - stveg_leaf) > 0.001:
                endrun(msg=" ERROR: ImplicitFluxProfileSolution: Leaf heat storage error")

            if abs(float(_tleaf[ic, isun]) - tleaf_implic[ic, isun]) > 1.0e-6:
                endrun(msg=" ERROR: ImplicitFluxProfileSolution: Leaf temperature error (sunlit)")

            if abs(float(_tleaf[ic, isha]) - tleaf_implic[ic, isha]) > 1.0e-6:
                endrun(msg=" ERROR: ImplicitFluxProfileSolution: Leaf temperature error (shaded)")

    # Compare soil fluxes — Fortran lines 386-398
    if abs(float(mlcanopy_inst.shsoi_soil[p]) - sh0) > 0.001:
        endrun(msg=" ERROR: ImplicitFluxProfileSolution: Soil sensible heat flux error")

    if abs(lambda_ * (float(mlcanopy_inst.etsoi_soil[p]) - et0)) > 0.001:
        endrun(msg=" ERROR: ImplicitFluxProfileSolution: Soil latent heat flux error")

    if abs(float(mlcanopy_inst.gsoi_soil[p]) - g0) > 0.001:
        endrun(msg=" ERROR: ImplicitFluxProfileSolution: Soil heat flux error")

    if abs(float(mlcanopy_inst.tg_soil[p]) - t0) > 1.0e-6:
        endrun(msg=" ERROR: ImplicitFluxProfileSolution: Soil surface temperature error")

    if abs(float(mlcanopy_inst.eg_soil[p]) - e0) > 1.0e-6:
        endrun(msg=" ERROR: ImplicitFluxProfileSolution: Soil surface vapor pressure error")

    return mlcanopy_inst


# ---------------------------------------------------------------------------
# Private: conservation error checks
# ---------------------------------------------------------------------------


def ErrorCheck02(
    p: int,
    lambda_: float,
    avail_energy: Array,
    shsrc: Array,
    etsrc: Array,
    stveg: Array,
    storage_sh: Array,
    storage_et: Array,
    sh0: float,
    et0: float,
    g0: float,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Conservation error checks for the implicit flux-profile solution.

    Mirrors Fortran subroutine ``ErrorCheck02`` (lines 392-470).

    Args:
        p: Patch index.
        lambda_: Latent heat of vaporization (J/mol).
        avail_energy: Available energy per leaf type (W/m2),
            shape ``(nlevmlcan, nleaf)``.
        shsrc: Canopy-layer leaf SH flux (W/m2), shape ``(nlevmlcan,)``.
        etsrc: Canopy-layer leaf ET flux (mol H2O/m2/s),
            shape ``(nlevmlcan,)``.
        stveg: Canopy-layer leaf storage flux (W/m2),
            shape ``(nlevmlcan,)``.
        storage_sh: Air SH storage flux (W/m2), shape ``(nlevmlcan,)``.
        storage_et: Air ET storage flux (mol H2O/m2/s),
            shape ``(nlevmlcan,)``.
        sh0: Ground SH flux (W/m2).
        et0: Ground ET flux (mol H2O/m2/s).
        g0: Soil heat flux (W/m2).
        mlcanopy_inst: Multilayer canopy state container.

    Returns:
        Unchanged ``mlcanopy_inst`` (or aborts on error).
    """
    n = int(mlcanopy_inst.ncan_canopy[p])
    ntop_p = int(mlcanopy_inst.ntop_canopy[p])

    # Pre-extract shair/etair as numpy — 2 syncs vs 4n in the loops below
    _shair = np.asarray(mlcanopy_inst.shair_profile[p])
    _etair = np.asarray(mlcanopy_inst.etair_profile[p])

    # Vegetation flux energy balance — Fortran lines 431-436
    for ic in range(1, n + 1):
        err = (
            avail_energy[ic, isun]
            + avail_energy[ic, isha]
            - shsrc[ic]
            - lambda_ * etsrc[ic]
            - stveg[ic]
        )
        if abs(err) > 0.001:
            endrun(msg=" ERROR: ImplicitFluxProfileSolution: Leaf energy balance error")

    # Flux conservation at each layer — Fortran lines 439-455
    for ic in range(1, n + 1):
        if ic == 1:
            err = storage_sh[ic] - (sh0 + shsrc[ic] - float(_shair[ic]))
        else:
            err = storage_sh[ic] - (float(_shair[ic - 1]) + shsrc[ic] - float(_shair[ic]))
        if abs(err) > 0.001:
            endrun(
                msg=" ERROR: ImplicitFluxProfileSolution: Sensible heat layer conservation error"
            )

        if ic == 1:
            err = storage_et[ic] - (et0 + etsrc[ic] - float(_etair[ic]))
        else:
            err = storage_et[ic] - (float(_etair[ic - 1]) + etsrc[ic] - float(_etair[ic]))
        err = err * lambda_
        if abs(err) > 0.001:
            endrun(msg=" ERROR: ImplicitFluxProfileSolution: Latent heat layer conservation error")

    # Canopy sensible heat conservation — Fortran lines 458-465
    sum_src_sh = float(np.sum(shsrc[1 : ntop_p + 1]))
    sum_storage_sh = float(np.sum(storage_sh[1 : ntop_p + 1]))
    err = (sh0 + sum_src_sh - sum_storage_sh) - float(_shair[ntop_p])
    if abs(err) > 0.001:
        endrun(msg=" ERROR: ImplicitFluxProfileSolution: Sensible heat canopy conservation error")

    # Canopy latent heat conservation — Fortran lines 468-475
    sum_src_et = float(np.sum(etsrc[1 : ntop_p + 1]))
    sum_storage_et = float(np.sum(storage_et[1 : ntop_p + 1]))
    err = ((et0 + sum_src_et - sum_storage_et) - float(_etair[ntop_p])) * lambda_
    if abs(err) > 0.001:
        endrun(msg=" ERROR: ImplicitFluxProfileSolution: Latent heat canopy conservation error")

    # Ground energy balance — Fortran lines 478-481
    err = float(mlcanopy_inst.rnsoi_soil[p]) - sh0 - lambda_ * et0 - g0
    if abs(err) > 0.001:
        endrun(msg=" ERROR: ImplicitFluxProfileSolution: Ground temperature energy balance error")

    return mlcanopy_inst


# ---------------------------------------------------------------------------
# Private: well-mixed canopy assumption
# ---------------------------------------------------------------------------


def WellMixed(
    num_filter: int,
    filter: Array,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Set canopy scalar profiles to reference-height values (well-mixed
    assumption) or read them from a dataset.

    Mirrors Fortran subroutine ``WellMixed`` (lines 484-580).

    Args:
        num_filter: Number of patches in the filter.
        filter: 1-D array of patch indices.
        mlcanopy_inst: Multilayer canopy state container.

    Returns:
        Updated :class:`mlcanopy_type` with profiles set and leaf/soil
        fluxes computed.
    """
    # Unpack inputs
    uref = mlcanopy_inst.uref_forcing
    tref = mlcanopy_inst.tref_forcing
    eref = mlcanopy_inst.eref_forcing
    co2ref = mlcanopy_inst.co2ref_forcing
    qref = mlcanopy_inst.qref_forcing
    ncan = mlcanopy_inst.ncan_canopy
    wind_data = mlcanopy_inst.wind_data_profile
    tair_data = mlcanopy_inst.tair_data_profile
    eair_data = mlcanopy_inst.eair_data_profile

    uaf = mlcanopy_inst.uaf_canopy
    taf = mlcanopy_inst.taf_canopy
    qaf = mlcanopy_inst.qaf_canopy
    gac0 = mlcanopy_inst.gac0_soil
    wind = mlcanopy_inst.wind_profile
    tair = mlcanopy_inst.tair_profile
    eair = mlcanopy_inst.eair_profile
    cair = mlcanopy_inst.cair_profile
    gac = mlcanopy_inst.gac_profile
    shair = mlcanopy_inst.shair_profile
    etair = mlcanopy_inst.etair_profile
    stair = mlcanopy_inst.stair_profile
    mflx = mlcanopy_inst.mflx_profile

    for fp in range(num_filter):
        p = int(filter[fp])
        n = int(ncan[p])

        for ic in range(n):

            # CO2 profile — Fortran line 535
            cair = cair.at[p, ic].set(co2ref[p])

            # Scalar profiles — Fortran lines 538-554
            if flux_profile_type == 0:
                # Well-mixed assumption
                wind = wind.at[p, ic].set(uref[p])
                tair = tair.at[p, ic].set(tref[p])
                eair = eair.at[p, ic].set(eref[p])
            elif flux_profile_type == -1:
                # Dataset profiles
                wind = wind.at[p, ic].set(wind_data[p, ic])
                tair = tair.at[p, ic].set(tair_data[p, ic])
                eair = eair.at[p, ic].set(eair_data[p, ic])

            # Vertical fluxes set to zero — Fortran lines 557-560
            shair = shair.at[p, ic].set(0.0)
            etair = etair.at[p, ic].set(0.0)
            stair = stair.at[p, ic].set(0.0)
            mflx = mflx.at[p, ic].set(0.0)

            # Update state before calling leaf flux routines
            mlcanopy_inst = mlcanopy_inst._replace(
                wind_profile=wind,
                tair_profile=tair,
                eair_profile=eair,
                cair_profile=cair,
                shair_profile=shair,
                etair_profile=etair,
                stair_profile=stair,
                mflx_profile=mflx,
            )

            # Leaf fluxes — Fortran lines 563-564
            mlcanopy_inst = LeafFluxes(p, ic, isun, mlcanopy_inst)
            mlcanopy_inst = LeafFluxes(p, ic, isha, mlcanopy_inst)

            # Re-read any fields written by LeafFluxes
            wind = mlcanopy_inst.wind_profile
            tair = mlcanopy_inst.tair_profile
            eair = mlcanopy_inst.eair_profile
            shair = mlcanopy_inst.shair_profile
            etair = mlcanopy_inst.etair_profile
            stair = mlcanopy_inst.stair_profile
            mflx = mlcanopy_inst.mflx_profile

        # Soil conductance: large resistance -> small soil fluxes — Fortran lines 568-569
        # gac0 = (1/100) * 42.3  mol/m2/s
        gac0 = gac0.at[p].set((1.0 / 100.0) * 42.3)
        mlcanopy_inst = mlcanopy_inst._replace(gac0_soil=gac0)
        mlcanopy_inst = SoilFluxes(p, mlcanopy_inst)

        # Output-only fields — Fortran lines 573-578
        uaf = uaf.at[p].set(uref[p])
        taf = taf.at[p].set(tref[p])
        qaf = qaf.at[p].set(qref[p])

        gac0 = mlcanopy_inst.gac0_soil
        gac = mlcanopy_inst.gac_profile

        for ic in range(n):
            gac = gac.at[p, ic].set((1.0 / 10.0) * 42.3)  # small non-zero resistance

        mlcanopy_inst = mlcanopy_inst._replace(
            uaf_canopy=uaf,
            taf_canopy=taf,
            qaf_canopy=qaf,
            gac_profile=gac,
        )

    return mlcanopy_inst
