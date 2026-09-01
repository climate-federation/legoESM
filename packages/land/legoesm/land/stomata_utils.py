"""Stomatal / carbon dispatch utilities.

Shared by the slab land and multi-layer land models when running with
the **SimpleSEB** surface scheme.  Computes an effective moisture
availability ``beta`` — starting from the bucket / root-zone-weighted
soil beta — and optionally down-regulates it via Jarvis (carbon off)
or coupled Leuning Farquhar + Ball-Berry / Medlyn (carbon on).

The TwoLeafCanopy surface scheme bypasses this module and uses its own
Newton closure in ``canopy/solver.py`` — it does not need a beta proxy
because LE is computed directly from leaf ↔ canopy-air humidity gradients.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.canopy.sif import leaf_sif
from legoesm.land.carbon.config import CarbonState
from legoesm.land.stomata import (
    compute_stomatal_beta,
    jarvis_gs,
    solve_coupled_farquhar_ci,
)


def compute_effective_beta(
    T_sfc: jnp.ndarray,
    forcing: AtmToSurface,
    beta_soil: jnp.ndarray,
    config,
    carbon_state: CarbonState | None,
    dt: float,
    land_params=None,
    pmodel_acclim=None,
    phydro_supply=None,
) -> tuple[jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None]:
    """Compute effective moisture availability beta, with optional stomatal/carbon coupling.

    When stomatal conductance is disabled, returns ``beta_soil`` directly.
    When enabled with the differland carbon scheme, uses the coupled
    Leuning Farquhar-stomata solver.  Otherwise falls back to the Jarvis
    multiplicative model.

    Parameters
    ----------
    T_sfc : jnp.ndarray
        Surface temperature [K].
    forcing : AtmToSurface
        Atmospheric forcing fields (provides sw_down, co2_ppmv,
        q_lowest, p_surface).
    beta_soil : jnp.ndarray
        Bucket / root-zone soil moisture beta [0-1].
    config : LandConfig or MultiLayerLandConfig
        Land configuration (must have ``stomata`` and ``carbon`` attributes).
    carbon_state : CarbonState or None
        Current carbon pools (needed for differland LAI).
    dt : float
        Time step [s] (reserved for future use).
    land_params : LandSurfaceParams or None
        Spatially-varying parameters.  When provided, overrides
        ``config.stomata.Vc_max25``, ``config.stomata.g1_bb``,
        and ``config.carbon.LCMA`` with per-cell arrays.

    Returns
    -------
    (beta, gpp_farq, sif)
        Effective moisture factor [0-1], Farquhar GPP [gC/m2/s] (or None),
        and observed top-of-canopy SIF [umol/m2/s] (or None — only the coupled
        Farquhar path with ``stomata.sif`` set produces it; the Jarvis fallback
        has no Ci/An to invert).
    """
    # Fail-early validation of the static switch fields (typos, inconsistent
    # P-model combinations, switches on disabled stomata) — the config is
    # static, so this runs at trace time, not per step.
    config.stomata.validate()

    # Override stomatal / carbon config fields with spatial arrays if provided.
    # ``_fC4`` (per-column C4 area fraction) drives the canonical-FvCB C3/C4 blend
    # in the coupled solver; it is a per-cell PFT-derived field, so it is carried
    # on ``land_params`` (not a scalar config knob). A column with no C4 data
    # (land_params is None, or its fC4 is None on the prescribed-PFT path) runs
    # pure C3 — the correct default, and identical to the pre-FvCB behaviour.
    if land_params is not None:
        _stomata = config.stomata._replace(
            Vc_max25=land_params.Vc_max25,
            g1_bb=land_params.g1,
        )
        _carbon = config.carbon._replace(LCMA=land_params.LCMA)
        _fC4 = land_params.fC4 if land_params.fC4 is not None else 0.0
    else:
        _stomata = config.stomata
        _carbon = config.carbon
        _fC4 = 0.0

    gpp_farq = None
    sif = None

    # --- P-model optimality parameter source (big-leaf full set) ---
    # Acclimated Vcmax25 replaces the prescribed capacity, the predicted
    # Medlyn slope replaces g1_med (when selected), and the coordination
    # rjv25 + acclimated growth temperature thread into the FvCB solve.
    _pm_on = (config.stomata.capacity_scheme == "p_model"
              or config.stomata.g1_source == "p_model")
    _rjv25 = None
    _tgc_c = None
    if _pm_on:
        if not (config.carbon.scheme == "differland"
                and carbon_state is not None and config.stomata.enabled):
            raise ValueError(
                "P-model big-leaf switches require the coupled Farquhar "
                "path (stomata.enabled with carbon.scheme='differland' and "
                "a carbon state); the Jarvis fallback has no Vcmax/g1 for "
                "the P model to supply.")
        from legoesm.land.p_model import (
            acclimated_capacities, acclimated_capacities_c4)
        _phydro_on = getattr(
            config, "transpiration_stress", "beta_theta") == "phydro"
        if _phydro_on:
            if phydro_supply is None:
                raise ValueError(
                    "transpiration_stress='phydro' but no PhydroSupply was "
                    "threaded to the big-leaf stomata (phydro_supply=None).")
            from legoesm.land.phydro import phydro_optimum, slope_for_model
            from legoesm.land.p_model import optimal_chi as _ochi
            _hcaps = phydro_optimum(
                pmodel_acclim, phydro_supply, config.phydro, config.p_model)
            _, _, _, _gs_pa_h, _ = _ochi(
                pmodel_acclim.t_mean_K, pmodel_acclim.vpd_mean_pa,
                pmodel_acclim.co2_mean_ppm, pmodel_acclim.ps_ema,
                config.p_model)
            _caps = None
        else:
            _caps = acclimated_capacities(pmodel_acclim, config.p_model)
        _caps4 = acclimated_capacities_c4(pmodel_acclim, config.p_model)
        # The big-leaf kernel shares ONE Vcmax25 (and one Medlyn slope)
        # across its C3/C4 branches and area-blends the RATES by fC4; in
        # production fC4 is the dominant-PFT 0/1 flag, so the linear
        # capacity blend below is EXACT at both endpoints and inherits the
        # kernel's own single-capacity approximation for intermediate fC4
        # (codex design review).
        _vc3 = _hcaps.vcmax25_leaf if _phydro_on else _caps.vcmax25_leaf
        _rjv_src = _hcaps.rjv25 if _phydro_on else _caps.rjv25
        if config.stomata.capacity_scheme == "p_model":
            _stomata = _stomata._replace(
                Vc_max25=((1.0 - _fC4) * _vc3
                          + _fC4 * _caps4.vcmax25_c4_leaf))
            _rjv25 = _rjv_src  # consumed by the C3 branch only
            _tgc_c = pmodel_acclim.t_mean_K - constants.T_freeze
        if config.stomata.g1_source == "p_model":
            if _phydro_on:
                # Slope for the ACTIVE model from the profit optimum's chi;
                # blended with the C4 least-cost slope as before.  Note the
                # slope slot per model: g1_med (medlyn), g1_bb (ball_berry),
                # a1_leuning (leuning).
                _slope3 = slope_for_model(
                    config.stomata.stomata_model, _hcaps.chi, pmodel_acclim,
                    _gs_pa_h,
                    d0_leuning_kpa=config.stomata.d0_leuning_kpa)
                _sm = config.stomata.stomata_model
                if _sm == "medlyn":
                    _stomata = _stomata._replace(
                        g1_med=((1.0 - _fC4) * _slope3
                                + _fC4 * _caps4.g1_c4_kpa))
                elif _sm == "ball_berry":
                    _stomata = _stomata._replace(g1_bb=_slope3)
                else:  # leuning (validate() restricts the set)
                    _stomata = _stomata._replace(a1_leuning=_slope3)
            else:
                _stomata = _stomata._replace(
                    g1_med=((1.0 - _fC4) * _caps.g1_kpa
                            + _fC4 * _caps4.g1_c4_kpa))

    if config.stomata.enabled:
        if config.carbon.scheme == "differland" and carbon_state is not None:
            LAI = carbon_state.C_fol / _carbon.LCMA
            # Under phydro the water limitation is inside the optimum: the
            # solver's beta_soil capacity fold is neutralised (ones) so the
            # stress is not double-counted; beta_theta keeps the legacy fold.
            _beta_for_solver = (jnp.ones_like(beta_soil)
                                if _pm_on and getattr(
                                    config, "transpiration_stress",
                                    "beta_theta") == "phydro"
                                else beta_soil)
            leaf = solve_coupled_farquhar_ci(
                T_sfc, forcing.sw_down, forcing.co2_ppmv,
                forcing.q_lowest, forcing.p_surface, LAI, _beta_for_solver,
                _stomata, _fC4, rjv25=_rjv25, TgC_C=_tgc_c)
            gs, gpp_farq = leaf.gs, leaf.gpp
            beta = compute_stomatal_beta(
                gs, LAI, beta_soil, _stomata)
            # SIF from the same solve (big-leaf).  Static gate on _stomata.sif.
            # fesc clamped to [0,1] (a probability) so a hand-set / perturbed
            # config can't produce negative or amplified SIF. C3-style electron-
            # transport inversion (a documented approximation for the C4-blended
            # fraction, whose SIF physics differ), consistent with the two-leaf
            # canopy's passive SIF diagnostic.
            if _stomata.sif is not None:
                fesc = jnp.clip(_stomata.sif.escape_probability, 0.0, 1.0)
                sif = leaf_sif(
                    leaf.A_net, leaf.Ci, leaf.gamma_star, leaf.APAR_umol,
                    _stomata.sif) * fesc
        else:
            gs = jarvis_gs(
                T_sfc, forcing.sw_down, forcing.q_lowest,
                forcing.p_surface, beta_soil, _stomata)
            beta = compute_stomatal_beta(
                gs, None, beta_soil, _stomata)
    else:
        beta = beta_soil

    return beta, gpp_farq, sif
