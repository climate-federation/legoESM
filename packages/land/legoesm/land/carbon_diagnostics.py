"""Reconstruct the DifferLand carbon-flux breakdown for one coupled land step.

The coupled multi-layer land step (:func:`step_multilayer_land`) returns only
the net CO2 flux, not the GPP/NPP/respiration/allocation/turnover breakdown.
This helper re-derives that breakdown from the END-OF-STEP soil state — the
model's own end-of-step surface temperature and root-zone ``beta_soil`` — by
re-evaluating the same carbon step with diagnostics on.  It is the single
source of truth used by BOTH the land-carbon equilibrium validator and the
``run_lmip`` semi-analytic spin-up (so the two never re-derive this glue).

Leaf module: it depends on ``multilayer_land`` (beta helper), ``stomata_utils``
(effective-beta / GPP), and ``carbon.carbon_cycle`` (the diagnostic carbon
step); nothing in those imports this module, so there is no import cycle.

The reconstruction is faithful to a few percent — the coupled model computes
GPP inside its surface-energy solve (``surface_out.gpp``, at its own iterated
surface temperature), whereas this re-derives GPP post-hoc via
``compute_effective_beta`` (the SimpleSEB + Farquhar/Ball-Berry stomata path).
It is exact enough for the mean-annual slow-pool fluxes the semi-analytic
spin-up needs (Xia et al. 2012); the model's pools and net CO2 flux remain the
authoritative quantities.  NOTE: for the ``TwoLeafCanopy`` / CLM-ML canopy
surface schemes the model's GPP comes from the canopy solver, so this SimpleSEB
re-derivation is only an approximation there — use it with the SimpleSEB
surface scheme (the ``run_lmip`` / validator default) for the tightest match.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.land.carbon.carbon_cycle import step_carbon_differland
from legoesm.land.multilayer_land import root_zone_beta_soil
from legoesm.land.stomata_utils import compute_effective_beta


def reconstruct_carbon_diagnostics(
    new_state,
    forcing,
    carbon_state,
    config,
    root_frac,
    theta_wp,
    theta_fc,
    beta_min: float,
    lat: jnp.ndarray,
    doy,
    dt: float,
    spatial: bool = False,
):
    """Return the :class:`CarbonDiagnostics` for one coupled land step.

    Parameters
    ----------
    new_state : MultiLayerLandState
        The END-OF-STEP land state (provides surface T and soil moisture).
    forcing : AtmToSurface
        The atmospheric forcing used for this step.
    carbon_state : CarbonState
        The carbon pools at the START of the step (as the model's carbon step
        consumes them).
    config : MultiLayerLandConfig
        Land config (``.stomata`` / ``.carbon`` sub-configs).
    root_frac, theta_wp, theta_fc, beta_min :
        Root-zone parameters for :func:`root_zone_beta_soil` (same values the
        model used).
    lat, doy, dt :
        Latitude [rad], day-of-year, timestep [s].
    spatial : bool
        Whether ``theta_wp``/``theta_fc`` are per-column arrays.
    """
    T_sfc_new = new_state.T_soil[:, 0]
    beta_soil_new, _ = root_zone_beta_soil(
        new_state.theta_soil, root_frac, theta_wp, theta_fc, beta_min,
        spatial=spatial)
    _, gpp_override, _ = compute_effective_beta(
        T_sfc_new, forcing, beta_soil_new, config, carbon_state, dt)
    _, _, diag = step_carbon_differland(
        carbon_state, forcing.sw_down, T_sfc_new, forcing.co2_ppmv,
        beta_soil_new, lat, doy, forcing.precip_total, config.carbon, dt,
        gpp_override=gpp_override, return_diagnostics=True)
    return diag
